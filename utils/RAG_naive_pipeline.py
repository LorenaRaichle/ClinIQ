# RAG_pipeline.py

import os
import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import INDEX_NAME, EMBEDDING_MODEL_NAME

from tqdm import tqdm


class RAGPipeline:
    def __init__(self, test_data, full_data, pubmed_data=None,  k=5, pc=None, embedding_pipe=None, model_pipeline=None):
        self.test_data = test_data
        self.full_data = full_data
        self.pubmed_data = pubmed_data or {}
        self.k = k
        self.pc = pc
        self.embedding_pipe = embedding_pipe or HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.model_pipeline = model_pipeline
        self.index = self.pc.Index(INDEX_NAME)
        self.retriever = self._get_retriever()
        self.chain = self._build_qa_chain()

    def _load_deepseek_model(self):
        model_id = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)


    def _get_prompt(self):
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Context:
            {context}
            
            {question}
            """.strip()
                    )

    # base retriever
    def _get_retriever(self):
        vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embedding_pipe,
            text_key="page_content"
        )
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.k})


    def _build_qa_chain(self):
        if self.model_pipeline is None:
            print("Loading model into memory.")
            pipe = self._load_deepseek_model()
            self.model_pipeline = HuggingFacePipeline(pipeline=pipe)
        else:
            print("Using shared model pipeline.")


        llm = self.model_pipeline
        prompt = self._get_prompt()

        def log_and_format(inputs, prompt):
            content = []

            for doc in inputs['context']:
                doc_id = doc.metadata['id']

                try:
                    prefix, idx_str = doc_id.split("_", 1)
                    if prefix in {"mc", "sa", "tf", "mh"}:
                        idx = int(idx_str)

                        if prefix == "mc":
                            if idx >= len(self.full_data["multiple_choice"]):
                                raise IndexError(f"Index {idx} out of range for multiple_choice")
                            example = self.full_data["multiple_choice"][idx]
                            content.append(example.get('question', '') + " " + example.get('correct_answer', ''))

                        elif prefix == "sa":
                            if idx >= len(self.full_data["short_answer"]):
                                raise IndexError(f"Index {idx} out of range for short_answer")
                            example = self.full_data["short_answer"][idx]
                            content.append(example.get('question', ''))

                        elif prefix == "tf":
                            if idx >= len(self.full_data["true_false"]):
                                raise IndexError(f"Index {idx} out of range for true_false")
                            example = self.full_data["true_false"][idx]
                            content.append(example.get('question', ''))

                        elif prefix == "mh":
                            if idx >= len(self.full_data["multi_hop"]):
                                raise IndexError(f"Index {idx} out of range for multi_hop")
                            example = self.full_data["multi_hop"][idx]
                            content.append(example.get('question', ''))

                    elif prefix == "pubmed":
                        doc = self.pubmed_data.get(doc_id)
                        if doc:
                            content.append(doc.get("content", ""))
                        else:
                            print(f"[Warning] pubmed ID '{doc_id}' not found in pubmed_data.")

                    else:
                        print(f"[Warning] Unrecognized ID prefix in: {doc_id}")

                except Exception as e:
                    print(f"[Warning] Skipping doc with id '{doc_id}': {e}")
                    continue

            if not content:
                print("[Warning] No content extracted from retrieved docs.")

            inputs['context'] = "\n".join(content)
            return prompt.format(**inputs)

        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | RunnableLambda(lambda x: log_and_format(x, self._get_prompt()))
            | llm
            | StrOutputParser()
        )

    def run(self, question_data):
        prompt = question_data['prompt']
        question = question_data['question']
        gold_answer = question_data['answer']
        generated_answer = self.chain.invoke(prompt)

        return {
            "question": question,
            "true_answer": gold_answer,
            "generated_answer": generated_answer
        }

    def batch_run(self, dataset_slice, output_path):
        results = []
        for example in tqdm(dataset_slice, desc="Generating answers"):
            result = self.run(example)
            results.append(result)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved {len(results)} results to {output_path}")



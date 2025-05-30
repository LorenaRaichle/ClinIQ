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
    def __init__(self, data, k=5, pc=None, embedding_pipe=None):
        self.data = data
        self.k = k
        self.pc = pc
        self.embedding_pipe = embedding_pipe or HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.model_pipeline = None
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
            pipe = self._load_deepseek_model()
            self.model_pipeline = HuggingFacePipeline(pipeline=pipe)

        llm = self.model_pipeline
        prompt = self._get_prompt()

        def log_and_format(inputs, prompt):
            content = []


            id_index = {ex["id"]: ex for ex in self.data}

            for doc in inputs['context']:
                doc_id = doc.metadata['id']
                example = id_index.get(doc_id)

                if not example:
                    raise KeyError(f"No example found with id '{doc_id}'")

                qtype = example["type"]

                if qtype == "multiple_choice":
                    content.append(example.get('question', '') + " " + example.get('correct_answer', ''))
                elif qtype in ["true_false", "short_answer", "multi_hop"]:
                    content.append(example.get('question', ''))

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

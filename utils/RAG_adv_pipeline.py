# RAG_adv_pipeline.py

import os
from peft import PeftModel
from typing import Any
import torch
from langchain_core.runnables.config import RunnableConfig
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.llms import HuggingFacePipeline
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
# from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
import pinecone
from langchain.schema import Document
from config import INDEX_NAME, EMBEDDING_MODEL_NAME
from utils.RAG_metadata import extract_keywords_and_entities, extract_age_gender, categorize_age
from config import GENERATION_CONFIGS
from tqdm import tqdm


class RAGAdvPipeline:
    def __init__(self, test_data, full_data, pubmed_data=None,  k=5, pc=None, embedding_pipe=None, model_pipeline=None, question_type=None):
        self.test_data = test_data
        self.full_data = full_data
        self.pubmed_data = pubmed_data or {}
        self.k = k
        self.pc = pc
        self.embedding_pipe = embedding_pipe or HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.model_pipeline = model_pipeline
        self.index = self.pc.Index(INDEX_NAME)
        self.question_type = question_type
        self.retriever = self.get_hybrid_retriever()
        self.chain = self.build_qa_chain()

    def get_prompt(self):
        template = """
        Context:
        {context}

        {question}
        Please answer with the correct letter (A, B, C, D, or E).

        Answer:
        """

        return PromptTemplate(
            input_variables=["context", "question"],
            template=template.strip()
        )



    def get_hybrid_retriever(self):
        class HybridRetriever(Runnable):
            def __init__(self, index, embedder):
                self.index = index
                self.embedder = embedder

            def invoke(self, input: Any, config: RunnableConfig | None = None):
                question_text = input if isinstance(input, str) else input.get("question", "")
                return self.get_relevant_documents(question_text)

            def get_relevant_documents(self, query: str, k: int = 5):
                # extracting metadata from query
                question_text = query if isinstance(query, str) else query.get("question", "")
                kw, diseases, symptoms, procedures = extract_keywords_and_entities(question_text)
                ages, gender = extract_age_gender(question_text)

                # print(f"\nQuery: {query!r}")
                # print("  → keywords:  ", kw)
                # print("  → diseases:  ", diseases)
                # print("  → symptoms:  ", symptoms)
                # print("  → procedures:", procedures)
                # print("  → ages:      ", ages)
                # print("  → gender:    ", gender)

                # Pinecone filter only on relevant metadata fields
                flt = {}
                if diseases:   flt["diseases"] = {"$in": diseases}
                if symptoms:   flt["symptoms"] = {"$in": symptoms}
                if procedures: flt["procedures"] = {"$in": procedures}
                if kw:         flt["keywords"] = {"$in": kw}
                if ages:       flt["age"] = {"$in": ages}
                if gender:     flt["gender"] = {"$eq": gender}

                # query embedding
                vec = self.embedder.embed_query(query)

                # call Pinecone with vector + filter
                filtered_resp = self.index.query(
                    vector=vec,
                    top_k=k,
                    filter=flt,
                    include_metadata=True
                )
                # search without filter, only vector search
                unfiltered_resp = self.index.query(
                    vector=vec,
                    top_k=k,
                    include_metadata=True
                )

                # collect unique results (ids) from both queries: with and without metadata
                seen_ids = set()
                docs = []

                def add_matches(matches):
                    for match in matches:
                        if match.id not in seen_ids:
                            seen_ids.add(match.id)
                            docs.append(Document(
                                page_content="", metadata=match.metadata or {"id": match.id}
                            ))

                add_matches(filtered_resp.matches)
                add_matches(unfiltered_resp.matches)

                return docs[:k]

        return HybridRetriever(self.index, self.embedding_pipe)



    # Build QA chain
    def build_qa_chain(self):
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline

        if self.model_pipeline is None:
            # BASELINE
            # model_id = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
            # tokenizer = AutoTokenizer.from_pretrained(model_id)
            # model = AutoModelForCausalLM.from_pretrained(model_id,
            #                                             device_map="auto",
            #                                             torch_dtype=torch.float16)

            # FINETUNED
            # check models/ft_multiple_choice_v2_balanced_and_shuffled-20250531T154625Z-1-001.zip
           ##
            model_path = "/content/drive/MyDrive/LORENA/RAG/DS/ft_multiple_choice_v2_balanced_and_shuffled"
           # model_path = "/content/drive/MyDrive/NLP/03_Training/ft_v21_balanced/ft_multiple_choice_v2_balanced_and_shuffled"

            base_model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"

            adapter_path = "/content/drive/MyDrive/LORENA/RAG/DS/ft_multiple_choice_v2_balanced_and_shuffled"

            # Load base model & tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)

            model = PeftModel.from_pretrained(base_model, adapter_path)

            gen_config = GENERATION_CONFIGS.get(self.question_type, {"max_new_tokens": 100, "temperature": 0.7})

            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                            max_new_tokens=gen_config["max_new_tokens"],
                            temperature=gen_config["temperature"],
                            top_p=0.9,
                            top_k=50,
                            do_sample=True
                            )

            self.model_pipeline = HuggingFacePipeline(pipeline=pipe)

        llm = self.model_pipeline
        prompt = self.get_prompt()

        def log_and_format(inputs, prompt):
            # Extracting ids from the retrieved contexts and retrieving context from files
            input_str = inputs['context']
            content = []

            sources = set()
            for split, split_data in self.full_data.items():
                for entry in split_data:
                    sources.add(entry['source'])

            sources = dict.fromkeys(sources, 0)
            sources['PubMed'] = 0


            pubmed_data_dict = {doc["id"]: doc for doc in self.pubmed_data}

            for string in input_str:
                id = string.metadata['id']
                # print("ID von context: ", id)
                if id.startswith("mc"):
                    doc = self.full_data['multiple_choice'][int(id[3:])]
                    content.append(doc.get('question', '') + " " + doc.get('answer', ''))
                    source = doc['source']
                    sources[source] += 1

                elif id.startswith("sa"):
                    doc = self.full_data['short_answer'][int(id[3:])]
                    content.append(doc.get('question', '') + " " + doc.get('answer', ''))
                    source = doc['source']
                    sources[source] += 1

                elif id.startswith("tf"):
                    doc = self.full_data['true_false'][int(id[3:])]
                    content.append(doc.get('question', '') + " " + doc.get('answer', ''))
                    source = doc['source']
                    sources[source] += 1

                elif id.startswith("mh"):
                    doc = self.full_data['multi_hop'][int(id[3:])]
                    content.append(doc.get('question', '') + " " + doc.get('answer', ''))
                    source = doc['source']
                    sources[source] += 1

                elif id.startswith("pubmed"):
                    doc = pubmed_data_dict[id]["content"]
                    content.append(doc)
                    source = "PubMed"
                    sources[source] += 1

            content = {
                'role': 'system',
                'content': "\n".join(content)
            }
            if not content:
                print("Warning: No content extracted!")
            inputs['context'] = content['content']

            rendered = prompt.format(**inputs)
            return rendered


        chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | RunnableLambda(lambda x: log_and_format(x, prompt))
                | llm
                | StrOutputParser()
        )

        return chain




def run_rag_pipeline(question_data, full_data, pubmed_data, pinecone_client, question_type):
    rag_pipeline = RAGAdvPipeline(
        test_data=[question_data],
        full_data=full_data,
        pubmed_data=pubmed_data,
        pc=pinecone_client,
        question_type=question_type
    )

    qa_chain = rag_pipeline.chain
    question = question_data["question"]
    gold_answer = question_data["answer"]

    generated_answer = qa_chain.invoke({"question": question})

    return {
        "question": question,
        "true_answer": gold_answer,
        "generated_answer": generated_answer
    }

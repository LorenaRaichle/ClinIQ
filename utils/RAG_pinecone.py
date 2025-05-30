# RAG_pinecone.py

import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from config import INDEX_NAME, DIMENSION, EMBEDDING_MODEL_NAME
from .RAG_metadata import extract_keywords_and_entities, extract_age_gender


load_dotenv()
pinecone_key = os.getenv("PINECONE")
pc = Pinecone(api_key=pinecone_key, environment="us-west1-gcp")


def init_index(pc):
    """
    Initialize Pinecone index using passed Pinecone client.
    Input: pc (Pinecone client instance)
    Output: Pinecone Index object
    """
    indices = [index["name"] for index in pc.list_indexes()]
    if INDEX_NAME not in indices:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{INDEX_NAME}' created.")
    else:
        print(f"ℹ Index '{INDEX_NAME}' already exists.")
    return pc.Index(INDEX_NAME)



class RAGIndexer:
    """Handles encoding and upserting QA data into Pinecone.

    Input: Pinecone index, optional embedding model name
    Output: None (methods operate on the index directly)
    """

    def __init__(self, index, embedding_model_name=EMBEDDING_MODEL_NAME):
        """Initialize with Pinecone index and embedding model."""
        self.index = index
        self.model = SentenceTransformer(embedding_model_name)

    def insert_questions(self, data: dict):
        """Encodes and inserts questions + metadata into Pinecone.

        Input: dict (keys: QA types like 'multiple_choice', values: list of QA dicts)
        Output: None (data is uploaded to Pinecone index)
        """
        vectors_to_upsert = []
        id_counter = 0

        for qtype, questions in data.items():
            for item in tqdm(questions, desc=f"Vectorizing '{qtype}'", unit="q"):
                question = item.get("question")
                answer = item.get("correct_answer" if qtype == "multiple_choice" else "answer")
                reasoning = None
                if qtype == "multi_hop":
                    reasoning = "\n".join(item.get("reasoning", []))

                if not question or not answer:
                    continue

                text = f"{question} Answer: {answer}"
                if reasoning:
                    text += f"\nReasoning: {reasoning}"

                vector = self.model.encode(text)

                #  Metadata extracted using functions defined in utils/RAG_metadata.py
                keywords, diseases, symptoms, procedures = extract_keywords_and_entities(question + " " + answer)
                age, gender = extract_age_gender(question + " " + answer)

                metadata = {
                    "id": item.get("id"),
                    "age": age or [],
                    "gender": gender or [],
                    "keywords": keywords or [],
                    "diseases": diseases or [],
                    "symptoms": symptoms or [],
                    "procedures": procedures or [],
                    "page_content": item.get("id")
                }

                vectors_to_upsert.append((str(id_counter), vector.tolist(), metadata))
                id_counter += 1

        # Batch upsert
        for i in tqdm(range(0, len(vectors_to_upsert), 100), desc="Upserting"):
            batch = vectors_to_upsert[i:i + 100]
            self.index.upsert(vectors=batch)

        print(f"Upserted {len(vectors_to_upsert)} items to Pinecone.")



    def describe_index(self):
        """Returns basic index statistics like vector count and metadata configuration."""
        stats = self.index.describe_index_stats()
        print(stats)
        return stats
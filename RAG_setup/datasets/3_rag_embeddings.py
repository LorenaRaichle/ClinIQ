from langchain_community.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
import os
import json
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


# Load environment variables and embedding model
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
PG_CONN_STRING = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
engine = create_engine(PG_CONN_STRING)


def fetch_rows():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, question, answer, reasoning, options, question_type FROM rag_questions"))
        return result.fetchall()

# def fetch_rows(limit=20):
#     with engine.connect() as conn:
#         result = conn.execute(text(f"""
#             SELECT id, question, answer, reasoning, options, question_type
#             FROM rag_questions
#             LIMIT {limit}
#         """))
#         return result.fetchall()


# Convert questions into Document chunks
def build_chunks(rows):
    docs = []
    for row in tqdm(rows, desc="Building chunks"):
        qtype = row.question_type
        base_text = row.question or ""

        if qtype in ("short_answer", "true_false") and row.answer:
            base_text += "\nAnswer:" + row.answer

        elif qtype == "multiple_choice":
            if row.answer:
                base_text += f"\nAnswer: {row.answer}"

        elif qtype == "multi_hop":
            base_text += f"\n{row.answer or ''}"
            if row.reasoning:
                base_text += "\nAnswer:" + row.reasoning.strip()

        docs.append(Document(
            page_content=base_text.strip(),
            metadata={"question_id": row.id, "type": qtype}
        ))
    return docs

# Store embeddings into rag_chunks
def store_embeddings(docs, batch_size=1000):
    print(f"Storing {len(docs)} documents into rag_chunks in batches...")

    for i in tqdm(range(0, len(docs), batch_size), desc="Storing batches"):
        batch = docs[i:i + batch_size]
        PGVector.from_documents(
            documents=batch,
            embedding=embedding,
            collection_name="rag_trainingset",
            connection_string=PG_CONN_STRING
        )

    print("Embeddings inserted.")



if __name__ == "__main__":
    rows = fetch_rows()
    docs = build_chunks(rows)
    store_embeddings(docs)
    print("Embeddings stored in rag_trainingset.")

# preview of transforming rag_questions content into vector store (rag_chunks) + embeddings

from langchain_community.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

from sqlalchemy import create_engine, text
import os
import json

from dotenv import load_dotenv
load_dotenv()




embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
PG_CONN_STRING = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
engine = create_engine(PG_CONN_STRING)

# 1. Fetch 2 sample rows per type
def fetch_rows():
    query = """
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY question_type ORDER BY id) AS rn
            FROM rag_questions
        ) AS sub
        WHERE rn BETWEEN 18 AND 20;
        """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.fetchall()

# 2. Build chunks
def build_chunks(rows):
    docs = []
    for row in rows:
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


# 3. Preview
if __name__ == "__main__":
    rows = fetch_rows()
    docs = build_chunks(rows)

    print("Previewing document content for embedding:\n")
    for doc in docs:
        print(f"--- Question ID: {doc.metadata['question_id']} | Type: {doc.metadata['type']} ---")
        print(doc.page_content)
        print("\n")

# filling rag_questions DB with preprocessed training set specific for RAG: train_dataset_rag.json

import json
import psycopg2
from psycopg2.extras import Json
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
)

cursor = conn.cursor()

with open("train_dataset_rag.json", "r") as f:
    data = json.load(f)

def insert_question(question, answer, reasoning, options, qtype, source):
    cursor.execute(
        """
        INSERT INTO rag_questions (question, answer, reasoning, options, question_type, source)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (question, answer, reasoning, Json(options) if options else None, qtype, source)
    )

for qtype, questions in data.items():
    for item in questions:
        question = item.get("question")
        source = item.get("source")

        if qtype == "multiple_choice":
            answer = item.get("correct_answer")
            # Skip multiple_choice if no correct_answer (NONE types have been filtered (none of the above,...)
            if not answer:
                continue
        else:
            answer = item.get("answer")

        reasoning = "\n".join(item.get("reasoning", [])) if qtype == "multi_hop" else None
        options = item.get("options") if qtype == "multiple_choice" else None

        insert_question(question, answer, reasoning, options, qtype, source)

conn.commit()
cursor.close()
conn.close()
print("Data inserted into rag_questions successfully.")

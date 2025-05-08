import psycopg2
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

cursor.execute("SELECT COUNT(*) FROM rag_questions;")
total_rows = cursor.fetchone()[0]


cursor.execute("""
    SELECT question_type, COUNT(*) 
    FROM rag_questions 
    GROUP BY question_type 
    ORDER BY question_type;
""")
type_counts = cursor.fetchall()

print(f"Total rows in rag_questions: {total_rows}")
print("Count by question type:")
for qtype, count in type_counts:
    print(f" - {qtype}: {count}")

cursor.close()
conn.close()

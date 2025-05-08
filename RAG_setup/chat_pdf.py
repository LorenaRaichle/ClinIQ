# chat_pdf.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging
from sqlalchemy import create_engine, text


set_debug(True)
set_verbose(True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from dotenv import load_dotenv
import os

load_dotenv()

PG_CONN_STRING = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"



class QA:
    """ question answering using RAG."""

    def __init__(self, llm_model: str = "deepseek-r1:latest", embedding_model: str = "mxbai-embed-large"):
        """
        InitializeLLM and embedding model.
        """
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        logger.info(f"🔍 Initializing ChatPDF with LLM: {llm_model} and Embeddings: {embedding_model}")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.
            Context:
            {context}
            
            Question:
            {question}
            
            Answer concisely and accurately in three sentences or less.
            """
        )
        self.vector_store = None
        self.retriever = None

    # def ingest(self, pdf_file_path: str):
    #     """
    #     Ingest a PDF file, split its contents, and store the embeddings in the vector store.
    #     """
    #     logger.info(f"Starting ingestion for file: {pdf_file_path}")
    #     docs = PyPDFLoader(file_path=pdf_file_path).load()
    #     chunks = self.text_splitter.split_documents(docs)
    #     chunks = filter_complex_metadata(chunks)
    #
    #     self.vector_store = Chroma.from_documents(
    #         documents=chunks,
    #         embedding=self.embeddings,
    #         persist_directory="chroma_db",
    #     )
    #     logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None


    def test_pg_connection(self):
        try:
            engine = create_engine(PG_CONN_STRING)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                print(" Connected to Postgres:")
                print(result.fetchone()[0])
        except Exception as e:
            print("Failed to connect to Postgres:")
            print(e)

    # def ask_multiple_choice_question(self, question_data: dict):
    #     """
    #     Ask the LLM a multiple-choice question using the structured input format.
    #     """
    #     question_text = question_data["question"]
    #     options = question_data["options"]
    #
    #     # Format the options cleanly
    #     options_formatted = "\n".join([f"{key}. {value}" for key, value in options.items()])
    #
    #     # Final prompt passed to the model
    #     full_question = (
    #         f"{question_text}\n\n"
    #         f"Options:\n{options_formatted}\n\n"
    #         f"Please select the best answer (A-E) and explain briefly."
    #     )
    #
    #     formatted_input = {
    #         "context": "",  # Can be filled with retrieved context later
    #         "question": full_question,
    #     }
    #
    #     # Build and run the RAG-style chain
    #     chain = (
    #             RunnablePassthrough()
    #             | self.prompt
    #             | self.model
    #             | StrOutputParser()
    #     )
    #
    #     logger.info("Asking LLM a multiple-choice question.")
    #     try:
    #         output = chain.invoke(formatted_input)
    #         print("LLM Output:\n", output)
    #         return output
    #     except Exception as e:
    #         logger.error(" Error during LLM execution: %s", e)
    #         return "An error occurred during question processing."


if __name__ == "__main__":
    assistant = QA()
    assistant.test_pg_connection()

    # question_data = {
    #     "correct_answer": "E",
    #     "options": {
    #         "A": "Ampicillin",
    #         "B": "Ceftriaxone",
    #         "C": "Ciprofloxacin",
    #         "D": "Doxycycline",
    #         "E": "Nitrofurantoin"
    #     },
    #     "question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?",
    #     "type": "multiple_choice"
    # }
    #
    # print("\n Answering multiple choice question:")
    # print(assistant.ask_multiple_choice_question(question_data))

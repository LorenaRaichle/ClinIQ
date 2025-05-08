import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda


load_dotenv()
import logging

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_prompt():
    template = """
You are a helpful medical assistant.

Use the following retrieved context to answer the question.
If you don’t know the answer, say “I don’t know.” Do not make up answers.

Context:
{context}

Question:
{question}

Answer:"""

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template.strip()
    )


def get_retriever():
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    PG_CONN_STRING = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@" \
                     f"{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"

    vectorstore = PGVector(
        connection_string=PG_CONN_STRING,
        embedding_function=embedding,
        collection_name="rag_trainingset"
    )
    logger.info("Vector store connected.")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever


def build_qa_chain(retriever):
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        device=0,  # Apple M2 -> uses MPS
        max_new_tokens=512,
        temperature=0.2,
        do_sample=False  # match your config
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = get_prompt()

    # Add logging layer
    def log_prompt(inputs):
        rendered = prompt.format(**inputs)
        logger.info(f"\n📄 Prompt sent to model:\n{rendered}\n")
        return rendered



    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(lambda x: log_and_format(x, prompt))
            | llm
            | StrOutputParser()
    )

    def log_and_format(inputs, prompt):
        rendered = prompt.format(**inputs)
        logger.info(f"\n📄 Prompt sent to model:\n{rendered}\n")
        return rendered

    return chain


#### VERSION 1 - raw answer
# def build_qa_chain(retriever):
#     from langchain_community.llms import HuggingFacePipeline
#     from transformers import pipeline
#
#     pipe = pipeline(
#         "text-generation",
#         model="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
#         device=0,
#         max_new_tokens=512,
#         temperature=0.2
#     )
#
#     llm = HuggingFacePipeline(pipeline=pipe)
#
#     prompt = get_prompt()
#
#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#
#     return chain


def run_rag_pipeline():
    retriever = get_retriever()
    qa_chain = build_qa_chain(retriever)

    print("Ask a question (type 'exit' to quit):")
    while True:

        question = input("\n> ")
        if question.lower() in {"exit", "quit"}:
            break
        logger.info(f"Running QA chain for question: {question}")
        answer = qa_chain.invoke(question)
        print(f"\nAnswer:\n{answer}")
        logger.info(f"Answer generated.")


if __name__ == "__main__":
    run_rag_pipeline()

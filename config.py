# config.py

# Pinecone index settings
# index before presentation (400k)
#INDEX_NAME = "rag-trainset-index"

# index including improvements after final presentation (200k)
INDEX_NAME = "balanced-index"

DIMENSION = 1024

# SentenceTransformer model
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"



GENERATION_CONFIGS = {
    "mc": {"max_new_tokens": 20, "temperature": 0.1},
    "sa": {"max_new_tokens": 100, "temperature": 0.7},
    "tf": {"max_new_tokens": 20, "temperature": 0.1},
    "mh": {"max_new_tokens": 200, "temperature": 0.7},
}

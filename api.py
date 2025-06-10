from pinecone import Pinecone
from groq import Groq
import os
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "labse": SentenceTransformer("sentence-transformers/LaBSE", device=device),
    "bge-m3": SentenceTransformer("BAAI/bge-m3", device=device),
    "distiluse": SentenceTransformer("distiluse-base-multilingual-cased-v2", device=device),
}

PINECONE_API_KEY = "pcsk_3B7xYL_n353PLw3NEZ7KxFncYou4UTHMiMjh7XXGCJZCPVdcW4tcbPWs76jbyqLMrW65L"
PINECONE_ENV = "us-east-1"
labse_name = "index1"
bge_name = "bge-m3"
distiluse_name = "distiluse"

pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

index_labse = pinecone_client.Index(name=labse_name)
index_bge = pinecone_client.Index(name=bge_name)
index_distiluse = pinecone_client.Index(name=distiluse_name)

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)



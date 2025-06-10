from pinecone import Pinecone
from groq import Groq
import os
from sentence_transformers import SentenceTransformer
import torch
import streamlit as st

def get_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "labse": SentenceTransformer("sentence-transformers/LaBSE", device=device),
        "bge-m3": SentenceTransformer("BAAI/bge-m3", device=device),
        "distiluse": SentenceTransformer("distiluse-base-multilingual-cased-v2", device=device),
    }

def get_pinecone_indexes():
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
    PINECONE_ENV = "us-east-1"
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    return {
        "labse": pinecone_client.Index(name="index1"),
        "bge-m3": pinecone_client.Index(name="bge-m3"),
        "distiluse": pinecone_client.Index(name="distiluse"),
    }

def get_groq_client():
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    return Groq(api_key=GROQ_API_KEY)

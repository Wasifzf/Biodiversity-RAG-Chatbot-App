from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "labse": SentenceTransformer("sentence-transformers/LaBSE", device=device),
    "bge-m3": SentenceTransformer("BAAI/bge-m3", device=device),
    "distiluse": SentenceTransformer("distiluse-base-multilingual-cased-v2", device=device),
}

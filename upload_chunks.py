from api import upload_embeddings
from api import models, index_labse, index_bge, index_distiluse
from chunking import token_chunks_all, char_chunks_all, sentence_chunks_all, semantic_chunks_all

def upload_all_chunks(chunks, chunking_type):
    for model_name, model in models.items():
        if model_name == "labse":
            index = index_labse
        elif model_name == "bge-m3":
            index = index_bge
        elif model_name == "distiluse":
            index = index_distiluse
        else:
            print(f"⚠️ No index found for model: {model_name}, skipping...")
            continue

        upload_embeddings(chunks, model_name, model, index, chunking_type=chunking_type)

# Upload all chunk types
upload_all_chunks(token_chunks_all, chunking_type="token")
upload_all_chunks(char_chunks_all, chunking_type="char")
upload_all_chunks(sentence_chunks_all, chunking_type="sentence")
upload_all_chunks(semantic_chunks_all, chunking_type="semantic")

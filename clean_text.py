import re
import unicodedata
from langdetect import detect

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def preprocess_documents(documents, filename):
    cleaned_docs = []
    for i, doc in enumerate(documents):
        text = clean_text(doc.page_content)
        try:
            lang = detect(text)
        except:
            lang = "unknown"

        doc.page_content = text
        doc.metadata["language"] = lang
        doc.metadata["source"] = filename
        doc.metadata["chunk_index"] = i

        if "page" not in doc.metadata:
            doc.metadata["page"] = i

        cleaned_docs.append(doc)
    return cleaned_docs

def upload_embeddings(chunks, model_name, model, index, chunking_type):
    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        texts = [c.page_content for c in batch_chunks]
        embeddings = model.encode(texts, show_progress_bar=False)

        vectors = []
        for j, embedding in enumerate(embeddings):
            chunk_id = f"{model_name}_{chunking_type}_chunk_{i + j}"
            metadata = batch_chunks[j].metadata.copy()
            metadata["text"] = batch_chunks[j].page_content
            metadata["chunking"] = chunking_type
            vectors.append((chunk_id, embedding.tolist(), metadata))

        index.upsert(vectors=vectors)
        print(f"✅ Uploaded {len(vectors)} vectors to {model_name} ({chunking_type})")
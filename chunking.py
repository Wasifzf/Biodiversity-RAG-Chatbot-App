from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.document_loaders import TextLoader, PyMuPDFLoader
from clean_text import preprocess_documents
from api import models

docs = [
    '/content/drive/MyDrive/nlpkbase/end_AIforwildlife.pdf',
    '/content/drive/MyDrive/nlpkbase/eng_effectsofclimatechange.pdf',
    '/content/drive/MyDrive/nlpkbase/eng_wildlife_conservation.pdf',
    '/content/drive/MyDrive/nlpkbase/technology for wildlife.txt',
    '/content/drive/MyDrive/nlpkbase/spanish_climatechange.pdf',
    '/content/drive/MyDrive/nlpkbase/spanish_strat_wildlife.pdf',
    '/content/drive/MyDrive/nlpkbase/2020 IPBES GLOBAL REPORT (CHAPTER 2.2)_V6_SINGLE.pdf',
    '/content/drive/MyDrive/nlpkbase/climate-science-risk-solutions.pdf'
]

def semantic_splitter(documents, model, threshold=0.5, min_chunk_size=100):
    semantic_chunks = []
    for doc in documents:
        segments = re.split(r'(?<=[.?!])\s+|\n\n', doc.page_content)
        segments = [s.strip() for s in segments if s.strip()]
        if not segments:
            continue

        segment_embeddings = model.encode(segments)

        current_chunk_text = ""
        current_chunk_segments = []
        chunk_start_index = 0

        for i in range(len(segments)):
            if not current_chunk_text:
                current_chunk_text = segments[i]
                current_chunk_segments = [segments[i]]
                chunk_start_index = i
            else:
                last_segment_embedding = segment_embeddings[i-1].reshape(1, -1)
                current_segment_embedding = segment_embeddings[i].reshape(1, -1)
                similarity = cosine_similarity(last_segment_embedding, current_segment_embedding)[0][0]

                if similarity < threshold:
                    if len(" ".join(current_chunk_segments)) >= min_chunk_size:
                        chunk_content = " ".join(current_chunk_segments)
                        new_metadata = doc.metadata.copy()
                        new_metadata["chunk_index"] = f"semantic_{doc.metadata.get('page', 0)}_{chunk_start_index}"
                        new_metadata["chunking"] = "semantic"
                        semantic_chunks.append(Document(page_content=chunk_content, metadata=new_metadata))

                    current_chunk_text = segments[i]
                    current_chunk_segments = [segments[i]]
                    chunk_start_index = i
                else:
                    current_chunk_text += " " + segments[i]
                    current_chunk_segments.append(segments[i])

        if current_chunk_segments and len(" ".join(current_chunk_segments)) >= min_chunk_size:
            chunk_content = " ".join(current_chunk_segments)
            new_metadata = doc.metadata.copy()
            new_metadata["chunk_index"] = f"semantic_{doc.metadata.get('page', 0)}_{chunk_start_index}"
            new_metadata["chunking"] = "semantic"
            semantic_chunks.append(Document(page_content=chunk_content, metadata=new_metadata))

    return semantic_chunks

# ✅ Functionalized chunking strategies
def char_chunking(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def token_chunking(documents, tokens_per_chunk=256, chunk_overlap=50):
    splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=tokens_per_chunk, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def sentence_chunking(documents, chunk_size=400, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "]
    )
    return splitter.split_documents(documents)

# ✅ Processing loop
char_chunks_all = []
token_chunks_all = []
sentence_chunks_all = []
semantic_chunks_all = []

for doc_path in docs:
    filename = os.path.basename(doc_path)

    if doc_path.endswith(".txt"):
        loader = TextLoader(doc_path)
    elif doc_path.endswith(".pdf"):
        loader = PyMuPDFLoader(doc_path)
    else:
        print(f"Skipping unsupported file: {doc_path}")
        continue

    raw_docs = loader.load()
    documents = preprocess_documents(raw_docs, filename)

    char_chunks_all.extend(char_chunking(documents))
    token_chunks_all.extend(token_chunking(documents))
    sentence_chunks_all.extend(sentence_chunking(documents))
    semantic_chunks_all.extend(semantic_splitter(documents, models["bge-m3"], threshold=0.5, min_chunk_size=100))

# ✅ Summary
print("✅ Total char chunks:", len(char_chunks_all))
print("✅ Total token chunks:", len(token_chunks_all))
print("✅ Total sentence chunks:", len(sentence_chunks_all))
print("✅ Total semantic chunks:", len(semantic_chunks_all))

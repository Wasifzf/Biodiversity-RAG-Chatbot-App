from api import client
from api import models, index_labse, index_bge, index_distiluse

def search_pinecone(query, model, index, top_k=10, chunking_type=None):
    from langdetect import detect

    try:
        query_lang = detect(query)
    except:
        query_lang = "unknown"

    query_embedding = model.encode([query], convert_to_numpy=True)[0]

    filter = {"language": {"$eq": query_lang}}
    if chunking_type:
        filter["chunking"] = {"$eq": chunking_type}

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=filter
    )

    for i, match in enumerate(results['matches']):
        print(f"\n🔹 Match #{i+1}")
        print(f"Score: {match['score']:.4f}")
        print(f"Language: {match['metadata'].get('language', 'unknown')}")
        print(f"Chunking: {match['metadata'].get('chunking', 'N/A')}")
        print(f"Text: {match['metadata']['text']}")

    return results

def generate_answer_groq(matches, query):
    context = "\n\n".join([match['metadata']['text'] for match in matches])

    system_prompt = """You are a highly knowledgeable AI assistant specializing in environmental science, biodiversity, and AI for conservation. 
You respond like an expert science communicator — clear, professional, and reader-friendly. 

When generating answers:
- Organize responses with **headings** (e.g., 🌍 Overview, 📉 Impacts).
- **Bold important terms** like climate change, biodiversity, or habitat loss.
- Avoid bullet points unless listing clear examples.
- Write 2–4 sentence paragraphs to improve readability.
- Synthesize content from different sources, avoid redundancy.
- If there’s not enough relevant context, say so and politely request the user to clarify or rephrase."""

    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content


query = "How is AI used in conservation to support biodiversity and protect endangered species?"
query_embedding = models["bge-m3"].encode(query, normalize_embeddings=True)

results = search_pinecone(query, model=models["bge-m3"], index=index_bge, top_k=10, chunking_type=None)
answer = generate_answer_groq(results['matches'], query)
print("💡 Answer:", answer)
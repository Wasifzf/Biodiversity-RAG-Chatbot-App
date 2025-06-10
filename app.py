import streamlit as st
from api import models, index_labse, index_bge, index_distiluse, client
from langdetect import detect

# --- Utility functions ---

def get_index(model_name):
    return {
        "labse": index_labse,
        "bge-m3": index_bge,
        "distiluse": index_distiluse
    }.get(model_name)

def search_pinecone(query, model_name, chunking_type=None, top_k=10):
    model = models[model_name]
    index = get_index(model_name)

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
    return results


def generate_answer_groq(matches, query):
    context = "\n\n".join([match['metadata']['text'] for match in matches])

    system_prompt = (
        "You are an expert environmental researcher. Based on the following context extracted "
        "from scientific papers, provide a clear, well-structured, and thoughtful answer to the question below. "
        "Make your answer engaging and informative, using **headings**, **bold emphasis**, and **paragraphs** "
        "where appropriate to help with readability. Avoid bullet points unless strictly necessary. "
        "Synthesize information across sources like a skilled human researcher."
    )

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    return response.choices[0].message.content


# --- Streamlit App ---

st.set_page_config(page_title="AI for Conservation Q&A", layout="wide")
st.title("🌿 AI for Biodiversity & Conservation")
st.write("Ask a question related to **biodiversity**, **climate change**, or **AI in conservation**.")

query = st.text_area("🔍 Enter your question:", height=100)

model_name = st.selectbox("🧠 Choose an embedding model:", ["bge-m3", "labse", "distiluse"])
chunking_type = st.selectbox("📦 Select chunking method:", [None, "token", "char", "sentence", "semantic"])
top_k = st.slider("📈 Top K matches to retrieve:", min_value=1, max_value=10, value=10)

if st.button("💬 Get Answer"):
    if query.strip() == "":
        st.warning("❗ Please enter a question.")
    else:
        with st.spinner("🔎 Retrieving relevant context and generating answer..."):
            try:
                results = search_pinecone(query, model_name, chunking_type, top_k)
                matches = results['matches']

                if not matches:
                    st.warning("No relevant documents found.")
                else:
                    answer = generate_answer_groq(matches, query)
                    st.subheader("💡 Answer")
                    st.markdown(answer, unsafe_allow_html=True)

                    # Optional: Show source texts if user wants
                    with st.expander("📚 Show Retrieved Context"):
                        for i, match in enumerate(matches):
                            st.markdown(f"**Source {i+1}:** {match['metadata'].get('source', 'N/A')}")
                            st.markdown(match['metadata']['text'])

            except Exception as e:
                st.error(f"❌ An error occurred:\n\n{str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for biodiversity and climate research.")

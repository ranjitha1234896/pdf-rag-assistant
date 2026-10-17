import streamlit as st
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.title("📄 AskMyPDF — PDF Q&A Assistant")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def get_chroma_client():
    return chromadb.Client()


model = load_embedding_model()
client = get_chroma_client()
gemini_model = genai.GenerativeModel("gemini-3.1-flash-lite")


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunk_pages(pdf):
    chunks = []
    pages = []
    for page_number, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        for chunk in chunk_text(text):
            chunks.append(chunk)
            pages.append(page_number)
    return chunks, pages


def build_collection(chunks, pages):
    # Delete the old collection so the previous PDF's chunks are gone
    try:
        client.delete_collection(name="pdf_chunks")
    except Exception:
        pass  # nothing to delete the first time

    collection = client.create_collection(name="pdf_chunks")
    embeddings = model.encode(chunks)
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"page": p} for p in pages]
    )
    return collection


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Only process the PDF again when a NEW file is uploaded
    if st.session_state.get("current_file") != uploaded_file.file_id:
        with pdfplumber.open(uploaded_file) as pdf:
            chunks, pages = chunk_pages(pdf)

        if len(chunks) == 0:
            st.session_state["current_file"] = None
            st.session_state["num_chunks"] = 0
            st.error("No text found in this PDF. It may be image-based (scanned).")
        else:
            build_collection(chunks, pages)
            st.session_state["current_file"] = uploaded_file.file_id
            st.session_state["num_chunks"] = len(chunks)

    if st.session_state.get("num_chunks", 0) > 0:
        st.success(f"Extracted and split into {st.session_state['num_chunks']} chunks.")

        collection = client.get_collection(name="pdf_chunks")
        question = st.text_input("Ask a question about this PDF:")

        if st.button("Get Answer") and question:
            question_embedding = model.encode([question])
            results = collection.query(
                query_embeddings=question_embedding.tolist(),
                n_results=min(11, collection.count())
            )
            context = "\n\n".join(results['documents'][0])
            with st.expander("🔍 Retrieved chunks (debug)"):
                for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    st.write(f"**Chunk {i+1} (Page {meta['page']}):** {doc}...")

            prompt = f"""Answer the question based only on the context below.
If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

            response = gemini_model.generate_content(prompt)
            st.subheader("Answer:")
            st.write(response.text)
            top_docs = results['documents'][0][:3]
            top_metas = results['metadatas'][0][:3]

            pages_used = sorted(set(m['page'] for m in top_metas))
            st.caption("📄 Sources: Page " + ", ".join(str(p) for p in pages_used))

            with st.expander("📄 Sources used (top 3 matches)"):
                for doc, meta in zip(top_docs, top_metas):
                    st.markdown(f"**Page {meta['page']}**")
                    st.write(doc)
                    st.divider()
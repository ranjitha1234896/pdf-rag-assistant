import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="pdf_chunks")
gemini_model = genai.GenerativeModel("gemini-3.8-flash")

question = input("Ask a question about your PDF: ")
question_embedding = model.encode([question])

results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=3
)

print("Question:", question)
print("\nRetrieved chunks:\n")
for i, doc in enumerate(results['documents'][0]):
    print(f"--- Chunk {i+1} ---")
    print(doc)
    print()

context = "\n\n".join(results['documents'][0])

prompt = f"""Answer the question based only on the context below.
If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

response = gemini_model.generate_content(prompt)

print("\nGemini's Answer:\n")
print(response.text)



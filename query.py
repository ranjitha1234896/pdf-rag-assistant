import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="pdf_chunks")

question = "What is this document about?"
question_embedding = model.encode([question])

results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=2
)

print("Question:", question)
print("\nMost relevant chunk:\n")
print(results['documents'][0][0])
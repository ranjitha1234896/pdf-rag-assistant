import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

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

# Step 1: Extract text from PDF
with pdfplumber.open("Sample_Resume_Aarav_Sharma.pdf") as pdf:
    full_text = ""
    for page in pdf.pages:
        full_text += page.extract_text() + "\n"

# Step 2: Chunk the text
chunks = chunk_text(full_text)
print("Number of chunks:", len(chunks))

# Step 3: Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
print("Shape of embeddings:", embeddings.shape)

# Step 4: Store in ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="pdf_chunks")

ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=ids
)

print("Stored", collection.count(), "chunks in ChromaDB")





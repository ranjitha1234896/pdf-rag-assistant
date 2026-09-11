import pdfplumber
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

with pdfplumber.open("Sample_Resume_Aarav_Sharma.pdf") as pdf:
    full_text = ""
    for page in pdf.pages:
        full_text += page.extract_text() + "\n"

chunks = chunk_text(full_text)
print("Number of chunks:", len(chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

print("Shape of embeddings:", embeddings.shape)
print("First embedding (first 10 numbers):", embeddings[0][:10])





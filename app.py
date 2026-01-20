import os
import PyPDF2
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

model = genai.GenerativeModel("gemini-1.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

faiss_index = None
text_chunks = []

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# UTILS
# =========================
def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def build_faiss(chunks):
    global faiss_index, text_chunks
    text_chunks = chunks
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings))

def retrieve_chunks(query, k=4):
    q_emb = embedder.encode([query])
    _, idx = faiss_index.search(np.array(q_emb), k)
    return "\n".join([text_chunks[i] for i in idx[0]])

# =========================
# ROUTES
# =========================
@app.route("/upload", methods=["POST"])
def upload_pdf():
    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    text = extract_text(path)
    chunks = chunk_text(text)
    build_faiss(chunks)

    return jsonify({"message": "PDF uploaded and indexed successfully"})

@app.route("/chat", methods=["POST"])
def chat():
    if faiss_index is None:
        return jsonify({"response": "Please upload a PDF first."})

    query = request.json.get("query")
    context = retrieve_chunks(query)

    prompt = f"""
Answer using ONLY the context below.
If not found, say "Not available in the document".

Context:
{context}

Question:
{query}
"""

    response = model.generate_content(prompt)
    return jsonify({"response": response.text})

# =========================
if __name__ == "__main__":
    app.run(debug=True)

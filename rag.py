import numpy as np
import re
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader


class RAGSystem:
    def __init__(self, pdf_path="heart_rag_large.pdf"):
        # Embedding model
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load PDF
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

        # Clean text
        text = re.sub(r'\s+', ' ', text)

        # Sentence-based chunking
        sentences = text.split(". ")
        self.chunks = []
        chunk = ""

        for sent in sentences:
            if len(chunk) + len(sent) < 400:
                chunk += sent + ". "
            else:
                self.chunks.append(chunk)
                chunk = sent + ". "
        if chunk:
            self.chunks.append(chunk)

        # Embeddings
        self.embeddings = self.embed_model.encode(self.chunks)

        # FAISS index
        self.index = faiss.IndexFlatL2(len(self.embeddings[0]))
        self.index.add(np.array(self.embeddings))

        # 🔥 GENERATION MODEL (ChatGPT-like)
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=200
        )

    def query(self, question):
        # Retrieve context
        q_emb = self.embed_model.encode([question])
        _, I = self.index.search(np.array(q_emb), k=5)

        context = " ".join([self.chunks[i] for i in I[0]])
        context = context[:1200]

        # 🔥 PROMPT ENGINEERING (VERY IMPORTANT)
        prompt = f"""
You are a medical assistant.

Answer the question clearly and completely using ONLY the given context.

If the answer is not in the context, say: "Answer not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.generator(prompt)[0]['generated_text']

        return response.strip()
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGSystem:
    def __init__(self, knowledge_file='security_knowledge.txt', model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.knowledge_file = knowledge_file
        self.documents = []
        self.doc_embeddings = None
        self.index = None
        self._build_index()

    def _build_index(self):
        if not os.path.exists(self.knowledge_file):
            print(f"[ERROR] {self.knowledge_file} not found.")
            return

        with open(self.knowledge_file, 'r', encoding='utf-8') as f:
            self.documents = [line.strip() for line in f if line.strip()]

        if not self.documents:
            print("[WARNING] Knowledge base is empty.")
            return

        # Compute embeddings
        self.doc_embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        dim = self.doc_embeddings.shape[1]

        # Build FAISS index
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.doc_embeddings)
        print("[INFO] FAISS index built successfully.")

    def retrieve_context(self, query, k=1):
        if self.index is None:
            return "Knowledge base not initialized."

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)
        results = [self.documents[i] for i in indices[0]]
        return results[0] if results else "No relevant context found."

# -----------------------------
# Usage Example
# -----------------------------
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    rag = RAGSystem('security_knowledge.txt')
    context = rag.retrieve_context("DDoS Attack")
    print(f"Retrieved Context: {context}")
import os
import faiss
import requests
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# RAG SYSTEM (FAISS)
# -----------------------------
class RAGSystem:
    def __init__(self, knowledge_file='security_knowledge.txt', model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.knowledge_file = knowledge_file
        self.documents = []
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

        embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        print("[INFO] FAISS index built successfully.")

    def retrieve_context(self, query, k=1):
        if self.index is None:
            return "Knowledge base not initialized."

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)

        if len(indices) > 0:
            return self.documents[indices[0][0]]

        return "No relevant context found."


# -----------------------------
# OLLAMA LLM (API)
# -----------------------------
class LLMExplainer:
    def __init__(self, model_name="tinyllama"):
        self.host = "http://localhost:11434/api/generate"
        self.model_name = model_name
        print(f"[INFO] Using Ollama model: {self.model_name}")

    def generate_explanation(self, attack_type, context):
        prompt = f"""
You are a cybersecurity expert. Explain the attack in **simple terms in 2-3 sentences**.
Include how it affects the system and 1-2 easy mitigation steps.

Attack Type: {attack_type}
Context: {context}

Answer concisely:
"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.host, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No response received from Ollama.")

        except requests.exceptions.ConnectionError:
            return "[ERROR] Ollama not running. Start it using: ollama run tinyllama"

        except requests.exceptions.Timeout:
            return "[ERROR] Ollama request timed out."

        except Exception as e:
            return f"[ERROR] {str(e)}"


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Set working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\n[STEP 1] Initializing RAG...")
    rag = RAGSystem('security_knowledge.txt')

    print("\n[STEP 2] Retrieving context...")
    attack_detected = "DDoS Attack"
    context = rag.retrieve_context(attack_detected)

    print(f"[RAG OUTPUT] {context}")

    print("\n[STEP 3] Generating explanation using Ollama...")
    explainer = LLMExplainer(model_name="tinyllama")

    explanation = explainer.generate_explanation(attack_detected, context)

    print("\n" + "="*60)
    print("FINAL LLM EXPLANATION & MITIGATION")
    print("="*60)
    print(explanation)
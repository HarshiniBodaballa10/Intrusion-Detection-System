import os
import pandas as pd
import joblib
import requests
from sentence_transformers import SentenceTransformer
import faiss

# =============================
# RAG SYSTEM
# =============================
class RAGSystem:
    def __init__(self, knowledge_file='rag/security_knowledge.txt'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_file = knowledge_file
        self.documents = []
        self.index = None
        self._build_index()

    def _build_index(self):
        if not os.path.exists(self.knowledge_file):
            print("[ERROR] Knowledge file missing.")
            return

        with open(self.knowledge_file, 'r', encoding='utf-8') as f:
            self.documents = [line.strip() for line in f if line.strip()]

        if not self.documents:
            print("[WARNING] Empty knowledge base.")
            return

        embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        print("[INFO] RAG initialized.")

    def retrieve_context(self, query, k=2):
        if self.index is None:
            return "No knowledge available."

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(query_embedding, k)

        return " ".join([self.documents[i] for i in indices[0] if i < len(self.documents)])


# =============================
# LLM EXPLAINER (SHORT OUTPUT)
# =============================
class LLMExplainer:
    def __init__(self, model_name="tinyllama"):
        self.model = model_name
        self.url = "http://localhost:11434/api/generate"

    def generate(self, attack, context):
        prompt = f"""
You are a cybersecurity expert.

Attack: {attack}

Context:
{context}

Explain briefly:
- What the attack is (1-2 lines)
- Impact (1 line)
- Prevention (2 points)
"""

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 120,
                    "stream": False
                },
                timeout=60
            )

            result = response.json().get("response", "No response")
            return result.strip()

        except requests.exceptions.ConnectionError:
            return "⚠️ Ollama not running"
        except Exception as e:
            return f"⚠️ Error: {str(e)}"


# =============================
# ATTACK DETECTOR
# =============================
class AttackDetector:
    def __init__(self):
        self.model = joblib.load("models/intrusion_model.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
        self.encoder = joblib.load("models/label_encoder.pkl")

    def preprocess(self, df):
        df = pd.get_dummies(df)

        # Fix column mismatch
        if hasattr(self.scaler, "feature_names_in_"):
            for col in self.scaler.feature_names_in_:
                if col not in df:
                    df[col] = 0

            df = df[self.scaler.feature_names_in_]

        return self.scaler.transform(df)

    def predict(self, df):
        processed = self.preprocess(df)
        preds = self.model.predict(processed)
        return self.encoder.inverse_transform(preds)


# =============================
# MAIN PIPELINE
# =============================
class DynamicPipeline:
    def __init__(self):
        print("\n[INIT] Starting system...")

        self.detector = AttackDetector()
        self.rag = RAGSystem()
        self.llm = LLMExplainer("tinyllama")

        self.seen_attacks = set()
        self.results = []

        print("[INIT] System Ready!\n")

    def run_on_dataset(self, file_path):
        if not os.path.exists(file_path):
            print("[ERROR] Dataset file not found.")
            return

        df = pd.read_csv(file_path)

        # Reduce size (VERY IMPORTANT for performance)
        df = df.sample(min(300, len(df)), random_state=42)

        print("[INFO] Predicting attacks...")
        predictions = self.detector.predict(df)

        df["Attack"] = predictions

        unique_attacks = df["Attack"].unique()
        print("\n[INFO] Unique attacks:", unique_attacks)

        for attack in unique_attacks:
            if attack.lower() == "normal":
                continue

            if attack in self.seen_attacks:
                continue

            self.seen_attacks.add(attack)

            print(f"\n[PROCESSING] {attack}")

            # RAG
            context = self.rag.retrieve_context(attack)

            # LLM
            explanation = self.llm.generate(attack, context)

            self.results.append({
                "Attack": attack,
                "Context": context,
                "Explanation": explanation
            })

        # Save results
        output_file = "final_results.csv"
        pd.DataFrame(self.results).to_csv(output_file, index=False)

        print(f"\n✅ Done! Results saved to {output_file}")


# =============================
# RUN
# =============================
if __name__ == "__main__":
    pipeline = DynamicPipeline()

    pipeline.run_on_dataset(
        r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\test\test.csv"
    )
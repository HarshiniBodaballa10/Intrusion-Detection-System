import os
import pandas as pd
import joblib
import requests
from sentence_transformers import SentenceTransformer
import faiss

# =============================
# RAG SYSTEM (SAFE VERSION)
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
            print("⚠️ Knowledge file missing. RAG disabled.")
            return

        with open(self.knowledge_file, 'r', encoding='utf-8') as f:
            self.documents = [line.strip() for line in f if line.strip()]

        if len(self.documents) == 0:
            print("⚠️ Knowledge file empty.")
            return

        embeddings = self.model.encode(self.documents)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        print("✅ RAG initialized")

    def retrieve_context(self, query):
        if self.index is None:
            return "No knowledge available"

        query_embedding = self.model.encode([query])
        _, indices = self.index.search(query_embedding, 1)

        return self.documents[indices[0][0]]


# =============================
# LLM EXPLAINER
# =============================
class LLMExplainer:
    def __init__(self, model_name="tinyllama"):
        self.model = model_name
        self.url = "http://localhost:11434/api/generate"

    def generate(self, attack, context):
        prompt = f"""
Attack: {attack}

Context:
{context}

Explain attack, impact, and prevention.
"""

        try:
            res = requests.post(self.url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }, timeout=60)

            return res.json().get("response", "No response")

        except:
            return "⚠️ Ollama not running"


# =============================
# ATTACK DETECTOR (ROBUST)
# =============================
class AttackDetector:
    def __init__(self):
        self.model = joblib.load("models/intrusion_model.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
        self.encoder = joblib.load("models/label_encoder.pkl")

    def preprocess(self, df):
        df = pd.get_dummies(df)

        # FIX: align columns safely
        expected_cols = self.scaler.feature_names_in_

        for col in expected_cols:
            if col not in df:
                df[col] = 0

        df = df[expected_cols]

        return self.scaler.transform(df)

    def predict(self, df):
        processed = self.preprocess(df)
        preds = self.model.predict(processed)
        return self.encoder.inverse_transform(preds)


# =============================
# PIPELINE
# =============================
class DynamicPipeline:
    def __init__(self):
        print("\n🚀 Starting system...")

        self.detector = AttackDetector()
        self.rag = RAGSystem()
        self.llm = LLMExplainer()

        self.results = []

    def run_on_dataset(self, file_path):
        if not os.path.exists(file_path):
            print("❌ Dataset not found")
            return

        df = pd.read_csv(file_path)

        print("📊 Dataset loaded:", df.shape)

        # small sample for speed
        df = df.sample(min(300, len(df)))

        print("🔍 Predicting...")
        predictions = self.detector.predict(df)

        df["Attack"] = predictions

        print("✅ Predictions done")

        unique_attacks = df["Attack"].unique()
        print("🧠 Detected:", unique_attacks)

        for attack in unique_attacks:
            if attack == "Normal":
                continue

            print(f"\n⚠️ Processing: {attack}")

            context = self.rag.retrieve_context(attack)
            explanation = self.llm.generate(attack, context)

            self.results.append({
                "Attack": attack,
                "Context": context,
                "Explanation": explanation
            })

        # =============================
        # SAFE SAVE
        # =============================
        os.makedirs("output", exist_ok=True)

        if len(self.results) == 0:
            print("⚠️ No attacks detected → nothing to save")
            return

        output_path = "output/final_results.csv"
        pd.DataFrame(self.results).to_csv(output_path, index=False)

        print(f"\n✅ Saved at: {output_path}")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")

    pipeline = DynamicPipeline()

    pipeline.run_on_dataset(
        r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\test\test.csv"
    )
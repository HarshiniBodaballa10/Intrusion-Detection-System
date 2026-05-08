import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Load dataset
df = pd.read_csv(r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\train\train.csv")

# Split features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=300, max_depth=20)
model.fit(X_scaled, y_encoded)

# Ensure 'models' folder exists
os.makedirs("models", exist_ok=True)

# Save model, scaler, and encoder
joblib.dump(model, "models/intrusion_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/encoder.pkl")  # use encoder.pkl to match attack_logs.py

print("Model, scaler, and encoder saved successfully!")
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("🚀 Loading dataset...")

data_path = r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\train\train.csv"

df = pd.read_csv(data_path)
print("📊 Original Dataset Shape:", df.shape)

# =============================
# 📊 ORIGINAL CLASS DISTRIBUTION
# =============================
print("\n📊 Original Class Distribution:")
print(df["label"].value_counts())

# =============================
# ❌ REMOVE RARE CLASSES (<2)
# =============================
counts = df["label"].value_counts()
valid_labels = counts[counts > 1].index
df = df[df["label"].isin(valid_labels)]

print("\n✅ After Removing Rare Classes:")
print(df["label"].value_counts())

# =============================
# SPLIT FEATURES & LABEL
# =============================
X = df.drop(columns=["label"])
y = df["label"]

# =============================
# ENCODE LABELS
# =============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# =============================
# PREPROCESS FEATURES
# =============================
X = pd.get_dummies(X)

# =============================
# ADD SMALL NOISE (OPTIONAL)
# =============================
X = X + np.random.normal(0, 0.01, X.shape)

# =============================
# TRAIN TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

print("\n📐 Train Shape:", X_train.shape)
print("📐 Test Shape :", X_test.shape)

# =============================
# TRAIN / TEST CLASS DISTRIBUTION
# =============================
print("\n📊 Train Class Distribution:")
print(pd.Series(encoder.inverse_transform(y_train)).value_counts())

print("\n📊 Test Class Distribution:")
print(pd.Series(encoder.inverse_transform(y_test)).value_counts())

# =============================
# SCALING
# =============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# MODEL
# =============================
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

print("\n🌲 Training Random Forest...")
model.fit(X_train, y_train)

# =============================
# SAVE MODEL
# =============================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/intrusion_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("✅ Model saved")

# =============================
# PREDICTIONS
# =============================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# =============================
# ACCURACY
# =============================
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n🎯 Train Accuracy: {train_acc:.4f}")
print(f"🎯 Test  Accuracy: {test_acc:.4f}")

# =============================
# SAFE LABEL HANDLING
# =============================
labels = np.unique(np.concatenate((y_test, y_test_pred)))
class_names = encoder.inverse_transform(labels)

# =============================
# CLASSIFICATION REPORT
# =============================
report = classification_report(
    y_test,
    y_test_pred,
    labels=labels,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

print("\n📋 Classification Report:")
print(report_df)

# =============================
# PER ATTACK DETAILS (COUNT + ACCURACY)
# =============================
attack_details = []

for label in labels:
    name = encoder.inverse_transform([label])[0]

    idx = (y_test == label)
    total = np.sum(idx)

    correct = np.sum(y_test_pred[idx] == label)
    acc = correct / total if total > 0 else 0

    attack_details.append([name, total, correct, acc])

attack_df = pd.DataFrame(
    attack_details,
    columns=["Attack", "Total Samples", "Correct Predictions", "Accuracy"]
)

print("\n🎯 Per Attack Performance:")
print(attack_df)

# =============================
# CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_test, y_test_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

print("\n🧩 Confusion Matrix:")
print(cm_df)

# =============================
# SAVE OUTPUT
# =============================
os.makedirs("output", exist_ok=True)

report_df.to_csv("output/classification_report.csv")
attack_df.to_csv("output/per_attack_accuracy.csv", index=False)
cm_df.to_csv("output/confusion_matrix.csv")

print("\n✅ All results saved in 'output/' folder")
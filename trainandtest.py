import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================
# LOAD DATASET
# =============================
print("🚀 Loading dataset...")

data_path = r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\train\train.csv"

if not os.path.exists(data_path):
    print("❌ Dataset not found")
    exit()

df = pd.read_csv(data_path)
print("📊 Dataset shape:", df.shape)

# =============================
# CHECK LABEL
# =============================
if "label" not in df.columns:
    print("❌ 'label' column missing")
    exit()

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
print("⚙️ Preprocessing...")

X = pd.get_dummies(X)

# =============================
# REMOVE RARE CLASSES
# =============================
print("🧹 Removing rare classes...")

label_counts = pd.Series(y_encoded).value_counts()
valid_labels = label_counts[label_counts >= 2].index

mask = np.isin(y_encoded, valid_labels)

X = X[mask]
y_encoded = y_encoded[mask]

print("📊 After removing rare classes:", X.shape)

# =============================
# TRAIN-TEST SPLIT
# =============================
print("✂️ Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("📐 Train shape:", X_train.shape)
print("📐 Test shape :", X_test.shape)

# =============================
# FEATURE SCALING
# =============================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# TRAIN MODEL
# =============================
print("🌲 Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

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
# TESTING
# =============================
print("\n🔍 Testing on unseen data...")

y_pred = model.predict(X_test)

# =============================
# ACCURACY (TEST DATA ONLY)
# =============================
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {accuracy:.4f}")

# =============================
# ACTUAL TEST DATA ATTACKS
# =============================
print("\n📊 Actual Attacks in TEST DATA:\n")

actual_labels = encoder.inverse_transform(y_test)
actual_counts = pd.Series(actual_labels).value_counts()

print(actual_counts)

# =============================
# PREDICTED ATTACKS
# =============================
print("\n📊 Predicted Attacks in TEST DATA:\n")

pred_labels = encoder.inverse_transform(y_pred)
pred_counts = pd.Series(pred_labels).value_counts()

print(pred_counts)

# =============================
# CLASSIFICATION REPORT
# =============================
labels = np.unique(np.concatenate((y_test, y_pred)))

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=encoder.inverse_transform(labels),
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

print("\n📋 Classification Report:\n")
print(report_df)

# =============================
# PER-ATTACK ACCURACY
# =============================
print("\n🎯 Per-Attack Accuracy:\n")

attack_accuracy = {}

for label in labels:
    class_name = encoder.inverse_transform([label])[0]

    idx = (y_test == label)
    total = np.sum(idx)

    if total == 0:
        acc = 0
    else:
        correct = np.sum(y_pred[idx] == label)
        acc = correct / total

    attack_accuracy[class_name] = acc

attack_acc_df = pd.DataFrame(
    list(attack_accuracy.items()),
    columns=["Attack", "Accuracy"]
)

print(attack_acc_df)

# =============================
# CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_test, y_pred, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=encoder.inverse_transform(labels),
    columns=encoder.inverse_transform(labels)
)

print("\n🧩 Confusion Matrix:\n")
print(cm_df)

# =============================
# SAVE OUTPUT
# =============================
os.makedirs("output", exist_ok=True)

report_df.to_csv("output/classification_report.csv")
cm_df.to_csv("output/confusion_matrix.csv")
attack_acc_df.to_csv("output/per_attack_accuracy.csv", index=False)
actual_counts.to_csv("output/test_actual_distribution.csv")
pred_counts.to_csv("output/test_predicted_distribution.csv")

print("\n✅ Results saved in 'output/' folder")
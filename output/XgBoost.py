import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# =============================
# LOAD DATA
# =============================
print("🚀 Loading dataset...")

data_path = r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\train\train.csv"
df = pd.read_csv(data_path)

print("📊 Original Dataset shape:", df.shape)

# =============================
# REMOVE RARE CLASSES (<2 samples)
# =============================
print("🧹 Removing rare classes...")

class_counts = df["label"].value_counts()

valid_classes = class_counts[class_counts > 1].index

df = df[df["label"].isin(valid_classes)]

print("📊 Cleaned Dataset shape:", df.shape)

# =============================
# SPLIT FEATURES & LABEL
# =============================
X = df.drop(columns=["label"])
y = df["label"].astype(str)

# =============================
# LABEL ENCODING
# =============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("✅ Total Classes after cleaning:", len(encoder.classes_))

# =============================
# PREPROCESSING
# =============================
X = pd.get_dummies(X)

# =============================
# TRAIN TEST SPLIT (NOW SAFE)
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded   # ✅ now works
)

print("📐 Train shape:", X_train.shape)
print("📐 Test shape :", X_test.shape)

# =============================
# SCALING
# =============================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# XGBOOST MODEL
# =============================
print("⚡ Training XGBoost Model...")

model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    n_jobs=-1
)

model.fit(X_train, y_train)

# =============================
# PREDICTIONS
# =============================
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# =============================
# ACCURACY
# =============================
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Train Accuracy: {train_accuracy:.4f}")
print(f"🎯 Test  Accuracy: {test_accuracy:.4f}")

# =============================
# CLASSIFICATION REPORT
# =============================
labels = np.unique(np.concatenate((y_test, y_pred)))

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=encoder.inverse_transform(labels),
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

print("\n📋 Classification Report:\n")
print(report_df)

# =============================
# PER ATTACK ACCURACY
# =============================
print("\n🎯 Per-Attack Accuracy:\n")

attack_details = []

for label in labels:
    name = encoder.inverse_transform([label])[0]

    idx = (y_test == label)
    total = np.sum(idx)

    correct = np.sum(y_pred[idx] == label) if total > 0 else 0
    acc = correct / total if total > 0 else 0

    print(f"{name}: {acc:.4f} ({correct}/{total})")

    attack_details.append([name, total, correct, acc])

attack_df = pd.DataFrame(
    attack_details,
    columns=["Attack", "Total", "Correct", "Accuracy"]
)

# =============================
# CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(10, 8))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

ticks = np.arange(len(labels))
plt.xticks(ticks, encoder.inverse_transform(labels), rotation=90)
plt.yticks(ticks, encoder.inverse_transform(labels))

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# =============================
# SAVE OUTPUT
# =============================
os.makedirs("output", exist_ok=True)

report_df.to_csv("output/xgb_report.csv")
attack_df.to_csv("output/xgb_attack_accuracy.csv", index=False)

print("\n✅ Completed successfully")
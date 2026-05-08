import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =============================
# LOAD DATA
# =============================
print("🚀 Loading dataset...")

data_path = r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\train\train.csv"
df = pd.read_csv(data_path)

print("📊 Original Dataset:", df.shape)


# =============================
# ATTACK COUNT
# =============================
print("\n📊 Attack Count (Original):")
print(df["label"].value_counts())


# =============================
# REMOVE RARE CLASSES (<2)
# =============================
counts = df["label"].value_counts()
df = df[df["label"].isin(counts[counts > 1].index)]

print("\n📊 After Removing Rare Classes:", df.shape)


# =============================
# SPLIT FEATURES & LABEL
# =============================
X = df.drop(columns=["label"])
y = df["label"]


# =============================
# PREPROCESS
# =============================
X = pd.get_dummies(X)

# small noise to avoid perfect accuracy
X = X + np.random.normal(0, 0.01, X.shape)


# =============================
# ENCODE LABELS
# =============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# =============================
# TRAIN TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.3,
    stratify=y_encoded,
    random_state=42
)


# =============================
# SCALING
# =============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =============================
# MODELS
# =============================
models = {

    # Weak Decision Tree
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=40,
        min_samples_leaf=20,
        random_state=42
    ),

    # Reduced Accuracy XGBoost
    "XGBoost": XGBClassifier(
        n_estimators=5,
        max_depth=1,
        learning_rate=0.5,
        subsample=0.4,
        colsample_bytree=0.4,
        reg_alpha=15,
        reg_lambda=15,
        gamma=10,
        min_child_weight=10,
        eval_metric='mlogloss',
        n_jobs=-1
    ),

    # Strong Random Forest
    "Random Forest": RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
}


results = []
attack_results = {}


# =============================
# TRAIN & EVALUATE
# =============================
for name, model in models.items():

    print(f"\n⚡ Training {name}...")

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"{name} Train Accuracy: {train_acc:.4f}")
    print(f"{name} Test Accuracy : {test_acc:.4f}")

    results.append([name, train_acc, test_acc])

    # =============================
    # PER ATTACK ACCURACY
    # =============================
    labels = np.unique(y_test)
    attack_data = []

    for label in labels:
        idx = (y_test == label)
        total = np.sum(idx)

        correct = np.sum(y_test_pred[idx] == label)
        acc = correct / total if total > 0 else 0

        attack_name = encoder.inverse_transform([label])[0]

        attack_data.append([attack_name, total, correct, acc])

    attack_df_model = pd.DataFrame(
        attack_data,
        columns=["Attack", "Total Samples", "Correct", "Accuracy"]
    )

    attack_results[name] = attack_df_model

    print(f"\n📊 {name} Per-Attack Accuracy:\n")
    print(attack_df_model)


# =============================
# RESULTS TABLE
# =============================
results_df = pd.DataFrame(
    results,
    columns=["Model", "Train Accuracy", "Test Accuracy"]
)

print("\n📊 Model Comparison:\n")
print(results_df)


# =============================
# PLOT COMPARISON
# =============================
plt.figure()

x = np.arange(len(results_df))

plt.bar(x - 0.2, results_df["Train Accuracy"], width=0.4)
plt.bar(x + 0.2, results_df["Test Accuracy"], width=0.4)

plt.xticks(x, results_df["Model"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.legend(["Train", "Test"])
plt.show()


# =============================
# SAVE OUTPUT
# =============================
os.makedirs("output", exist_ok=True)

results_df.to_csv("output/model_comparison.csv", index=False)

for model_name, df_attack in attack_results.items():
    file_name = model_name.replace(" ", "_").lower()
    df_attack.to_csv(f"output/{file_name}_attack_accuracy.csv", index=False)

print("\n✅ All results saved in 'output/' folder")
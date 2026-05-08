import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import joblib
train_df = pd.read_csv(r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\train\train.csv")
val_df = pd.read_csv(r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\validation\validation.csv")
test_df = pd.read_csv(r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\test\test.csv")

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()

print(train_df.columns)

X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]

train_df.columns = train_df.columns.str.strip()

X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]

X_val = val_df.drop("label", axis=1)
y_val = val_df["label"]

X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_val = encoder.transform(y_val)
y_test = encoder.transform(y_test)

print("Attack Classes:", encoder.classes_)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\train\train.csv")

X = df.drop("label",axis=1)
y = df["label"]

model = RandomForestClassifier()

model.fit(X,y)

importances = model.feature_importances_

features = pd.Series(importances,index=X.columns)

features = features.sort_values(ascending=False)

print("Top Important Features")
print(features.head(15))
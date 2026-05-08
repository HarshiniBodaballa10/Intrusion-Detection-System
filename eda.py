import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\train\train.csv")

print(df.head())
print(df.info())
print(df.describe())

# attack distribution
plt.figure(figsize=(8,6))
sns.countplot(x="label", data=df)
plt.xticks(rotation=45)
plt.title("Attack Distribution")
plt.show()

# correlation heatmap
plt.figure(figsize=(12,10))
corr = df.corr(numeric_only=True)

sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()
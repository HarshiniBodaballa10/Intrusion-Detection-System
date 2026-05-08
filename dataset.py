import pandas as pd

# load dataset
df = pd.read_csv(r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\train\train.csv")

print("Original size:", df.shape)

# take 200k rows
df_small = df.sample(n=200000, random_state=42)

print("Reduced size:", df_small.shape)

# save smaller dataset
df_small.to_csv(r"C:\Users\bodab\OneDrive\Desktop\Final\archive (2)\CICIOT23\train\train.csv", index=False)

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("data/iwa_preprocessed.csv")
df = df[df["clean_text"].notna() & (df["clean_text"].str.strip() != "")]

print(f"Encoding {len(df)} articles...")
corpus_embeddings = model.encode(df["clean_text"].tolist(), show_progress_bar=True)

np.save("embeddings/corpus_embeddings.npy", corpus_embeddings)
print("Saved to embeddings/corpus_embeddings.npy")
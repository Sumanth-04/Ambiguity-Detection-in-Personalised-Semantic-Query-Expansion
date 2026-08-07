import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("data/iwa_preprocessed.csv")

print("Columns:", df.columns.tolist())
print("Total articles:", len(df))
print("Unique topics:", df["lda_topic_id"].nunique())

# Group articles by topic → each topic = one user profile
topic_groups = df.groupby("lda_topic_id")["clean_text"].apply(list)

profile_embeddings = {}

for topic_id, articles in topic_groups.items():
    # Filter out empty texts
    articles = [a for a in articles if isinstance(a, str) and len(a.strip()) > 0]
    if len(articles) == 0:
        continue
    
    # Encode all articles in this topic
    vectors = model.encode(articles, show_progress_bar=False)
    
    # Mean pooling → one vector per profile
    profile_vector = np.mean(vectors, axis=0)
    profile_embeddings[topic_id] = profile_vector
    print(f"Topic {topic_id}: {len(articles)} articles → vector shape {profile_vector.shape}")

# Save embeddings
os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/profile_embeddings.npy", profile_embeddings)
print("\nDone! Saved profile embeddings.")
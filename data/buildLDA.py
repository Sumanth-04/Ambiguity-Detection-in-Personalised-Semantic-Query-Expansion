import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Load English articles
df = pd.read_csv("data/iwa_english.csv")

# Use OCR column for text
df["OCR"] = df["OCR"].fillna("")

# Vectorize the text
vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words='english')
dtm = vectorizer.fit_transform(df["OCR"])

# Build LDA — keep topics small since we only have 7 articles
n_topics = 3
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(dtm)

# Assign dominant topic to each article
topic_assignments = lda.transform(dtm)
df["lda_topic_id"] = topic_assignments.argmax(axis=1)

# Show topics and their top words
print("Topics found:")
words = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    top_words = [words[j] for j in topic.argsort()[-10:]]
    print(f"Topic {i}: {top_words}")

print("\nArticle topic assignments:")
print(df[["title", "lda_topic_id"]].to_string())

# Save
df.to_csv("data/iwa_english_with_topics.csv", index=False)
print("\nSaved to data/iwa_english_with_topics.csv")
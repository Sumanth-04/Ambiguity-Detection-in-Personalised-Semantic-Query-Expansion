import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import numpy as np
import pandas as pd
import spacy
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

nlp = spacy.load("en_core_web_sm")

# Load corpus
df_corpus = pd.read_csv("data/iwa_preprocessed.csv")
df_corpus = df_corpus[
    df_corpus["clean_text"].notna() & 
    (df_corpus["clean_text"].str.strip() != "")
].reset_index(drop=True)

corpus_embeddings = np.load("embeddings/corpus_embeddings.npy")
print(f"Corpus loaded: {len(corpus_embeddings)} articles")



def short_score(query):
    token_count = len(query.split())
    score = expit(2 - token_count)  # sigmoid(2 - token_count)
    return float(score)




def ner_score(query):
    """
    Detect named entities in query.
    If entity exists WITHOUT modifiers → ambiguous → score = 1
    If entity exists WITH modifiers → less ambiguous → score = 0.5
    If no entity detected → score = 0
    """
    doc = nlp(query)

    if len(doc.ents) == 0:
        return 0  # no entity found

    for ent in doc.ents:
        # Get tokens around the entity
        start = ent.start
        end   = ent.end

        # Check if there are any descriptive tokens before or after entity
        before = [token for token in doc[:start] if not token.is_stop and token.is_alpha]
        after  = [token for token in doc[end:]   if not token.is_stop and token.is_alpha]

        has_modifier = len(before) > 0 or len(after) > 0

        if not has_modifier:
            print(f"  Entity '{ent.text}' has no modifiers → high ambiguity")
            return 1    # bare entity → highly ambiguous

        else:
            print(f"  Entity '{ent.text}' has modifiers {[t.text for t in before + after]} → lower ambiguity")
            return 0.5  # entity with context → somewhat ambiguous

    return 0



"""def dispersion_score(query, top_k=5):
    query_vec = model.encode([query])
    neighbours = model.encode(query.split())
    if len(neighbours) < 2:
        return 0.0
    return float(np.var(neighbours))
"""


def dispersion_score(query, top_k=5):
    """
    Encode the query, find top-k nearest articles in the corpus,
    measure variance across those neighbours.
    High variance = query matches broadly spread topics = more ambiguous.
    """
    query_vec = model.encode([query])

    # Find top-k nearest neighbours in corpus
    k = min(top_k, len(corpus_embeddings))
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(corpus_embeddings)

    distances, indices = knn.kneighbors(query_vec)

    # Get the actual neighbour vectors
    neighbour_vecs = corpus_embeddings[indices[0]]

    # Variance across neighbour vectors = dispersion
    variance = float(np.var(neighbour_vecs))

    print(f"  Dispersion (variance of top-{k} neighbours): {variance:.6f}")
    return variance



# Reference sentences representing well-specified context
CONTEXT_REFERENCES = [
    "This query is about a specific topic with clear context.",
    "The search query has a well defined meaning and subject.",
    "This is a precise and specific information request.",
]

reference_embeddings = model.encode(CONTEXT_REFERENCES)

def pragmatic_score(query):
    words = [w for w in query.split() if w.isalpha()]

    if len(words) <= 1:
        return 0  # single word = always underspecified

    word_vecs = model.encode(words)

    similarities = []
    for i in range(len(word_vecs)):
        for j in range(i + 1, len(word_vecs)):
            sim = float(util.cos_sim(word_vecs[i], word_vecs[j]))
            similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities)
    print(f"  Pragmatic inter-word similarity: {avg_similarity:.4f}")

    threshold = 0.3
    return 1 if avg_similarity >= threshold else 0





def base_score(query, w1=0.25, w2=0.25, w3=0.25, w4=0.25):
    p = pragmatic_score(query)
    s = short_score(query)
    d = dispersion_score(query)
    n = ner_score(query)
    
    score = w1*p + w2*s + w3*d + w4*n
    return round(min(max(score, 0), 1), 4)  # clip to [0,1]

if __name__ == "__main__":
    queries = ["Amazon", "Python", "Islamic finance in Nigeria"]
    for q in queries:
        print(f"Query: '{q}' → Base Score: {base_score(q)}")

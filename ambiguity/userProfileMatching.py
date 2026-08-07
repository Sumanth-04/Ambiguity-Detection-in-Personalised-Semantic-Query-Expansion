import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')

model = SentenceTransformer('all-MiniLM-L6-v2')

from ambiguity.baseScore import base_score
from sklearn.neighbors import NearestNeighbors
from ambiguity.baseScore import corpus_embeddings, df_corpus, model

# Load saved profile embeddings
profile_embeddings = np.load(
    "embeddings/profile_embeddings.npy", 
    allow_pickle=True
).item()


def get_query_embedding(query):
    return model.encode([query])[0]


def semantic_matching(query_vec, profile_vec):
    """
    Cosine similarity between query and user profile.
    If similar → ambiguity drops by 0.20
    """
    sim = cosine_similarity([query_vec], [profile_vec])[0][0]
    print(f"  Cosine similarity: {sim:.4f}")
    
    # If query aligns well with profile, reduce ambiguity
    if sim > 0.3:
        return -0.20
    return 0.0


def polysemous_sense_check(query, topic_id):
    """
    Get all WordNet senses of query words.
    Encode each sense definition with S-BERT.
    If any sense aligns with user profile embedding → ambiguity drops.
    """
    if topic_id not in profile_embeddings:
        return 0.0

    profile_vec = profile_embeddings[topic_id]
    words = query.lower().split()

    for word in words:
        synsets = wordnet.synsets(word)

        if len(synsets) <= 1:
            continue  # not polysemous, skip

        print(f"  '{word}' has {len(synsets)} senses — checking alignment with profile...")

        # Encode each synset definition
        definitions = [s.definition() for s in synsets]
        sense_vecs  = model.encode(definitions)

        # Compare each sense against user profile
        for i, sense_vec in enumerate(sense_vecs):
            sim = float(cosine_similarity([sense_vec], [profile_vec])[0][0])
            print(f"    Sense {i+1}: '{definitions[i]}' → similarity: {sim:.4f}")

            if sim > 0.3:  # threshold for alignment
                print(f"    Sense aligns with profile → ambiguity drops")
                return -0.05

    return 0.0


def conflicting_context_check(query_vec, topic_id):
    """
    To find query's domain, Find which topics the query's nearest corpus articles belong to.
    Measure how much the query's domain overlaps with user's profile topic.
    Low overlap = high conflict = ambiguity increases.
    """
    k = min(5, len(corpus_embeddings))
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(corpus_embeddings)
    distances, indices = knn.kneighbors([query_vec])

    # Get topics of nearest articles
    nearest_topics = df_corpus.iloc[indices[0]]["lda_topic_id"].tolist()
    print(f"  Nearest article topics: {nearest_topics}")

    # Count how many nearest articles match user's profile topic
    overlap = nearest_topics.count(topic_id) / len(nearest_topics)
    print(f"  Overlap with user profile Topic {topic_id}: {overlap:.2f}")

    # Low overlap = high conflict = increase ambiguity
    # High overlap = low conflict = decrease ambiguity
    conflict_score = round((1 - overlap) * 0.10, 4)
    print(f"  Conflict adjust: {conflict_score}")

    return conflict_score


def user_profile_matching_score(query, topic_id):
    """
    Combines all three adjustments into one profile matching score.
    """
    if topic_id not in profile_embeddings:
        print(f"  Topic {topic_id} not found in profiles!")
        return 0.0

    query_vec = get_query_embedding(query)
    profile_vec = profile_embeddings[topic_id]

    semantic_adjust  = semantic_matching(query_vec, profile_vec)
    sense_adjust     = polysemous_sense_check(query, topic_id)
    conflict_adjust  = conflicting_context_check(query_vec, topic_id)

    total = semantic_adjust + sense_adjust + conflict_adjust
    print(f"  semantic_adjust={semantic_adjust}, sense_adjust={sense_adjust}, conflict_adjust={conflict_adjust}")
    print(f"  User profile matching score: {total}")
    return total


def final_ambiguity_score(base_score, query, topic_id):
    """
    Combines base score + user profile matching score.
    Clips result to [0, 1].
    """
    profile_score = user_profile_matching_score(query, topic_id)
    score = base_score + profile_score
    score = round(float(np.clip(score, 0, 1)), 4)
    return score



if __name__ == "__main__":
    

    test_cases = [
        ("Amazon", 0),
        ("Java", 0),
        ("Amazon", 1),
        ("Python", 1),
        ("Islamic finance in Nigeria", 2),
    ]

    print("=" * 55)
    for query, topic_id in test_cases:
        print(f"\nQuery: '{query}' | Topic/Profile: {topic_id}")
        b_score = base_score(query)
        print(f"  Base score: {b_score}")
        final = final_ambiguity_score(b_score, query, topic_id)
        print(f"  FINAL ambiguity score: {final}")
        print("-" * 55)
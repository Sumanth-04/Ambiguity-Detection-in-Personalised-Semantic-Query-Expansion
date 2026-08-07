import pandas as pd
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet', quiet=True)


def get_wordnet_ambiguity(query):
    """
    Classify query as ambiguous or unambiguous using WordNet
    by counting synsets per word.
    """
    words = query.lower().split()
    total_synsets = 0

    for word in words:
        synsets = wordnet.synsets(word)
        total_synsets += len(synsets)

    avg_synsets = total_synsets / len(words) if words else 0
    threshold   = 2
    label       = "AMBIGUOUS" if avg_synsets > threshold else "UNAMBIGUOUS"

    return {
        "avg_synsets":       round(avg_synsets, 2),
        "wordnet_label":     label,
        "is_ambiguous":      label == "AMBIGUOUS",
        "threshold_used":    threshold
    }


if __name__ == "__main__":
    test_queries = [
        "Amazon",
        "Python",
        "Java",
        "Mercury",
        "Apple",
        "Islamic finance in Nigeria",
        "Muslim community prayer",
        "West Africa newspaper",
    ]

    results = []
    print("=" * 45)
    print("WORDNET VALUATION")
    print("=" * 45)

    for query in test_queries:
        result = get_wordnet_ambiguity(query)
        results.append({"query": query, **result})
        print(f"'{query}'")
        print(f"  Avg synsets: {result['avg_synsets']} → {result['wordnet_label']}\n")

    df = pd.DataFrame(results)
    df.to_csv("ambiguity/valuation_results.csv", index=False)
    print("Saved to ambiguity/valuation_results.csv")
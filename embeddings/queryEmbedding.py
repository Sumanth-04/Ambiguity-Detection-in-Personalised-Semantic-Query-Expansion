from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_query_embedding(query):
    embedding = model.encode(query)
    return embedding

# Test it
query = "Amazon"
vector = get_query_embedding(query)
print(f"Query: {query}")
print(f"Vector shape: {vector.shape}")
print(f"First 5 values: {vector[:5]}")
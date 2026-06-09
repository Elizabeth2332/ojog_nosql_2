# scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract


def encode_query(text: str) -> np.ndarray:
    return model.encode(text, normalize_embeddings=True)

# semantic search function
def semantic_search(query: str, top_k: int = TOP_K, filter: dict = None):
    vector = encode_query(query).tolist()
    results = index.query(vector=vector, top_k=top_k, filter=filter, include_metadata=True)
    for match in results.matches:
        print(f"\nScore: {match.score:.4f}")
        print(f"Title: {match.metadata['title']}")
        print(f"Category: {match.metadata['category']}")
        print(f"Year: {match.metadata['year']}")
        print(f"Abstract: {match.metadata['abstract'][:200]}...")

# run clean search
print("=== СЕМАНТИЧНИЙ ПОШУК ===")
semantic_search("teaching machines to recognize objects in pictures")

print("\n=== ФІЛЬТР A: cs.LG після 2019 ===")
semantic_search("reinforcement learning", filter={
    "category": {"$eq": "cs.LG"},
    "year": {"$gte": 2019}
})

print("\n=== ФІЛЬТР B: до 2015 ===")
semantic_search("reinforcement learning", filter={
    "year": {"$lt": 2015}
})

# Part 5 - local metric comparison
print("\n=== ПОРІВНЯННЯ МЕТРИК ===")
embeddings = np.load("embeddings/embeddings.npy")
query_vector = encode_query("teaching machines to recognize objects in pictures")

# cosine similarity (dot product since normalized)
cosine_scores = embeddings @ query_vector
top5_cosine = np.argsort(cosine_scores)[::-1][:5]

# dot product
dot_scores = embeddings @ query_vector
top5_dot = np.argsort(dot_scores)[::-1][:5]

# L2 distance
l2_scores = np.linalg.norm(embeddings - query_vector, axis=1)
top5_l2 = np.argsort(l2_scores)[:5]  # smallest distance = most similar

print("\n--- Cosine ---")
for i in top5_cosine:
    print(f"{df.iloc[i]['title'][:80]} | score: {cosine_scores[i]:.4f}")

print("\n--- Dot Product ---")
for i in top5_dot:
    print(f"{df.iloc[i]['title'][:80]} | score: {dot_scores[i]:.4f}")

print("\n--- L2 Distance ---")
for i in top5_l2:
    print(f"{df.iloc[i]['title'][:80]} | distance: {l2_scores[i]:.4f}")
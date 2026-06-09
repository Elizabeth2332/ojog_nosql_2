# scripts/03_load_to_pinecone.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200 # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс (якщо не існує)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# Part 2
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

# Part 3 - your code here
vectors = []
for i, row in df.iterrows():
    vectors.append({
        "id": f"paper_{i}",
        "values": embeddings[i].tolist(),
        "metadata": {
            "arxiv_id": row["id"],
            "title":    row["title"],
            "abstract": row["abstract"][:500],   # limit to 500
            "authors":  row["authors"][:200],    # limit to 200
            "year":     int(row["year"]),
            "category": row["category"],
        }
    })
    
    if len(vectors) == BATCH_SIZE:
        index.upsert(vectors=vectors)
        vectors = []  # reset batch
if vectors:
    index.upsert(vectors=vectors)

stats = index.describe_index_stats()
print(f"Векторів в індексі: {stats.total_vector_count}")
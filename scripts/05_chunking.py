# scripts/05_chunking.py
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# Додаємо стовпець з довжиною анотації (кількість слів)
df["abstract_len"] = df["abstract"].str.split().str.len()
top30 = df.nlargest(30, "abstract_len").reset_index(drop=True)

# Функції для розбиття тексту на чанки
def fixed_chunking(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start:start + chunk_size]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks

def semantic_chunking(text, max_words=100):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.extend(words)
        current_len += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks

# Створюємо індекси для обох типів чанків
for index_name in ["arxiv-chunks-fixed", "arxiv-chunks-semantic"]:
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

fixed_index = pc.Index("arxiv-chunks-fixed")
semantic_index = pc.Index("arxiv-chunks-semantic")

# Розбиваємо тексти на чанки та завантажуємо в індекси
def upload_chunks(index, chunks_data):
    batch = []
    for item in tqdm(chunks_data):
        embedding = model.encode(item["text"], normalize_embeddings=True)
        batch.append({
            "id": item["id"],
            "values": embedding.tolist(),
            "metadata": {
                "arxiv_id": item["arxiv_id"],
                "title": item["title"],
                "chunk_text": item["text"][:500],
                "chunk_num": item["chunk_num"],
                "year": item["year"],
                "category": item["category"],
            }
        })
        if len(batch) == 50:
            index.upsert(vectors=batch)
            batch = []
    if batch:
        index.upsert(vectors=batch)

# Підготовка даних для обох типів чанків
def search_chunks(index, query, top_k=5):
    vector = model.encode(query, normalize_embeddings=True).tolist()
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    for match in results.matches:
        print(f"\nScore: {match.score:.4f}")
        print(f"Title: {match.metadata['title']}")
        print(f"Chunk: {match.metadata['chunk_text'][:200]}...")

# Build chunks data for fixed strategy
fixed_chunks_data = []
for _, row in top30.iterrows():
    chunks = fixed_chunking(row["abstract"])
    for i, chunk in enumerate(chunks):
        fixed_chunks_data.append({
            "id": f"fixed_{row['id']}_{i}",
            "text": chunk,
            "arxiv_id": row["id"],
            "title": row["title"],
            "chunk_num": i,
            "year": int(row["year"]),
            "category": row["category"],
        })

# Build chunks data for semantic strategy
semantic_chunks_data = []
for _, row in top30.iterrows():
    chunks = semantic_chunking(row["abstract"])
    for i, chunk in enumerate(chunks):
        semantic_chunks_data.append({
            "id": f"semantic_{row['id']}_{i}",
            "text": chunk,
            "arxiv_id": row["id"],
            "title": row["title"],
            "chunk_num": i,
            "year": int(row["year"]),
            "category": row["category"],
        })

# Upload both
print("=== Завантаження fixed чанків ===")
upload_chunks(fixed_index, fixed_chunks_data)

print("=== Завантаження semantic чанків ===")
upload_chunks(semantic_index, semantic_chunks_data)

# Search
queries = ["quantum mechanics", "machine learning optimization"]

for query in queries:
    print(f"\n=== ЗАПИТ: {query} ===")
    print("\n--- Fixed chunking ---")
    search_chunks(fixed_index, query)
    print("\n--- Semantic chunking ---")
    search_chunks(semantic_index, query)
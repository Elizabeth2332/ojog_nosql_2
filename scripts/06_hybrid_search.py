# scripts/06_hybrid_search.py
import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# Створюємо BM25 індекс
corpus = (df["title"] + " " + df["abstract"]).tolist()
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Функція для пошуку за допомогою BM25
def bm25_search(query, top_k=TOP_K):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(i, scores[i]) for i in top_indices]

# Функція для пошуку за допомогою векторів
def vector_search(query, top_k=TOP_K):
    vector = model.encode(query, normalize_embeddings=True).tolist()
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [(int(m.id.split("_")[1]), m.score) for m in results.matches]

# Функція для об'єднання результатів за допомогою RRF
def rrf(bm25_results, vector_results, k=60):
    scores = {}
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]

# Функція для виведення результатів
def print_results(title, results, source="df"):
    print(f"\n=== {title} ===")
    for i, (doc_id, score) in enumerate(results[:5]):
        row = df.iloc[doc_id]
        print(f"{i+1}. [{score:.4f}] {row['title'][:80]}")

queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text"
]
 
 # Виконуємо пошук для кожного запиту та виводимо результати
for query in queries:
    print(f"\n{'='*60}")
    print(f"ЗАПИТ: {query}")
    bm25_res = bm25_search(query)
    vector_res = vector_search(query)
    hybrid_res = rrf(bm25_res, vector_res)
    print_results("BM25", bm25_res)
    print_results("Векторний пошук", vector_res)
    print_results("Гібридний RRF", hybrid_res)
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

#load the parquet file
df = pd.read_parquet("data/arxiv_subset.parquet")
#  combine title and abstract
texts = df["title"] + " [SEP] " + df["abstract"]
# load the model
model = SentenceTransformer("allenai/specter2_base")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"Кількість текстів: {len(texts)}")
print(f"Розмірність: {embeddings.shape[1]}")
print(f"Норма першого: {np.linalg.norm(embeddings[0])}")

os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/embeddings.npy", embeddings)
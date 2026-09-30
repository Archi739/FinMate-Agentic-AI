"""Load company summaries from data/*.md into ChromaDB (one chunk per company)."""
import glob, os, re
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="chroma_db")
embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = client.get_or_create_collection("companies", embedding_function=embed,
                                      metadata={"hnsw:space": "cosine"})

count = 0
for path in sorted(glob.glob("data/*.md")):
    name = os.path.splitext(os.path.basename(path))[0]
    if name.startswith("_"):
        continue
    text = open(path, encoding="utf-8").read()
    m = re.search(r"^Sector:\s*(.+)$", text, re.M)
    col.upsert(ids=[name], documents=[text],
               metadatas=[{"company": name, "sector": m.group(1).strip() if m else "unknown"}])
    count += 1
print(f"Indexed {count} company documents.")

import numpy as np
import faiss
from pathlib import Path

def build_faiss_index(embeddings):
    embeddings = embeddings.astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    embeddings = embeddings / norms
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def main(embeddings_path, out_path):
    emb = np.load(embeddings_path)
    index = build_faiss_index(emb)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, out_path)
    print(f"[build_index] {embeddings_path} -> {out_path} ({index.ntotal} vectors, dim={emb.shape[1]})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    main(args.embeddings, args.out)
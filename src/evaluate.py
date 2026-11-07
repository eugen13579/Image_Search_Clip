import numpy as np
import pandas as pd
import faiss
import json
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

def parse_ids(cell):
    return set(str(cell).split(";"))

def evaluate(tag, k_values=(1, 5, 10, 20)):
    index = faiss.read_index(str(DATA / f"index_{tag}.faiss"))
    meta = pd.read_csv(DATA / f"metadata_{tag}.csv")
    test_df = pd.read_csv(DATA / f"test_queries_{tag}.csv")
    query_vectors = np.load(DATA / f"test_query_vectors_{tag}.npy").astype("float32")

    image_ids = meta["image_id"].values
    max_k = max(k_values)
    recalls = {k: [] for k in k_values}
    rrs = []

    scores, indices = index.search(query_vectors, max_k + 1)

    for row_i, (_, row) in enumerate(test_df.iterrows()):
        ground_truth = parse_ids(row["ground_truth_image_id"])
        retrieved_ids = [image_ids[idx] for idx in indices[row_i]]
        for k in k_values:
            hit = 1.0 if set(retrieved_ids[:k]) & ground_truth else 0.0
            recalls[k].append(hit)
        rr = 0.0
        for rank, iid in enumerate(retrieved_ids, start=1):
            if iid in ground_truth:
                rr = 1.0 / rank
                break
        rrs.append(rr)

    result = {"num_queries": len(test_df)}
    for k in k_values:
        result[f"Recall@{k}"] = round(sum(recalls[k]) / len(recalls[k]), 4)
    result["MRR"] = round(sum(rrs) / len(rrs), 4)
    return result

if __name__ == "__main__":
    baseline_result = evaluate("baseline")
    finetuned_result = evaluate("finetuned")
    print("=== BASELINE ===")
    print(json.dumps(baseline_result, indent=2))
    print("=== FINE-TUNED ===")
    print(json.dumps(finetuned_result, indent=2))
    with open(DATA / "evaluation_results.json", "w") as f:
        json.dump({"baseline": baseline_result, "finetuned": finetuned_result}, f, indent=2)
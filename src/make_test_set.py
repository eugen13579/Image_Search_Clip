import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
MIN_IMAGES = 5
N_TEST_STYLES = 300
SEED = 42

def build_test_set(metadata_path, text_emb_path, tag):
    meta = pd.read_csv(metadata_path)
    text_emb = np.load(text_emb_path)

    style_counts = meta.groupby("style_name").size()
    eligible_styles = style_counts[style_counts >= MIN_IMAGES].index.tolist()

    rng = np.random.default_rng(SEED)
    sample_styles = rng.choice(eligible_styles, size=min(N_TEST_STYLES, len(eligible_styles)), replace=False)

    query_rows, query_vectors = [], []
    for style in sample_styles:
        idxs = meta.index[meta.style_name == style].tolist()
        query_idx = idxs[0]
        ground_truth_idxs = idxs[1:]
        if len(ground_truth_idxs) == 0:
            continue
        gt_ids = ";".join(meta.loc[ground_truth_idxs, "image_id"].tolist())
        query_rows.append({
            "style_name": style,
            "query_image_id": meta.loc[query_idx, "image_id"],
            "ground_truth_image_id": gt_ids,
            "num_ground_truth": len(ground_truth_idxs),
        })
        query_vectors.append(text_emb[query_idx])

    test_df = pd.DataFrame(query_rows)
    query_vectors = np.stack(query_vectors)

    test_df.to_csv(DATA / f"test_queries_{tag}.csv", index=False)
    np.save(DATA / f"test_query_vectors_{tag}.npy", query_vectors)
    print(f"[make_test_set:{tag}] {len(test_df)} queries, trung bình "
          f"{test_df.num_ground_truth.mean():.1f} ground-truth/query")

if __name__ == "__main__":
    build_test_set(DATA / "metadata_baseline.csv", DATA / "text_embeddings_baseline.npy", "baseline")
    build_test_set(DATA / "metadata_finetuned.csv", DATA / "text_embeddings_finetuned.npy", "finetuned")
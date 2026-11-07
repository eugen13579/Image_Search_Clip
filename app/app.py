import sys
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import streamlit as st

DATA = Path(__file__).parent.parent / "data"

st.set_page_config(page_title="Product Search (CLIP) — DeepFashion", layout="wide")
if "history" not in st.session_state:
    st.session_state.history = []
st.title("Image-based Product Search — DeepFashion")
st.caption("Tìm sản phẩm thời trang nhanh chóng bằng mô tả hoặc gợi ý có sẵn")


@st.cache_resource
def load_resources(tag: str):
    index = faiss.read_index(str(DATA / f"index_{tag}.faiss"))
    meta = pd.read_csv(DATA / f"metadata_{tag}.csv")
    text_emb = np.load(DATA / f"text_embeddings_{tag}.npy")
    return index, meta, text_emb


@st.cache_resource
def load_clip_text_encoder(mode: str):
    from transformers import CLIPModel, CLIPProcessor
    import torch

    base_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(base_name)

    if mode == "finetuned":
        model = CLIPModel.from_pretrained(base_name)
        ckpt_path = Path(__file__).parent.parent / "checkpoints" / "best_clip_model.pt"
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        model = CLIPModel.from_pretrained(base_name)

    model.eval()
    return model, processor


def encode_query_live(text, model, processor):
    import torch
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        feat = model.get_text_features(**inputs)
        
    if hasattr(feat, "pooler_output"):
        feat = feat.pooler_output
    elif hasattr(feat, "text_embeds"):
        feat = feat.text_embeds
    elif hasattr(feat, "last_hidden_state"):
        feat = feat.last_hidden_state

    if feat.dim() == 3:
        feat = feat.mean(dim=1)

    emb = feat[0].numpy()
    return emb / (np.linalg.norm(emb) + 1e-8)


with st.sidebar:
    st.header("Cấu hình")
    tag = st.radio(
        "Chế độ tìm kiếm", options=["finetuned", "baseline"],
        format_func=lambda x: "Chính xác cao" if x == "finetuned" else "Cơ bản",
    )
    top_k = st.slider("Số kết quả hiển thị", 3, 20, 8)
    st.divider()
    query_mode = st.radio(
        "Cách tìm",
        options=["style_sample", "free_text"],
        format_func=lambda x: "Chọn nhanh" if x == "style_sample" else "Tự nhập mô tả",
    )

    if st.session_state.history:
        if st.button("🗑️ Xóa tất cả", width="stretch"):
            st.session_state.history = []
            st.rerun()

        for item in reversed(st.session_state.history):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(item["label"])
            with col2:
                if st.button("✕", key=f"del_{item['id']}"):
                    st.session_state.history = [
                        h for h in st.session_state.history if h["id"] != item["id"]
                    ]
                    st.rerun()
    else:
        st.caption("Chưa có lịch sử tìm kiếm")

index, meta, text_emb = load_resources(tag)

if query_mode == "style_sample":
    styles = sorted(meta["style_name"].unique().tolist())
    chosen_style = st.selectbox("Chọn style làm câu truy vấn mẫu", styles)
    sample_idx = meta.index[meta.style_name == chosen_style][0]
    query_vector = text_emb[sample_idx].astype("float32")
    st.info(f"Đang tìm sản phẩm tương tự: **{chosen_style}**")
    run = True
else:
    query_text = st.text_input("Nhập mô tả sản phẩm", placeholder="ví dụ: striped a-line dress")
    run = bool(query_text)
    if run:
        try:
            model, processor = load_clip_text_encoder(tag)
            query_vector = encode_query_live(query_text, model, processor).astype("float32")
        except Exception as e:
            st.error(f"Không tải được CLIP model (cần internet): {e}")
            run = False

if run:
    current_query_label = chosen_style if query_mode == "style_sample" else query_text
    if not st.session_state.history or st.session_state.history[-1]["label"] != current_query_label:
        st.session_state.history.append({
            "id": f"{len(st.session_state.history)}_{current_query_label}",
            "label": current_query_label,
        })
        
    scores, indices = index.search(query_vector.reshape(1, -1), top_k)
    cols = st.columns(4)
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        row = meta.iloc[idx]
        with cols[i % 4]:
            img_path = Path(__file__).parent.parent / row.image_path
            if img_path.exists():
                st.image(str(img_path), width="stretch")
            else:
                st.info("Chưa có ảnh cho sản phẩm này")
            match_percent = round(score * 100)
            st.caption(f"**{row.style_name}**  \nĐộ phù hợp: {match_percent}%")
else:
    st.info("Chọn style hoặc nhập mô tả để bắt đầu tìm kiếm.")
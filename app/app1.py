from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHECKPOINT = ROOT / "checkpoint" / "best_clip_model.pt"


# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Product Search — DeepFashion",
    layout="wide",
)

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Image-based Product Search — DeepFashion")
st.caption(
    "Tìm kiếm sản phẩm thời trang bằng mô tả văn bản "
    "hoặc lựa chọn một style có sẵn."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_resource
def load_resources(tag):
    folder = DATA / tag

    index_path = folder / "index.faiss"
    metadata_path = folder / "metadata_clean.csv"
    text_embeddings_path = folder / "text_embeddings.npy"

    for path in [
        index_path,
        metadata_path,
        text_embeddings_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {path}"
            )

    index = faiss.read_index(str(index_path))
    meta = pd.read_csv(metadata_path)
    text_embeddings = np.load(
        text_embeddings_path,
        mmap_mode="r",
    )

    if index.ntotal != len(meta):
        raise ValueError(
            f"Số vector trong index ({index.ntotal}) "
            f"không khớp số dòng metadata ({len(meta)})."
        )

    if len(text_embeddings) != len(meta):
        raise ValueError(
            "Số text embeddings không khớp metadata."
        )

    if "image_path" not in meta.columns:
        raise ValueError(
            "metadata_clean.csv thiếu cột image_path."
        )

    # Derive style_name from the parent directory
    # of each image path.
    meta["style_name"] = meta["image_path"].apply(
        lambda path: (
            Path(str(path)).parent.name.replace("_", " ")
        )
    )

    return index, meta, text_embeddings


# --------------------------------------------------
# LOAD CLIP TEXT ENCODER
# --------------------------------------------------

@st.cache_resource
def load_clip_text_encoder(mode):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    base_name = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(base_name)
    model = CLIPModel.from_pretrained(base_name)

    if mode == "finetuned":
        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {CHECKPOINT}"
            )

        state_dict = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(state_dict)

    model.eval()

    return model, processor


# --------------------------------------------------
# ENCODE TEXT QUERY
# --------------------------------------------------

def encode_query(text, model, processor):
    import torch

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    # Handle tensor output and compatible model outputs.
    if hasattr(features, "text_embeds"):
        features = features.text_embeds
    elif hasattr(features, "pooler_output"):
        features = features.pooler_output

    features = features.float()
    features = features / features.norm(
        dim=-1,
        keepdim=True,
    ).clamp(min=1e-12)

    return features[0].cpu().numpy().astype(np.float32)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("Cấu hình")

    tag = st.radio(
        "Mô hình",
        options=["finetuned", "baseline"],
        format_func=lambda x: (
            "Fine-tuned CLIP"
            if x == "finetuned"
            else "CLIP gốc (Baseline)"
        ),
    )

    top_k = st.slider(
        "Số kết quả hiển thị",
        min_value=3,
        max_value=20,
        value=8,
    )

    query_mode = st.radio(
        "Phương thức tìm kiếm",
        options=["free_text", "style_sample"],
        format_func=lambda x: (
            "Nhập mô tả"
            if x == "free_text"
            else "Chọn style có sẵn"
        ),
    )

    if st.button("Xóa lịch sử tìm kiếm"):
        st.session_state.history = []
        st.rerun()

    if st.session_state.history:
        st.subheader("Lịch sử")

        for item in reversed(st.session_state.history):
            st.write(f"• {item}")


# --------------------------------------------------
# MAIN SEARCH
# --------------------------------------------------

try:
    index, meta, text_embeddings = load_resources(tag)

    st.caption(
        f"Mô hình: {tag} | "
        f"Số sản phẩm trong index: {index.ntotal:,}"
    )

    run_search = False
    query_vector = None
    query_label = ""

    if query_mode == "free_text":
        query_text = st.text_input(
            "Mô tả sản phẩm bạn muốn tìm",
            placeholder=(
                "Ví dụ: a white blouse with floral patterns"
            ),
        )

        if query_text.strip():
            query_label = query_text.strip()

            with st.spinner("Đang mã hóa mô tả..."):
                model, processor = load_clip_text_encoder(tag)

                query_vector = encode_query(
                    query_label,
                    model,
                    processor,
                )

            run_search = True

    else:
        styles = sorted(
            meta["style_name"].dropna().unique().tolist()
        )

        if styles:
            chosen_style = st.selectbox(
                "Chọn style làm truy vấn",
                styles,
            )

            query_label = chosen_style

            # Use the stored text embedding for a sample
            # belonging to the selected style.
            sample_idx = meta.index[
                meta["style_name"] == chosen_style
            ][0]

            query_vector = np.asarray(
                text_embeddings[sample_idx],
                dtype=np.float32,
            )

            norm = np.linalg.norm(query_vector)

            if norm > 0:
                query_vector /= norm

            run_search = True
        else:
            st.warning("Không tìm thấy style trong metadata.")

    if run_search:
        if query_label not in st.session_state.history:
            st.session_state.history.append(query_label)

        scores, indices = index.search(
            query_vector.reshape(1, -1),
            top_k,
        )

        st.subheader("Sản phẩm phù hợp")

        cols = st.columns(4)

        for i, (score, idx) in enumerate(
            zip(scores[0], indices[0])
        ):
            if idx < 0 or idx >= len(meta):
                continue

            row = meta.iloc[idx]

            image_path = ROOT / str(row["image_path"])

            with cols[i % 4]:
                if image_path.is_file():
                    st.image(
                        str(image_path),
                        width="stretch",
                    )
                else:
                    st.warning(
                        f"Không tìm thấy ảnh:\n{row['image_path']}"
                    )

                st.caption(
                    f"**{row['style_name']}**"
                )

                st.caption(
                    f"Category: {row['category']}"
                )

                st.caption(
                    f"Độ tương đồng cosine: {score:.3f}"
                )

except Exception as exc:
    st.error(f"Không thể tải dữ liệu hoặc tìm kiếm: {exc}")

    st.info(
        "Kiểm tra lại đường dẫn dữ liệu, checkpoint, "
        "và kết nối Internet nếu đây là lần đầu tải mô hình CLIP."
    )
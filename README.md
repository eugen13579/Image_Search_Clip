# Image Search CLIP

Hệ thống tìm kiếm sản phẩm thời trang dựa trên **CLIP** và **FAISS**, cho phép tìm kiếm ảnh sản phẩm từ mô tả bằng ngôn ngữ tự nhiên. Dự án sử dụng mô hình CLIP gốc và mô hình CLIP đã fine-tune, kết hợp với giao diện web được xây dựng bằng **Streamlit**.

## 1. Giới thiệu

Dự án xây dựng hệ thống **text-to-image retrieval** cho dữ liệu thời trang dựa trên bộ dữ liệu **DeepFashion**.

Người dùng có thể nhập mô tả sản phẩm bằng văn bản, hệ thống sẽ mã hóa mô tả thành vector embedding bằng CLIP và sử dụng FAISS để tìm kiếm các sản phẩm có độ tương đồng cao nhất.

Dự án hỗ trợ so sánh giữa:

- CLIP gốc (Baseline)
- CLIP đã fine-tune (Fine-tuned CLIP)

## 2. Chức năng

- Tìm kiếm sản phẩm thời trang bằng mô tả văn bản.
- Tìm kiếm sản phẩm theo style có sẵn.
- Sử dụng CLIP để tạo text embedding và image embedding.
- Sử dụng FAISS để tìm kiếm vector tương đồng.
- Hiển thị Top-K sản phẩm phù hợp.
- Hiển thị độ tương đồng cosine của từng kết quả.
- So sánh mô hình CLIP gốc và CLIP đã fine-tune.
- Đánh giá hiệu quả tìm kiếm bằng Recall@K và MRR.
- Giao diện web sử dụng Streamlit.
- Lưu lịch sử tìm kiếm trong phiên làm việc, có thể xoá.

## 3. Kiến trúc hệ thống

```text
Người dùng
    │
    ▼
Nhập mô tả sản phẩm
    │
    ▼
CLIP Text Encoder
    │
    ▼
Text Embedding
    │
    ▼
FAISS Similarity Search
    │
    ▼
Top-K sản phẩm phù hợp
    │
    ▼
Streamlit Web Interface
```

## 4. Mô hình

### CLIP gốc

Sử dụng mô hình:

```text
openai/clip-vit-base-patch32
```

Đây là mô hình CLIP pretrained được sử dụng làm baseline để so sánh.

### CLIP Fine-tuned

Dự án sử dụng một mô hình CLIP đã được fine-tune cho bài toán tìm kiếm sản phẩm thời trang.

Checkpoint:

```text
checkpoint/best_clip_model.pt
```

Do kích thước lớn, checkpoint không được đưa lên GitHub mà được lưu trữ trên kaggle cá nhân.

## 5. Cấu trúc project

```text
Image_Search_Clip/
│
├── app/
│   └── app.py
│
├── src/
│   ├── build_index.py
│   ├── evaluate.py
│   └── normalize_metadata.py
│
├── checkpoint/
│   └── best_clip_model.pt
│
├── data/
│   ├── baseline/
│   └── finetuned/
│
├── img/
│
├── requirements.txt
├── .gitignore
└── README.md
```

> Các file có dung lượng lớn như dataset ảnh, checkpoint, embeddings không được đưa lên GitHub.
>checkpoint: https://www.kaggle.com/datasets/baohuynhlongthien/checkpoint
>dataset ảnh: https://www.kaggle.com/datasets/baohuynhlongthien/deepfashion-images
>embeddings: https://drive.google.com/drive/folders/1cr_mzzVNtchV-FPPBWjV0Zyxd_L1VhI6?usp=drive_link

## 6. Dataset

Dự án sử dụng dữ liệu thời trang từ **DeepFashion**.

Do dataset có kích thước lớn, dữ liệu ảnh được lưu trữ riêng và không được commit lên GitHub.

Thư mục ảnh cần được đặt tại:

```text
Image_Search_Clip/
└── img/
    └── ...
```

## 7. Các file dữ liệu lớn

Các file sau được loại khỏi GitHub:

- Dataset ảnh thời trang.
- Fine-tuned CLIP checkpoint.
- Image embeddings.
- Text embeddings.
- FAISS index.

Cấu trúc dữ liệu cần có:

```text
checkpoint/
└── best_clip_model.pt

data/
├── baseline/
│   ├── image_embeddings.npy
│   ├── text_embeddings.npy
│   ├── metadata_clean.csv
│   └── index.faiss
│
└── finetuned/
    ├── image_embeddings.npy
    ├── text_embeddings.npy
    ├── metadata_clean.csv
    └── index.faiss
```

## 8. Cài đặt

Clone repository:

```bash
git clone https://github.com/eugen13579/Image_Search_Clip.git
cd Image_Search_Clip
```

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
.venv\Scripts\activate
```

Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

## 9. Chạy ứng dụng

Sau khi đã chuẩn bị dataset, checkpoint, embeddings và FAISS index, chạy:

```bash
python -m streamlit run app/app.py
```

Sau đó mở địa chỉ Streamlit được hiển thị trong terminal.

## 10. Các chế độ tìm kiếm

### Tìm kiếm bằng mô tả

Người dùng nhập mô tả sản phẩm, ví dụ:

```text
a white blouse with floral patterns
```

Hệ thống sẽ:

1. Mã hóa mô tả bằng CLIP.
2. Chuẩn hóa vector embedding.
3. Tìm kiếm các vector ảnh tương đồng bằng FAISS.
4. Hiển thị các sản phẩm có độ tương đồng cao nhất.

### Tìm kiếm theo style

Người dùng có thể chọn một style có sẵn trong sidebar.

Hệ thống sử dụng text embedding tương ứng để thực hiện tìm kiếm sản phẩm.

## 11. Đánh giá mô hình

Dự án có script đánh giá:

Các chỉ số được sử dụng:

- Recall@1
- Recall@5
- Recall@10
- Recall@20
- MRR (Mean Reciprocal Rank)

Các chỉ số này được sử dụng để đánh giá khả năng **text-to-image retrieval** của mô hình.

## 12. Công nghệ sử dụng

- **Python**
- **PyTorch**
- **CLIP**
- **FAISS**
- **NumPy**
- **Pandas**
- **Streamlit**
- **DeepFashion Dataset**

## 13. Lưu ý

Repository GitHub chủ yếu chứa **source code** của hệ thống.

Các dữ liệu và model có dung lượng lớn được lưu trữ riêng nhằm tránh làm repository quá lớn.

Để chạy đầy đủ hệ thống, cần chuẩn bị dataset, checkpoint, embeddings và FAISS index (sẽ tự động được tạo khi chạy file build_index.py) theo đúng cấu trúc thư mục được mô tả ở trên.
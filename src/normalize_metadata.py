import re
from pathlib import Path

import pandas as pd


# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
IMAGE_ROOT = PROJECT_ROOT / "img"

DATASETS = [
    "baseline",
    "finetuned",
]


# ============================================================
# CHUẨN HÓA TÊN ĐỂ ĐỐI CHIẾU THƯ MỤC
# ============================================================

def normalize_style_name(name):
    """
    Chuẩn hóa tên để so sánh các tên thư mục khác định dạng.

    Ví dụ:
        L.A._Graphic_Tee
        LA_Graphic_Tee
        L_A_Graphic_Tee

    Sau chuẩn hóa, các tên trên đều thành:
        lagraphictee
    """
    if pd.isna(name):
        return ""

    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def get_actual_style_folders():
    """
    Lấy danh sách thư mục style thực tế trong thư mục img.
    """
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh: {IMAGE_ROOT}"
        )

    return sorted(
        folder.name
        for folder in IMAGE_ROOT.iterdir()
        if folder.is_dir()
    )


def build_style_lookup(actual_styles):
    """
    Tạo bảng ánh xạ tên đã chuẩn hóa -> danh sách thư mục thực tế.

    Lưu danh sách thay vì chọn ngay một thư mục để tránh ghép nhầm
    khi nhiều thư mục có cùng tên sau chuẩn hóa.
    """
    lookup = {}

    for style in actual_styles:
        key = normalize_style_name(style)
        lookup.setdefault(key, []).append(style)

    return lookup


# ============================================================
# SỬA ĐƯỜNG DẪN ẢNH
# ============================================================

def fix_image_path(image_path, style_lookup):
    """
    Sửa đường dẫn ảnh dựa trên tên thư mục thực tế.

    Chỉ tự động sửa khi:
      1. Tìm thấy thư mục khớp chính xác; hoặc
      2. Có duy nhất một thư mục khớp sau chuẩn hóa.

    Trả về:
        (đường_dẫn_ảnh_đã_sửa, tên_thư_mục_thực_tế, ảnh_tồn_tại)
    """
    if pd.isna(image_path) or not str(image_path).strip():
        return image_path, None, False

    # Chuẩn hóa dấu phân cách đường dẫn.
    raw_path = str(image_path).strip().replace("\\", "/")
    path = Path(raw_path)

    # Hỗ trợ cả đường dẫn dạng img/style/file.jpg
    # và đường dẫn tương đối chỉ gồm style/file.jpg.
    parts = path.parts

    if len(parts) < 2:
        return raw_path, None, False

    # Tìm vị trí thư mục style trong đường dẫn.
    if parts[0].lower() == "img":
        if len(parts) < 3:
            return raw_path, None, False
        original_style = parts[1]
        filename = parts[-1]
    else:
        original_style = parts[-2]
        filename = parts[-1]

    # 1. Ưu tiên tên thư mục khớp chính xác.
    actual_styles = {
        style
        for styles in style_lookup.values()
        for style in styles
    }

    if original_style in actual_styles:
        matched_style = original_style
    else:
        # 2. Nếu không khớp, so sánh tên đã chuẩn hóa.
        key = normalize_style_name(original_style)
        matches = style_lookup.get(key, [])

        # Chỉ tự sửa nếu tìm được đúng một kết quả.
        if len(matches) != 1:
            return raw_path, None, False

        matched_style = matches[0]

    # Tạo đường dẫn mới theo thư mục thực tế.
    corrected_path = Path("img") / matched_style / filename
    full_path = PROJECT_ROOT / corrected_path

    exists = full_path.is_file()

    return corrected_path.as_posix(), matched_style, exists


# ============================================================
# CHUẨN HÓA FILE METADATA
# ============================================================

def normalize_metadata(metadata_path):
    """
    Chuẩn hóa metadata và cập nhật trực tiếp file metadata_clean.csv.

    Các cột khác được giữ nguyên.
    """
    if not metadata_path.exists():
        print(f"[SKIP] Không tìm thấy file: {metadata_path}")
        return

    print(f"\n[INFO] Đang xử lý: {metadata_path}")

    meta = pd.read_csv(metadata_path)

    if "image_path" not in meta.columns:
        print(
            f"[ERROR] File {metadata_path.name} không có cột image_path."
        )
        return

    actual_styles = get_actual_style_folders()
    style_lookup = build_style_lookup(actual_styles)

    corrected_paths = []
    style_names = []
    image_exists = []

    for image_path in meta["image_path"]:
        corrected_path, style_name, exists = fix_image_path(
            image_path,
            style_lookup,
        )

        corrected_paths.append(corrected_path)
        style_names.append(
            style_name.replace("_", " ") if style_name else None
        )
        image_exists.append(exists)

    # Cập nhật đường dẫn ảnh.
    meta["image_path"] = corrected_paths

    # Cập nhật style_name.
    meta["style_name"] = style_names

    # Ghi nhận ảnh có thực sự tồn tại hay không.
    meta["image_exists"] = image_exists

    # Tạo image_id nếu chưa có.
    # Không ghi đè image_id hiện có để tránh làm thay đổi ID.
    if "image_id" not in meta.columns:
        meta.insert(
            0,
            "image_id",
            [f"img_{i:08d}" for i in range(len(meta))],
        )

    # Ghi lại file CSV.
    meta.to_csv(metadata_path, index=False)

    # Thống kê kết quả.
    total = len(meta)
    existing = sum(image_exists)
    missing = total - existing
    styles_count = meta["style_name"].nunique(dropna=True)

    print(f"[OK] Đã lưu: {metadata_path}")
    print(f"     Tổng số ảnh:       {total:,}")
    print(f"     Ảnh tồn tại:       {existing:,}")
    print(f"     Ảnh bị thiếu:      {missing:,}")
    print(f"     Số style:          {styles_count:,}")

    if missing:
        print("\n[WARNING] Một số đường dẫn ảnh vẫn chưa khớp.")

        missing_rows = meta.loc[
            ~meta["image_exists"],
            ["image_path", "style_name"],
        ]

        print("Một số đường dẫn cần kiểm tra:")
        print(missing_rows.head(15).to_string(index=False))


# ============================================================
# CHẠY CHO BASELINE VÀ FINETUNED
# ============================================================

if __name__ == "__main__":
    for dataset in DATASETS:
        metadata_path = (
            DATA_ROOT / dataset / "metadata_clean.csv"
        )

        normalize_metadata(metadata_path)

    print("\n[FINISHED] Đã xử lý metadata.")

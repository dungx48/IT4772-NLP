# -*- coding: utf-8 -*-
"""Chuẩn bị đặc trưng ENRICO cho thí nghiệm 2 (DAIN đơn giản hóa).

Bước 1: giải nén screenshots.zip và hierarchies.zip vào data/enrico/.
Bước 2: với mỗi màn hình có trong design_topics.csv:
    - modality ảnh : screenshot -> resize 64x64, chuẩn hóa [0,1];
    - modality text : từ view hierarchy lấy class name, resource-id, nội dung text
      của các phần tử UI -> bag-of-words băm 3000 chiều (TF kiểu log);
    - nhãn: design topic.
Lưu thành data/enrico_features.npz để huấn luyện không phải xử lý lại ảnh.

GHI RÕ: encoder gốc của paper DAIN là ResNet-50 (ảnh) + BERT-base (text) với
embedding 512 chiều. Ở đây dùng encoder nhẹ để chạy được trên CPU. Đây là sự
ĐƠN GIẢN HÓA có chủ ý và được ghi rõ trong báo cáo; số liệu không so sánh
trực tiếp với số của paper.

Chạy:  python -m src.exp2.prepare_enrico
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from ..common.utils import DATA_DIR

HASH_DIM = 3000
IMG_SIZE = 64
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def unzip(src: Path, dst: Path) -> None:
    if dst.exists() and any(dst.iterdir()):
        return
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(dst)


def hierarchy_tokens(path: Path) -> list[str]:
    """Rút token văn bản từ view hierarchy: class, resource-id và text hiển thị."""
    toks: list[str] = []
    try:
        tree = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return toks

    def walk(node):
        if isinstance(node, dict):
            for key in ("class", "resource-id"):
                v = node.get(key)
                if isinstance(v, str) and v:
                    toks.extend(_TOKEN.findall(v.replace("$", " ").replace(".", " ")))
            v = node.get("text")
            if isinstance(v, str) and v:
                toks.extend(_TOKEN.findall(v))
            for child in node.get("children", []):
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(tree)
    return toks


def hash_features(tokens: list[str]) -> np.ndarray:
    """Bag-of-words băm: mỗi token chiếu vào một chiều ngẫu nhiên cố định
    (dựa trên hash của chính token — xác định, chạy lại được)."""
    vec = np.zeros(HASH_DIM, dtype=np.float32)
    counts = Counter(tokens)
    for tok, c in counts.items():
        h = int(hashlib_md5(tok), 16)
        vec[h % HASH_DIM] += c
    if vec.sum() > 0:
        vec = np.log1p(vec)  # chuẩn hóa log mềm, giảm ảnh hưởng token lặp nhiều
    return vec


def hashlib_md5(tok: str) -> str:
    import hashlib

    return hashlib.md5(tok.encode("utf-8")).hexdigest()


def load_screenshot(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # (3, H, W)


def main() -> None:
    shots = DATA_DIR / "enrico" / "screenshots"
    hier = DATA_DIR / "enrico" / "hierarchies"
    unzip(DATA_DIR / "enrico_screenshots.zip", shots)
    unzip(DATA_DIR / "enrico_hierarchies.zip", hier)

    topics: dict[str, str] = {}
    with open(DATA_DIR / "enrico_design_topics.csv", encoding="utf-8") as f:
        next(f)  # bỏ dòng header
        for line in f:
            sid, topic = line.strip().split(",")
            topics[sid] = topic
    print(f"design_topics.csv: {len(topics)} màn hình, {len(set(topics.values()))} nhãn")

    xs_img, xs_txt, ys, ids = [], [], [], []
    skipped = 0
    for sid, topic in sorted(topics.items()):
        # Zip gốc giải nén ra screenshots/<sid>.jpg và hierarchies/<sid>.json
        # (có thể lồng một cấp thư mục cùng tên — dùng rglob để bắt cả hai trường hợp)
        img_path = next(iter(shots.rglob(f"{sid}.jpg")), None)
        hier_path = next(iter(hier.rglob(f"{sid}.json")), None)
        if img_path is None or hier_path is None:
            skipped += 1
            continue
        xs_img.append(load_screenshot(img_path))
        xs_txt.append(hash_features(hierarchy_tokens(hier_path)))
        ys.append(topic)
        ids.append(sid)

    X_img = np.stack(xs_img)
    X_txt = np.stack(xs_txt)
    y = np.array(ys)
    print(f"Xuất {len(y)} mẫu (bỏ {skipped} thiếu file); ảnh {X_img.shape}, text {X_txt.shape}")

    # Chia train/val/test theo tỉ lệ 70/10/20, phân tầng theo nhãn, seed cố định.
    # (Paper DAIN dùng 'established benchmark splits'; ở đây dùng chia ngẫu nhiên
    # phân tầng có seed vì không xác minh được đúng split gốc.)
    idx_all = np.arange(len(y))
    idx_train, idx_tmp = train_test_split(
        idx_all, test_size=0.30, random_state=42, stratify=y
    )
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=2 / 3, random_state=42, stratify=y[idx_tmp]
    )

    out = DATA_DIR / "enrico_features.npz"
    np.savez_compressed(
        out,
        X_img=X_img,
        X_txt=X_txt,
        y=y,
        ids=np.array(ids),
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )
    classes = sorted(set(y))
    print("Nhãn:", classes)
    print(f"Lưu {out} — train/val/test = {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")


if __name__ == "__main__":
    main()

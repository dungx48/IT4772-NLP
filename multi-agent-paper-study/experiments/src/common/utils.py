# -*- coding: utf-8 -*-
"""Hàm dùng chung cho hai thí nghiệm."""

import json
import random
import sys
from pathlib import Path

import numpy as np

# Console Windows thường dùng cp1252 — ép UTF-8 để in tiếng Việt không lỗi
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[3]  # .../multi-agent-paper-study
DATA_DIR = ROOT / "experiments" / "data"
RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "experiments" / "figures"


def set_seed(seed: int) -> None:
    """Cố định seed cho random, numpy để chạy lại được."""
    random.seed(seed)
    np.random.seed(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def token_count(text: str) -> int:
    """Số token xấp xỉ = số từ (tách theo khoảng trắng). Đủ dùng để so sánh tương đối."""
    return len(text.split())


def f1_binary(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# -*- coding: utf-8 -*-
"""Bộ ghép câu trả lời mô phỏng (thí nghiệm 1).

Vì không chạy LLM thật, tác vụ trả lời được mô phỏng bằng bài toán nhận diện
câu trả lời đúng trong một tập ứng viên: mỗi mẫu có 1 câu trả lời vàng
(answer của HotpotQA) + K-1 ứng viên nhiễu lấy ngẫu nhiên từ các mẫu khác
(cách dựng phổ biến cho bài toán answer selection).

Bộ ghép chấm điểm mỗi ứng viên bằng mức khớp từ ngữ giữa ứng viên và NGỮ CẢNH
ĐƯỢC CHỌN (số lần các từ của ứng viên xuất hiện trong ngữ cảnh, chuẩn hóa theo
độ dài). Ứng viên có hỗ trợ nhiều nhất trong ngữ cảnh được chọn sẽ thắng.

Đây là một "reader" từ vựng đơn giản — chỉ đo được việc ngữ cảnh được chọn có
còn giữ lại bằng chứng ủng hộ câu trả lời đúng hay không. KHÔNG phải phép đo
năng lực suy luận của LLM.
"""

from __future__ import annotations

import re

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def score_candidate(answer: str, context_text: str, idf: dict[str, float]) -> float:
    """Điểm khớp = tổng IDF của các từ ứng viên xuất hiện trong ngữ cảnh,
    chia cho tổng IDF của toàn bộ ứng viên (chuẩn hóa theo độ dài ứng viên)."""
    toks = tokenize(answer)
    if not toks:
        return 0.0
    ctx = set(tokenize(context_text))
    total = sum(idf.get(t, 1.0) for t in set(toks))
    if total == 0:
        return 0.0
    hit = sum(idf.get(t, 1.0) for t in set(toks) if t in ctx)
    return hit / total


def build_idf(candidate_pool: list[str]) -> dict[str, float]:
    """IDF thô: tanh many câu trả lời ứng viên chứa từ đó."""
    df: dict[str, int] = {}
    for ans in candidate_pool:
        for t in set(tokenize(ans)):
            df[t] = df.get(t, 0) + 1
    n = max(1, len(candidate_pool))
    return {t: float(np.log((n + 1) / (c + 1))) for t, c in df.items()}


def pick_answer(
    candidates: list[str],
    context_sentences: list[dict],
    idf: dict[str, float],
) -> tuple[str, float]:
    """Chọn ứng viên có điểm hỗ trợ cao nhất trong ngữ cảnh được chọn.
    Trả về (ứng viên, điểm)."""
    ctx_text = " ".join(s["text"] for s in context_sentences)
    scores = [score_candidate(c, ctx_text, idf) for c in candidates]
    best = int(np.argmax(scores))
    return candidates[best], float(scores[best])

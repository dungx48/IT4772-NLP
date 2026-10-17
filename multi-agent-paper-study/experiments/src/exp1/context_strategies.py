# -*- coding: utf-8 -*-
"""Các chiến lược chọn ngữ cảnh cho thí nghiệm 1 (minh họa Agent-Radar).

Mỗi câu trong lịch sử trao đổi được biểu diễn dưới dạng một dict:
    {
        "text": str,          # nội dung câu
        "agent": int,         # tác giả của câu (id tác tử)
        "round": int,         # vòng trao đổi (bắt đầu từ 1)
        "hop": int,           # khoảng cách hop từ tác giả tới tác tử đang đọc
        "gold": bool,         # câu có phải evidence vàng (supporting fact) không
    }

Công thức điểm số bám theo paper Agent-Radar (arXiv:2605.30136):
    - Eq. 4:  phi_s = lambda_s^(d_G(v_j, v_i) - 1)     (hop = 1 hoặc chính mình => 1)
    - Eq. 5:  phi_t = lambda_t^(t - tau - 1)           (tuổi của thông điệp)
    - Eq. 7:  phi_sem = cosine(Enc(s_k), Enc(q))
    - Eq. 8:  score = phi_s * phi_t * phi_sem
    - Eq. 9:  Z_hat = {s_k | score >= theta}  (truy vấn luôn được giữ lại)

LƯU Ý QUAN TRỌNG: đây là PHIÊN BẢN MINH HỌA. Paper gốc dùng cơ chế chọn ngữ cảnh
này kết hợp với attention steering (SPA) bên trong một LLM; ở đây không có LLM,
nên ta chỉ đánh giá chất lượng của BỘ CHỌN NGỮ CẢNH, không đánh giá hiệu ứng
khuếch đại attention lên câu trả lời của LLM.
"""

from __future__ import annotations

import numpy as np


def build_graph(n_agents: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Sinh đồ thị giao tiếp có hướng ngẫu nhiên (theo setup 'random topology' của paper).

    Trả về ma trận kề nhị phân A, A[i, j] = 1 nếu có cạnh từ i tới j.
    Khoảng cách hop được tính trên phiên bản vô hướng của đồ thị.
    """
    a = (rng.random((n_agents, n_agents)) < edge_prob).astype(int)
    np.fill_diagonal(a, 0)
    return a


def hop_distances(adj: np.ndarray) -> np.ndarray:
    """d[i, j] = số hop ngắn nhất từ i tới j trên đồ thị vô hướng; không tới được = lớn."""
    n = len(adj)
    undirected = ((adj + adj.T) > 0).astype(int)
    d = np.full((n, n), n + 1, dtype=int)  # "vô cực" = n+1
    np.fill_diagonal(d, 0)
    # BFS từ mỗi đỉnh
    for src in range(n):
        frontier = [src]
        seen = {src}
        dist = 0
        while frontier:
            dist += 1
            nxt = []
            for u in frontier:
                for v in np.nonzero(undirected[u])[0]:
                    if v not in seen:
                        seen.add(int(v))
                        nxt.append(int(v))
                        d[src, v] = dist
            frontier = nxt
    return d


def spatial_decay(hop: int, lambda_s: float) -> float:
    """Eq. 4: phi_s = lambda_s^(hop - 1). hop <= 1 (láng giềng trực tiếp hoặc chính mình) => 1."""
    if hop <= 1:
        return 1.0
    return float(lambda_s ** (hop - 1))


def temporal_decay(round_t: int, current_round: int, lambda_t: float) -> float:
    """Eq. 5: phi_t = lambda_t^(t - tau - 1). Thông điệp mới nhất (cùng vòng) => 1."""
    age = max(0, current_round - round_t - 1)
    return float(lambda_t ** age)


def semantic_similarities(sentences: list[str], query: str, encoder) -> np.ndarray:
    """Eq. 6-7: cosine similarity giữa từng câu và truy vấn hiện tại.

    `encoder` là đối tượng có phương thức encode(list[str]) -> np.ndarray.
    """
    if not sentences:
        return np.zeros(0)
    embs = encoder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    q = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    return embs @ q  # với vector đã chuẩn hóa, tích vô hướng = cosine


def select_full(sentences: list[str], **_) -> list[dict]:
    """Baseline 1: giữ toàn bộ lịch sử (không chọn lọc)."""
    return list(sentences)


def select_window(sentences: list[str], current_round: int, **_) -> list[dict]:
    """Baseline 2: chỉ giữ thông điệp ở vòng gần nhất."""
    return [s for s in sentences if s["round"] == current_round]


def select_semantic(
    sentences: list[str],
    sims: np.ndarray,
    theta: float,
    **_,
) -> list[dict]:
    """Baseline 3: chỉ lọc theo tương đồng ngữ nghĩa (tương ứng ablation
    'w/o spatial & temporal decay' trong paper)."""
    return [s for s, sim in zip(sentences, sims) if sim >= theta]


def select_radar(
    sentences: list[dict],
    sims: np.ndarray,
    current_round: int,
    lambda_s: float,
    lambda_t: float,
    theta: float,
) -> list[dict]:
    """Phiên bản minh họa Agent-Radar: score = phi_s * phi_t * cosine (Eq. 3-9)."""
    selected = []
    for s, sim in zip(sentences, sims):
        score = (
            spatial_decay(s["hop"], lambda_s)
            * temporal_decay(s["round"], current_round, lambda_t)
            * float(sim)
        )
        if score >= theta:
            selected.append(s)
    return selected


def run_selection(
    strategy: str,
    sentences: list[dict],
    sims: np.ndarray,
    cfg: dict,
) -> list[dict]:
    """Điều phối: chạy một chiến lược theo tên và cấu hình."""
    if strategy == "full":
        return select_full(sentences)
    if strategy == "window":
        return select_window(sentences, current_round=cfg["current_round"])
    if strategy == "semantic":
        return select_semantic(sentences, sims, theta=cfg["theta"])
    if strategy == "radar":
        return select_radar(
            sentences,
            sims,
            current_round=cfg["current_round"],
            lambda_s=cfg["lambda_s"],
            lambda_t=cfg["lambda_t"],
            theta=cfg["theta"],
        )
    if strategy == "semantic_topk":
        return select_topk(sentences, sims, k=cfg["k"], decay=None,
                           current_round=cfg["current_round"],
                           lambda_s=cfg["lambda_s"], lambda_t=cfg["lambda_t"])
    if strategy == "radar_topk":
        return select_topk(sentences, sims, k=cfg["k"], decay=True,
                           current_round=cfg["current_round"],
                           lambda_s=cfg["lambda_s"], lambda_t=cfg["lambda_t"])
    raise ValueError(f"Chiến lược không tồn tại: {strategy}")


def select_topk(
    sentences: list[dict],
    sims: np.ndarray,
    k: int,
    decay: bool | None,
    current_round: int,
    lambda_s: float,
    lambda_t: float,
) -> list[dict]:
    """So sánh xếp hạng tại cùng ngân sách k câu.

    decay=None : xếp theo cosine thuần (semantic top-k);
    decay=True : xếp theo cosine * suy giảm không-gian-thời gian (radar top-k).
    Cùng chọn đúng k câu nên khác biệt chỉ còn ở THỨ HẠNG — tách bạch đúng
    vai trò của hai hệ số suy giảm trong paper.
    """
    if not sentences:
        return []
    scores = np.asarray(sims, dtype=float).copy()
    if decay:
        for i, s in enumerate(sentences):
            scores[i] *= (
                spatial_decay(s["hop"], lambda_s)
                * temporal_decay(s["round"], current_round, lambda_t)
            )
    order = np.argsort(-scores)[:k]
    return [sentences[i] for i in order]

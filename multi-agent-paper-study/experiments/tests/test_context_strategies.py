# -*- coding: utf-8 -*-
"""Unit test cho các chiến lược chọn ngữ cảnh (thí nghiệm 1).

Kiểm tra các tính chất bắt buộc của công thức trong paper Agent-Radar:
- spatial decay: hop <= 1 => 1.0; giảm theo hop
- temporal decay: tin nhắn cùng vòng => 1.0; giảm theo tuổi
- radar: score >= theta mới được chọn; truy vấn luôn giữ (do runner đảm bảo)
"""

import numpy as np

from src.exp1.context_strategies import (
    hop_distances,
    run_selection,
    select_semantic,
    select_window,
    spatial_decay,
    temporal_decay,
)


def test_spatial_decay():
    assert spatial_decay(0, 0.9) == 1.0
    assert spatial_decay(1, 0.9) == 1.0
    assert spatial_decay(2, 0.9) == 0.9
    assert abs(spatial_decay(3, 0.92) - 0.92 ** 2) < 1e-12
    assert spatial_decay(4, 0.9) < spatial_decay(3, 0.9)


def test_temporal_decay():
    # tin nhắn ở vòng t khi đang ở vòng t (vừa nhận) => age = t - t - 1 < 0 => max(0,..) = 0
    assert temporal_decay(3, 3, 0.9) == 1.0
    assert temporal_decay(2, 3, 0.9) == 1.0  # age = 0
    assert temporal_decay(1, 3, 0.9) == 0.9  # age = 1
    assert abs(temporal_decay(1, 5, 0.92) - 0.92 ** 3) < 1e-12


def test_hop_distances_chain():
    # 0 - 1 - 2 (đường thẳng): d(0,2) = 2
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = adj[1, 2] = 1
    d = hop_distances(adj)
    assert d[0, 1] == 1
    assert d[0, 2] == 2
    assert d[1, 2] == 1
    assert d[2, 0] == 2  # vô hướng


def _fake_msgs():
    return [
        {"text": "gold nearby", "agent": 1, "round": 3, "hop": 1, "gold": True},
        {"text": "gold far old", "agent": 2, "round": 1, "hop": 3, "gold": True},
        {"text": "noise near recent", "agent": 3, "round": 3, "hop": 1, "gold": False},
        {"text": "noise far", "agent": 2, "round": 2, "hop": 3, "gold": False},
    ]


def _sims():
    return np.array([0.9, 0.9, 0.9, 0.9])


def test_window_keeps_only_last_round():
    out = select_window(_fake_msgs(), current_round=3)
    assert all(m["round"] == 3 for m in out)
    assert len(out) == 2


def test_semantic_threshold():
    sims = np.array([0.9, 0.5, 0.7, 0.1])
    out = select_semantic(_fake_msgs(), sims, theta=0.65)
    assert len(out) == 2
    assert out[0]["text"] == "gold nearby"
    assert out[1]["text"] == "noise near recent"


def test_radar_demotes_far_old_messages():
    cfg = {"current_round": 3, "lambda_s": 0.92, "lambda_t": 0.92, "theta": 0.65}
    # Mọi câu đều có cosine 0.9: chỉ câu gần & mới qua ngưỡng.
    # score câu hop=3, round=1: 0.92^2 * 0.92^(3-1-1) * 0.9 ~ 0.75 (qua)
    # score câu hop=3, round=2: 0.92^2 * 0.92^0 * 0.9 ~ 0.762 (qua)
    # => hạ lambda_s xuống thấp hơn để thấy rõ sự loại bỏ
    cfg["lambda_s"] = 0.5
    out = run_selection("radar", _fake_msgs(), _sims(), cfg)
    texts = {m["text"] for m in out}
    assert "gold nearby" in texts          # hop=1, mới
    assert "noise near recent" in texts    # hop=1, mới (nhiễu gần vẫn qua — đúng hành vi paper)
    assert "gold far old" not in texts     # 0.5^2 * 0.5^1 * 0.9 = 0.1125 < 0.65
    assert "noise far" not in texts        # 0.5^2 * 0.9 = 0.225 < 0.65


def test_run_selection_unknown():
    try:
        run_selection("khongton tai", [], np.zeros(0), {})
        assert False, "phải raise ValueError"
    except ValueError:
        pass

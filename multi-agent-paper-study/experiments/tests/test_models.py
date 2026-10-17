# -*- coding: utf-8 -*-
"""Unit test cho các mô hình thí nghiệm 2 (chạy trên tensor giả, CPU)."""

import torch

from src.exp2.models import DAIN, DAINCore, EarlyFusion, LateFusion, SparseMoE, StaticMoE

B, TXT_DIM, C = 4, 3000, 6


def _batch():
    torch.manual_seed(0)
    return torch.rand(B, 3, 64, 64), torch.rand(B, TXT_DIM), torch.randint(0, C, (B,))


def test_dain_core_shapes():
    core = DAINCore(n_modalities=2, n_agents=8, max_tau=4)
    e1, e2 = torch.rand(B, 256), torch.rand(B, 256)
    e_c = core.context_vector([e1, e2])
    assert e_c.shape == (B, 256)
    g = core.gates(e_c)
    assert g.shape == (B, 8)
    # Đúng đúng tau agent được kích hoạt (g > 0)
    assert ((g > 0).sum(1) == 4).all()
    Phi = core.comm_graph(e_c)
    assert Phi.shape == (B, 8, 8)
    assert (Phi >= 0).all() and (Phi <= 1).all()
    # Đường chéo không có tự-nối sau agent_stage
    h1 = core.agent_stage(e_c, g, Phi)
    assert h1.shape == (B, 8, 256)
    feat = core.consensus(h1, g)
    assert feat.shape == (B, 256)
    # Consensus chỉ dùng tác tử đang bật: gate tắt hết một agent phải không lỗi
    g2 = g.clone()
    g2[:, 0] = 0.0
    feat2 = core.consensus(h1, g2)
    assert feat2.shape == (B, 256)


def test_dain_full_training_step():
    model = DAIN(TXT_DIM, C, max_tau=4)
    xi, xt, y = _batch()
    model.train()
    logits, stats = model(xi, xt, y)
    assert logits.shape == (B, C)
    stats["loss"].backward()  # gradient chảy qua toàn đường truyền


def test_dain_variants():
    xi, xt, y = _batch()
    for variant, expect_tau in [("static", 8), ("nocomm", 4), ("full", 4)]:
        model = DAIN(TXT_DIM, C, max_tau=4, variant=variant)
        model.eval()
        with torch.no_grad():
            _, stats = model(xi, xt)
        assert abs(stats["tau_bar"] - expect_tau) < 1e-6, (variant, stats)
        if variant == "nocomm":
            assert stats["rho_bar"] >= 0  # Phi vẫn được tính, chỉ không dùng


def test_dain_static_always_active():
    core = DAINCore(n_modalities=2, always_active=True)
    g = core.gates(torch.rand(B, 256))
    assert (g == 1).all()


def test_baselines():
    xi, xt, y = _batch()
    for cls in [EarlyFusion, LateFusion, StaticMoE]:
        m = cls(TXT_DIM, C)
        m.train()
        logits, stats = m(xi, xt, y)
        stats["loss"].backward()
    m = SparseMoE(TXT_DIM, C, tau=4)
    m.eval()
    with torch.no_grad():
        _, stats = m(xi, xt)
    assert abs(stats["tau_bar"] - 4) < 1e-6


def test_specialization_loss_positive():
    core = DAINCore(n_modalities=2)
    e1, e2 = torch.rand(B, 256), torch.rand(B, 256)
    ls = core.specialization_loss([e1, e2])
    assert ls.item() >= 0.0
    assert ls.item() < 2.0  # D chuẩn hóa, không thể quá lớn


def test_agent_types_allocation():
    core = DAINCore(n_modalities=2, n_agents=8, synergy=3, redundancy=3)
    assert core.agent_types.count("synergy") == 3
    assert core.agent_types.count("uniqueness") == 2  # n=2 modality
    assert core.agent_types.count("redundancy") == 3

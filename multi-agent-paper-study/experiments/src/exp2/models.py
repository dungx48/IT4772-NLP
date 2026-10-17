# -*- coding: utf-8 -*-
"""Các mô hình cho thí nghiệm 2: DAIN (đơn giản hóa) và các baseline.

Kiến trúc DAIN bám theo paper arXiv:2606.30189, Eq. 1-11 và Algorithm 1:
    Eq. 1:  e_c = W_c [e1; ...; en] + b_c
    Eq. 2:  g   = Top_tau(Softmax(G_gate(e_c)))        (chọn top-tau agent)
    Eq. 3:  Phi = sigma(G_comm(e_c))                   (đồ thị giao tiếp K x K)
    Eq. 4:  h_k0 = f_k(e_c; theta_k)                   (trạng thái ban đầu)
    Eq. 5:  m_i<-j = B(h_j0; phi_ij), B(h; phi) = phi * (W_B h)
    Eq. 6:  h_i1 = LayerNorm(h_i0 + ReLU(W_u * sum_j m_i<-j))
    Eq. 7:  y_hat = C({h_k1 | g_k > 0})                (multi-head attention)
    Eq. 11: L = L_task + alpha*L_spec + lambda_act*||g||_1 + lambda_comm*||Phi||_F^2

GHI RÕ HAI ĐIỂM ĐƠN GIẢN HÓA SO VỚI GỐC (nêu lại trong báo cáo):
1. Eq. 2 của paper dùng Softmax rồi Top_tau. Nếu giữ Softmax thì tổng ||g||_1
   luôn bằng 1, nên thành phần lambda_act*||g||_1 (Eq. 10) trở thành hằng số
   vô nghĩa. Ở đây dùng gate sigmoid và chọn top-tau theo logit — giữ nguyên
   tính chất "kích hoạt thưa phụ thuộc ngữ cảnh" và làm ||g||_1 có ý nghĩa.
2. Encoder ảnh/text là bản nhẹ (CNN nhỏ + bag-of-words băm) thay cho
   ResNet-50 + BERT-base của paper gốc; embedding 256 thay vì 512.
3. Với n = 2 modality (ảnh, text), phân bổ K = 8 agent là
   3 synergy + 2 uniqueness + 3 redundancy. Paper ghi rõ phân bổ này linh hoạt
   theo task (gốc 3/3/2 với n = 3 modality).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM = 256
LOW_RANK = 32


# --------------------------------------------------------------------------- #
# Encoder nhẹ (đơn giản hóa so với ResNet-50 + BERT-base của paper)
# --------------------------------------------------------------------------- #
class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = EMB_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),  # 32 -> 16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = nn.Linear(32 * 2 * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class TextEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = EMB_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# --------------------------------------------------------------------------- #
# DAIN lõi
# --------------------------------------------------------------------------- #
class DAINCore(nn.Module):
    """Đường truyền Eq. 1-7: context vector, gating, giao tiếp, consensus."""

    def __init__(
        self,
        n_modalities: int,
        emb_dim: int = EMB_DIM,
        n_agents: int = 8,
        synergy: int = 3,
        redundancy: int = 3,  # uniqueness = K - synergy - redundancy
        max_tau: int = 4,
        always_active: bool = False,
    ):
        super().__init__()
        self.K = n_agents
        self.tau = max_tau
        self.n_modalities = n_modalities
        self.always_active = always_active  # dùng cho biến thể DAIN-Static
        self.agent_types = (
            ["synergy"] * synergy
            + ["uniqueness"] * (n_agents - synergy - redundancy)
            + ["redundancy"] * redundancy
        )

        # Eq. 1: context vector từ concatenation các embedding modality
        self.context = nn.Linear(n_modalities * emb_dim, emb_dim)
        # Eq. 2: gate (sigmoid — xem ghi chú đầu tệp)
        self.gate = nn.Linear(emb_dim, n_agents)
        # Eq. 3: đồ thị giao tiếp
        self.comm = nn.Linear(emb_dim, n_agents * n_agents)
        # Eq. 4: hàm xử lý riêng của từng agent
        self.agents = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, emb_dim))
                for _ in range(n_agents)
            ]
        )
        # Eq. 5: bottleneck low-rank W_B (d -> r -> d)
        self.wb_down = nn.Linear(emb_dim, LOW_RANK, bias=False)
        self.wb_up = nn.Linear(LOW_RANK, emb_dim, bias=False)
        # Eq. 6: W_u
        self.update = nn.Linear(emb_dim, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        # Eq. 7: consensus multi-head attention với query học được
        self.query = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)

    def context_vector(self, embs: list[torch.Tensor]) -> torch.Tensor:
        return self.context(torch.cat(embs, dim=-1))  # Eq. 1

    # --- Eq. 2: top-tau gate bằng sigmoid ---
    def gates(self, e_c: torch.Tensor) -> torch.Tensor:
        if self.always_active:  # DAIN-Static: mọi agent luôn bật
            return torch.ones(e_c.size(0), self.K, device=e_c.device)
        logits = self.gate(e_c)  # (B, K)
        sig = torch.sigmoid(logits)
        topk = torch.topk(logits, k=min(self.tau, self.K), dim=-1).indices  # (B, tau)
        mask = torch.zeros_like(sig)
        mask.scatter_(1, topk, 1.0)
        return sig * mask  # g_k > 0 chỉ với top-tau

    # --- Eq. 3: đồ thị giao tiếp ---
    def comm_graph(self, e_c: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.comm(e_c)).view(-1, self.K, self.K)  # (B, K, K)

    # --- Eq. 4-6: chạy agent + trao đổi nén ---
    def agent_stage(self, e_c: torch.Tensor, g: torch.Tensor, Phi: torch.Tensor,
                    use_comm: bool = True) -> torch.Tensor:
        h0 = torch.stack([agent(e_c) for agent in self.agents], dim=1)  # (B, K, d)
        active = (g > 0).float().unsqueeze(-1)                          # (B, K, 1)
        if use_comm:
            # Eq. 5 + dòng 15 Algorithm 1: m_i = sum_{j!=i} Phi_ij * g_j * W_B h_j0
            hb = self.wb_up(self.wb_down(h0))                            # (B, K, d)
            eye = torch.eye(self.K, device=e_c.device).unsqueeze(0)
            phi = Phi * (1 - eye) * active.transpose(1, 2)  # loại tự-nối và agent tắt
            m = torch.bmm(phi, hb)                                       # (B, K, d)
            h1 = self.ln(h0 + F.relu(self.update(m)))                    # Eq. 6
        else:
            h1 = self.ln(h0)
        return h1 * active  # agent không kích hoạt: zero, không tham gia consensus

    # --- Eq. 7: consensus fusion ---
    def consensus(self, h1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # Eq. 7: C({h_k^(1) | g_k > 0}) — chỉ hợp nhất các tác tử đang bật,
        # nên các vị trí không kích hoạt bị che khỏi attention (không chỉ là 0).
        q = self.query.unsqueeze(0).expand(h1.size(0), 1, -1)  # (B, 1, d)
        padding = g <= 0  # True = bị che
        out, _ = self.attn(q, h1, h1, key_padding_mask=padding)
        return out.squeeze(1)  # (B, d)

    def forward_core(self, embs: list[torch.Tensor], use_comm: bool = True):
        """Chạy Eq. 1-7. Trả về (đặc trưng consensus, g, Phi)."""
        e_c = self.context_vector(embs)
        g = self.gates(e_c)
        Phi = self.comm_graph(e_c)
        h1 = self.agent_stage(e_c, g, Phi, use_comm=use_comm)
        feat = self.consensus(h1, g)
        return feat, g, Phi

    @torch.no_grad()
    def specialization_loss(self, embs: list[torch.Tensor]) -> torch.Tensor:
        """Eq. 9: trung bình độ phân kỳ đầu ra agent khi masking từng modality.

        P = tập phép nhiễu (ở đây: zero từng modality một);
        D(a, b) = ||a - b||^2 / (||a||^2 + ||b||^2 + eps).
        Hàm không lan truyền gradient (chỉ là tín hiệu điều tiết, giống tinh thần
        'perturbation-based strategy' của paper).
        """
        e_c = self.context_vector(embs)
        g = self.gates(e_c)
        Phi = self.comm_graph(e_c)
        h_base = self.agent_stage(e_c, g, Phi)  # (B, K, d)
        total = h_base.new_zeros(())
        for m in range(self.n_modalities):
            embs_m = [e if i == m else torch.zeros_like(e) for i, e in enumerate(embs)]
            e_c_m = self.context_vector(embs_m)
            h_m = self.agent_stage(e_c_m, self.gates(e_c_m), self.comm_graph(e_c_m))
            num = (h_base - h_m).pow(2).sum(-1)
            den = h_base.pow(2).sum(-1) + h_m.pow(2).sum(-1) + 1e-8
            total = total + (num / den).mean()
        return total / self.n_modalities


class DAIN(nn.Module):
    """DAIN đầy đủ + các biến thể ablation.

    variant:
      - "full"   : gating top-tau + giao tiếp (mặc định)
      - "static" : mọi agent luôn bật (DAIN-Static)
      - "nocomm" : gating top-tau, không giao tiếp (DAIN-NoComm)
    """

    def __init__(
        self,
        text_in_dim: int,
        n_classes: int,
        n_agents: int = 8,
        max_tau: int = 4,
        variant: str = "full",
        alpha: float = 0.1,
        lambda_act: float = 0.01,
        lambda_comm: float = 0.001,
        use_spec_loss: bool = True,
    ):
        super().__init__()
        self.variant = variant
        self.alpha = alpha
        self.lambda_act = lambda_act
        self.lambda_comm = lambda_comm
        self.use_spec_loss = use_spec_loss
        self.img_enc = ImageEncoder()
        self.txt_enc = TextEncoder(text_in_dim)
        self.core = DAINCore(
            n_modalities=2, n_agents=n_agents, max_tau=max_tau,
            always_active=(variant == "static"),
        )
        self.head = nn.Sequential(nn.LayerNorm(EMB_DIM), nn.Linear(EMB_DIM, n_classes))

    def forward(self, x_img: torch.Tensor, x_txt: torch.Tensor, y: torch.Tensor = None):
        embs = [self.img_enc(x_img), self.txt_enc(x_txt)]
        use_comm = self.variant != "nocomm"
        feat, g, Phi = self.core.forward_core(embs, use_comm=use_comm)
        logits = self.head(feat)

        eye = torch.eye(self.core.K, device=Phi.device)
        stats = {
            "tau_bar": (g > 0).float().sum(1).mean().item(),
            "rho_bar": (Phi * (1 - eye)).mean().item(),
        }
        if not self.training:
            return logits, stats

        loss = F.cross_entropy(logits, y)  # Eq. 8
        if self.use_spec_loss:
            loss = loss + self.alpha * self.core.specialization_loss(embs)  # Eq. 9
        loss = loss + self.lambda_act * g.abs().sum(1).mean()               # Eq. 10 (L1 trên g)
        loss = loss + self.lambda_comm * (Phi * (1 - eye)).pow(2).sum(dim=(1, 2)).mean()
        stats["loss"] = loss
        return logits, stats


# --------------------------------------------------------------------------- #
# Baseline fusion tĩnh
# --------------------------------------------------------------------------- #
class EarlyFusion(nn.Module):
    """Baseline 1: nối đặc trưng hai modality rồi MLP phân loại."""

    def __init__(self, text_in_dim: int, n_classes: int):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.txt_enc = TextEncoder(text_in_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * EMB_DIM, EMB_DIM), nn.ReLU(), nn.Linear(EMB_DIM, n_classes)
        )

    def forward(self, x_img, x_txt, y: torch.Tensor = None):
        feat = torch.cat([self.img_enc(x_img), self.txt_enc(x_txt)], dim=-1)
        logits = self.head(feat)
        if not self.training:
            return logits, {}
        return logits, {"loss": F.cross_entropy(logits, y)}


class LateFusion(nn.Module):
    """Baseline 2: mỗi modality ra dự đoán riêng, hợp nhất bằng attention có trọng số."""

    def __init__(self, text_in_dim: int, n_classes: int):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.txt_enc = TextEncoder(text_in_dim)
        self.img_head = nn.Linear(EMB_DIM, n_classes)
        self.txt_head = nn.Linear(EMB_DIM, n_classes)
        self.attn = nn.Linear(2 * EMB_DIM, 2)

    def forward(self, x_img, x_txt, y: torch.Tensor = None):
        fi, ft = self.img_enc(x_img), self.txt_enc(x_txt)
        pi, pt = self.img_head(fi), self.txt_head(ft)
        w = torch.softmax(self.attn(torch.cat([fi, ft], dim=-1)), dim=-1)
        logits = w[:, :1] * pi + w[:, 1:] * pt
        if not self.training:
            return logits, {}
        return logits, {"loss": F.cross_entropy(logits, y)}


class _MoEBase(nn.Module):
    """Phần dùng chung của MoE / Sparse MoE: encoder + context vector Eq. 1 + experts."""

    def __init__(self, text_in_dim: int, n_classes: int, n_experts: int = 8):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.txt_enc = TextEncoder(text_in_dim)
        self.context = nn.Linear(2 * EMB_DIM, EMB_DIM)
        self.router = nn.Linear(EMB_DIM, n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(EMB_DIM, 128), nn.ReLU(), nn.Linear(128, EMB_DIM))
                for _ in range(n_experts)
            ]
        )
        self.head = nn.Sequential(nn.LayerNorm(EMB_DIM), nn.Linear(EMB_DIM, n_classes))

    def _route(self, e_c: torch.Tensor):
        raise NotImplementedError

    def forward(self, x_img, x_txt, y: torch.Tensor = None):
        e_c = self.context(torch.cat([self.img_enc(x_img), self.txt_enc(x_txt)], dim=-1))
        w = self._route(e_c)                                      # (B, K)
        h = torch.stack([ex(e_c) for ex in self.experts], dim=1)  # (B, K, d)
        feat = (w.unsqueeze(-1) * h).sum(1)
        logits = self.head(feat)
        if not self.training:
            return logits, {"tau_bar": (w > 0).float().sum(1).mean().item()}
        return logits, {"loss": F.cross_entropy(logits, y)}


class StaticMoE(_MoEBase):
    """Baseline 3: MoE — router softmax trên cả K expert, tổng có trọng số."""

    def _route(self, e_c: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.router(e_c), dim=-1)


class SparseMoE(_MoEBase):
    """Baseline 4: Sparse MoE — chỉ top-tau expert được tổng hợp (renormalize)."""

    def __init__(self, text_in_dim: int, n_classes: int, n_experts: int = 8, tau: int = 4):
        super().__init__(text_in_dim, n_classes, n_experts)
        self.tau = tau

    def _route(self, e_c: torch.Tensor) -> torch.Tensor:
        logits_r = self.router(e_c)
        topk = torch.topk(logits_r, k=min(self.tau, logits_r.size(1)), dim=-1)
        w = torch.softmax(topk.values, dim=-1)
        return torch.zeros_like(logits_r).scatter(1, topk.indices, w)

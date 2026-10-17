# -*- coding: utf-8 -*-
"""Sinh biểu đồ từ kết quả thực nghiệm (không dùng số tự nghĩ).

Vẽ từ các file CSV trong results/:
    exp1_rounds_summary.csv, exp1_agents_summary.csv, exp2_main.csv, exp2_tau_sweep.csv
Xuất PNG 300dpi vào figures/ để chèn vào báo cáo LaTeX.

Chạy:  python -m src.make_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common.utils import FIGURES_DIR, RESULTS_DIR

# Bảng màu phân loại đã qua kiểm định CVD (dataviz skill) — thứ tự cố định theo
# thực thể: full=blue, window=orange, semantic=aqua, radar=yellow
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4e0"

LABEL_VI = {
    "full": "Full history",
    "window": "Cửa sổ vòng cuối",
    "semantic": "Semantic (chỉ tương đồng)",
    "radar": "Radar (minh họa)",
}

MODEL_LABEL_VI = {
    "early": "Early Fusion",
    "late": "Late Fusion",
    "moe": "Static MoE",
    "sparse_moe": "Sparse MoE",
    "dain_static": "DAIN-Static",
    "dain_nocomm": "DAIN-NoComm",
    "dain": "DAIN (full)",
}


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK2)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8, labelcolor=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_exp1_rounds():
    """Hình 1: Evidence-F1 và số token theo số vòng trao đổi (hai panel riêng)."""
    df = pd.read_csv(RESULTS_DIR / "exp1_rounds_summary.csv")
    strategies = ["full", "window", "semantic", "radar"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax = axes[0]
    for i, s in enumerate(strategies):
        d = df[df["strategy"] == s].sort_values("n_rounds")
        ax.plot(d["n_rounds"], d["evidence_f1"], color=CAT[i], linewidth=2,
                marker="o", markersize=4, label=LABEL_VI[s])
    ax.set_xlabel("Số vòng trao đổi", fontsize=9, color=INK)
    ax.set_ylabel("Evidence-F1", fontsize=9, color=INK)
    style_ax(ax)

    ax = axes[1]
    for i, s in enumerate(strategies):
        d = df[df["strategy"] == s].sort_values("n_rounds")
        ax.plot(d["n_rounds"], d["selected_tokens"], color=CAT[i], linewidth=2,
                marker="o", markersize=4, label=LABEL_VI[s])
    # Nhiễu tham chiếu: tổng token lịch sử (như nhau giữa các chiến lược)
    d0 = df[df["strategy"] == "full"].sort_values("n_rounds")
    ax.plot(d0["n_rounds"], d0["history_tokens"], color=INK2, linewidth=1.2,
            linestyle="--", label="Toàn bộ lịch sử")
    ax.set_xlabel("Số vòng trao đổi", fontsize=9, color=INK)
    ax.set_ylabel("Số từ trong ngữ cảnh được chọn", fontsize=9, color=INK)
    style_ax(ax)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp1_rounds.png", dpi=300)
    plt.close(fig)


def fig_exp1_agents():
    """Hình 2: Evidence-F1 theo số tác tử."""
    df = pd.read_csv(RESULTS_DIR / "exp1_agents_summary.csv")
    strategies = ["full", "window", "semantic", "radar"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, s in enumerate(strategies):
        d = df[df["strategy"] == s].sort_values("n_agents")
        ax.plot(d["n_agents"], d["evidence_f1"], color=CAT[i], linewidth=2,
                marker="o", markersize=4, label=LABEL_VI[s])
    ax.set_xlabel("Số tác tử", fontsize=9, color=INK)
    ax.set_ylabel("Evidence-F1", fontsize=9, color=INK)
    style_ax(ax)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp1_agents.png", dpi=300)
    plt.close(fig)


def fig_exp2_main():
    """Hình 3: accuracy các biến thể (bar, mean ± std qua seed)."""
    df = pd.read_csv(RESULTS_DIR / "exp2_main.csv").sort_values("acc_mean")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ypos = range(len(df))
    colors = [CAT[4] if m == "dain" else "#9ec5f4" if m.startswith("dain") else CAT[0]
              for m in df["model"]]
    # DAIN nổi bật bằng slot khác; baseline dùng cùng một màu nhạt để không
    # biến màu thành thứ hạng (màu theo nhóm thực thể: baseline/ablation/DAIN)
    ax.barh(ypos, df["acc_mean"], xerr=df["acc_std"], color=colors,
            height=0.62, error_kw={"linewidth": 1, "ecolor": INK2})
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([MODEL_LABEL_VI[m] for m in df["model"]], fontsize=7.5, color=INK)
    for y, (acc, std) in enumerate(zip(df["acc_mean"], df["acc_std"])):
        ax.text(acc + std + 0.004, y, f"{acc:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.set_xlabel("Accuracy test (trung bình 5 seed)", fontsize=9, color=INK)
    ax.set_xlim(0, min(1.0, df["acc_mean"].max() + 3 * df["acc_std"].max() + 0.08))
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp2_main.png", dpi=300)
    plt.close(fig)


def fig_exp2_tau():
    """Hình 4: accuracy và tau_bar theo tau — hai panel riêng (không dùng trục kép)."""
    df = pd.read_csv(RESULTS_DIR / "exp2_tau_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    ax.errorbar(df["tau"], df["acc_mean"], yerr=df["acc_std"], color=CAT[0],
                linewidth=2, marker="o", markersize=5, capsize=3)
    ax.set_xlabel("tau (số agent kích hoạt tối đa)", fontsize=9, color=INK)
    ax.set_ylabel("Accuracy test", fontsize=9, color=INK)
    style_ax(ax)

    ax = axes[1]
    ax.plot(df["tau"], df["tau_bar"], color=CAT[1], linewidth=2, marker="o", markersize=5)
    ax.plot(df["tau"], df["tau"], color=INK2, linewidth=1.2, linestyle="--",
            label="Trần tau (kích hoạt đầy đủ)")
    ax.set_xlabel("tau (số agent kích hoạt tối đa)", fontsize=9, color=INK)
    ax.set_ylabel("Agent kích hoạt trung bình", fontsize=9, color=INK)
    style_ax(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp2_tau.png", dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_exp1_rounds()
    fig_exp1_agents()
    fig_exp2_main()
    fig_exp2_tau()
    print("Đã sinh biểu đồ vào", FIGURES_DIR)


if __name__ == "__main__":
    main()

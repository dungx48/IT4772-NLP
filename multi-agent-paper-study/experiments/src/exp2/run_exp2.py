# -*- coding: utf-8 -*-
"""Chạy thí nghiệm 2: DAIN (đơn giản hóa) và các baseline trên ENRICO.

Quy trình:
1. Nạp đặc trưng đã chuẩn bị (src/exp2/prepare_enrico.py).
2. Huấn luyện từng biến thể: EarlyFusion, LateFusion, StaticMoE, SparseMoE,
   DAIN-Static, DAIN-NoComm, DAIN (full) — với các seed như paper
   (42, 123, 456, 789, 1024), Adam lr 1e-4, batch 32, early stopping patience 10,
   tối đa 50 epoch (giống §4.1 của paper).
3. Đo accuracy test (mean ± std qua seed), tau_bar (agent kích hoạt trung bình),
   rho_bar (mật độ đồ thị giao tiếp).
4. Quét tau trong {1, 2, 4, 6, 8} cho DAIN full (giống Table 5 của paper).
5. Xuất results/exp2_*.csv và exp2_summary.json.

Chạy:  python -m src.exp2.run_exp2 --config src/exp2/config.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..common.utils import DATA_DIR, RESULTS_DIR, load_json, save_json, set_seed
from .models import DAIN, EarlyFusion, LateFusion, SparseMoE, StaticMoE


def count_params(model: nn.Module, effective: bool = False) -> int:
    """Đếm tham số (tổng); effective=True chỉ tính tham số có gradient khác None
    sau một lần backward thưa — đơn giản hóa: với DAIN ta báo tổng + tỷ lệ gate."""
    return sum(p.numel() for p in model.parameters())


def make_loaders(data: dict, batch_size: int, seed: int):
    set_seed(seed)
    x_img = torch.from_numpy(data["X_img"]).float()
    x_txt = torch.from_numpy(data["X_txt"]).float()
    labels = sorted(set(data["y"].tolist()))
    lab2id = {l: i for i, l in enumerate(labels)}
    y = torch.from_numpy(np.array([lab2id[v] for v in data["y"]])).long()

    def ds(idx):
        return TensorDataset(x_img[idx], x_txt[idx], y[idx])

    gen = torch.Generator().manual_seed(seed)
    return (
        DataLoader(ds(data["idx_train"]), batch_size=batch_size, shuffle=True, generator=gen),
        DataLoader(ds(data["idx_val"]), batch_size=batch_size),
        DataLoader(ds(data["idx_test"]), batch_size=batch_size),
    )


def evaluate(model, loader, device) -> tuple[float, float, float]:
    """Trả về (accuracy, tau_bar, rho_bar)."""
    model.eval()
    correct, total = 0, 0
    tau_vals, rho_vals = [], []
    with torch.no_grad():
        for x_img, x_txt, y in loader:
            x_img, x_txt, y = x_img.to(device), x_txt.to(device), y.to(device)
            logits, stats = model(x_img, x_txt)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
            if "tau_bar" in stats:
                tau_vals.append(stats["tau_bar"])
            if "rho_bar" in stats:
                rho_vals.append(stats["rho_bar"])
    acc = correct / max(1, total)
    return acc, float(np.mean(tau_vals)) if tau_vals else float("nan"), \
        float(np.mean(rho_vals)) if rho_vals else float("nan")


def train_one(model, loaders, device, cfg, seed: int) -> dict:
    tr, va, te = loaders
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999))
    best_val, best_state, patience = -1.0, None, 0
    t0 = time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for x_img, x_txt, y in tr:
            x_img, x_txt, y = x_img.to(device), x_txt.to(device), y.to(device)
            _, stats = model(x_img, x_txt, y)
            opt.zero_grad()
            stats["loss"].backward()
            opt.step()
        val_acc, _, _ = evaluate(model, va, device)
        if val_acc > best_val + 1e-4:
            best_val, patience = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, tau_bar, rho_bar = evaluate(model, te, device)
    return {
        "test_acc": test_acc,
        "val_acc": best_val,
        "tau_bar": tau_bar,
        "rho_bar": rho_bar,
        "epochs": epoch + 1,
        "train_seconds": round(time.time() - t0, 1),
        "total_params": count_params(model),
    }


def build_model(name: str, text_dim: int, n_classes: int, cfg: dict, tau: int):
    if name == "early":
        return EarlyFusion(text_dim, n_classes)
    if name == "late":
        return LateFusion(text_dim, n_classes)
    if name == "moe":
        return StaticMoE(text_dim, n_classes)
    if name == "sparse_moe":
        return SparseMoE(text_dim, n_classes, tau=tau)
    if name == "dain_static":
        return DAIN(text_dim, n_classes, max_tau=tau, variant="static",
                    alpha=cfg["alpha"], lambda_act=cfg["lambda_act"],
                    lambda_comm=cfg["lambda_comm"])
    if name == "dain_nocomm":
        return DAIN(text_dim, n_classes, max_tau=tau, variant="nocomm",
                    alpha=cfg["alpha"], lambda_act=cfg["lambda_act"],
                    lambda_comm=cfg["lambda_comm"])
    if name == "dain":
        return DAIN(text_dim, n_classes, max_tau=tau, variant="full",
                    alpha=cfg["alpha"], lambda_act=cfg["lambda_act"],
                    lambda_comm=cfg["lambda_comm"])
    raise ValueError(name)


MODELS = ["early", "late", "moe", "sparse_moe", "dain_static", "dain_nocomm", "dain"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=str(Path(__file__).parent / "config.json"))
    args = ap.parse_args()
    cfg = load_json(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị: {device}")

    data = dict(np.load(DATA_DIR / "enrico_features.npz", allow_pickle=True))
    labels = sorted(set(data["y"].tolist()))
    n_classes, text_dim = len(labels), data["X_txt"].shape[1]
    print(f"{len(data['y'])} mẫu, {n_classes} lớp, text dim {text_dim}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---- Phần 1: so sánh các biến thể với tau mặc định ----
    for name in MODELS:
        accs, taus, rhos, secs = [], [], [], []
        for seed in cfg["seeds"]:
            set_seed(seed)
            loaders = make_loaders(data, cfg["batch_size"], seed)
            model = build_model(name, text_dim, n_classes, cfg, cfg["tau"]).to(device)
            torch.manual_seed(seed)
            res = train_one(model, loaders, device, cfg, seed)
            accs.append(res["test_acc"])
            taus.append(res["tau_bar"])
            rhos.append(res["rho_bar"])
            secs.append(res["train_seconds"])
        row = {
            "model": name,
            "tau": cfg["tau"],
            "acc_mean": round(float(np.mean(accs)), 4),
            "acc_std": round(float(np.std(accs)), 4),
            "tau_bar": round(float(np.nanmean(taus)), 2),
            "rho_bar": round(float(np.nanmean(rhos)), 3),
            "train_seconds_mean": round(float(np.mean(secs)), 1),
            "total_params": res["total_params"],
        }
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp2_main.csv", index=False)

    # ---- Phần 2: quét tau cho DAIN (giống Table 5 của paper) ----
    tau_rows = []
    for tau in cfg["sweep_tau"]:
        accs, taus = [], []
        for seed in cfg["seeds"]:
            set_seed(seed)
            loaders = make_loaders(data, cfg["batch_size"], seed)
            model = build_model("dain", text_dim, n_classes, cfg, tau).to(device)
            torch.manual_seed(seed)
            res = train_one(model, loaders, device, cfg, seed)
            accs.append(res["test_acc"])
            taus.append(res["tau_bar"])
        tau_rows.append(
            {
                "tau": tau,
                "acc_mean": round(float(np.mean(accs)), 4),
                "acc_std": round(float(np.std(accs)), 4),
                "tau_bar": round(float(np.mean(taus)), 2),
            }
        )
        print(tau_rows[-1])
    pd.DataFrame(tau_rows).to_csv(RESULTS_DIR / "exp2_tau_sweep.csv", index=False)

    save_json(
        {
            "config": cfg,
            "note": (
                "Số liệu do chương trình chạy thực tế trên ENRICO (encoder đơn giản hóa, "
                "CPU/GPU nhỏ). KHÔNG so sánh trực tiếp với số của paper DAIN vì khác "
                "encoder, khác split và chỉ dùng 1/5 dataset."
            ),
            "device": device,
        },
        RESULTS_DIR / "exp2_summary.json",
    )
    print("Đã lưu kết quả vào", RESULTS_DIR)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Thí nghiệm 1b: tách bạch vai trò của hệ số suy giảm qua so sánh top-k.

Ở thí nghiệm chính, các chiến lược dùng ngưỡng theta như bộ lọc cứng, nên
radar (cosine * decay >= theta) chọn ít câu hơn semantic (cosine >= theta)
trên cùng ngưỡng — sự khác biệt gồm cả NGƯỠNG lẫn THỨ HẠNG.

Thí nghiệm 1b loại bỏ yếu tố ngưỡng: mọi chiến lược chọn đúng k câu, khác
nhau chỉ ở điểm xếp hạng (cosine thuần so với cosine * decay). Nếu decay
có vai trò như paper mô tả, radar top-k sẽ đạt precision cao hơn ở recall
tương đương (hoặc recall cao hơn ở precision tương đương).

Chạy:  python -m src.exp1.run_exp1b
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.utils import DATA_DIR, RESULTS_DIR, load_json, set_seed, token_count
from .answer_matcher import build_idf, pick_answer
from .context_strategies import run_selection, semantic_similarities
from .run_exp1 import build_candidates, evidence_metrics
from .simulate_debate import simulate_history

STRATEGIES = ["semantic_topk", "radar_topk"]


def main() -> None:
    cfg = load_json(Path(__file__).parent / "config.json")
    cfg["n_rounds"] = 3
    cfg["n_agents"] = 5
    set_seed(cfg["seed"])

    data = load_json(DATA_DIR / cfg["data_file"])
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(data), size=cfg["n_examples"], replace=False)
    examples = [data[i] for i in idx]

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(cfg["encoder"])

    idf = build_idf([e["answer"] for e in examples])
    rows = []
    for k in cfg["sweep_k"]:
        for idx_i, ex in enumerate(examples):
            hist = simulate_history(
                ex, n_agents=cfg["n_agents"], n_rounds=cfg["n_rounds"],
                edge_prob=cfg["edge_prob"], expert_frac=cfg["expert_frac"],
                base_noise=cfg["base_noise"], echo_prob=cfg["echo_prob"],
                seed=cfg["seed"] + idx_i,
            )
            messages = hist["messages"]
            if not messages:
                continue
            texts = [m["text"] for m in messages]
            sims = semantic_similarities(texts, hist["question"], encoder)
            candidates = build_candidates(examples, idx_i, cfg["n_candidates"],
                                          np.random.default_rng(cfg["seed"]))
            for strat in STRATEGIES:
                scfg = {
                    "current_round": cfg["n_rounds"],
                    "theta": cfg["theta"],
                    "lambda_s": cfg["lambda_s"],
                    "lambda_t": cfg["lambda_t"],
                    "k": k,
                }
                t0 = time.perf_counter()
                selected = run_selection(strat, messages, sims, scfg)
                elapsed = time.perf_counter() - t0
                precision, recall = evidence_metrics(selected, messages)
                pred, _ = pick_answer(candidates, selected, idf)
                rows.append({
                    "k": k,
                    "strategy": strat,
                    "evidence_precision": round(precision, 4),
                    "evidence_recall": round(recall, 4),
                    "evidence_f1": round(2 * precision * recall / (precision + recall)
                                         if precision + recall else 0.0, 4),
                    "answer_correct": int(pred == hist["answer"]),
                    "selected_tokens": sum(token_count(s["text"]) for s in selected),
                    "selection_time_ms": round(elapsed * 1000, 3),
                })
        print(f"[k={k}] xong")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "exp1b_topk_raw.csv", index=False)
    summary = df.groupby(["k", "strategy"]).mean(numeric_only=True).reset_index()
    summary.to_csv(RESULTS_DIR / "exp1b_topk_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Chạy thí nghiệm 1: so sánh các chiến lược quản lý ngữ cảnh (minh họa Agent-Radar).

Quy trình:
1. Nạp HotpotQA validation (distractor), lấy mẫu con theo seed.
2. Với mỗi mẫu: mô phỏng lịch sử trao đổi P tác tử x T vòng.
3. Với từng chiến lược {full, window, semantic, radar}: chọn ngữ cảnh cho
   tác tử đọc (agent 0) ở vòng cuối, đo:
     - Evidence-P/R/F1 so với supporting facts vàng,
     - số từ trong ngữ cảnh được chọn và trong toàn bộ lịch sử,
     - độ chính xác nhận diện câu trả lời đúng trong tập ứng viên,
     - thời gian chạy bộ chọn ngữ cảnh.
4. Quét tham số: số vòng T (khi P cố định) và số tác tử P (khi T cố định).
5. Xuất kết quả ra results/exp1_*.csv và exp1_summary.json.

Chạy:  python -m src.exp1.run_exp1 --config src/exp1/config.json
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.utils import DATA_DIR, RESULTS_DIR, load_json, save_json, set_seed, token_count
from .answer_matcher import build_idf, pick_answer
from .context_strategies import run_selection, semantic_similarities
from .simulate_debate import simulate_history

STRATEGIES = ["full", "window", "semantic", "radar"]


def evidence_metrics(selected: list[dict], context: list[dict]) -> tuple[float, float]:
    """Precision/Recall của tập câu được chọn so với các câu vàng trong lịch sử.

    - Precision: trong số câu được chọn, bao nhiêu % là vàng.
    - Recall: trong số câu vàng có trong lịch sử, bao nhiêu % được giữ lại.
    (Recall tính trên gold có mặt trong lịch sử để công bằng với 'window',
    vốn không thể giữ lại những gì không nằm trong cửa sổ.)
    """
    gold_in_ctx = [s for s in context if s["gold"]]
    if not selected:
        return 0.0, 0.0
    n_gold_selected = sum(1 for s in selected if s["gold"])
    precision = n_gold_selected / len(selected)
    recall = n_gold_selected / len(gold_in_ctx) if gold_in_ctx else 0.0
    return float(precision), float(recall)


def build_candidates(examples: list[dict], idx: int, k: int, rng) -> list[str]:
    """1 ứng viên vàng + (k-1) ứng viên nhiễu từ các mẫu khác."""
    others = [examples[j]["answer"] for j in rng.choice(len(examples), size=k, replace=False)
              if j != idx]
    return [examples[idx]["answer"]] + others[: k - 1]


def evaluate_example(
    example: dict,
    examples: list[dict],
    idx: int,
    encoder,
    cfg: dict,
    idf: dict,
    rng,
) -> list[dict]:
    """Chạy tất cả chiến lược trên một mẫu, trả về list kết quả từng dòng đo."""
    hist = simulate_history(
        example,
        n_agents=cfg["n_agents"],
        n_rounds=cfg["n_rounds"],
        edge_prob=cfg["edge_prob"],
        expert_frac=cfg["expert_frac"],
        base_noise=cfg["base_noise"],
        echo_prob=cfg["echo_prob"],
        seed=cfg["seed"] + idx,
    )
    messages = hist["messages"]
    if not messages:
        return []

    texts = [m["text"] for m in messages]
    sims = semantic_similarities(texts, hist["question"], encoder)

    candidates = build_candidates(examples, idx, cfg["n_candidates"], rng)
    rows = []
    for strat in STRATEGIES:
        scfg = {
            "current_round": cfg["n_rounds"],
            "theta": cfg["theta"],
            "lambda_s": cfg["lambda_s"],
            "lambda_t": cfg["lambda_t"],
        }
        t0 = time.perf_counter()
        selected = run_selection(strat, messages, sims, scfg)
        elapsed = time.perf_counter() - t0

        precision, recall = evidence_metrics(selected, messages)
        pred, _ = pick_answer(candidates, selected, idf)
        rows.append(
            {
                "example_id": hist.get("id", idx),
                "question_type": example.get("type", ""),
                "strategy": strat,
                "evidence_precision": round(precision, 4),
                "evidence_recall": round(recall, 4),
                "evidence_f1": round(
                    2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                    4,
                ),
                "answer_correct": int(pred == hist["answer"]),
                "selected_tokens": sum(token_count(s["text"]) for s in selected),
                "history_tokens": sum(token_count(s["text"]) for s in messages),
                "selection_time_ms": round(elapsed * 1000, 3),
                "n_rounds": cfg["n_rounds"],
                "n_agents": cfg["n_agents"],
            }
        )
    return rows


def run_sweep(examples: list[dict], encoder, base_cfg: dict, sweep: str, values: list) -> pd.DataFrame:
    """Chạy quét một tham số (n_rounds hoặc n_agents); trả về DataFrame mọi dòng đo."""
    all_rows = []
    for v in values:
        cfg = dict(base_cfg)
        if sweep == "n_rounds":
            cfg["n_rounds"] = int(v)
        else:
            cfg["n_agents"] = int(v)
        rng = np.random.default_rng(cfg["seed"])
        idf = build_idf([e["answer"] for e in examples])
        for idx, ex in enumerate(examples):
            all_rows.extend(evaluate_example(ex, examples, idx, encoder, cfg, idf, rng))
        print(f"[sweep {sweep}={v}] xong {len(examples)} mẫu")
    return pd.DataFrame(all_rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Trung bình các chỉ số theo (sweep giá trị, chiến lược)."""
    keys = ["n_rounds", "n_agents", "strategy"]
    metrics = [
        "evidence_precision",
        "evidence_recall",
        "evidence_f1",
        "answer_correct",
        "selected_tokens",
        "history_tokens",
        "selection_time_ms",
    ]
    return df.groupby(keys, dropna=False)[metrics].mean().reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=str(Path(__file__).parent / "config.json"))
    args = ap.parse_args()
    cfg = load_json(args.config)

    set_seed(cfg["seed"])

    # 1. Nạp dữ liệu và lấy mẫu con theo seed
    data = load_json(DATA_DIR / cfg["data_file"])
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(data), size=cfg["n_examples"], replace=False)
    examples = [data[i] for i in idx]
    print(f"Nạp {len(examples)} mẫu HotpotQA (từ {len(data)} mẫu validation)")

    # 2. Encoder ngữ nghĩa: dùng đúng all-MiniLM-L6-v2 như footnote 2 của paper
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(cfg["encoder"])
    print(f"Encoder: {cfg['encoder']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Quét số vòng trao đổi
    df_rounds = run_sweep(examples, encoder, cfg, "n_rounds", cfg["sweep_rounds"])
    df_rounds.to_csv(RESULTS_DIR / "exp1_rounds_raw.csv", index=False)
    sum_rounds = summarize(df_rounds)
    sum_rounds.to_csv(RESULTS_DIR / "exp1_rounds_summary.csv", index=False)
    print(sum_rounds.to_string(index=False))

    # 4. Quét số tác tử
    df_agents = run_sweep(examples, encoder, cfg, "n_agents", cfg["sweep_agents"])
    df_agents.to_csv(RESULTS_DIR / "exp1_agents_raw.csv", index=False)
    sum_agents = summarize(df_agents)
    sum_agents.to_csv(RESULTS_DIR / "exp1_agents_summary.csv", index=False)
    print(sum_agents.to_string(index=False))

    # 5. Tóm tắt
    best_rounds = sum_rounds[sum_rounds["n_rounds"] == cfg["n_rounds"]]
    save_json(
        {
            "config": cfg,
            "note": "Số liệu do chương trình chạy thực tế sinh ra; mô phỏng lịch sử trao đổi, không chạy LLM.",
            "main_config_summary": best_rounds.to_dict(orient="records"),
        },
        RESULTS_DIR / "exp1_summary.json",
    )
    print("Đã lưu kết quả vào", RESULTS_DIR)


if __name__ == "__main__":
    main()

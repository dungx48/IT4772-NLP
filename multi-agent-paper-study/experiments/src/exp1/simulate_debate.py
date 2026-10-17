# -*- coding: utf-8 -*-
"""Mô phỏng lịch sử trao đổi đa tác tử trên HotpotQA (thí nghiệm 1).

Thiết kế mô phỏng (ghi rõ trong báo cáo: đây là MÔ PHỎNG, không phải chạy LLM thật):

- Mỗi mẫu HotpotQA (distractor) có 1 câu hỏi + 10 đoạn văn (2 đoạn vàng, 8 gây nhiễu)
  và supporting facts chỉ ra chính xác các câu vàng.
- P tác tử trao đổi trên đồ thị ngẫu nhiên có hướng trong T vòng.
- "Tác giả" (agent 0) được gán vai trò nắm giữ câu hỏi; các "expert" là các tác tử
  có khả năng truy xuất đoạn văn vàng (mô phỏng việc một tác tử tìm đúng tài liệu).
- Thông điệp của một tác tử ở vòng t gồm:
    + tín hiệu: câu vàng — tác tử expert tự truy xuất được; tác tử thường chỉ
      "nghe lại" câu vàng từ tin nhắn vòng trước của láng giềng (với xác suất
      echo_prob, mô phỏng lan truyền thông tin);
    + nhiễu: (base_noise + t) câu từ các đoạn gây nhiễu — số lượng nhiễu tăng
      theo vòng, mô phỏng hiện tượng lịch sử phình to và loãng dần mà paper
      mô tả (§1 và Figure 2).
- Nhãn gold của từng câu được lưu lại để đo chất lượng chọn ngữ cảnh.

Mọi tham số mô phỏng nằm trong config để chạy lại được với cùng cấu hình.
"""

from __future__ import annotations

import numpy as np

from ..common.utils import set_seed
from .context_strategies import build_graph, hop_distances


def flatten_example(example: dict) -> dict:
    """Tách một mẫu HotpotQA thành danh sách câu kèm nhãn gold (supporting fact).

    Trả về dict gồm:
        question, answer,
        gold_sentences: list[str]  (các câu thuộc supporting facts)
        noise_sentences: list[str] (các câu trong đoạn gây nhiễu + câu thường của đoạn vàng)
    """
    sf = set(zip(example["sf_titles"], example["sf_sent_ids"]))
    gold, noise = [], []
    for title, sents in zip(example["context_titles"], example["context_sentences"]):
        for sid, s in enumerate(sents):
            s = s.strip()
            if not s:
                continue
            if (title, sid) in sf:
                gold.append(s)
            else:
                noise.append(s)
    return {
        "question": example["question"],
        "answer": example["answer"],
        "gold_sentences": gold,
        "noise_sentences": noise,
    }


def simulate_history(
    example: dict,
    n_agents: int,
    n_rounds: int,
    edge_prob: float,
    expert_frac: float,
    base_noise: int,
    echo_prob: float,
    seed: int,
) -> dict:
    """Sinh lịch sử trao đổi mô phỏng cho một mẫu.

    Trả về dict:
        messages: list[dict] — mỗi phần tử một câu {text, agent, round, hop, gold}
        dist_matrix: ma trận hop giữa các tác tử
        question, answer
    """
    rng = np.random.default_rng(seed)
    set_seed(seed)

    flat = flatten_example(example)
    adj = build_graph(n_agents, edge_prob, rng)
    dist = hop_distances(adj)

    # Agent 0 là "tác giả nhận truy vấn"; chọn nhóm expert (trực tiếp truy xuất
    # được đoạn vàng) ngẫu nhiên, mô phỏng vai trò của Wiki Searcher trong paper.
    n_experts = max(1, int(round(expert_frac * n_agents)))
    experts = set(rng.choice(n_agents, size=n_experts, replace=False).tolist())

    messages: list[dict] = []
    prev_round_gold: dict[int, list[str]] = {}  # câu vàng mỗi tác tử nghe được ở vòng trước

    for t in range(1, n_rounds + 1):
        heard_this_round: dict[int, list[str]] = {}
        for j in range(n_agents):
            own_gold: list[str] = []
            if j in experts and flat["gold_sentences"]:
                # Expert tự truy xuất: lấy ngẫu nhiên 2 câu vàng (nếu đủ)
                k = min(2, len(flat["gold_sentences"]))
                idx = rng.choice(len(flat["gold_sentences"]), size=k, replace=False)
                own_gold = [flat["gold_sentences"][i] for i in idx]

            # Tác tử thường "nghe lại" câu vàng từ tin nhắn vòng trước (echo)
            echoed: list[str] = []
            if j in prev_round_gold and rng.random() < echo_prob:
                echoed = list(prev_round_gold[j])

            # Nhiễu tăng theo vòng: mô phỏng hội thoại loãng dần
            n_noise = base_noise + t
            if flat["noise_sentences"]:
                idx = rng.choice(
                    len(flat["noise_sentences"]),
                    size=min(n_noise, len(flat["noise_sentences"])),
                    replace=False,
                )
                noise = [flat["noise_sentences"][i] for i in idx]
            else:
                noise = []

            for text in own_gold + echoed + noise:
                messages.append(
                    {
                        "text": text,
                        "agent": int(j),
                        "round": t,
                        "hop": int(dist[j, 0]),  # khoảng cách tới tác tử đọc (agent 0)
                        "gold": text in set(flat["gold_sentences"]),
                    }
                )
            heard_this_round[j] = [x["text"] for x in messages
                                   if x["agent"] == j and x["round"] == t and x["gold"]]

        prev_round_gold = heard_this_round

    return {
        "messages": messages,
        "dist_matrix": dist.tolist(),
        "question": flat["question"],
        "answer": flat["answer"],
    }

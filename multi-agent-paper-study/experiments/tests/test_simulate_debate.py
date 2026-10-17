# -*- coding: utf-8 -*-
"""Unit test cho mô phỏng lịch sử trao đổi và bộ ghép câu trả lời."""

from src.exp1.simulate_debate import flatten_example, simulate_history
from src.exp1.answer_matcher import build_idf, pick_answer, score_candidate

EXAMPLE = {
    "question": "Who is the chancellor of Germany?",
    "answer": "Angela Merkel",
    "type": "bridge",
    "sf_titles": ["Angela Merkel", "Germany"],
    "sf_sent_ids": [0, 1],
    "context_titles": ["Angela Merkel", "Germany", "Other"],
    "context_sentences": [
        ["Angela Merkel was chancellor.", "She was born in Hamburg.", "Third sentence."],
        ["Germany is a country.", "Its capital is Berlin.", "Extra."],
        ["Unrelated paragraph.", "More unrelated."],
    ],
}


def test_flatten_example():
    flat = flatten_example(EXAMPLE)
    assert flat["gold_sentences"] == ["Angela Merkel was chancellor.", "Its capital is Berlin."]
    assert len(flat["noise_sentences"]) == 6
    assert flat["answer"] == "Angela Merkel"


def test_simulate_history_growth():
    hist = simulate_history(EXAMPLE, n_agents=4, n_rounds=3, edge_prob=0.5,
                            expert_frac=0.5, base_noise=3, echo_prob=0.5, seed=0)
    per_round = {}
    for m in hist["messages"]:
        per_round[m["round"]] = per_round.get(m["round"], 0) + 1
    # Nhiễu tăng theo vòng => số câu mỗi vòng không giảm
    assert per_round[2] > per_round[1] or per_round[3] > per_round[1]
    # Có câu vàng trong lịch sử
    assert any(m["gold"] for m in hist["messages"])
    # Hop luôn >= 1 và hop của agent 0 là 0
    assert all(m["hop"] >= 0 for m in hist["messages"])


def test_deterministic():
    h1 = simulate_history(EXAMPLE, n_agents=4, n_rounds=2, edge_prob=0.5,
                          expert_frac=0.5, base_noise=3, echo_prob=0.5, seed=7)
    h2 = simulate_history(EXAMPLE, n_agents=4, n_rounds=2, edge_prob=0.5,
                          expert_frac=0.5, base_noise=3, echo_prob=0.5, seed=7)
    assert h1["messages"] == h2["messages"]


def test_answer_matcher():
    ctx = [
        {"text": "Angela Merkel served as chancellor of Germany for many years."},
        {"text": "Berlin is the capital."},
    ]
    pool = ["Angela Merkel", "Berlin", "Paris", "Angela", "Merkel admin"]
    idf = build_idf(pool)
    pred, score = pick_answer(["Angela Merkel", "Berlin", "Paris"], ctx, idf)
    assert pred == "Angela Merkel"
    assert score > score_candidate("Paris", " ".join(c["text"] for c in ctx), idf)

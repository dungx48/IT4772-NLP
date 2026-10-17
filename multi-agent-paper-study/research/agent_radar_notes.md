# Research Notes — Agent-Radar

> Nguồn: arXiv:2605.30136v1 [cs.AI], 28/05/2026, Preprint.
> **Enhancing Multi-Agent Communication through Attention Steering with Context Relevance**
> Tác giả: Hongxiang Zhang, Yuan Tian, Tianyi Zhang (Purdue University).
> PDF cục bộ: `papers/agent_radar.pdf` (36 trang).
> Mã nguồn chính thức: **không xác minh được** — paper không nêu link GitHub; tìm kiếm không thấy repo chính thức. Trang cá nhân tác giả chính (harrison4ride.github.io) không có link repo cho bài này.

## 1. Bài toán và động lực

- Hệ multi-agent dựa trên LLM cộng tác qua nhiều vòng thì lịch sử trao đổi phình to nhanh; thông tin liên quan bị "pha loãng" trong ngữ cảnh, gây hiệu ứng *lost-in-the-middle* (trích dẫn Liu et al. 2024a, Cemri et al. 2025, §1).
- Hai hướng xử lý cũ đều làm mất thông tin:
  1. **Nén/tóm tắt định kỳ** lịch sử (Wang et al. 2025a; Wu et al. 2025) — tóm tắt có thể bỏ sót tín hiệu tinh tế.
  2. **Cắt tỉa (pruning)** tác tử hoặc cạnh giao tiếp (AgentPrune — Zhang et al. 2025a; AgentDropout — Wang et al. 2025b) — thay đổi cấu trúc và bỏ bớt thông điệp.
- Vấn đề này xấu đi khi tăng số tác tử và số vòng (§1).

## 2. Ý tưởng cốt lõi

Không nén, không cắt — **giữ nguyên toàn bộ lịch sử (full transcript) và topology**, chỉ chọn ra một tập câu liên quan và "lái" attention của LLM về tập đó trong lúc sinh (§1, §3.2, Algorithm 1 — paper nhấn mạnh ở trang 16: "AGENT-RADAR does not delete, rewrite, or summarize messages").

## 3. Công thức hóa bài toán (§2)

- Hệ MAS là đồ thị có hướng `G = (V, E, F)`; `V` là N tác tử, `E` là topology giao tiếp, `F` là tập hàm suy luận.
- `d_G(v_j, v_i)`: khoảng cách hop ngắn nhất từ `v_j` đến `v_i`. Láng giềng k-hop: `N_k(v_i) = {v_j ∈ V | d_G(v_j, v_i) ≤ k}`. Tập mọi tác tử tới được `v_i`: `N_{N−1}(v_i)`.
- Tại vòng `t`, tác tử đang hoạt động `v_i` sinh `o_i^(t) = f_i(q^(t), Z_i^(t))` (Eq. 1), với `q^(t)` là truy vấn và `Z_i^(t)` là lịch sử tích lũy (Eq. 2: mọi `o_j^(τ)` với `v_j ∈ N_{N−1}(v_i)`, `1 ≤ τ < t`).
- Phương pháp bổ sung tham số `Ẑ_i^(t)`: ngữ cảnh được chọn, đưa vào `f_i` như một tham số thêm để lái attention (Eq. 10).

## 4. Cơ chế chọn ngữ cảnh động (§3.1)

Với mỗi câu `s_k` từ thông điệp `o_j^(τ)`:

1. **Spatial decay** (Eq. 4, cảm hứng từ lý thuyết mạng tổ chức — Borgatti 2009, Allen 1984; giải thích "mỗi chặng truyền lại chỉ giữ một tỉ lệ λ_s của tín hiệu"):

   `φ_s(v_j, v_i) = λ_s^(d_G(v_j,v_i) − 1)`, `0 < λ_s < 1`

   Tác tử là láng giềng trực tiếp hoặc chính nó: `φ_s = 1`.

2. **Temporal decay** (Eq. 5, dựa trên hiệu ứng gần đây trong trí nhớ người — Finkenbinder 1913):

   `φ_t(τ; t) = λ_t^(t − τ − 1)`, `0 < λ_t < 1`

   Thông điệp mới nhất có trọng số 1.

3. **Điểm liên quan mức thông điệp**: `r_i(o_j^(τ); t) = φ_s · φ_t` (Eq. 3).

4. **Truy xuất mức câu**: cắt `o_j^(τ)` thành câu `S_{j,τ} = {s_k}`; mã hóa bằng SentenceTransformer **all-MiniLM-L6-v2** (footnote 2, trang 4); tương đồng cosine với truy vấn (Eq. 6–7): `φ_sem(s_k, q^(t)) = e_k^T e_q / (‖e_k‖‖e_q‖)`.

5. **Điểm tổng hợp** (Eq. 8): `score(s_k | o_j^(τ), v_i, t, q^(t)) = r_i(o_j^(τ); t) · φ_sem(s_k, q^(t))`.

6. **Ngữ cảnh được chọn** (Eq. 9): `Ẑ_i^(t) = {q^(t)} ∪ {s_k | score ≥ θ}`. Truy vấn hiện tại **luôn** nằm trong Ẑ để ngữ cảnh chọn ra không lấn át mục tiêu suy luận. Mặc định `θ = 0.65` (Appendix A.3).

## 5. Attention steering và suy luận (§3.2)

- Suy luận cập nhật thành `o_i^(t) = f_i(q^(t), Z_i^(t), Ẑ_i^(t))` (Eq. 10) — cả lịch sử đầy đủ lẫn ngữ cảnh được chọn đều được cung cấp.
- Backend lái attention: **Selective Prompt Anchoring (SPA)** (Tian & Zhang 2025) — khuếch đại attention vào các câu trong Ẑ, không sửa trọng số LLM. Phương án thay thế đã thử: **PASTA** (Zhang et al. 2024b), chỉ chạy trên Llama; Agent-Radar vẫn tăng điểm với backend này (§4.3).
- Toàn bộ pipeline **training-free**, cắm thêm (plug-in) vào các MAS khác nhau.

## 6. Thuật toán (Appendix B, Algorithm 1, trang 16)

1. Dự pool thông điệp `M_i^(t)` từ láng giềng có thể tới được.
2. Khởi tạo `Ẑ = {q}`; mã hóa truy vấn.
3. Với mỗi thông điệp: tính `φ_s`, `φ_t`, `r = φ_s·φ_t`; cắt câu; với mỗi câu tính `φ_sem`, `score = r·φ_sem`; nếu `score ≥ θ` thì thêm vào `Ẑ`.
4. Áp attention steering lên các span trong `Ẑ`; sinh output với cả `q`, `Ẑ`, `Z`.

## 7. Thiết lập thực nghiệm của tác giả (§4.1, Appendix C)

- **Benchmarks**: HotpotQA, 2WikiMultihopQA, MuSiQue (báo cáo F1); MATH-500, MMLU-Pro (báo cáo accuracy).
- **Base model chính**: Qwen3-4B-Instruct-2507; kiểm chứng thêm trên Llama-3.1-8B-Instruct và Qwen3-32B (Appendix A.1, Table 5).
- **Baselines**: Vanilla, CoT-SC (n=8); Compaction (nén), AgentPrune, AgentDropout (pruning); tích hợp vào GPTSwarm, AutoGen, MAD.
- **Topology mặc định**: random graph (theo setup AgentPrune); thêm layered và fully connected (§4.2, Table 3; định nghĩa topology ở Appendix C.1, Figure 4).
- **Cấu hình**: 5 tác tử, 2 vòng giao tiếp (debate T=3); temperature = 1; suy luận lượng tử hóa trên NVIDIA A5500; 5 tác tử gán persona (Knowledge Expert, Wiki Searcher, Critic, Math Solver... — Appendix C.3).
- **Siêu tham số**: `λ_s = λ_t = 0.92` tốt nhất trong vùng thử `[0.88, 0.96]` (Appendix A.2, Figure 3); `θ = 0.65` tốt nhất trong `[0.55, 0.75]` (A.3).

## 8. Kết quả tác giả công bố (số liệu của tác giả, không phải của tôi)

**Table 1 (so với các phương pháp quản lý ngữ cảnh, Qwen3-4B):**

| Phương pháp | HotpotQA | 2Wiki | MuSiQue | MATH-500 | MMLU-Pro |
|---|---|---|---|---|---|
| Vanilla | 75.07 | 71.14 | 35.47 | 80.60 | 60.20 |
| CoT-SC (n=8) | 76.51 | 72.03 | 36.33 | 79.80 | 62.60 |
| Compaction | 74.33 | 73.34 | 34.43 | 83.60 | 56.40 |
| AgentPrune | 75.31 | 73.03 | 37.05 | 80.80 | 63.60 |
| AgentDropout | 73.14 | 78.45 | 37.06 | 82.60 | 67.40 |
| **Agent-Radar** | **80.78** | **80.81** | **39.72** | **88.80** | **69.40** |

- So với AgentDropout (baseline mạnh nhất): +7.64 HotpotQA, +2.36 2Wiki, +2.66 MuSiQue, +6.20 MATH-500, +2.00 MMLU-Pro. Trung bình +7.41 so với Vanilla.
- Mức tăng cao nhất bài nêu "+7.64 absolute points" là trên HotpotQA so với AgentDropout; "+12.87" là trên MuSiQue khi cắm vào GPTSwarm (Table 2: GPTSwarm 26.85 → 39.72).

**Table 2 (cắm vào các hệ MAS)**: cải thiện nhất quán cả 3 framework. Ví dụ GPTSwarm + Agent-Radar: MuSiQue 26.85→39.72; AutoGen: 79.63→84.29 (HotpotQA); MAD: 75.32→77.38.

**Table 3 (topology)**: Agent-Radar mạnh nhất gần như mọi cấu hình layered/fully-connected; ngoại lệ duy nhất: fully connected + MMLU-Pro, AgentDropout nhỉnh hơn (69.80 vs 69.40).

**Figure 2 (robustness với MAD trên MuSiQue)**: vanilla MAD đạt đỉnh ở 4 tác tử rồi giảm; Agent-Radar đỉnh ở 5 tác tử, giữ lợi thế mọi cấu hình; theo số vòng: cả hai tăng rồi bão hòa, Agent-Radar luôn cao hơn.

**Table 4 (ablation, trên MuSiQue / MATH-500 / MMLU-Pro):**

| Biến thể | MuSiQue | MATH-500 | MMLU-Pro |
|---|---|---|---|
| AgentDropout | 37.06 | 82.60 | 67.40 |
| BM25 (thay matcher) | 36.76 | 85.20 | 68.40 |
| w/o attention steering | 35.27 | 81.40 | 63.20 |
| w/o spatial decay | 28.41 | 84.40 | 66.20 |
| w/o temporal decay | 32.83 | 84.60 | 67.80 |
| w/o cả hai decay | 33.97 | 85.40 | 67.00 |
| **Agent-Radar đầy đủ** | **39.72** | **88.80** | **69.40** |

- Nhận xét đáng chú ý của tác giả: bỏ cả hai decay (chỉ semantic) có khi tốt hơn bỏ một decay — vì một decay đơn lẻ có thể overweight/underweight một nhóm thông điệp (trang 8–9).
- Thay SPA bằng PASTA (Llama backbone): MuSiQue 10.41→15.32; MATH-500 25.60→28.20; MMLU-Pro 12.40→15.20 — vẫn có lợi, chứng minh không phụ thuộc một backend.

**Table 6 (hiệu năng, giây/task)**: MAS 968.8; Compaction 1757.1; AgentPrune 1089.2; AgentDropout 1382.4; **Agent-Radar 1196.5 (+227.7)** — chi phí trung bình, nhanh hơn Compaction/AgentDropout, chậm hơn AgentPrune.

## 9. Hạn chế (§6 của paper + nhận xét dựa trên bằng chứng)

- Cần chọn `λ_s, λ_t, θ`; paper cho thấy khá bền với lựa chọn này nhưng vẫn là siêu tham số thủ công.
- Phụ thuộc chất lượng bộ truy xuất ngữ nghĩa.
- Cần backend attention steering (SPA/PASTA) — làm giới hạn khả năng tương thích với các LLM chỉ gọi qua API (không can thiệp được attention/logits). Đây là suy luận của tôi từ đặc điểm kỹ thuật của SPA/PASTA, paper không nói thẳng điều này.
- SPA/PASTA chỉ hỗ trợ một số backbone; PASTA thậm chí chỉ chạy với Llama trong thí nghiệm của paper.
- Overhead +23% thời gian (968.8→1196.5 s/task).

## 10. Điểm cần nhớ khi triển khai minh họa

- Điểm số câu = `cosine_semantic × λ_s^(hop−1) × λ_t^(age)`; ngưỡng θ; truy vấn luôn giữ lại.
- Spatial decay với hop=1 hoặc chính mình → trọng số 1 (mũ bằng 0).
- "Retained" nghĩa là: lịch sử đầy đủ vẫn được đưa vào prompt; các câu được chọn chỉ được khuếch đại attention. Trong thực nghiệm minh họa không có LLM, ta chỉ đo được **chất lượng chọn ngữ cảnh** và **lượng token được xử lý**, không đo được hiệu ứng khuếch đại attention.

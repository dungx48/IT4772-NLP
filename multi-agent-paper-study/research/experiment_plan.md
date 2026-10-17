# Kế hoạch thực nghiệm

## Đánh giá tính khả thi

### Agent-Radar

- Thực nghiệm gốc cần: LLM cục bộ (Qwen3-4B/32B, Llama-3.1-8B), GPU (A5500/A100), backend SPA/PASTA can thiệp attention của model, 5 benchmark QA lớn. → **Không khả thi** trong môi trường này (chi phí GPU/API cao, SPA cần truy cập internals của model).
- Phần **có thể làm trung thực**: cơ chế chọn ngữ cảnh (công thức Eq. 3–9) hoàn toàn tự chứa, không cần LLM — chỉ cần sentence encoder. Có thể đo:
  - chất lượng ngữ cảnh được chọn (so với supporting facts vàng của dataset),
  - số token đưa vào ngữ cảnh được chọn,
  - thời gian chạy bộ chọn,
  - biến thiên khi tăng số vòng/tác tử.
- Phần **không thể đo trung thực** mà không có LLM: hiệu ứng khuếch đại attention lên đầu ra của tác tử. Do đó thí nghiệm phải dừng ở mức "đánh giá bộ chọn ngữ cảnh", KHÔNG tuyên bố tái hiện hiệu năng của Agent-Radar.
- Quyết định: xây **phiên bản minh họa (illustrative)** của cơ chế chọn ngữ cảnh, chạy trên HotpotQA (distractor dev, có supporting facts), mô phỏng lịch sử trao đổi nhiều vòng nhiều tác tử với tỷ lệ nhiễu tăng dần. So 4 chiến lược: full-history, sliding window, semantic-only, spatio-temporal (minh họa Agent-Radar). Encoder: thử sentence-transformers all-MiniLM-L6-v2 (đúng paper); nếu tải được dùng, không thì fallback TF-IDF (ghi rõ).

### DAIN

- Thực nghiệm gốc cần: ResNet-50 + BERT-base, A100, các dataset y tế cần đăng ký (ADNI, MIMIC-IV). → Không tái hiện 1:1 được.
- Phần **có thể làm**: kiến trúc DAIN (Eq. 1–11) là mô hình nhỏ, mọi công thức tường minh. Chọn **ENRICO** (screenshot UI + metadata, công khai, nhỏ, tải được qua GitHub) làm dataset thật. Encoder thay bằng bản nhẹ (text: bag-of-words hashed; image: CNN nhỏ) — ghi rõ đây là **đơn giản hóa** so với ResNet-50/BERT-base gốc.
- Quy mô: Huấn luyện đủ nhanh trên CPU/GPU nhỏ với mô hình < 2M params.
- So sánh: Early Fusion, Late Fusion, Static MoE, DAIN-Static, DAIN-NoComm, Full DAIN + sweep τ + đo τ̄, ρ̄. Đây là mức so sánh giống cấu trúc Table 4/5/3 của paper, trên 1/5 dataset gốc.
- Nếu ENRICO tải không được: dùng dữ liệu tổng hợp hai modality và ghi rõ là mô phỏng, không phải dữ liệu thật.

## Thiết kế chi tiết

### Thí nghiệm 1 (Agent-Radar minh họa): chọn ngữ cảnh trong lịch sử trao đổi mô phỏng

**Dữ liệu**: HotpotQA distractor dev — mỗi câu hỏi có 10 đoạn văn (2 vàng + 8 gây nhiễu), supporting facts chỉ ra câu vàng chính xác. Lấy mẫu con cố định (seed rõ) ~200 mẫu.

**Mô phỏng trao đổi**: P tác tử (mặc định 5) trên đồ thị ngẫu nhiên có hướng (giống setup paper), T vòng (1–6). Tại vòng t, tác tử j "viết" thông điệp gồm vài câu: một phần câu từ supporting facts (tín hiệu), một phần câu từ đoạn gây nhiễu (nhiễu); tỷ lệ nhiễu tăng theo vòng (mô phỏng hội thoại loãng dần — hiện tượng paper mô tả ở Figure 2). Thông điệp của tác tử ở vòng sau có thể lặp lại/suy biến câu từ vòng trước (redundancy).

**Các chiến lược chọn ngữ cảnh cho tác tử đọc ở vòng t**:
1. `full`: toàn bộ lịch sử (baseline 1 của paper);
2. `window`: chỉ vòng gần nhất (baseline 2);
3. `semantic`: chỉ lọc theo cosine ngữ nghĩa ≥ θ (tương ứng ablation "w/o spatial & temporal decay" của paper);
4. `radar-illustrative`: cosine × λ_s^(hop−1) × λ_t^(age) ≥ θ (công thức Eq. 3–9, đúng paper, nhưng backend là bộ chọn ngữ cảnh thuần vì không có LLM/SPA).

**Chỉ số**:
- Evidence-F1: F1 giữa tập câu được chọn và supporting facts vàng (đo "lượng thông tin ngữ cảnh được giữ lại" — đúng hướng paper).
- Tokens: số từ trong ngữ cảnh được chọn (và trong full history để so).
- Answer-F1/EM: trả lời bằng reader trích xuất đơn giản (BM25-style scoring câu theo truy vấn) hoạt động trên ngữ cảnh được chọn — chỉ số yếu nhưng đo được, ghi rõ là reader mô phỏng, không phải LLM.
- Thời gian chọn ngữ cảnh (giây).
- Quét: T ∈ {1..6} với P=5; P ∈ {2,4,6,8} với T=3.

**Cấu hình**: `λ_s = λ_t = 0.92, θ = 0.65` (giá trị mặc định của paper), seed cố định.

### Thí nghiệm 2 (DAIN đơn giản hóa): điều phối động tác tử trên ENRICO

**Dữ liệu**: ENRICO — ~1,5k màn hình Android, 6 chủ đề; modality 1: ảnh screenshot; modality 2: văn bản nhãn phần tử UI. Chia train/val/test theo paper gốc của ENRICO.

**Kiến trúc**: hiện thực đúng Eq. 1–11 với K=8 (3 synergy, 3 uniqueness, 2 redundancy). Encoder nhẹ thay cho ResNet-50/BERT-base:
- text: bag-of-words 256 chiều (hashed) → linear;
- image: ảnh 64×64 → CNN 2 tầng conv → 256 chiều.

**Các biến thể**: Early Fusion (concat+MLP), Late Fusion (attention trên hai encoder), Static MoE (8 expert luôn chạy, router softmax không thưa), DAIN-Static (gating bỏ top-τ, mọi agent chạy), DAIN-NoComm (Φ=0), Full DAIN.

**Chỉ số**: accuracy test (mean±std qua ≥5 seed), τ̄, ρ̄, sweep τ ∈ {1,2,4,6,8}, số tham số effective.

## Output mong đợi

- `experiments/results/*.csv|json` — số đo thực.
- `experiments/figures/*.png` — biểu đồ sinh từ số liệu đã chạy (không có số tự nghĩ).
- Bảng + hình đưa vào report/ LaTeX.

# Multi-Agent Paper Study

Báo cáo tìm hiểu và thực nghiệm hai công trình về giao tiếp và phối hợp tác tử
trong hệ thống đa tác tử:

1. **Agent-Radar** — *Enhancing Multi-Agent Communication through Attention
   Steering with Context Relevance* (arXiv:2605.30136)
2. **DAIN** — *Dynamic Agent-Based Interaction Network for Efficient and
   Collaborative Multimodal Reasoning* (arXiv:2606.30189)

Đây là bài báo cáo học tập: phần báo cáo trình bày lại hai phương pháp bằng
cách hiểu của người đọc, phần thực nghiệm minh họa những ý tưởng kỹ thuật có
thể kiểm chứng với tài nguyên thông thường. Hai paper đều **chưa công bố mã
nguồn chính thức** (đã kiểm tra arXiv, trang cá nhân tác giả và tìm kiếm).

## Cấu trúc dự án

```
multi-agent-paper-study/
├── papers/                 # PDF hai paper gốc (tải từ arXiv)
├── research/               # Ghi chú nghiên cứu, bảng so sánh, kế hoạch thực nghiệm
├── extracted/              # Văn bản trích xuất từ PDF (phục vụ đọc/kiểm tra)
├── experiments/
│   ├── data/               # HotpotQA + ENRICO (tải tự động hoặc theo Makefile)
│   ├── src/
│   │   ├── common/         # tiện ích dùng chung
│   │   ├── exp1/           # thí nghiệm 1: chọn ngữ cảnh (minh họa Agent-Radar)
│   │   ├── exp2/           # thí nghiệm 2: DAIN đơn giản hóa trên ENRICO
│   │   └── make_figures.py # sinh biểu đồ từ kết quả
│   ├── tests/              # unit test (pytest)
│   ├── results/            # CSV/JSON kết quả đo được
│   └── figures/            # PNG cho báo cáo
├── report/                 # LaTeX (IEEEtran, tiếng Việt, biên dịch bằng Tectonic)
├── requirements.txt
├── Makefile
└── README.md
```

## Môi trường

- Python 3.13 (đã thử với bản Anaconda); CPU là đủ, không cần GPU.
- Cài thư viện: `pip install -r requirements.txt`
  (torch bản CPU là đủ: `pip install torch --index-url https://download.pytorch.org/whl/cpu`)
- Tải dữ liệu lần đầu cần Internet (HuggingFace cho HotpotQA, aalto.fi cho ENRICO,
  HuggingFace Hub cho encoder all-MiniLM-L6-v2 — ~90MB).
- Biên dịch báo cáo cần [Tectonic](https://tectonic-typesetting.github.io)
  (hoặc XeLaTeX/LuaLaTeX bất kỳ có font Times New Roman/Arial/Consolas —
  trên Linux đổi sang TeX Gyre Termes trong `report/main.tex`).

## Chạy lại thực nghiệm

```bash
# 1. Dữ liệu
make data          # tải HotpotQA (validation distractor) + ENRICO, trích đặc trưng

# 2. Unit test
make tests

# 3. Thí nghiệm (CPU, mỗi cái khoảng 10-30 phút)
make exp1          # thí nghiệm 1: chọn ngữ cảnh + quét số vòng/số tác tử
python -m src.exp1.run_exp1b   # (chạy trong experiments/) thí nghiệm 1b: so sánh top-k
make exp2          # thí nghiệm 2: DAIN + baseline trên ENRICO

# 4. Biểu đồ + báo cáo
make figures
make report        # xuất report/main.pdf
```

Mọi cấu hình nằm trong `experiments/src/exp1/config.json` và
`experiments/src/exp2/config.json`; seed cố định nên chạy lại cho ra cùng số.

## Phạm vi và giới hạn (đọc trước khi dùng số liệu)

- **Thí nghiệm 1 là mô phỏng, không chạy LLM.** Lịch sử trao đổi được dựng từ
  câu văn bản thật của HotpotQA, nhưng "hành vi tác tử" là quy tắc mô phỏng.
  Kết quả chỉ nói về chất lượng của bộ chọn ngữ cảnh (công thức của paper),
  KHÔNG nói về hiệu ứng lái attention lên LLM — phần đó không thể kiểm chứng
  nếu không chạy SPA trên một LLM cục bộ.
- **Thí nghiệm 2 là bản DAIN đơn giản hóa**: giữ nguyên cấu trúc thuật toán
  (Eq. 1–11 của paper), thay encoder ResNet-50/BERT-base bằng CNN nhỏ +
  bag-of-words băm, embedding 256 (paper: 512), gate sigmoid thay Softmax+Top-τ
  (lý do ghi trong `models.py`), và chỉ chạy trên 1/5 dataset gốc (ENRICO).
  Số liệu **không so sánh trực tiếp được** với số của paper.
- Hai paper chưa có mã nguồn chính thức để đối chiếu từng dòng; mọi mức
  "tái hiện" trong dự án này hiểu theo nghĩa hiện thực lại từ mô tả trong paper.

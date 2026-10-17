# Makefile cho project multi-agent-paper-study
# Dùng trên Windows (Git Bash) hoặc Linux/macOS.
# Biên dịch LaTeX dùng Tectonic (hỗ trợ XeTeX engine, cần cho tiếng Việt).
# Cài Tectonic: https://tectonic-typesetting.github.io

PY ?= python
TECTONIC ?= tectonic

.PHONY: data tests exp1 exp2 figures report clean

# Tải và chuẩn bị dữ liệu (HotpotQA qua HuggingFace, ENRICO qua aalto.fi)
data:
	$(PY) -c "from datasets import load_dataset; load_dataset('hotpotqa/hotpot_qa', 'distractor', split='validation')"
	cd experiments/data && curl -L -o enrico_screenshots.zip http://userinterfaces.aalto.fi/enrico/resources/screenshots.zip
	cd experiments/data && curl -L -o enrico_hierarchies.zip http://userinterfaces.aalto.fi/enrico/resources/hierarchies.zip
	cd experiments/data && curl -L -o enrico_design_topics.csv https://raw.githubusercontent.com/luileito/enrico/master/design_topics.csv
	cd experiments && $(PY) -m src.exp2.prepare_enrico

tests:
	cd experiments && $(PY) -m pytest tests/ -q

# Thí nghiệm 1: chọn ngữ cảnh minh họa Agent-Radar (khoảng 15-30 phút CPU)
exp1:
	cd experiments && $(PY) -m src.exp1.run_exp1

# Thí nghiệm 2: DAIN đơn giản hóa trên ENRICO (khoảng 10-20 phút CPU)
exp2:
	cd experiments && $(PY) -m src.exp2.run_exp2

# Sinh biểu đồ từ kết quả đã chạy
figures:
	cd experiments && $(PY) -m src.make_figures

# Biên dịch báo cáo LaTeX (chạy sau khi exp1, exp2, figures đã xong)
report:
	cd report && $(TECTONIC) main.tex

clean:
	rm -rf experiments/results/*.csv experiments/results/*.json experiments/figures/*.png

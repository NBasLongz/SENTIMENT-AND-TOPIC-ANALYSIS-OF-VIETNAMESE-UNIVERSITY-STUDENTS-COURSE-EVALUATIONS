# 🎓 Vietnamese Student Feedback Analysis

Phân tích đánh giá của sinh viên Việt Nam bằng mô hình Multi-task BERT cho **Sentiment Classification** (3 lớp) và **Topic Classification** (4 lớp, multi-label).

## 📊 Tổng quan dự án

Dự án này sử dụng các mô hình Transformer (PhoBERT và mBERT) để phân tích đồng thời:
- **Sentiment**: negative, neutral, positive
- **Topics**: lecturer, training_program, facility, others

### 🎯 Điểm nổi bật
- Multi-task learning cho 2 tasks cùng lúc
- Hỗ trợ 2 models: PhoBERT (Vietnamese-specific) và mBERT (multilingual)
- Multi-label classification cho topics
- Demo UI với Streamlit
- Đầy đủ utilities cho training, evaluation, và inference

## 📁 Cấu trúc dự án

```
.
├── datasets/
│   ├── processed/          # Dữ liệu đã xử lý (train/val/test.csv)
│   └── raw/               # Dữ liệu gốc
├── notebooks/             # Jupyter notebooks cho EDA, training
│   ├── EDA_Preprocessing.ipynb
│   ├── Train_multitask_phobert.ipynb
│   ├── Train_multitask_mbert.ipynb
│   └── Threshold_tuning_baseline.ipynb
├── outputs/
│   ├── models/            # Models đã train
│   │   ├── phobert_multitask/final_model/
│   │   ├── mbert_multitask/final_model/
│   │   └── baseline_lr/
│   ├── metrics/           # Kết quả đánh giá
│   └── figures/           # Biểu đồ, visualization
├── src/                   # Source code modules
│   ├── model_utils.py     # MultiTaskModel, load/save functions
│   ├── data_utils.py      # Dataset, data loading utilities
│   ├── eval_utils.py      # Evaluation metrics, threshold tuning
│   └── ui_app.py          # Streamlit demo app
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone <your-repo-url>
cd "CS221_S-T_ ANALYSIS OF VNST"
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## 📦 Yêu cầu hệ thống

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- Streamlit 1.20+
- scikit-learn, pandas, numpy

Xem chi tiết trong [requirements.txt](requirements.txt)

## 🎮 Sử dụng

### 1. Demo UI với Streamlit

Chạy ứng dụng demo:

```bash
streamlit run src/ui_app.py
```

Giao diện cho phép:
- Chọn model (PhoBERT hoặc mBERT)
- Điều chỉnh threshold cho topic classification
- Nhập text và xem kết quả phân tích real-time
- Xem xác suất của từng topic

### 2. Sử dụng trong code Python

#### Load model và inference

```python
from src.model_utils import load_model
import torch

# Load model
model, tokenizer, sent_map, topic_map, config = load_model(
    "outputs/models/phobert_multitask/final_model",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Inference
text = "Giảng viên nhiệt tình, phòng học tốt"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    sent_logits, topic_logits = model(**inputs)

# Get predictions
sent_pred = sent_logits.argmax(-1).item()
topic_probs = torch.sigmoid(topic_logits).numpy()[0]

print(f"Sentiment: {sent_map[sent_pred]}")
print(f"Topic probabilities: {topic_probs}")
```

#### Load và xử lý data

```python
from src.data_utils import load_processed_data, get_data_statistics

# Load data
train_df = load_processed_data("datasets/processed/train.csv")

# Get statistics
stats = get_data_statistics("datasets/processed/train.csv")
print(stats)
```

#### Đánh giá model

```python
from src.eval_utils import evaluate_model
from src.data_utils import prepare_data_for_training

# Prepare data
_, _, test_loader, sent_map, topic_map = prepare_data_for_training(
    "datasets/processed/train.csv",
    "datasets/processed/validation.csv",
    "datasets/processed/test.csv",
    tokenizer,
    batch_size=16
)

# Evaluate
metrics = evaluate_model(
    model, 
    test_loader, 
    device="cuda",
    topic_threshold=0.35,
    sent_labels=list(sent_map.values()),
    topic_labels=list(topic_map.values())
)

# Print results
from src.eval_utils import print_metrics_summary
print_metrics_summary(metrics)
```

## 📊 Dataset

Dataset gồm feedback của sinh viên UIT về:
- **Sentiment labels**: negative (0), neutral (1), positive (2)
- **Topic labels**: lecturer (0), training_program (1), facility (2), others (3)

### Phân chia dữ liệu
- **Train**: ~11,000 samples
- **Validation**: ~1,400 samples  
- **Test**: ~1,400 samples

### Format CSV
```csv
sentence,sentiment_id,sentiment,topic_id,topic
"slide giáo trình đầy đủ .",2,positive,1,training_program
"nhiệt tình giảng dạy",2,positive,0,lecturer
```

## 🧠 Models

### PhoBERT Multi-task
- **Backbone**: `vinai/phobert-base`
- **Architecture**: Shared encoder + 2 task-specific heads
- **Performance**: Xem [outputs/metrics/phobert_multitask_test_metrics.json](outputs/metrics/phobert_multitask_test_metrics.json)

### mBERT Multi-task
- **Backbone**: `bert-base-multilingual-cased`
- **Architecture**: Shared encoder + 2 task-specific heads
- **Performance**: Xem [outputs/metrics/mbert_multitask_test_metrics.json](outputs/metrics/mbert_multitask_test_metrics.json)

### Model files
Mỗi model folder chứa:
- `pytorch_model.bin`: Model weights
- `label_maps.json`: Label mappings
- `tokenizer files`: Tokenizer config và vocab
- `bundle_config.json`: Model configuration

## 📈 Kết quả

Xem chi tiết trong `outputs/metrics/`:
- `phobert_multitask_test_metrics.json`: PhoBERT metrics
- `mbert_multitask_test_metrics.json`: mBERT metrics
- `results_summary.csv`: Tổng hợp so sánh các models
- `best_threshold.json`: Optimal threshold cho topic classification

## 🔧 Training

Để train lại models, xem notebooks trong `notebooks/`:
1. `EDA_Preprocessing.ipynb`: Phân tích và tiền xử lý dữ liệu
2. `Train_multitask_phobert.ipynb`: Training PhoBERT
3. `Train_multitask_mbert.ipynb`: Training mBERT
4. `Threshold_tuning_baseline.ipynb`: Tìm optimal threshold

## 🤝 Contributing

Mọi đóng góp đều được hoan nghênh! Vui lòng:
1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

## 👥 Authors

- **Your Name** - *Initial work*

## 🙏 Acknowledgments

- Dataset: Vietnamese Students' Feedback (UIT)
- PhoBERT: VinAI Research
- Transformers: HuggingFace

## 📞 Contact

Nếu có câu hỏi, vui lòng liên hệ: [your-email@example.com]

---

**Last Updated**: December 2025
 

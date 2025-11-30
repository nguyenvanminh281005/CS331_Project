# Age-Invariant Face Recognition
## Temporal-Aware Deep Learning Approach

Đồ án về nhận dạng khuôn mặt bất biến theo tuổi tác sử dụng Deep Learning với Temporal Contrastive Learning.

## 📋 Mô tả Dự án

Dự án này nghiên cứu và phát triển mô hình nhận dạng khuôn mặt có khả năng duy trì hiệu suất cao ngay cả khi khoảng cách thời gian giữa ảnh đăng ký (enrollment) và ảnh kiểm tra (probe) tăng lên (aging effect).

### Datasets Hỗ trợ:

**1. MORPH-2 (Khuyến nghị - có phân tích temporal)**
- Có thông tin age, gender trong metadata
- Hỗ trợ phân tích performance degradation theo time gap
- Location: `/mnt/data/KHTN2023/CS331/dataset/morph_2/`

**2. AgeDB-30**
- Ảnh đã được aligned sẵn
- **Hạn chế**: Không có metadata age, không thể phân tích temporal
- Location: `/mnt/data/KHTN2023/CS331/dataset/agedb_30/`

## 🚀 Quick Start

### Sử dụng MORPH-2 (Mặc định - với phân tích temporal)

```bash
# Chạy toàn bộ pipeline với MORPH-2
python main_pipeline.py --dataset morph_2

# Hoặc chạy từng bước:
python main_pipeline.py --dataset morph_2 --step preprocess
python main_pipeline.py --dataset morph_2 --step pairs
python main_pipeline.py --dataset morph_2 --step train_baseline
python main_pipeline.py --dataset morph_2 --step evaluate
python main_pipeline.py --dataset morph_2 --step visualize
```

### Sử dụng AgeDB-30

```bash
# Chạy với AgeDB-30 (không có temporal analysis)
python main_pipeline.py --dataset agedb_30 --step preprocess
python main_pipeline.py --dataset agedb_30 --step pairs
```

### Các thành phần chính:

1. **Data Pipeline**: Xử lý dữ liệu với face detection, alignment, và quality filtering
2. **Baseline Models**: ArcFace và MagFace
3. **Temporal-Aware Model**: Mô hình mới với Temporal Contrastive Loss (TCL)
4. **Evaluation Framework**: Đánh giá TAR@FAR, degradation rate (chỉ với MORPH-2)
5. **Visualization**: ROC curves, TAR vs Time Gap (chỉ với MORPH-2)

## 🗂️ Cấu trúc Thư mục

```
project/
├── config.py                 # Cấu hình toàn bộ dự án
├── requirements.txt          # Dependencies
├── main_pipeline.py          # Pipeline chính
├── train.py                  # Script huấn luyện
├── evaluate.py              # Script đánh giá
│
├── models/                   # Các mô hình
│   ├── arcface_model.py     # ArcFace baseline
│   ├── magface_model.py     # MagFace baseline
│   ├── temporal_loss.py     # Temporal Contrastive Loss
│   └── temporal_model.py    # Temporal-Aware model
│
├── utils/                    # Utilities
│   ├── face_detector.py     # MTCNN face detection
│   ├── data_processor.py    # Data preprocessing
│   ├── pair_generator.py    # Generate temporal pairs
│   ├── metrics.py           # Evaluation metrics
│   ├── visualization.py     # Plotting functions
│   └── statistical_analysis.py  # Statistical tests
│
├── outputs/                  # Kết quả đầu ra
│   ├── processed_images/    # Ảnh đã xử lý
│   ├── pairs/               # Cặp ảnh test
│   ├── features/            # Embeddings
│   ├── results/             # Kết quả đánh giá
│   └── visualizations/      # Biểu đồ
│
└── saved_models/            # Model checkpoints
```

## 🚀 Cài đặt

```bash
# Clone repository
cd /mnt/data/KHTN2023/CS331/project

# Cài đặt dependencies
pip install -r requirements.txt
```

## 📊 Dataset

Dự án sử dụng **AgeDB-30** dataset:
- Đường dẫn: `/mnt/data/KHTN2023/CS331/dataset/agedb_30`
- Format: Ảnh 112x112 đã được aligned
- Annotation file: `agedb_30_ann.txt`

## 🎯 Sử dụng

### 1. Xử lý dữ liệu (Data Preprocessing)

```bash
python main_pipeline.py --step preprocess
```

Bước này sẽ:
- Phát hiện khuôn mặt với MTCNN
- Align theo 5-point landmarks
- Lọc chất lượng
- Tạo metadata.csv

### 2. Tạo cặp ảnh (Pair Generation)

```bash
python main_pipeline.py --step pairs
```

Tạo cặp ảnh theo time-gap:
- Positive pairs: cùng người, khác thời điểm
- Negative pairs: khác người
- Time gaps: 1, 2, 4, 6, 8, 10 năm

### 3. Huấn luyện Baseline Models

```bash
python main_pipeline.py --step train_baseline
```

Huấn luyện:
- ArcFace: với ArcMargin loss
- MagFace: với magnitude-aware margin

### 4. Huấn luyện Temporal-Aware Model

```bash
python main_pipeline.py --step train_temporal
```

Huấn luyện mô hình với **Temporal Contrastive Loss**:

$$L_{TCL} = (1 - \cos(f(x_t), f(x_{t+\Delta t}))) + \alpha \max(0, m - \cos(f(x_t), f(x_{neg})))$$

### 5. Đánh giá (Evaluation)

```bash
python main_pipeline.py --step evaluate
```

Đánh giá:
- TAR @ các mức FAR (0.1%, 1%, 10%)
- EER (Equal Error Rate)
- AUC (Area Under Curve)
- Degradation rate theo time gap

### 6. Visualization & Analysis

```bash
python main_pipeline.py --step visualize
```

Tạo:
- ROC curves
- TAR vs Time Gap plots
- Embedding trajectories (t-SNE/UMAP)
- Statistical analysis reports

### Chạy toàn bộ pipeline

```bash
python main_pipeline.py --step all
```

## 📈 Công thức Chính

### 1. Temporal Contrastive Loss (TCL)

```python
L_TCL = (1 - cos(f(x_t), f(x_{t+Δt}))) + α * max(0, m - cos(f(x_t), f(x_neg)))
```

Với:
- `f(x_t)`: Embedding tại thời điểm t
- `f(x_{t+Δt})`: Embedding cùng người tại t+Δt
- `f(x_neg)`: Embedding người khác
- `α`: Weight cho negative loss
- `m`: Margin

### 2. Degradation Rate

```python
D = (TAR_t - TAR_{t+Δt}) / Δt
```

Đo tốc độ suy giảm hiệu suất theo thời gian (năm).

### 3. Combined Loss

```python
L_total = L_ArcFace + λ * L_TCL
```

## 📊 Metrics Đánh giá

1. **TAR @ FAR**: True Accept Rate tại các mức False Accept Rate
   - FAR = 0.1%: Tiêu chuẩn nghiêm ngặt
   - FAR = 1%: Tiêu chuẩn thông thường

2. **EER**: Equal Error Rate (FAR = FRR)

3. **Degradation Rate**: Suy giảm hiệu suất/năm

4. **Statistical Tests**:
   - Pearson/Spearman correlation
   - Linear Mixed-Effects Model (LME)
   - Hypothesis testing (t-test, Wilcoxon)

## 🔬 Phân tích Thống kê

### Linear Mixed-Effects Model

```python
match_score ~ time_gap + age + gender + (1 | identity)
```

Phân tích ảnh hưởng của:
- Time gap (khoảng cách thời gian)
- Age (tuổi tác)
- Gender (giới tính)

## 📝 Kết quả Mong đợi

1. **Baseline Performance**:
   - ArcFace: TAR@FAR=0.1% ≈ 85-90%
   - MagFace: TAR@FAR=0.1% ≈ 87-92%

2. **Temporal-Aware Model**:
   - Giảm degradation rate
   - Cải thiện TAR tại time gap lớn
   - Embeddings ổn định hơn theo thời gian

3. **Visualizations**:
   - ROC curves cho comparison
   - TAR decay over time
   - Embedding trajectories của cùng identity

## 🛠️ Tùy chỉnh

Chỉnh sửa `config.py` để thay đổi:
- Hyperparameters (learning rate, batch size)
- Model architecture
- Loss weights (λ_tcl, α, margin)
- Evaluation protocols

## 📚 Tài liệu Tham khảo

1. **ArcFace**: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
2. **MagFace**: Meng et al., "MagFace: A Universal Representation for Face Recognition and Quality Assessment" (CVPR 2021)
3. **AgeDB**: Moschoglou et al., "AgeDB: The First Manually Collected, In-the-Wild Age Database" (CVPRW 2017)

## 🤝 Đóng góp

Sinh viên: Nguyễn Văn Minh - Nguyễn Văn Hồng Thái
Môn học: CS331 - Thị giác máy tính Nâng cao
Trường: Đại học Công nghệ Thông tin, ĐHQG-HCM

## 📄 License

Dự án này dành cho mục đích học tập và nghiên cứu.
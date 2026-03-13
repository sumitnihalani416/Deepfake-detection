# 🛡️ DeepfakeShield

A production-grade deepfake detection system with face extraction, transfer learning, temporal analysis, and a Streamlit dashboard.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Face Detection** | MTCNN (facenet-pytorch) + RetinaFace — handles occlusion, small faces |
| **Architectures** | EfficientNet-B4 (best), Xception, ResNet18 (fastest) |
| **Attention** | CBAM (channel + spatial) on top of EfficientNet |
| **Training** | Mixed precision, staged fine-tuning, cosine LR, label smoothing, early stopping |
| **Augmentation** | Albumentations — compression artifacts, blur, color jitter, coarse dropout |
| **Inference** | Frame-level + temporal smoothing (Savitzky-Golay) + video-level aggregation |
| **Dashboard** | Streamlit — upload video, see verdict, confidence gauge, frame timeline |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess your dataset
python preprocess.py \
    --input ./raw_data \          # must contain real/ and fake/ subdirs
    --output ./data/processed \
    --detector mtcnn \
    --frames 30

# 3. Train a model
python train.py \
    --data ./data/processed \
    --arch efficientnet_b4 \
    --epochs 20 \
    --batch-size 32

# 4. Launch dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
deepfake_detector/
├── app.py                   # Streamlit dashboard
├── train.py                 # Training script
├── inference.py             # DeepfakePredictor class
├── preprocess.py            # Dataset preprocessing CLI
├── requirements.txt
├── configs/
│   └── config.py            # Dataclass-based config
├── models/
│   └── detector.py          # EfficientNet-B4, Xception, ResNet18 + CBAM
└── utils/
    ├── face_detector.py     # MTCNN / RetinaFace wrappers
    ├── video_processor.py   # Frame sampling + face extraction pipeline
    └── dataset.py           # PyTorch Dataset + Albumentations transforms
```

---

## 🗂️ Dataset Format

The preprocessing script expects:
```
raw_data/
├── real/      (or 'original', 'pristine')
│   ├── video1.mp4
│   └── ...
└── fake/      (or 'manipulated', 'deepfake')
    ├── video2.mp4
    └── ...
```

After preprocessing, face crops are saved in `data/processed/real/` and `data/processed/fake/`.

### Recommended Datasets

| Dataset | Link | Notes |
|---|---|---|
| **FaceForensics++** | [github](https://github.com/ondyari/FaceForensics) | 1000 real + 4000 fake videos |
| **Celeb-DF v2** | [paper](https://arxiv.org/abs/1909.12962) | High quality deepfakes |
| **DFDC (Kaggle)** | [kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) | 100k+ clips, diverse |
| **WildDeepfake** | [github](https://github.com/deepfakeinthewild/deepfake-in-the-wild) | In-the-wild samples |

---

## ⚙️ Configuration

All settings live in `configs/config.py` as Python dataclasses:

```python
from configs.config import Config, ModelConfig, TrainingConfig

cfg = Config(
    model=ModelConfig(architecture="efficientnet_b4", dropout_rate=0.4),
    training=TrainingConfig(num_epochs=30, learning_rate=5e-5),
)
```

---

## 🧠 Model Details

### EfficientNet-B4 + CBAM (recommended)
- ImageNet-pretrained backbone (1792-dim features)
- Channel + spatial attention (CBAM) on feature maps
- 3-layer classification head with BatchNorm and Dropout
- ~19M parameters

### Xception
- Depthwise separable convolutions — great at detecting GAN artifacts
- ~22M parameters

### ResNet18 (lightweight)
- Fast baseline for prototyping / low-resource environments
- ~11M parameters

---

## 📊 Inference Pipeline

```
Video file
    │
    ▼
Frame sampling (uniform / random, N frames)
    │
    ▼
Face detection (MTCNN or RetinaFace)
    │
    ▼
Preprocessing (resize + normalize)
    │
    ▼
Batch inference (model → softmax)
    │
    ▼
Temporal smoothing (Savitzky-Golay filter)
    │
    ▼
Video-level aggregation (mean / max / weighted-mean)
    │
    ▼
REAL / FAKE verdict + confidence score
```

---

## 🖥️ Streamlit Dashboard

The dashboard provides:
- **Video upload** (MP4, AVI, MOV, MKV, WebM)
- **Model & detector selection** from the sidebar
- **Confidence gauge** (Plotly)
- **Frame-by-frame timeline** with raw + smoothed scores
- **Score distribution histogram**
- **Summary statistics** (faces found, high-risk frames, FPS)

---

## 📝 License

MIT — for research and educational use only. Respect privacy laws when using on real individuals.

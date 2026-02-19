
## Speech Emotion Recognition using Hybrid Feature Fusion (Wav2Vec2 + Traditional Features)

## Overview

This project presents a **high-performance Speech Emotion Recognition (SER) system** built using a hybrid deep learning framework that combines:

* 🎧 **Wav2Vec2 embeddings (self-supervised speech representation)**
* 📊 **Traditional acoustic features (MFCC, Mel-Spectrogram, ZCR, RMSE)**
* 🧠 **Deep Feature Fusion architecture**
* 🔎 **Explainable AI (Grad-CAM + LIME)**

The system achieves:

> ✅ **90.66% Test Accuracy**
> ✅ **0.9071 Macro F1 Score**
> ✅ Strong generalization across multiple benchmark datasets

---

# 🏗 Architecture Overview

### Feature Extraction Pipeline

* Wav2Vec2 (facebook/wav2vec2-base-960h)
* MFCC (40 coefficients)
* Mel Spectrogram (128 bands)
* Prosodic Features (ZCR, RMSE)

All features are fused through a dense projection and classification head.

### Model Highlights

* Fusion-based deep learning architecture
* Batch Normalization for stability
* Dropout for regularization
* Early stopping + LR scheduling
* Class imbalance handling with weighted loss

---

# 📊 Datasets Used

| Dataset   | Samples  |
| --------- | -------- |
| RAVDESS   | 1440     |
| TESS      | 2800     |
| EMODB     | 535      |
| **Total** | **4775** |

Emotion Classes:

* Neutral
* Happy
* Sad
* Angry
* Fear
* Disgust

---

# 📈 Performance

### Final Test Metrics

* **Accuracy:** 90.66%
* **Macro F1:** 0.9071
* Balanced precision-recall across all classes

The model shows strong generalization with minimal overfitting.

---

# 🔎 Explainability (XAI)

To make the model interpretable:

### 🎯 Grad-CAM

* Highlights time-frequency regions of spectrograms responsible for predictions.
* Identifies emotionally relevant acoustic patterns.

### 🧠 LIME

* Explains local decision boundaries.
* Identifies which spectrogram regions positively/negatively influence prediction.

This ensures the system is **not a black box** and increases trustworthiness.

---

# 🛠 Training Strategy

* Optimizer: Adam
* Loss: CrossEntropy with class weights
* Early Stopping
* Learning Rate Scheduler
* Batch Size: 64
* Training on Google Colab T4 GPU (15GB)

---

# 📁 Project Structure

```
Speech-Emotion-Recognition-Fusion/
│
├── data/
├── features/
├── models/
├── training/
├── explainability/
├── results/
├── README.md
└── requirements.txt
```

---

# 💡 Why This Project is Strong

✔ Combines self-supervised learning with handcrafted features
✔ Hybrid feature fusion approach
✔ Multi-dataset training
✔ Advanced regularization techniques
✔ Class imbalance handling
✔ Explainable AI integration
✔ Production-level code structure
✔ GitHub-ready modular architecture

---

# 🚀 Future Enhancements

* Real-time audio inference
* Deployment using FastAPI
* Model quantization
* Cross-language emotion generalization
* Transformer-based fusion architecture

---

# 👨‍💻 Author

Speech Emotion Recognition Research & Implementation Project
Patent filed for the proposed hybrid fusion framework.


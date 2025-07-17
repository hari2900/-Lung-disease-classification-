# 🫁 Lung Disease Detection using VGG19 and Grad-CAM

This project applies deep learning to classify chest X-ray images into three categories — **COVID-19**, **Normal**, and **Pneumonia**. We utilize transfer learning with **VGG19**, achieve around **97% validation accuracy**, interpret the results using **Grad-CAM**, and support decision-making with a **Decision Tree**.

---

## 📁 Project Structure

```
.
├── Lung cancer.ipynb            # Jupyter notebook with all code
├── vgg19_pneumonia_model.h5     # Trained model weights
├── dataset/
│   ├── train/
│   │   ├── COVID/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
```

---

## 🚀 Overview of Steps

### 1. 🧹 Data Loading & Preprocessing
- Chest X-ray dataset loaded from `train/` and `test/` folders.
- Image size standardized to **224x224**.
- Data augmentation:
  - Rotation, width/height shifts, shear, zoom, horizontal flip.
- Preprocessing done using `ImageDataGenerator`.

### 2. 🧠 Model: VGG19 + Custom Classifier
- **VGG19** loaded with pretrained `imagenet` weights.
- Top layers replaced with:
  - `Flatten → Dense(256) → Dropout(0.5) → Dense(3 softmax)`
- All layers except the last 4 frozen to prevent overfitting.

### 3. 📈 Training
- Optimizer: `Adam`
- Loss Function: `categorical_crossentropy`
- Epochs: 20
- Callback: `ReduceLROnPlateau` (for dynamic learning rate adjustment)

### 4. 📊 Evaluation
- Achieved ~97% validation accuracy.
- Model saved as `vgg19_pneumonia_model.h5`.

### 5. 🔍 Grad-CAM Visualization
- Applied **Grad-CAM** to visualize areas of the image that influenced predictions.
- This helps build trust in the model’s predictions, especially in clinical applications.

### 6. 🌲 Decision Tree Classifier
- Extracted features and predictions were used to train a **Decision Tree** classifier.
- Helps provide rule-based explanations for predictions.

---

## 🧪 Results

| Metric        | Value      |
|---------------|------------|
| Accuracy       | ~97%       |
| Model Used     | VGG19      |
| Classes        | COVID, Normal, Pneumonia |
| Explainability | Grad-CAM + Decision Tree |

---

## 📌 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn

Install dependencies:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

---

## 📎 Dataset Source

Ensure the dataset follows the structure:
```
dataset/
├── train/
│   ├── COVID/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── COVID/
    ├── NORMAL/
    └── PNEUMONIA/
```

> 📌 Dataset used in this project "https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia"

---

## ✅ Future Work

- Experiment with EfficientNet, ResNet.
- Try ensemble models.
- Integrate with a Flask web app or Streamlit for live predictions.
- Convert model to **TFLite/ONNX** for edge deployment.

---

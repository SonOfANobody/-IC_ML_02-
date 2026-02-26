### Facial Emotion Recognition (FER) Using MobileNetV2

### 📌 Project Overview

This project focuses on the automated detection and classification of human emotions from facial images. Using a deep learning–based computer vision approach, the system analyzes $48 \times 48$ grayscale images and classifies them into seven distinct emotional states. The project leverages MobileNetV2 and specialized transfer learning strategies to overcome the challenges of low-resolution data and significant class imbalances, providing a robust framework for affective computing.

### 🎯 Objectives

Classify facial images into 7 categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

Implement Upscaling (96x96) to enhance micro-expression feature extraction.

Utilize Transfer Learning with a two-phase training approach (Warm-up & Fine-tuning).

Address class imbalance using Weighted Loss and Label Smoothing ($0.1$).

### 🧠 Dataset

Source: FER-2013 or similar image-based facial emotion datasets.

Format: Grayscale images at $48 \times 48$ resolution.

Structure: Images are organized into class-labeled folders (e.g., train/angry, train/happy).

Note: This system is optimized for real-time application and research in human-computer interaction.

### 🛠️ Technologies & Tools

Core: Python, TensorFlow, Keras

Processing: NumPy, OpenCV, Pillow

Visualization: Matplotlib, Seaborn (Confusion Matrices, Loss Curves)

Metrics: Scikit-learn

### ⚙️Key Features

Face detection using OpenCV

Image preprocessing and normalization

CNN-based emotion classification

Real-time emotion recognition via webcam


### ⚙️ Methodology

Data Preprocessing – Grayscale to RGB conversion and pixel normalization to the [-1, 1] range.

Resolution Enhancement – Adaptive Resizing to $96 \times 96$ to allow pre-trained MobileNetV2 filters to function at peak efficiency.

Model Architecture – MobileNetV2 backbone with Global Average Pooling and a 512-unit Dense head with Dropout (0.5).

Phase 1: Warm-up – Training the top-level classifier for 20 epochs with frozen base weights.

Phase 2: Fine-tuning – Unfreezing the base model with a microscopic learning rate ($1 \times 10^{-5}) for surgical adjustments.

Optimization – Utilizing ReduceLROnPlateau and EarlyStopping to manage convergence.

### 📊 Evaluation

MetricsAccuracy: Overall prediction success.

Confusion Matrix: Vital for identifying "Neutral" bias and "Fear/Surprise" confusion.

Loss Curves: Monitoring the gap between training and validation to catch overfitting.

Class-Specific Recall: Focusing on difficult minority classes like Disgust.

### 📁 Project StructurePlaintextFacial-Emotion-Recognition/
│── data/
│   ├── train/          # Class-labeled folders (Angry, Fear, etc.)
│   └── test/           # Validation images
│── models/
│   └── ultimate_model.keras
│── notebooks/
│   └── training_pipeline.ipynb
│── README.md
└── requirements.txt

### Dataset

https://www.kaggle.com/datasets/msambare/fer2013

### 🚀 How to Run

Clone the repository.

Install dependencies:pip install -r requirements.txt

Define Dataset Path: Set the PATH variable in your script to point to your local dataset.

Execute Training: Run the notebook to perform the Two-Phase training.

### 📌 Results

The integration of upscaling and label smoothing significantly reduces the model's tendency to default to "Neutral" predictions. Transfer learning with MobileNetV2 provides a sophisticated feature-extraction base that outperforms standard CNNs on small, noisy facial datasets.

### ⚠️ Disclaimer

This model is intended for research and educational purposes. Accuracy can vary based on lighting, camera angle, and ethnic diversity within the training data.

### 👤 Author

Muhammad Abdulkareem

### ⭐ Acknowledgements

Special thanks to the Google Gemini and TensorFlow communities for insights into transfer learning optimization for low-resolution image data.

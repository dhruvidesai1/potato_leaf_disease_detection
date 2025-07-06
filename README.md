# potato_leaf_disease_detection
Deep learning-based detection of potato leaf diseases using CNN and Autoencoder + Classifier models across Early Blight, Late Blight, and Healthy classes.

# 🥔 Potato Leaf Disease Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red?logo=keras)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Used-orange?logo=tensorflow)
![Accuracy](https://img.shields.io/badge/CNN%20Test%20Accuracy-99.6%25-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This project focuses on detecting diseases in potato leaves using deep learning models. Two different approaches are implemented:

- A **Convolutional Neural Network (CNN)** model
- An **Autoencoder + Classifier** pipeline

Both models classify leaf images into three categories:  
**Early Blight**, **Late Blight**, and **Healthy**.

---

## 📂 Dataset

- The dataset is organized into three folders: `Early_Blight`, `Late_Blight`, and `Healthy`.
- Each folder contains labeled images of potato leaves corresponding to that class.
- A train-validation-test split is used for model training and evaluation.

---

## 🧠 Models Used

### 1. CNN Model
- Built using multiple convolutional, pooling, and dense layers.
- Trained for end-to-end classification directly from images.

**Performance:**
- ✅ Train Accuracy: 97.9%  
- ✅ Validation Accuracy: 96.3%  
- ✅ Test Accuracy: 99.6%

---

### 2. Autoencoder + Classifier
- Autoencoder learns compressed representations of input images.
- The encoder output is passed to a separate classifier for final prediction.

**Performance:**
- ✅ Train Accuracy: 91.2%  
- ✅ Validation Accuracy: 90.8%  
- ✅ Test Accuracy: 89.7%  
- ✅ Final Loss: 0.0051

![CNN Accuracy](https://img.shields.io/badge/CNN%20Accuracy-99.6%25-brightgreen)
![AE+Classifier Accuracy](https://img.shields.io/badge/AE%2BClassifier%20Accuracy-89.7%25-blue)

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy, Matplotlib  
- Scikit-learn  
- Google Colab / Jupyter Notebook

---

## 📈 Evaluation Metrics

- Accuracy  
- Loss curves  
- Confusion Matrix  
- Precision, Recall, F1-Score

---

## 🔍 Project Highlights

- Comparison between CNN and Autoencoder-based approaches
- Visualizations of training/validation curves
- Model generalization on unseen test data
- Clean dataset structure and modular codebase

---

## 🚧 Challenges Faced

- Handling intra-class visual similarities
- Tuning the autoencoder for optimal feature representation
- Preventing overfitting on the CNN model

---

## 📎 Future Improvements

- Try transfer learning using pre-trained models (e.g., ResNet, MobileNet)
- Data augmentation to improve robustness
- Deploy the model as a web or mobile application for farmers

---

## 📌 How to Run

1. Clone the repo  
   ```bash
   git clone https://github.com/yourusername/potato-leaf-disease-detection.git
   cd potato-leaf-disease-detection

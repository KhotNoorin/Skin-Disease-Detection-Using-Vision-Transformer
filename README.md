# 🧠 Skin Disease Detection using Vision Transformer (ViT)

This project presents a deep learning-based solution for automated skin disease classification using the **Vision Transformer (ViT)** architecture. Leveraging the **HAM10000** dataset—a comprehensive collection of dermatoscopic images—the model is trained to recognize and classify seven distinct types of skin lesions, including melanoma, with high accuracy. The aim is to assist dermatologists and medical practitioners by enabling early detection, improving diagnostic efficiency, and supporting clinical decision-making through a reliable, AI-powered tool.

## 📁 Dataset

- **Name**: HAM10000 (Human Against Machine with 10000 training images)
- **Source**: [Kaggle - HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Classes**:
  - Melanocytic nevi (nv)
  - Melanoma (mel)
  - Benign keratosis-like lesions (bkl)
  - Basal cell carcinoma (bcc)
  - Actinic keratoses (akiec)
  - Vascular lesions (vasc)
  - Dermatofibroma (df)

---

## 🧠 Model Architecture

- **Base Model**: Vision Transformer (ViT)
- **Framework**: PyTorch
- **Fine-tuning**: Final classification head adapted for 7 skin disease classes.

---

## 🚀 Project Pipeline

1. **Data Preparation**
   - Metadata parsing
   - Image pre-processing and augmentation
   - Directory structuring

2. **Model Training**
   - Transfer learning using ViT
   - Training with learning rate scheduler, optimizer, and cross-entropy loss

3. **Evaluation**
   - Accuracy, confusion matrix, precision, recall, F1-score
   - Visualization of predictions

4. **Deployment**
   - Streamlit web application for real-time image upload and classification

---

## 🌐 Streamlit App

To run the web app:
bash: streamlit run app.py
Upload an image of a skin lesion, and the model will predict the most probable class.

---

## 📌 Key Features:

✔️ Uses Vision Transformer (ViT)

✔️ 7-class skin disease classification

✔️ Preprocessing and data augmentation

✔️ Streamlit-based user interface

✔️ Easy-to-follow modular codebase

---

## 👩‍💻 Author:
Noorin Nasir Khot
# 🌿 Plant Disease Detection

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Deep%20Learning-success.svg)

An AI-based plant disease detection system utilizing deep learning and computer vision to identify healthy and diseased plant leaves.

<div align="center">
  <img width="1257" alt="System Overview 1" src="https://github.com/user-attachments/assets/fcd83a12-8baa-4658-8b9f-dcc36dd23b84" />
  <img width="2560" alt="System Overview 2" src="https://github.com/user-attachments/assets/368aa1a9-bcec-43e3-ae63-8fa3aae9ecc7" />
</div>

---

## 📊 Data Pipeline

### 1. Data Collection
The dataset features multiple plant species, covering both healthy and diseased conditions, to ensure a diverse and balanced dataset for training a robust classification model. Target species include:
* Grape
* Lime
* Apricot blossom (*hoa mai*)
* Rose leaves

<div align="center">
  <img width="200" src="https://github.com/user-attachments/assets/5bc381f2-dfac-4aae-b792-176e32804c07" />
  <img width="200" src="https://github.com/user-attachments/assets/d9e4e381-2dd8-4e22-a7e1-787c62f37574" />
  <img width="200" src="https://github.com/user-attachments/assets/54de03cc-9aa3-447a-8633-ff204a0e3b6b" />
  <img width="200" src="https://github.com/user-attachments/assets/ee841206-732d-4a2f-a185-4dcd91272610" />
</div>

### 2. Data Filtering and Organization
Raw images were carefully filtered to remove low-quality, blurry, duplicate, and irrelevant samples. The cleaned dataset is organized into structured directories based on plant type and disease category to maintain consistency and streamline data loading.

### 3. Data Preprocessing
Standardization techniques were applied to improve model stability and training efficiency:
* **Image Resizing:** Scaled to a fixed resolution.
* **Normalization:** Scaled pixel values for consistent neural network input.
* **Noise Reduction:** Background isolation and noise removal.

<div align="center">
  <img width="400" alt="Preprocessing Output 1" src="https://github.com/user-attachments/assets/f9f77b7d-2672-4c7e-9bbe-aeb0a07a0d45" />
  <img width="400" alt="Preprocessing Output 2" src="https://github.com/user-attachments/assets/25e7e658-11d7-4804-aa98-9617cb76a9fb" />
</div>

### 4. Dataset Caching
A caching mechanism stores preprocessed images in a serialized format. This significantly reduces redundant preprocessing during active training and accelerates I/O data loading speeds, which is highly beneficial for large-scale agricultural datasets.

### 5. Data Augmentation
To prevent overfitting and enhance the model's ability to generalize to real-world variations, extensive data augmentation was applied. Techniques include:
* Rotation
* Horizontal and vertical flipping
* Zooming and spatial shifting
* Brightness and contrast adjustments

<div align="center">
  <img width="500" alt="Augmentation Examples" src="https://github.com/user-attachments/assets/f378385f-a6a4-45d8-82b6-2fbb721053a5" />
</div>

---

## 🧠 Model Architecture

A Convolutional Neural Network (CNN) architecture was designed specifically for plant disease detection. The network extracts hierarchical visual features from the preprocessed leaf images to perform accurate multi-class classification across the various plant/disease combinations.

<div align="center">
  <img width="753" alt="CNN Architecture Diagram" src="https://github.com/user-attachments/assets/5cfac86b-dd79-45ab-a590-7e3cdc46fbd0" />
</div>

---

## 📈 Training and Evaluation

The model was implemented in **Python** utilizing modern deep learning frameworks (TensorFlow/PyTorch). 

* **Optimization:** Iterative hyperparameter tuning was conducted to optimize the learning rate, batch size, and network depth.
* **Metrics:** Performance was rigorously evaluated using standard classification metrics, primarily focusing on validation **Accuracy** and categorical cross-entropy **Loss**, ensuring high reliability for practical agricultural deployment.

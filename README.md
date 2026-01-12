# 👁️ Neural Lens: Image Captioning with InceptionV3 & LSTM

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)
![Flickr8k](https://img.shields.io/badge/Dataset-Flickr8k-blue?logo=kaggle&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-InceptionV3-blueviolet)
![NLP](https://img.shields.io/badge/NLP-LSTM%20RNN-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Neural Lens** is an end-to-end deep learning pipeline designed to bridge the gap between computer vision and natural language processing. By combining a deep Convolutional Neural Network (CNN) for visual understanding with a Long Short-Term Memory (LSTM) network for linguistic generation, this project automatically generates descriptive, human-like captions for any given image.

---

## 🧠 Overview

This project implements a **Merge Architecture** model for image captioning. Instead of just "translating" an image into a sentence, the system processes visual and textual data through two distinct "heads" before fusing them:

* **The Encoder (InceptionV3):** Uses a pre-trained CNN (on ImageNet) to extract a 2048-dimensional feature vector, representing the "semantic essence" of the image.
* **The Decoder (LSTM):** A language model that takes the visual vector and previously generated words to predict the next word in the sequence.

---

## ✨ Features

* 🖼️ **Deep Feature Extraction:** Leverages **InceptionV3** with transfer learning for high-accuracy object and scene recognition.
* 🔡 **Advanced Tokenization:** Full text-cleaning pipeline including punctuation removal, lowercasing, and `start`/`end` sequence marking.
* 🔍 **Dual Search Strategies:**
* **Greedy Search:** Fast, efficient word-by-word generation.
* **Beam Search (K-beams):** Heuristic search that explores multiple sentence paths for higher linguistic quality.


* 📊 **Performance Analytics:** Quantitative evaluation using **BLEU-1 and BLEU-2** scores with Chen-Cherry smoothing.
* ⚡ **Memory Optimized:** Custom Python generator and `tf.data.Dataset` integration to handle the Flickr8k dataset without memory overflow.

---

## 🏛️ High Level Architecture

The model uses a "Merge" strategy where the image features are injected into the RNN after the embedding layer, rather than only at the beginning.

---

## 📚 Table of Contents

* [🚀 Installation & Usage](https://www.google.com/search?q=%23-installation--usage)
* [🛠️ Tech Stack](https://www.google.com/search?q=%23%EF%B8%8F-tech-stack)
* [📈 Model Performance](https://www.google.com/search?q=%23-model-performance)
* [🖼️ Sample Results](https://www.google.com/search?q=%23-sample-results)

---

## 🚀 Installation & Usage

1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/neural-lens.git
cd neural-lens

```


2. **Download Dataset:**
Download the [Flickr8k dataset](https://www.google.com/search?q=https://www.kaggle.com/datasets/adityajn105/flickr8k) and place the `Images` folder and `captions.txt` in the project root.
3. **Install Dependencies:**
```bash
pip install tensorflow numpy matplotlib pandas tqdm scikit-learn nltk

```


4. **Run Inference:**
Load the pre-trained `.h5` model and pass any image through the `greedy_generator` or `beam_search_generator` functions.

---

## 🛠️ Tech Stack

| Category | Technologies |
| --- | --- |
| **Frameworks** | TensorFlow 2.x, Keras |
| **CNN Architecture** | InceptionV3 (Pre-trained) |
| **RNN Architecture** | LSTM (Long Short-Term Memory) |
| **NLP Metrics** | BLEU Score (NLTK), Tokenizer |
| **Data Handling** | NumPy, Pandas, tf.data |

---

## 📈 Model Performance

The model was trained for 15 epochs with an exponential learning rate decay and early stopping to prevent overfitting.

* **BLEU-1:** Measures individual word accuracy (~0.45 - 0.55 typical for Flickr8k).
* **BLEU-2:** Measures phrase and fluency accuracy (~0.25 - 0.35).

---

## 🖼️ Sample Results

| Image | Generated Caption |
| --- | --- |
| *[Image 1]* | "a brown dog is running through the tall grass" |
| *[Image 2]* | "two children are playing in a small pool" |
| *[Image 3]* | "a man in a red shirt is climbing a rock wall" |
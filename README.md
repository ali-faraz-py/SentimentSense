# 🧠 Sentiment Sense: Advanced Aspect-Based Sentiment AI

A professional-grade Natural Language Processing (NLP) dashboard built with **Python** and **Streamlit**. This tool utilizes **Transfer Learning** via **Hugging Face Transformers** to perform simultaneous sentiment analysis and aspect extraction with high contextual awareness.

---

## 🚀 Live Demo
**[Click here to try the Live App](https://sentiment-sense-ai.streamlit.app/)**

---

## 📺 Demo Preview
![Sentiment Sense Demo](assets/SentimentSense.gif)


---

## ✨ Features
* **Dual-Model Pipeline:** Combines a DistilBERT transformer for sentiment with a BART-Large model for zero-shot topic classification.
* **Aspect-Based Analysis (ABSA):** Automatically identifies specific categories (Food, Service, Price, etc.) mentioned in the text without manual labeling.
* **Contextual Nuance:** Correctively handles negations (e.g., "not great") and complex linguistic structures that traditional models miss.
* **Real-Time Inference:** High-speed processing using optimized PyTorch backends.
* **Enterprise UI:** Interactive metrics, progress bars, and sidebar technical documentation for an intuitive user experience.

---

## 🛠️ Tech Stack
* **Language:** Python 3.12+
* **Framework:** Streamlit (Web UI)
* **NLP Engines:** Hugging Face Transformers
* **Deep Learning Framework:** PyTorch
* **Models:** `distilbert-base-uncased` & `facebook/bart-large-mnli`
* **Deployment:** Streamlit Community Cloud

---

## 🚀 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ali-faraz-py/SentimentSense](https://github.com/ali-faraz-py/SentimentSense)
   cd SentimentSense

2. **Install dependencies:**
   ```bash
    pip install -r requirements.txt

3. **Run the application:**
   ```bash
    streamlit run app.py

---

## 📂 Project Structure

```text
sentimentsense/
├── app.py              # Main Streamlit application and dual-model logic
├── requirements.txt    # Project dependencies (Transformers, Torch, Streamlit)
├── .gitignore          # Prevents tracking of cache and environment files
└── explore.ipynb       # Initial research on LSTM-based sentiment analysis
```

---

## 🧠 Model Insights
The engine moves beyond simple "keyword matching" by using **Self-Attention mechanisms**. 

* **The Sentiment Stage:** Uses **DistilBERT**, which was trained on the SST-2 dataset, allowing it to achieve 90%+ accuracy on general sentiment tasks while remaining lightweight.
* **The Aspect Stage:** Uses **Zero-Shot Classification (BART)**. This allows the app to detect any custom category (like "Atmosphere" or "Health") by measuring the semantic similarity between the input text and the target labels.

---

### 👤 Author
**Syed Ali Faraz** - [GitHub Profile](https://github.com/ali-faraz-py)

*If you found this NLP pipeline useful, please give the repository a ⭐!*
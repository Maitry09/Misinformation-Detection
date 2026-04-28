# 🔍 VeriLang — Vernacular Misinformation Detector

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![MuRIL](https://img.shields.io/badge/Model-MuRIL-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-99.92%25-brightgreen?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-maitry30%2Fverilang--muril-yellow?style=flat-square)
![Languages](https://img.shields.io/badge/Languages-4%20Indian%20%2B%20Hinglish-purple?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> AI-powered fake news detector for WhatsApp forwards in Hindi, Gujarati, Marathi and Telugu — including Hinglish and Romanized text — using Google's MuRIL model.

---

## 🌐 Live Demo

👉 **[Try VeriLang Live](https://verilang.streamlit.app)**

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [What Makes This Unique](#-what-makes-this-unique)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Hinglish Support](#-hinglish-support)
- [Explainability — SHAP](#-explainability--shap)
- [Tech Stack](#-tech-stack)
- [How to Run Locally](#-how-to-run-locally)
- [Ethical Considerations](#-ethical-considerations)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🎯 Problem Statement

WhatsApp misinformation in Indian regional languages is a serious and growing problem. Over **500 million WhatsApp users in India** regularly receive and forward unverified news in their native languages. Existing fake news detectors work only for English — leaving Hindi, Gujarati, Marathi and Telugu speakers completely unprotected.

**VeriLang** solves this by detecting misinformation in 4 major Indian languages, with full support for Hinglish and Romanized text — the way people actually type on their phones.

---

## ✨ What Makes This Unique

| Feature | Other Projects | VeriLang |
|---|---|---|
| Language support | English only | 4 Indian languages + Hinglish |
| Model | Generic BERT | MuRIL — Indian language specialist |
| Romanized text | Not supported | Fully supported (8/8 accuracy) |
| Target domain | General news | WhatsApp forwards specifically |
| Explainability | None | SHAP word importance per language |
| Deployment | Jupyter only | Live web app with dark animated UI |
| Dataset | Small CSV | 49,426 real-world articles |

---

## 📊 Dataset

- **Source:** [Zenodo — Multilingual Fake News Detection Dataset](https://zenodo.org/records/11408513)
- **Total size:** 49,426 articles after cleaning
- **Labels:** Fake `0` / Real `1`
- **Columns:** `text`, `label`, `language`

### Distribution

| Language | Total | Fake | Real |
|---|---|---|---|
| Hindi | 15,051 | 7,599 | 7,452 |
| Gujarati | 14,830 | 6,145 | 8,685 |
| Telugu | 11,424 | 4,795 | 6,629 |
| Marathi | 8,121 | 707 | 7,414 |
| **Total** | **49,426** | **19,246** | **30,180** |

> Marathi had severe class imbalance (707 fake vs 7,414 real) handled using class weights during training.

---

## 📁 Project Structure

```
verilang-misinformation-detector/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── shap_all_languages.png          # SHAP word importance chart
│
├── notebooks/
│   ├── misinfo_detection.ipynb     # Notebook 1: EDA + baseline ML
│   └── misinfo_MuRIL.ipynb         # Notebook 2: MuRIL + Hinglish
│
└── outputs/
    ├── data_distribution.png
    ├── baseline_comparison.png
    ├── confusion_matrix.png
    ├── overfitting_check.png
    └── shap_all_languages.png
```

---

## 🔬 Methodology

### Phase 1 — Data Cleaning and EDA
- Removed URLs, HTML tags, extra whitespace
- Kept unicode scripts intact — critical for Indian languages
- Analyzed class balance per language
- Generated word clouds for fake vs real per language
- Zero missing values, zero duplicates after cleaning

### Phase 2 — Baseline ML Models
- Used **character n-gram TF-IDF** (2-4 grams, 50k features)
- Character n-grams outperform word n-grams for Indian scripts
- Trained Logistic Regression, SVM (LinearSVC), Naive Bayes
- Verified no overfitting using 5-fold cross-validation
- Train vs test gap under 0.20% for all models

### Phase 3 — MuRIL Fine-tuning
- Used `google/muril-base-cased` — trained on 17 Indian languages
- Fine-tuned for 3 epochs on Google Colab T4 GPU (free)
- Max sequence length 256 tokens
- Applied balanced class weights for Marathi imbalance
- Learning rate 2e-5 with linear warmup scheduler

### Phase 4 — Hinglish Augmentation
- Model initially failed on Romanized text (0/4 accuracy)
- Generated realistic Hinglish training data using:
  - Common word substitution maps per language
  - Manual WhatsApp-style sentence patterns
  - 35% of original dataset converted to Romanized form
- Fine-tuned on Hinglish data only (1 epoch, LR=1e-5)
- Result: **8/8 on Hinglish test set**

### Phase 5 — Explainability (SHAP)
- Used `shap.Explainer` with Text masker
- Generated word-level importance scores per language
- Shows which words drove fake vs real predictions

---

## 📈 Model Performance

### Overall Accuracy

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 99.63% | Character TF-IDF |
| Naive Bayes | 97.77% | ComplementNB |
| SVM (LinearSVC) | 99.84% | Best baseline |
| MuRIL v1 | 99.92% | Fine-tuned 3 epochs |
| **MuRIL v2 (final)** | **99.92%** | + Hinglish support |

### Per Language Accuracy

| Language | SVM | MuRIL v2 | Fake F1 | Real F1 |
|---|---|---|---|---|
| Hindi | 99.66% | 99.80% | 0.998 | 0.998 |
| Gujarati | 99.86% | 99.93% | 0.999 | 0.999 |
| Marathi | 99.94% | 100.00% | 1.000 | 1.000 |
| Telugu | 99.96% | 100.00% | 1.000 | 1.000 |

### Overfitting Check

| Model | Train | Test | Gap | Status |
|---|---|---|---|---|
| Logistic Reg | 99.60% | 99.63% | 0.03% | ✅ No overfit |
| SVM | 99.99% | 99.84% | 0.15% | ✅ No overfit |
| Naive Bayes | 97.57% | 97.77% | 0.20% | ✅ No overfit |

### 5-Fold Cross Validation (SVM)

```
Fold 1: 99.82%  Fold 2: 99.76%  Fold 3: 99.70%
Fold 4: 99.79%  Fold 5: 99.68%
Mean: 99.75%  |  Std Dev: 0.05%  →  Highly consistent
```

---

## 🗣 Hinglish Support

One of the key differentiators of VeriLang — full support for Romanized and Hinglish text as people type on WhatsApp.

### Hinglish Test Results (Before → After)

| Input | True | Before | After |
|---|---|---|---|
| "Yeh khabar bilkul jhooth hai share mat karo" | FAKE | REAL ❌ | FAKE ✅ |
| "Maharashtra government ne naya yojana shuru kiya" | REAL | FAKE ❌ | REAL ✅ |
| "Aa khabar khoti chhe share karva jaisi nathi" | FAKE | REAL ❌ | FAKE ✅ |
| "Gujarat sarkar e navi yojana sharu kari chhe" | REAL | FAKE ❌ | REAL ✅ |
| "Hi baatami khoti aahe share karu naka" | FAKE | REAL ❌ | FAKE ✅ |
| "Ee news fake undi share cheyakandi vaddhu" | FAKE | REAL ❌ | FAKE ✅ |
| "Yeh video fake hai aur sach nahi hai" | FAKE | — | FAKE ✅ |
| "Delhi mein aaj election results aaye hain" | REAL | — | REAL ✅ |

**Final Score: 8/8 (100%)** — all Hinglish inputs correctly classified

---

## 🔍 Explainability — SHAP

VeriLang uses **SHAP** to explain every prediction at the word level:

- 🔴 Red words pushed the model toward the prediction
- 🔵 Blue words pushed the model against the prediction

This makes the model transparent — critical for any real-world fact-checking application.

![SHAP](shap_all_languages.png)

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Core Language | Python 3.10 |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Main Model | google/muril-base-cased |
| ML Baseline | Scikit-learn, TF-IDF |
| Explainability | SHAP |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Web App | Streamlit |
| Model Hosting | HuggingFace Hub |
| Deployment | Streamlit Cloud |
| Training | Google Colab T4 GPU (free) |

---

## 💻 How to Run Locally

### Step 1 — Clone the repo
```bash
git clone https://github.com/Maitry09/verilang-misinformation-detector.git
cd verilang-misinformation-detector
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Add HuggingFace token
Create `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "your_huggingface_token_here"
```

### Step 4 — Run
```bash
streamlit run app.py
```

### Test with these inputs

**Native scripts:**
```
यह खबर बिल्कुल झूठी है और इसे शेयर मत करो
આ સમાચાર ખોટા છે અને સોશ્યલ મીડિયા પર વાયરલ થઈ રહ્યા છે
```

**Hinglish:**
```
Yeh khabar bilkul jhooth hai share mat karo
Aa khabar khoti chhe share karva jaisi nathi
Hi baatami khoti aahe share karu naka
Ee news fake undi share cheyakandi
```

---

## ⚖️ Ethical Considerations

- This tool is for **awareness and education only**
- It is **not a final judgment** on any news article
- Always verify from official sources before taking action
- Model trained on news articles — may behave differently on very short forwards
- Marathi fake detection had limited training data (707 samples)

### Trusted Fact-Check Sources
- [PIB Fact Check](https://pib.gov.in/factcheck.aspx) — Government of India
- [Boom Live](https://www.boomlive.in) — Independent
- [Alt News](https://www.altnews.in) — Independent
- [Vishvas News](https://www.vishvasnews.com) — Multilingual

---

## 🚀 Future Work

- Add Bengali, Tamil, Kannada, Malayalam support
- Build WhatsApp bot integration using Twilio API
- Add image-based misinformation detection
- Real-time verification via Google Fact Check API
- Mobile app for direct WhatsApp forward checking
- Federated learning for privacy-preserving on-device inference
- Audio misinformation detection for voice messages

---

## 📓 Notebooks Summary

### Notebook 1 — `misinfo_detection.ipynb`
Data loading → cleaning → EDA → TF-IDF → LR, SVM, NB training → per-language accuracy → overfitting check → model saving

**Outputs:** `lr_model.pkl`, `svm_model.pkl`, `nb_model.pkl`, `tfidf_vectorizer.pkl`, `test_data.csv`

### Notebook 2 — `misinfo_MuRIL.ipynb`
MuRIL loading → tokenization → fine-tuning → evaluation → SHAP → Hinglish augmentation → Hinglish fine-tuning → HuggingFace upload

**Outputs:** `muril_model_v2/`, `muril_test_predictions.csv`, `shap_all_languages.png`

---

## 👤 Author

**Maitry**
- GitHub: [@Maitry09](https://github.com/Maitry09)
- HuggingFace: [@maitry30](https://huggingface.co/maitry30)
- Model: [maitry30/verilang-muril](https://huggingface.co/maitry30/verilang-muril)
- Live App: [verilang.streamlit.app](https://verilang.streamlit.app)

---

## 📄 License

MIT License — free to use, modify and distribute with attribution.

---

> ⭐ Star this repo if you found it useful for your research or placement prep!
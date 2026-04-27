---
license: mit
---
# 🔍 VeriLang — Vernacular Misinformation Detector

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![MuRIL](https://img.shields.io/badge/Model-MuRIL-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-99.92%25-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square)
![Languages](https://img.shields.io/badge/Languages-4%20Indian-purple?style=flat-square)

> AI-powered fake news detector for WhatsApp forwards in Hindi,
> Gujarati, Marathi and Telugu using Google's MuRIL model.
---

## 🎯 Problem Statement

WhatsApp misinformation in Indian regional languages is a
serious problem affecting millions of people daily. Existing
fake news detectors work only for English. VeriLang solves
this by detecting misinformation in 4 major Indian languages
— Hindi, Gujarati, Marathi and Telugu.

---

## ✨ What Makes This Unique

| Feature | Other Projects | VeriLang |
|---|---|---|
| Languages | English only | 4 Indian languages |
| Model | Generic BERT | MuRIL — Indian language specialist |
| Target | General news | WhatsApp forwards specifically |
| Explainability | None | SHAP word importance |
| Deployment | Jupyter only | Live web app |

---

## 📊 Dataset

- **Source:** Extracted 4 different language dataset from Zenodo — Multilingual Fake News Detection
  (https://zenodo.org/records/11408513)
- **Size:** 49,426 articles
- **Labels:** Fake (0) / Real (1)
- **Languages:** Hindi, Gujarati, Marathi, Telugu

| Language | Total | Fake | Real |
|---|---|---|---|
| Hindi | 15,051 | 7,599 | 7,452 |
| Gujarati | 14,830 | 6,145 | 8,685 |
| Telugu | 11,424 | 4,795 | 6,629 |
| Marathi | 8,121 | 707 | 7,414 |

---
## 🔬 Methodology

### Phase 1 — Data Cleaning
- Removed URLs, HTML tags, extra spaces
- Kept unicode scripts intact for Indian languages
- No duplicates, no missing values

### Phase 2 — Baseline ML Models
- TF-IDF with character n-grams (2-4 grams)
- Character n-grams work better than word n-grams
  for Indian scripts
- Trained Logistic Regression, SVM, Naive Bayes

### Phase 3 — MuRIL Fine-tuning
- Used google/muril-base-cased from HuggingFace
- MuRIL is trained on 17 Indian languages
- Fine-tuned for 3 epochs on Google Colab T4 GPU
- Applied class weights to fix Marathi imbalance

---

## 📈 Model Performance

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Logistic Regression | 99.63% |
| Naive Bayes | 97.77% |
| SVM | 99.84% |
| **MuRIL (final)** | **99.92%** |

### Per Language Accuracy

| Language | SVM | MuRIL |
|---|---|---|
| Hindi | 99.66% | 99.80% |
| Gujarati | 99.86% | 99.93% |
| Marathi | 99.94% | 100.00% |
| Telugu | 99.96% | 100.00% |

### Overfitting Check

| Model | Train | Test | Gap | Status |
|---|---|---|---|---|
| Logistic Reg | 99.60% | 99.63% | 0.03% | No overfit |
| SVM | 99.99% | 99.84% | 0.15% | No overfit |
| MuRIL | — | 99.92% | — | CV std 0.05% |

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, HuggingFace Transformers |
| Model | google/muril-base-cased |
| ML Baseline | Scikit-learn, TF-IDF |
| Explainability | SHAP |
| Web App | Streamlit |
| Model Hosting | HuggingFace Hub |
| Training | Google Colab T4 GPU |

---

## 💻 How to Run Locally

```bash
git clone https://github.com/Maitry09/verilang-misinformation-detector
cd verilang-misinformation-detector
pip install -r requirements.txt
streamlit run app.py
```

Add `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "your_token_here"
```

---

## ⚖️ Ethical Considerations

- Tool is for awareness only — not a final judgment system
- Always verify from official sources like PIB, ANI, PTI
- Model may have bias toward certain writing styles
- Marathi fake detection improved with augmentation
  but still has fewer training samples

---

## 🚀 Future Work

- Add Bengali, Tamil, Kannada support
- Work on mixture of language
- Build WhatsApp bot integration using Twilio API
- Add image-based misinformation detection
- Real-time news verification via fact-check APIs
- Mobile app for direct WhatsApp forward checking

---

## 👤 Author

**Maitry**
- GitHub: [@Maitry09](https://github.com/Maitry09)
- Live App: [verilang.streamlit.app]

---

## 📄 License
MIT License

---

> ⭐ Star this repo if you found it useful!
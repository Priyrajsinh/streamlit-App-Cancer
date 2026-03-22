# 🔬 Breast Cancer Diagnosis Predictor

A machine learning web application that predicts whether a breast mass is **benign or malignant** based on cell nuclei measurements from Fine Needle Aspirate (FNA) cytology samples.

**Live App →** [priyrajsinh-app-cancer.streamlit.app](https://priyrajsinh-app-cancer-8za2dby7bewf6cyg2beqkf.streamlit.app/)

---

## 📌 What This App Does

1. A user uploads a cytology PDF report or manually adjusts the sidebar sliders
2. The app extracts 30 numeric cell nuclei measurements from the PDF
3. A trained Logistic Regression model predicts benign or malignant
4. The result is displayed alongside a radar chart and probability scores

---

## 🖼️ App Preview

| Sidebar Sliders | Radar Chart | Prediction |
|----------------|-------------|------------|
| 30 cell nuclei measurements | Visual comparison of mean, SE, worst values | Benign / Malignant with probability |

---

## 📂 Project Structure

```
streamlit-App-Cancer/
│
├── app/
│   └── main.py              # Main Streamlit application
│
├── model/
│   ├── main.py              # Model training script
│   ├── model.pkl            # Trained Logistic Regression model
│   └── scaler.pkl           # StandardScaler for input normalisation
│
├── data/
│   └── data.csv             # Wisconsin Breast Cancer Dataset
│
├── assests/
│   └── style.css            # Custom CSS styling
│
├── test_reports/            # Sample PDFs for testing the app
│   ├── patient1_benign.pdf
│   ├── patient2_malignant.pdf
│   ├── patient3_benign.pdf
│   ├── patient4_malignant.pdf
│   ├── patient5_benign.pdf
│   └── README.md
│
├── requirements.txt
├── packages.txt
└── README.md
```

---

## 🧠 How the Model Works

The app uses a **Logistic Regression** classifier trained on the [Wisconsin Breast Cancer (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

### Input Features (30 total)
For each of the 10 cell nucleus properties, three values are measured:

| Property | Mean | Standard Error | Worst |
|----------|------|---------------|-------|
| Radius | ✓ | ✓ | ✓ |
| Texture | ✓ | ✓ | ✓ |
| Perimeter | ✓ | ✓ | ✓ |
| Area | ✓ | ✓ | ✓ |
| Smoothness | ✓ | ✓ | ✓ |
| Compactness | ✓ | ✓ | ✓ |
| Concavity | ✓ | ✓ | ✓ |
| Concave Points | ✓ | ✓ | ✓ |
| Symmetry | ✓ | ✓ | ✓ |
| Fractal Dimension | ✓ | ✓ | ✓ |

### Model Performance
- **Algorithm:** Logistic Regression
- **Accuracy:** ~95% on test set
- **Train/Test Split:** 80/20

---

## 🧪 Testing the App

Sample reports are provided in the `/test_reports` folder. Download any of them and upload to the app:

| File | Expected Result |
|------|----------------|
| `patient1_benign.pdf` | 🟢 Benign |
| `patient2_malignant.pdf` | 🔴 Malignant |
| `patient3_benign.pdf` | 🟢 Benign |
| `patient4_malignant.pdf` | 🔴 Malignant |
| `patient5_benign.pdf` | 🟢 Benign |

---

## ⚠️ Important Limitation — Why Real Hospital Reports Don't Work

This is the most important thing to understand about this project.

### What real lab reports look like
Reports from hospitals like Apollo, SRL, Thyrocare, or any pathology lab look like this:

> *"Highly cellular smear showing atypical ductal cells with high N/C ratio, irregular nuclear membrane, coarse chromatin and prominent nucleoli. Features suspicious of malignancy."*

They contain **written descriptions**, not the 30 numeric measurements this model needs.

### Where do the 30 measurements come from?
These values are computed by **specialized digital microscopy imaging software** that scans cell nuclei at a pixel level. This software is installed in research labs and is **not part of standard hospital pathology workflows** in most countries including India.

### What this means
Real patient PDFs from any hospital will **not** contain these numbers, so this app cannot read them. The test PDFs in this repo were synthetically generated specifically for testing purposes.

---

## 🚀 What Would Be Required to Support Real Lab PDFs

To make this app work with real-world patient reports from any lab, the following would need to be built:

### Option 1 — Text Classification Model (Medium Effort)
- Collect thousands of real FNA pathology reports with confirmed diagnoses (benign/malignant)
- Train a **text classification model** (e.g. fine-tuned BERT or a smaller NLP model) directly on the report text
- Replace the current numeric model with the text classifier
- **Challenge:** Requires a large labelled dataset of real reports — needs hospital partnerships

### Option 2 — Lab Imaging Software Integration (High Effort)
- Integrate directly with microscopy imaging software (e.g. CellProfiler, ImageJ) used in pathology labs
- The software outputs the 30 measurements digitally which are then fed to this model
- **Challenge:** Requires hospital-level technical integration and data privacy compliance

### Option 3 — Image-Based Deep Learning (High Effort)
- Instead of numeric measurements, take the actual microscopy slide images as input
- Train a **Convolutional Neural Network (CNN)** directly on cell images
- This is how modern clinical AI tools actually work
- **Challenge:** Requires large datasets of labelled microscopy images (e.g. BreakHis dataset)

### Option 4 — Full Clinical Pipeline (Research Level)
- Partner with a pathology lab or hospital
- Build end-to-end: image capture → feature extraction → prediction → report generation
- Comply with medical device regulations (FDA, CDSCO in India)
- **Challenge:** Multi-year research and regulatory project

---

## 🛠️ Running Locally

### Prerequisites
- Python 3.9+

### Installation
```bash
git clone https://github.com/Priyrajsinh/streamlit-App-Cancer.git
cd streamlit-App-Cancer
pip install -r requirements.txt
```

### Train the model (optional — pre-trained model already included)
```bash
python model/main.py
```

### Run the app
```bash
streamlit run app/main.py
```

---

## 📦 Dependencies

```
streamlit
numpy
pandas
scikit-learn
plotly
pdfplumber
pypdf
google-generativeai
```

---

## 🔑 Deployment (Streamlit Cloud)

To deploy with AI-powered PDF extraction:

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Get a free Gemini API key from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
4. In your app settings → Secrets, add:
```toml
GEMINI_API_KEY = "AIza..."
```
5. Deploy — users just upload a PDF, no setup needed on their end

---

## 📊 Dataset

**Wisconsin Breast Cancer (Diagnostic) Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- Samples: 569 (357 benign, 212 malignant)
- Features: 30 numeric + 1 diagnosis label
- Original research: Dr. William H. Wolberg, University of Wisconsin

---

## ⚕️ Disclaimer

**This application is built for educational purposes only.** It is not intended for clinical use and should never be used as a substitute for professional medical diagnosis. Always consult a qualified medical professional for any health concerns.

---

## 👨‍💻 Author

**Priyrajsinh**
- GitHub: [@Priyrajsinh](https://github.com/Priyrajsinh)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

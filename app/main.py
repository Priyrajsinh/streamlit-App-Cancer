import streamlit as st
import pickle as pkl
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
import io
import json
import base64

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


# ── constants ──────────────────────────────────────────────────────────────────

SLIDER_LABELS = [
    ("Radius (mean)",             "radius_mean"),
    ("Texture (mean)",            "texture_mean"),
    ("Perimeter (mean)",          "perimeter_mean"),
    ("Area (mean)",               "area_mean"),
    ("Smoothness (mean)",         "smoothness_mean"),
    ("Compactness (mean)",        "compactness_mean"),
    ("Concavity (mean)",          "concavity_mean"),
    ("Concave points (mean)",     "concave points_mean"),
    ("Symmetry (mean)",           "symmetry_mean"),
    ("Fractal dimension (mean)",  "fractal_dimension_mean"),
    ("Radius (se)",               "radius_se"),
    ("Texture (se)",              "texture_se"),
    ("Perimeter (se)",            "perimeter_se"),
    ("Area (se)",                 "area_se"),
    ("Smoothness (se)",           "smoothness_se"),
    ("Compactness (se)",          "compactness_se"),
    ("Concavity (se)",            "concavity_se"),
    ("Concave points (se)",       "concave points_se"),
    ("Symmetry (se)",             "symmetry_se"),
    ("Fractal dimension (se)",    "fractal_dimension_se"),
    ("Radius (worst)",            "radius_worst"),
    ("Texture (worst)",           "texture_worst"),
    ("Perimeter (worst)",         "perimeter_worst"),
    ("Area (worst)",              "area_worst"),
    ("Smoothness (worst)",        "smoothness_worst"),
    ("Compactness (worst)",       "compactness_worst"),
    ("Concavity (worst)",         "concavity_worst"),
    ("Concave points (worst)",    "concave points_worst"),
    ("Symmetry (worst)",          "symmetry_worst"),
    ("Fractal dimension (worst)", "fractal_dimension_worst"),
]

ALL_KEYS = [k for _, k in SLIDER_LABELS]

FEATURE_MAP = {
    "radius":            "radius",
    "texture":           "texture",
    "perimeter":         "perimeter",
    "area":              "area",
    "smoothness":        "smoothness",
    "compactness":       "compactness",
    "concavity":         "concavity",
    "concave points":    "concave points",
    "symmetry":          "symmetry",
    "fractal dimension": "fractal_dimension",
}
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# ── data ───────────────────────────────────────────────────────────────────────

@st.cache_data
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
            if text.strip():
                return text
        except Exception:
            pass
    if HAS_PYPDF:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        except Exception:
            pass
    return text


# ── Gemini extraction (free AI) ────────────────────────────────────────────────

GEMINI_PROMPT = """You are a medical data extraction assistant.
Extract the following 30 breast cancer FNA (Fine Needle Aspirate) measurements from this lab report.
Return ONLY a valid JSON object with exactly these keys and numeric float values.
If a value is not found, use null.

Required keys:
radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean,
radius_se, texture_se, perimeter_se, area_se, smoothness_se,
compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se,
radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst

Return ONLY the JSON. No explanation. No markdown. No code blocks.
Example: {"radius_mean": 14.5, "texture_mean": 19.2, ...}
"""

def extract_with_gemini(file_bytes: bytes, api_key: str) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    b64 = base64.standard_b64encode(file_bytes).decode()
    response = model.generate_content([
        {"mime_type": "application/pdf", "data": b64},
        GEMINI_PROMPT,
    ])

    raw = response.text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    return json.loads(raw.strip())


# ── fallback rule-based parser ─────────────────────────────────────────────────

def parse_measurements_from_text(text: str) -> dict:
    extracted = {}
    for line in text.splitlines():
        line_low = line.strip().lower()
        for fname, fkey in FEATURE_MAP.items():
            if line_low.startswith(fname):
                nums = [float(x) for x in FLOAT_RE.findall(line)]
                if len(nums) >= 3:
                    extracted[f"{fkey}_mean"]  = nums[0]
                    extracted[f"{fkey}_se"]    = nums[1]
                    extracted[f"{fkey}_worst"] = nums[2]
                break
    return extracted


# ── chart & prediction ─────────────────────────────────────────────────────────

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(["diagnosis"], axis=1)
    return {
        key: (val - X[key].min()) / (X[key].max() - X[key].min())
        for key, val in input_dict.items()
    }


def get_radar_chart(input_data):
    scaled = get_scaled_values(input_data)
    categories = ["Radius","Texture","Perimeter","Area","Smoothness",
                  "Compactness","Concavity","Concave Points","Symmetry","Fractal Dimension"]
    groups = {
        "Mean Value": [
            "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
            "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
        ],
        "Standard Error": [
            "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
            "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
        ],
        "Worst Value": [
            "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
            "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst",
        ],
    }
    fig = go.Figure()
    for label, keys in groups.items():
        fig.add_trace(go.Scatterpolar(
            r=[scaled[k] for k in keys],
            theta=categories, fill="toself", name=label,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
    )
    return fig


def add_predictions(input_data):
    model  = pkl.load(open("model/model.pkl",  "rb"))
    scaler = pkl.load(open("model/scaler.pkl", "rb"))
    arr    = np.array(list(input_data.values())).reshape(1, -1)
    arr_sc = scaler.transform(arr)
    pred   = model.predict(arr_sc)[0]
    proba  = model.predict_proba(arr_sc)[0]

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is:")
    if pred == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)
    st.write(f"**Probability of Benign:** {proba[0]:.3f}")
    st.write(f"**Probability of Malignant:** {proba[1]:.3f}")
    st.info("This app assists medical professionals but should not replace professional medical advice.")


def add_sidebar(prefill=None):
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    input_dict = {}
    for label, key in SLIDER_LABELS:
        default = float(prefill[key]) if (prefill and prefill.get(key) is not None) else float(data[key].mean())
        input_dict[key] = st.sidebar.slider(
            label, min_value=float(0),
            max_value=float(data[key].max()), value=default,
        )
    return input_dict


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    with open("assests/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.title("Breast Cancer Predictor")
    st.write(
        "Upload your cytology lab report (PDF) and the app will automatically "
        "extract measurements and predict whether the cell mass is benign or malignant."
    )

    # ── Gemini API key input ───────────────────────────────────────────────────
    with st.expander("🔑 Set up Free AI Extraction (Google Gemini — 100% Free)"):
        st.markdown("""
**Get your free Gemini API key in 1 minute:**
1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with any Google account
3. Click **"Create API Key"** → Copy it
4. Paste it below — no credit card, no payment ever required

*Key is only stored in your browser session and never saved anywhere.*
        """)
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza...",
            help="Free forever — 15 requests/min, no credit card needed",
        )

    st.subheader("📄 Upload Lab Report (PDF)")
    uploaded_file = st.file_uploader(
        "Upload any cytology PDF report from any lab",
        type=["pdf"],
        help="Works with any lab format when Gemini key is provided.",
    )

    extracted_values = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        with st.spinner("📖 Extracting measurements from report..."):
            try:
                method_used = ""

                # ── Try Gemini first (handles ANY PDF format) ──────────────
                if gemini_key and gemini_key.strip().startswith("AIza") and HAS_GEMINI:
                    try:
                        result = extract_with_gemini(file_bytes, gemini_key.strip())
                        extracted_values = {
                            k: float(v) for k, v in result.items()
                            if v is not None and k in ALL_KEYS
                        }
                        method_used = "Gemini AI"
                    except Exception as e:
                        st.warning(f"⚠️ Gemini extraction failed ({e}), trying text parser...")

                # ── Fallback: plain text rule-based parser ─────────────────
                if not extracted_values:
                    text = extract_text_from_pdf(file_bytes)
                    if text.strip():
                        extracted_values = parse_measurements_from_text(text)
                        method_used = "text parser"

                # ── Report result ──────────────────────────────────────────
                if extracted_values:
                    missing = [k for k in ALL_KEYS if k not in extracted_values]
                    found   = len(ALL_KEYS) - len(missing)

                    if found == len(ALL_KEYS):
                        st.success(f"✅ All 30 measurements extracted successfully! *(via {method_used})*")
                    elif found > 0:
                        st.warning(
                            f"⚠️ Extracted {found}/30 via {method_used}. "
                            f"Missing: {', '.join(missing)} — using dataset averages."
                        )
                    else:
                        extracted_values = None
                        st.error("❌ Could not extract measurements.")

                    if extracted_values:
                        data = get_clean_data()
                        for k in [k for k in ALL_KEYS if k not in extracted_values]:
                            extracted_values[k] = float(data[k].mean())
                else:
                    st.error(
                        "❌ Could not extract measurements from this PDF.\n\n"
                        "**To fix this:** Add your free Gemini API key above — it can read "
                        "any lab report regardless of format."
                    )

            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

    input_data = add_sidebar(prefill=extracted_values)

    col1, col2 = st.columns([4, 1])
    with col1:
        st.plotly_chart(get_radar_chart(input_data))
    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()

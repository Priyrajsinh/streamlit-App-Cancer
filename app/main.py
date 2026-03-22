import streamlit as st
import pickle as pkl
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pdfplumber
import re
import io


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

# Maps lowercase feature name (as seen in PDF) → internal key prefix
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


# ── data helpers ───────────────────────────────────────────────────────────────

@st.cache_data
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


# ── PDF parser (free, no API key) ──────────────────────────────────────────────

def try_float(s):
    """Return float if string is a valid number, else None."""
    try:
        return float(str(s).strip().replace(",", "."))
    except (ValueError, TypeError):
        return None


def match_feature(cell_text):
    """Return the internal key prefix if cell_text matches a known feature name."""
    t = str(cell_text).strip().lower()
    for fname, fkey in FEATURE_MAP.items():
        if fname in t:
            return fkey
    return None


def extract_measurements_from_pdf(file_bytes: bytes) -> dict:
    """
    Parse all 30 FNA measurements from a PDF report.
    Looks for a table whose rows are: Feature | Mean | SE | Worst
    Also falls back to scanning raw text lines.
    """
    extracted = {}

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        full_text = ""
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_text += page_text + "\n"

            # ── Strategy 1: structured table rows ─────────────────────────
            for table in (page.extract_tables() or []):
                for row in table:
                    if not row or len(row) < 4:
                        continue

                    fkey = match_feature(row[0])
                    if not fkey:
                        continue

                    # Collect all numeric cells (skip the first/feature cell)
                    nums = [try_float(c) for c in row[1:] if try_float(c) is not None]
                    if len(nums) >= 3:
                        extracted[f"{fkey}_mean"]  = nums[0]
                        extracted[f"{fkey}_se"]    = nums[1]
                        extracted[f"{fkey}_worst"] = nums[2]

    # ── Strategy 2: scan raw text lines ────────────────────────────────────
    # Handles: "Radius 13.54000 0.26990 15.11000"
    if len(extracted) < 10:
        float_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
        for line in full_text.splitlines():
            fkey = match_feature(line)
            if not fkey:
                continue
            nums = [float(x) for x in float_re.findall(line)]
            if len(nums) >= 3 and f"{fkey}_mean" not in extracted:
                extracted[f"{fkey}_mean"]  = nums[0]
                extracted[f"{fkey}_se"]    = nums[1]
                extracted[f"{fkey}_worst"] = nums[2]

    # ── Strategy 3: key=value / key: value anywhere in text ────────────────
    if len(extracted) < 10:
        for key in ALL_KEYS:
            if key in extracted:
                continue
            pattern = re.compile(
                rf"{re.escape(key)}\s*[=:]\s*([-+]?\d*\.?\d+)",
                re.IGNORECASE,
            )
            m = pattern.search(full_text)
            if m:
                extracted[key] = float(m.group(1))

    return extracted


# ── chart ──────────────────────────────────────────────────────────────────────

def get_scaled_values(input_dict):
    data = get_clean_data()
    X    = data.drop(["diagnosis"], axis=1)
    return {
        key: (val - X[key].min()) / (X[key].max() - X[key].min())
        for key, val in input_dict.items()
    }


def get_radar_chart(input_data):
    scaled = get_scaled_values(input_data)
    categories = ["Radius","Texture","Perimeter","Area","Smoothness",
                  "Compactness","Concavity","Concave Points","Symmetry","Fractal Dimension"]
    groups = {
        "Mean Value":     ["radius_mean","texture_mean","perimeter_mean","area_mean",
                           "smoothness_mean","compactness_mean","concavity_mean",
                           "concave points_mean","symmetry_mean","fractal_dimension_mean"],
        "Standard Error": ["radius_se","texture_se","perimeter_se","area_se",
                           "smoothness_se","compactness_se","concavity_se",
                           "concave points_se","symmetry_se","fractal_dimension_se"],
        "Worst Value":    ["radius_worst","texture_worst","perimeter_worst","area_worst",
                           "smoothness_worst","compactness_worst","concavity_worst",
                           "concave points_worst","symmetry_worst","fractal_dimension_worst"],
    }
    fig = go.Figure()
    for label, keys in groups.items():
        fig.add_trace(go.Scatterpolar(
            r=[scaled[k] for k in keys],
            theta=categories,
            fill="toself",
            name=label,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
    )
    return fig


# ── prediction ─────────────────────────────────────────────────────────────────

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
    st.info("This app assists medical professionals in making a diagnosis but should not replace professional medical advice.")


# ── sidebar ────────────────────────────────────────────────────────────────────

def add_sidebar(prefill=None):
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    input_dict = {}
    for label, key in SLIDER_LABELS:
        default = float(prefill[key]) if (prefill and prefill.get(key) is not None) else float(data[key].mean())
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=default,
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
        "extract the measurements and predict whether the cell mass is benign or malignant. "
        "You can also adjust values manually using the sidebar sliders."
    )

    st.subheader("📄 Upload Lab Report (PDF)")
    uploaded_file = st.file_uploader(
        "Upload a cytology PDF report",
        type=["pdf"],
        help="The app reads the PDF directly — no API key or internet connection required.",
    )

    extracted_values = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        with st.spinner("📖 Reading your report..."):
            try:
                extracted_values = extract_measurements_from_pdf(file_bytes)
                missing = [k for k in ALL_KEYS if k not in extracted_values]
                found   = len(ALL_KEYS) - len(missing)

                if found == len(ALL_KEYS):
                    st.success("✅ All 30 measurements extracted successfully!")
                elif found > 0:
                    st.warning(
                        f"⚠️ Extracted {found}/30 measurements. "
                        f"Missing values will use dataset averages: {', '.join(missing)}"
                    )
                else:
                    st.error(
                        "❌ Could not extract measurements from this PDF. "
                        "Please make sure it contains a table with Feature, Mean, SE and Worst columns, "
                        "or use the sidebar sliders to enter values manually."
                    )
                    extracted_values = None

                if extracted_values is not None:
                    data = get_clean_data()
                    for k in missing:
                        extracted_values[k] = float(data[k].mean())

            except Exception as e:
                st.error(f"❌ Error reading PDF: {e}")
                extracted_values = None

    input_data = add_sidebar(prefill=extracted_values)

    col1, col2 = st.columns([4, 1])
    with col1:
        st.plotly_chart(get_radar_chart(input_data))
    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()

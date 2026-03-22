import streamlit as st
import pickle as pkl
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import anthropic
import base64
import json


# ── helpers ────────────────────────────────────────────────────────────────────

SLIDER_LABELS = [
    ("Radius (mean)",              "radius_mean"),
    ("Texture (mean)",             "texture_mean"),
    ("Perimeter (mean)",           "perimeter_mean"),
    ("Area (mean)",                "area_mean"),
    ("Smoothness (mean)",          "smoothness_mean"),
    ("Compactness (mean)",         "compactness_mean"),
    ("Concavity (mean)",           "concavity_mean"),
    ("Concave points (mean)",      "concave points_mean"),
    ("Symmetry (mean)",            "symmetry_mean"),
    ("Fractal dimension (mean)",   "fractal_dimension_mean"),
    ("Radius (se)",                "radius_se"),
    ("Texture (se)",               "texture_se"),
    ("Perimeter (se)",             "perimeter_se"),
    ("Area (se)",                  "area_se"),
    ("Smoothness (se)",            "smoothness_se"),
    ("Compactness (se)",           "compactness_se"),
    ("Concavity (se)",             "concavity_se"),
    ("Concave points (se)",        "concave points_se"),
    ("Symmetry (se)",              "symmetry_se"),
    ("Fractal dimension (se)",     "fractal_dimension_se"),
    ("Radius (worst)",             "radius_worst"),
    ("Texture (worst)",            "texture_worst"),
    ("Perimeter (worst)",          "perimeter_worst"),
    ("Area (worst)",               "area_worst"),
    ("Smoothness (worst)",         "smoothness_worst"),
    ("Compactness (worst)",        "compactness_worst"),
    ("Concavity (worst)",          "concavity_worst"),
    ("Concave points (worst)",     "concave points_worst"),
    ("Symmetry (worst)",           "symmetry_worst"),
    ("Fractal dimension (worst)",  "fractal_dimension_worst"),
]

ALL_KEYS = [k for _, k in SLIDER_LABELS]


@st.cache_data
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


def extract_measurements_with_claude(file_bytes: bytes, mime_type: str) -> dict:
    """Send the uploaded file to Claude and get back a dict of measurements."""
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

    b64 = base64.standard_b64encode(file_bytes).decode()

    field_list = json.dumps(ALL_KEYS, indent=2)
    prompt = (
        "You are analysing a breast-cancer fine-needle aspirate (FNA) cytology "
        "lab report. Extract the numeric values for the following 30 features and "
        "return ONLY a single valid JSON object - no markdown, no explanation.\n\n"
        f"Required fields:\n{field_list}\n\n"
        "Rules:\n"
        "- All values must be numbers (float). Use null if a value is genuinely "
        "absent.\n"
        "- Field names must match exactly (including spaces and underscores).\n"
        "- Do not add any extra keys.\n"
        "Return format example (truncated):\n"
        '{"radius_mean": 14.5, "texture_mean": 19.2, ...}'
    )

    if mime_type == "application/pdf":
        file_block = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
        }
    else:
        file_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": b64},
        }

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [file_block, {"type": "text", "text": prompt}],
            }
        ],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def get_scaled_values(input_dict: dict) -> dict:
    data = get_clean_data()
    X = data.drop(["diagnosis"], axis=1)
    scaled = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled[key] = (value - min_val) / (max_val - min_val)
    return scaled


def get_radar_chart(input_data: dict):
    input_data = get_scaled_values(input_data)
    categories = [
        "Radius", "Texture", "Perimeter", "Area",
        "Smoothness", "Compactness", "Concavity",
        "Concave Points", "Symmetry", "Fractal Dimension",
    ]

    fig = go.Figure()
    suffixes = {
        "mean":  ["radius_mean","texture_mean","perimeter_mean","area_mean",
                  "smoothness_mean","compactness_mean","concavity_mean",
                  "concave points_mean","symmetry_mean","fractal_dimension_mean"],
        "se":    ["radius_se","texture_se","perimeter_se","area_se",
                  "smoothness_se","compactness_se","concavity_se",
                  "concave points_se","symmetry_se","fractal_dimension_se"],
        "worst": ["radius_worst","texture_worst","perimeter_worst","area_worst",
                  "smoothness_worst","compactness_worst","concavity_worst",
                  "concave points_worst","symmetry_worst","fractal_dimension_worst"],
    }
    groups = [("mean", "Mean Value"), ("se", "Standard Error"), ("worst", "Worst Value")]
    for group, label in groups:
        fig.add_trace(go.Scatterpolar(
            r=[input_data[k] for k in suffixes[group]],
            theta=categories,
            fill="toself",
            name=label,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
    )
    return fig


def add_predictions(input_data: dict):
    model  = pkl.load(open("model/model.pkl",  "rb"))
    scaler = pkl.load(open("model/scaler.pkl", "rb"))

    input_array        = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction         = model.predict(input_array_scaled)[0]
    proba              = model.predict_proba(input_array_scaled)[0]

    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is:")

    if prediction == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)

    st.write(f"**Probability of Benign:** {proba[0]:.3f}")
    st.write(f"**Probability of Malignant:** {proba[1]:.3f}")
    st.info(
        "This app assists medical professionals in making a diagnosis "
        "but should not replace professional medical advice."
    )


def add_sidebar(prefill=None) -> dict:
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
        "Upload your cytology lab report (PDF or image) and the app will "
        "automatically extract measurements and predict whether the cell mass "
        "is benign or malignant. You can also fine-tune values with the sidebar sliders."
    )

    st.subheader("📄 Upload Lab Report")
    uploaded_file = st.file_uploader(
        "Upload a cytology report (PDF, PNG, JPG, JPEG)",
        type=["pdf", "png", "jpg", "jpeg"],
        help="The AI will read the report and fill in all 30 measurements automatically.",
    )

    extracted_values = None

    if uploaded_file is not None:
        mime_map = {
            "pdf":  "application/pdf",
            "png":  "image/png",
            "jpg":  "image/jpeg",
            "jpeg": "image/jpeg",
        }
        ext        = uploaded_file.name.rsplit(".", 1)[-1].lower()
        mime       = mime_map.get(ext, "image/jpeg")
        file_bytes = uploaded_file.read()

        with st.spinner("🔍 Analysing your report with AI..."):
            try:
                extracted_values = extract_measurements_with_claude(file_bytes, mime)

                missing = [k for k in ALL_KEYS if extracted_values.get(k) is None]
                found   = len(ALL_KEYS) - len(missing)

                if found == len(ALL_KEYS):
                    st.success("✅ All 30 measurements extracted successfully!")
                else:
                    st.warning(
                        f"⚠️ Extracted {found}/30 measurements. "
                        f"Missing fields will use dataset averages: "
                        f"{', '.join(missing)}"
                    )

                data = get_clean_data()
                for k in missing:
                    extracted_values[k] = float(data[k].mean())

            except json.JSONDecodeError:
                st.error("❌ Could not parse measurements from the report. Please try a clearer image or adjust sliders manually.")
                extracted_values = None
            except Exception as e:
                st.error(f"❌ Error analysing report: {e}")
                extracted_values = None

    input_data = add_sidebar(prefill=extracted_values)

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()

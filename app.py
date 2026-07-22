"""
AI-Based Crop Disease Prediction System — Premium Interactive App (v6)
-----------------------------------------------------------------------
Run locally with:  streamlit run app.py

Before running, place these two files (downloaded from the Colab notebook)
in the same folder as this script:
    - crop_disease_model.h5
    - class_names.json

WHAT'S NEW IN v6
    - Richer, layered animated background (drifting gradient blobs + grain + veins)
    - "About my AI" section explaining the model in plain language
    - Interactive flip-card gallery, one stylized card per disease
    - Click a disease (gallery card or sidebar) to open a full detail panel
    - Real reference photos (Wikimedia Commons, CC-licensed) for Early blight,
      Late blight, and Septoria leaf spot — the rest use a stylized diagnostic
      illustration until a photo is added
    - Bold big/small type scale: Fraunces (serif, big display headings) +
      Inter (small body text) + IBM Plex Mono (tiny uppercase labels)
    - Custom animated loading indicator while the model runs
    - Invalid-image detection: low-confidence uploads get flagged instead of
      guessed
    - Full post-detection breakdown: disease, cause, symptoms, treatment,
      precautions, and ongoing care tips
    - All "cards" now render as single solid blocks (no more empty boxes)

ADDING YOUR OWN DISEASE PHOTOS
    Create a folder called "images" next to this script, and drop in a photo
    named EXACTLY after the class (see class_names.json), e.g.:
        images/Tomato___Bacterial_spot.jpg
        images/Tomato___healthy.png
    Your own photo always takes priority over the built-in illustration or
    Wikimedia photo for that disease — no code changes needed.
"""

import json
from pathlib import Path
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from fpdf import FPDF

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 45.0  # below this, flag as "invalid image" rather than guess

# =========================================================================
# DISEASE KNOWLEDGE BASE
# =========================================================================
DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "description": "A bacterial infection causing small, dark, water-soaked spots on leaves, stems, and fruit.",
        "symptoms": "Small brown/black spots with yellow halos; spots merge and cause leaf drop.",
        "cause": "Xanthomonas bacteria, favored by warm, wet, humid conditions and splashing water.",
        "remedy": "Use copper-based bactericides. Avoid overhead watering. Rotate crops yearly. Remove infected debris.",
        "precautions": "Avoid working in the field when leaves are wet. Disinfect tools between plants. Use certified disease-free seed/transplants.",
        "care_tips": "Water at the base early in the day so foliage dries fast. Space plants for airflow. Mulch to stop soil splashing onto leaves.",
        "severity": "Moderate",
        "pattern": "spots",
    },
    "Tomato___Early_blight": {
        "description": "A common fungal disease affecting older, lower leaves first, with target-like concentric rings.",
        "symptoms": "Brown spots with concentric rings (bullseye pattern), yellowing around spots.",
        "cause": "The fungus Alternaria solani, spreads in warm, humid weather.",
        "remedy": "Remove infected leaves. Apply fungicide (chlorothalonil or copper-based). Improve air circulation.",
        "precautions": "Rotate crops on a 2-3 year cycle. Stake plants to keep foliage off the ground. Remove volunteer tomato/potato plants nearby.",
        "care_tips": "Feed plants well (stressed plants are more susceptible). Prune lower leaves once fruit sets. Water consistently to avoid stress.",
        "severity": "Moderate",
        "pattern": "rings",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Early%20blight%20on%20tomato%20leaves%20(7871930010).jpg",
        "image_credit": "Dwight Sipler, Wikimedia Commons (CC BY-SA)",
    },
    "Tomato___Late_blight": {
        "description": "A fast-spreading, highly destructive disease — the same pathogen behind the Irish Potato Famine.",
        "symptoms": "Large, irregular water-soaked grey-green spots turning brown/black rapidly; white mold underneath.",
        "cause": "Phytophthora infestans, spreads explosively in cool, wet, humid conditions.",
        "remedy": "Destroy infected plants immediately. Apply fungicide preventively in humid conditions. Avoid wet foliage.",
        "precautions": "Monitor local blight forecasts during cool wet spells. Never compost infected material. Isolate new plants before introducing to the main patch.",
        "care_tips": "Increase plant spacing for airflow. Water only at the base. Inspect plants daily during humid weather since this disease spreads fast.",
        "severity": "Severe",
        "pattern": "blotch",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Phytophthora%20infestans%20on%20Solanum%20lycopersicum%20in%20Dnipro%20by%20baby-bear.org.jpg",
        "image_credit": "baby-bear.org, Wikimedia Commons (CC BY-SA)",
    },
    "Tomato___Leaf_Mold": {
        "description": "A fungal disease common in humid greenhouse conditions, mainly affecting leaves.",
        "symptoms": "Pale yellow spots on top of leaves, olive-green to grey fuzzy mold on undersides.",
        "cause": "The fungus Passalora fulva, thrives in high humidity and poor air circulation.",
        "remedy": "Improve ventilation, reduce humidity, apply fungicide if severe.",
        "precautions": "Keep greenhouse humidity below 85%. Avoid overhead irrigation. Choose resistant varieties where possible.",
        "care_tips": "Prune to open up the canopy. Vent greenhouses/tunnels daily. Water in the morning so leaves dry before nightfall.",
        "severity": "Moderate",
        "pattern": "mold",
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "A common fungal disease that spreads quickly in wet conditions, causing widespread leaf spotting.",
        "symptoms": "Small circular spots with dark borders and light grey centers, tiny black dots inside spots.",
        "cause": "The fungus Septoria lycopersici, spreads via splashing water and wet foliage.",
        "remedy": "Remove affected leaves, apply fungicide, avoid wetting leaves while watering.",
        "precautions": "Clear old plant debris at season end. Rotate crops. Stake and mulch to limit soil splash.",
        "care_tips": "Remove and bin (don't compost) affected lower leaves as soon as spotted. Water at soil level, not overhead.",
        "severity": "Moderate",
        "pattern": "small_dots",
        "image_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Septoria%20leaf%20spot%20symptoms%20on%20tomato%20leaf%20(Septoria%20lycopersici%20on%20Solanum%20lycopersicum%20leaf).jpg",
        "image_credit": "Wikimedia Commons (CC BY-SA)",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Not a disease but a pest infestation — tiny mites that feed on leaf tissue.",
        "symptoms": "Tiny yellow/white speckling, fine webbing visible, leaves turn bronze and dry out.",
        "cause": "Tetranychus urticae mites, thrive in hot, dry conditions.",
        "remedy": "Use miticide or insecticidal soap. Introduce natural predators like ladybugs.",
        "precautions": "Avoid drought-stressing plants. Inspect leaf undersides regularly, especially in hot dry spells. Quarantine new plants before adding to the garden.",
        "care_tips": "Hose down foliage occasionally to knock mites off and raise humidity. Encourage predatory insects. Avoid excess nitrogen fertilizer.",
        "severity": "Moderate",
        "pattern": "webbing",
    },
    "Tomato___Target_Spot": {
        "description": "A fungal disease causing distinctive target-like lesions on leaves, stems, and fruit.",
        "symptoms": "Brown lesions with concentric rings similar to early blight, can also affect fruit.",
        "cause": "The fungus Corynespora cassiicola, favored by warm, humid weather.",
        "remedy": "Apply fungicide, remove crop debris, rotate crops.",
        "precautions": "Rotate crops away from tomato/related hosts. Remove and destroy crop residue after harvest. Avoid dense overcrowded planting.",
        "care_tips": "Improve airflow through pruning and spacing. Water at the base. Scout plants weekly during warm humid weather.",
        "severity": "Moderate",
        "pattern": "rings",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "A serious viral disease transmitted by whiteflies — one of the most damaging tomato diseases worldwide.",
        "symptoms": "Upward curling and yellowing of leaves, stunted growth, reduced fruit production.",
        "cause": "TYLCV, spread by whitefly (Bemisia tabaci) insects.",
        "remedy": "Control whitefly population. Remove infected plants. Use resistant varieties.",
        "precautions": "Use insect netting or reflective mulch to deter whiteflies. Remove weeds that host whiteflies nearby. Plant certified virus-free seedlings.",
        "care_tips": "Inspect leaf undersides for whiteflies weekly. Remove and destroy infected plants promptly to protect the rest of the crop.",
        "severity": "Severe",
        "pattern": "curl",
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "A highly contagious viral disease affecting leaf appearance and plant vigor.",
        "symptoms": "Mottled light/dark green pattern, leaf distortion, stunted plant growth.",
        "cause": "ToMV, spreads via contact, tools, hands, and infected seeds — highly persistent.",
        "remedy": "Remove and destroy infected plants. Disinfect tools. Avoid tobacco use near plants.",
        "precautions": "Wash hands and disinfect tools between plants. Use certified virus-free seed. Avoid handling plants after using tobacco products.",
        "care_tips": "Remove infected plants promptly to stop spread. Don't take cuttings/save seed from affected plants. Keep a dedicated tool set per section of the garden if possible.",
        "severity": "Severe",
        "pattern": "mosaic",
    },
    "Tomato___healthy": {
        "description": "No disease detected — the leaf shows no visible signs of infection or pest damage.",
        "symptoms": "Uniform green color, no spots, curling, or discoloration.",
        "cause": "N/A",
        "remedy": "No action needed. Keep monitoring regularly, maintain good watering and spacing practices.",
        "precautions": "Keep monitoring weekly. Maintain crop rotation. Avoid overhead watering as a preventive habit.",
        "care_tips": "Feed on a regular schedule, keep consistent soil moisture, and stake/prune for good airflow to keep the plant thriving.",
        "severity": "None",
        "pattern": "healthy",
    },
}

# =========================================================================
# PALETTE
# =========================================================================
COL_FOREST = "#16342b"
COL_FOREST_DEEP = "#0d211a"
COL_SAGE = "#8fbf9f"
COL_SAGE_LIGHT = "#d7ead9"
COL_CREAM = "#faf6ec"
COL_AMBER = "#e2a23a"
COL_TERRACOTTA = "#bf5b45"
COL_INK = "#20342c"
COL_PLUM = "#6a4c6e"
COL_TEAL = "#3f7a72"

SEVERITY_COLOR = {"None": COL_SAGE, "Moderate": COL_AMBER, "Severe": COL_TERRACOTTA}


# =========================================================================
# MODEL HELPERS
# =========================================================================
@st.cache_resource
def load_assets():
    model = load_model("crop_disease_model.h5")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


def predict(img: Image.Image, model, class_names):
    img_resized = img.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return class_names[idx], float(preds[idx]) * 100, preds


def clean_name(label: str) -> str:
    return label.replace("Tomato___", "").replace("_", " ").strip()


def radial_gauge_svg(percent: float, color: str) -> str:
    r = 54
    circumference = 2 * 3.14159 * r
    offset = circumference * (1 - percent / 100)
    return f"""
    <svg width="180" height="180" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r="{r}" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="12"/>
        <circle cx="70" cy="70" r="{r}" fill="none" stroke="{color}" stroke-width="12"
            stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{circumference}"
            transform="rotate(-90 70 70)">
            <animate attributeName="stroke-dashoffset" from="{circumference}" to="{offset}" dur="1.1s" fill="freeze" calcMode="spline" keySplines="0.25 0.1 0.25 1"/>
        </circle>
        <text x="70" y="65" text-anchor="middle" font-size="26" font-family="Fraunces, serif" font-weight="700" fill="{COL_CREAM}">{percent:.0f}%</text>
        <text x="70" y="86" text-anchor="middle" font-size="10" font-family="IBM Plex Mono, monospace" fill="{COL_SAGE_LIGHT}">CONFIDENCE</text>
    </svg>
    """


# =========================================================================
# DISEASE GALLERY — stylized SVG "leaf" illustrations (no external images
# needed, keeps everything offline-safe & on-brand)
# =========================================================================
def leaf_svg(pattern: str, color: str) -> str:
    """Return a small stylized leaf illustration whose markings hint at the
    disease pattern (spots, rings, curl, mold, mosaic, webbing, healthy)."""
    base_leaf = (
        f'<path d="M60 10 C95 25 100 70 60 110 C20 70 25 25 60 10 Z" '
        f'fill="#2c4f3f" stroke="{color}" stroke-width="2.5"/>'
        f'<path d="M60 14 L60 106" stroke="{color}" stroke-width="1.4" opacity="0.5"/>'
    )
    marks = ""
    if pattern == "spots":
        marks = "".join(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" opacity="0.85"/>'
            for cx, cy, r in [(45, 40, 4), (72, 55, 5), (50, 75, 3.5), (68, 88, 4)]
        )
    elif pattern == "rings":
        marks = "".join(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="2"/>'
            for cx, cy in [(50, 45), (70, 70)]
            for r in (4, 8)
        )
    elif pattern == "blotch":
        marks = f'<path d="M40 35 Q55 30 68 45 Q80 60 65 75 Q48 85 38 68 Q30 50 40 35 Z" fill="{color}" opacity="0.75"/>'
    elif pattern == "mold":
        marks = "".join(
            f'<ellipse cx="{cx}" cy="{cy}" rx="10" ry="6" fill="{color}" opacity="0.55"/>'
            for cx, cy in [(48, 50), (68, 68), (55, 85)]
        )
    elif pattern == "small_dots":
        marks = "".join(
            f'<circle cx="{40 + (i % 4) * 12}" cy="{35 + (i // 4) * 18}" r="2.4" fill="{color}"/>'
            for i in range(12)
        )
    elif pattern == "webbing":
        marks = (
            f'<path d="M35 35 Q60 55 40 90" stroke="{color}" stroke-width="1" fill="none" opacity="0.8"/>'
            f'<path d="M85 35 Q60 55 80 90" stroke="{color}" stroke-width="1" fill="none" opacity="0.8"/>'
            f'<path d="M60 20 Q60 55 60 100" stroke="{color}" stroke-width="1" fill="none" opacity="0.8"/>'
        )
    elif pattern == "curl":
        marks = f'<path d="M35 30 Q70 45 40 100" stroke="{color}" stroke-width="3" fill="none" opacity="0.8"/>'
    elif pattern == "mosaic":
        marks = "".join(
            f'<rect x="{35 + (i % 3) * 18}" y="{30 + (i // 3) * 20}" width="12" height="12" '
            f'fill="{color}" opacity="{0.3 + 0.15 * (i % 3)}" rx="2"/>'
            for i in range(9)
        )
    elif pattern == "healthy":
        marks = f'<path d="M45 55 Q55 68 75 40" stroke="{color}" stroke-width="4" fill="none" stroke-linecap="round"/>'
    return f'<svg viewBox="0 0 120 120" width="100%" height="100%">{base_leaf}{marks}</svg>'


def get_disease_photo(label: str):
    """Return (source, credit) for the best available real photo, or (None, None).
    Priority: a local file in ./images/<label>.jpg or .png (drop your own
    photos there — filename must exactly match the class name from
    class_names.json, e.g. images/Tomato___Early_blight.jpg) takes priority
    over the verified Wikimedia Commons URL baked into DISEASE_INFO.
    If neither exists, caller falls back to the stylized illustration."""
    for ext in ("jpg", "jpeg", "png"):
        local_path = Path("images") / f"{label}.{ext}"
        if local_path.exists():
            return str(local_path), "Your own photo"
    info = DISEASE_INFO.get(label, {})
    if info.get("image_url"):
        return info["image_url"], info.get("image_credit", "Wikimedia Commons")
    return None, None


def generate_diagnosis_pdf(disease_name: str, severity: str, confidence: float, info: dict) -> bytes:
    """Build a one-page PDF diagnosis report and return it as bytes."""

    def clean(text: str) -> str:
        # Core PDF fonts only support latin-1; swap any characters that fall
        # outside it (smart quotes, em dashes, etc.) so nothing crashes.
        return (text or "N/A").encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF(format="A4", unit="mm")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    # Header
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(22, 52, 43)
    pdf.cell(0, 12, "Verdant AI", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, "Crop Disease Diagnosis Report", ln=True)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", ln=True)
    pdf.ln(4)
    pdf.set_draw_color(143, 191, 159)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    # Result headline
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(22, 52, 43)
    pdf.cell(0, 10, clean(disease_name), ln=True)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(120, 80, 20)
    pdf.cell(0, 7, f"Severity: {clean(severity)}   |   Model confidence: {confidence:.1f}%", ln=True)
    pdf.ln(4)

    def section(title: str, body: str):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(22, 52, 43)
        pdf.cell(0, 8, clean(title), ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 6, clean(body))
        pdf.ln(3)

    section("What it is", info.get("description", "N/A"))
    section("Symptoms", info.get("symptoms", "N/A"))
    section("Reasons / Cause", info.get("cause", "N/A"))
    section("Recommended Treatment", info.get("remedy", "N/A"))
    section("Precautions", info.get("precautions", "N/A"))
    section("Ongoing Care Tips", info.get("care_tips", "N/A"))

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, clean(
        "Verdant AI is an educational diagnostic aid, not a substitute for professional agronomic advice."
    ))

    return bytes(pdf.output())


# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(page_title="Verdant AI — Crop Disease Diagnostics", page_icon="🌿", layout="wide")

# session state defaults
if "selected_disease" not in st.session_state:
    st.session_state.selected_disease = None

# =========================================================================
# STYLES
# =========================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;0,9..144,700;0,9..144,800;1,9..144,500&family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 16px;
    line-height: 1.6;
}}

/* SAFETY NET: force every plain-text Streamlit element to a visible light
   color, so nothing can silently render as dark-text-on-dark-background */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stApp .stMarkdown, .stApp .stCaption, .stApp .stText,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
    color: {COL_CREAM};
}}
.stApp .stButton button, .stApp .stFormSubmitButton button {{
    color: {COL_INK};
}}

/* ---------- LAYERED ANIMATED BACKGROUND ---------- */
.stApp {{
    position: relative;
    min-height: 100vh;
    background:
        radial-gradient(circle at 15% 10%, rgba(143,191,159,0.20) 0%, transparent 45%),
        radial-gradient(circle at 85% 90%, rgba(191,91,69,0.12) 0%, transparent 42%),
        radial-gradient(circle at 75% 15%, rgba(106,76,110,0.16) 0%, transparent 40%),
        linear-gradient(160deg, {COL_FOREST_DEEP} 0%, {COL_FOREST} 55%, #1d3b30 100%);
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.05;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Cpath d='M60 0 Q80 40 60 60 Q40 40 60 0' stroke='%23ffffff' stroke-width='1' fill='none'/%3E%3Cpath d='M60 60 Q80 80 60 120 Q40 80 60 60' stroke='%23ffffff' stroke-width='1' fill='none'/%3E%3C/svg%3E");
    z-index: -1;
}}

/* drifting soft blobs for a "living" background */
.stApp::after {{
    content: "";
    position: fixed;
    inset: -10%;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(circle 320px at 20% 30%, rgba(143,191,159,0.14), transparent 60%),
        radial-gradient(circle 260px at 80% 70%, rgba(226,162,58,0.10), transparent 60%),
        radial-gradient(circle 300px at 60% 20%, rgba(106,76,110,0.10), transparent 60%);
    animation: driftBlobs 26s ease-in-out infinite alternate;
    filter: blur(6px);
    z-index: -1;
}}
@keyframes driftBlobs {{
    0%   {{ transform: translate(0px, 0px) scale(1); }}
    50%  {{ transform: translate(3%, -4%) scale(1.06); }}
    100% {{ transform: translate(-3%, 3%) scale(1); }}
}}

/* ---------- TYPOGRAPHY SYSTEM ---------- */
h1, h2, h3, h4, h5, .hero-title {{
    font-family: 'Fraunces', Georgia, serif;
    color: {COL_CREAM};
    font-weight: 600;
    letter-spacing: -0.01em;
    line-height: 1.2;
}}
h3 {{ font-weight: 600; font-size: 1.35rem !important; }}
h4 {{ font-weight: 600; font-size: 1.1rem !important; }}
strong, b {{ font-weight: 700; }}
p, .stMarkdown p {{ line-height: 1.65; }}
[data-testid="stCaptionContainer"], .stApp small {{
    font-size: 0.82rem !important;
    opacity: 0.85;
}}

.hero-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-size: 0.72rem;
    font-weight: 500;
    color: {COL_SAGE};
    margin-bottom: 10px;
}}

.site-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(250,246,236,0.14);
}}
.site-logo {{ font-size: 2.6rem; line-height: 1; }}
.site-name {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 800;
    font-size: 2.2rem;
    letter-spacing: -0.02em;
    color: {COL_CREAM};
}}

.hero-title {{
    font-size: 4.4rem;
    font-weight: 700;
    line-height: 1.02;
    letter-spacing: -0.03em;
    margin: 0;
    background: linear-gradient(90deg, {COL_CREAM} 30%, {COL_SAGE} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.hero-sub {{
    color: #c9d9cd;
    font-size: 1.02rem;
    line-height: 1.6;
    max-width: 600px;
    margin-top: 16px;
    font-weight: 400;
}}

.section-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2.3rem;
    font-weight: 600;
    letter-spacing: -0.015em;
    color: {COL_CREAM};
    margin-bottom: 4px;
}}
.section-sub {{
    color: #c9d9cd;
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 22px;
    max-width: 640px;
}}
.section-kicker {{
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.65rem;
    font-weight: 500;
    color: {COL_AMBER};
    margin-bottom: 6px;
}}

.vein-divider {{ width: 100%; margin: 22px 0 30px 0; }}
.vein-path {{
    stroke: {COL_SAGE};
    stroke-width: 2;
    fill: none;
    stroke-dasharray: 900;
    stroke-dashoffset: 900;
    animation: draw 2.6s ease-out forwards;
}}
@keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
.fade-in {{ animation: fadeInUp 0.6s ease-out both; }}
.fade-in-1 {{ animation-delay: 0.05s; }}
.fade-in-2 {{ animation-delay: 0.15s; }}
.fade-in-3 {{ animation-delay: 0.25s; }}

/* Base: gentle fade-up on load — works everywhere, never leaves content hidden */
.scroll-reveal {{
    animation: fadeInUp 0.7s ease-out both;
}}

/* Enhancement: on browsers that support scroll-driven animations (Chrome/Edge 115+),
   sections instead animate in as they scroll into view. Wrapped in @supports so
   browsers without it simply keep the fade-in-on-load behavior above — content
   can never get stuck invisible. */
@supports (animation-timeline: view()) {{
    .scroll-reveal {{
        animation: scrollReveal linear both;
        animation-timeline: view();
        animation-range: entry 0% cover 35%;
    }}
}}
@keyframes scrollReveal {{
    from {{ opacity: 0; transform: translateY(36px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.glass-card {{
    background: rgba(250, 246, 236, 0.06);
    border: 1px solid rgba(250, 246, 236, 0.14);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 18px;
    transition: transform 0.25s ease, border-color 0.25s ease;
    position: relative;
    z-index: 1;
}}
.glass-card:hover {{ transform: translateY(-2px); border-color: rgba(143,191,159,0.5); }}

.paper-card {{
    background: {COL_CREAM};
    color: {COL_INK};
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    position: relative;
    z-index: 1;
}}
.paper-card b {{ color: {COL_FOREST}; }}

.badge {{
    display: inline-block;
    padding: 5px 16px;
    border-radius: 999px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: {COL_FOREST_DEEP};
    letter-spacing: 0.03em;
}}

.metric-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {COL_SAGE};
}}
.metric-value {{
    font-family: 'Fraunces', serif;
    font-size: 2.1rem;
    color: {COL_CREAM};
    font-weight: 600;
    letter-spacing: -0.01em;
}}

.gauge-wrap {{ display:flex; justify-content:center; align-items:center; padding: 10px 0; }}

section[data-testid="stSidebar"] {{ background: {COL_FOREST_DEEP}; border-right: 1px solid rgba(255,255,255,0.08); }}
section[data-testid="stSidebar"] * {{ color: {COL_SAGE_LIGHT} !important; }}

footer {{ visibility: hidden; }}

/* ---------- DISEASE GALLERY FLIP CARDS ---------- */
.flip-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 16px;
    margin-bottom: 8px;
}}
.flip-card {{
    background: transparent;
    perspective: 1000px;
    height: 190px;
}}
.flip-inner {{
    position: relative;
    width: 100%; height: 100%;
    text-align: center;
    transition: transform 0.6s;
    transform-style: preserve-3d;
}}
.flip-card:hover .flip-inner {{ transform: rotateY(180deg); }}
.flip-front, .flip-back {{
    position: absolute; inset: 0;
    -webkit-backface-visibility: hidden;
    backface-visibility: hidden;
    border-radius: 14px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 10px;
    border: 1px solid rgba(250,246,236,0.16);
}}
.flip-front {{ background: rgba(250,246,236,0.07); }}
.flip-front .icon-wrap {{ width: 70px; height: 70px; margin-bottom: 8px; }}
.flip-front .flip-name {{
    font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.78rem;
    color: {COL_CREAM}; line-height: 1.2;
}}
.flip-back {{
    background: {COL_CREAM};
    color: {COL_INK};
    transform: rotateY(180deg);
    font-size: 0.72rem;
    line-height: 1.35;
    overflow: hidden;
}}
.flip-back b {{ display:block; font-family:'Fraunces', serif; font-size: 0.86rem; margin-bottom: 4px; color: {COL_FOREST}; }}

/* ---------- ABOUT AI ---------- */
.ai-pillar {{
    background: rgba(250,246,236,0.05);
    border: 1px solid rgba(250,246,236,0.12);
    border-radius: 14px;
    padding: 18px 20px;
    height: 100%;
}}
.ai-pillar .pillar-emoji {{ font-size: 1.6rem; }}
.ai-pillar h4 {{ font-family:'Fraunces', serif; color:{COL_CREAM}; margin: 8px 0 6px 0; font-size: 1.05rem; }}
.ai-pillar p {{ color:#c9d9cd; font-size: 0.88rem; margin:0; }}

/* ---------- CUSTOM LOADER ---------- */
.leaf-loader-wrap {{
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 40px 0;
}}
.leaf-loader {{
    width: 64px; height: 64px;
    animation: spin 1.1s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.leaf-loader-text {{
    margin-top: 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {COL_SAGE_LIGHT};
}}

/* ---------- NATIVE BORDERED CONTAINERS (reliable — no HTML-parsing quirks) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: rgba(250, 246, 236, 0.06);
    border: 1px solid rgba(250, 246, 236, 0.14) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(14px);
}}

/* ---------- DISEASE DETAIL PANEL (click a disease to open) ---------- */
.detail-badge {{
    display:inline-block; padding: 5px 16px; border-radius: 999px;
    font-family:'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight:600;
    color: {COL_FOREST_DEEP};
}}
.diagram-frame {{
    background: radial-gradient(circle at 50% 40%, rgba(250,246,236,0.08), rgba(250,246,236,0.02));
    border: 1px solid rgba(250,246,236,0.16);
    border-radius: 14px;
    padding: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 180px;
    height: 180px;
    margin-bottom: 8px;
}}
.diagram-icon {{ width: 100%; height: 100%; }}
.diagram-caption {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {COL_SAGE};
    margin-bottom: 10px;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown("### 🌿 Verdant AI")
    st.caption("Tomato leaf diagnostics, powered by MobileNetV2")
    st.markdown("---")
    st.markdown("**Model**")
    st.write("Transfer-learned CNN · trained on PlantVillage (Tomato subset)")
    st.markdown("**Detects — click a disease to learn more**")
    for c in DISEASE_INFO.keys():
        if st.button(clean_name(c), key=f"nav_{c}", use_container_width=True):
            st.session_state.selected_disease = c
            st.rerun()
    st.markdown("---")
    st.caption("AI internship project · for educational use")

# =========================================================================
# TOP HEADER
# =========================================================================
st.markdown(
    '<div class="site-header">'
    '<span class="site-logo">🌿</span>'
    '<span class="site-name">Verdant AI</span>'
    '</div>',
    unsafe_allow_html=True,
)

# =========================================================================
# HERO
# =========================================================================
st.markdown('<div class="hero-eyebrow fade-in">AI-Powered Plant Diagnostics</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-title fade-in fade-in-1">Read the leaf.<br>Know the cure.</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub fade-in fade-in-2">Upload a photo of a tomato leaf and get an instant AI diagnosis — '
    'disease identity, severity, symptoms, and a treatment plan — in seconds.</p>',
    unsafe_allow_html=True,
)

st.markdown("""
<svg class="vein-divider" height="40" viewBox="0 0 1200 40" preserveAspectRatio="none">
    <path class="vein-path" d="M0 20 Q150 0 300 20 T600 20 T900 20 T1200 20" />
</svg>
""", unsafe_allow_html=True)
import traceback

try:
    model, class_names = load_assets()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error("Could not load model files. Real error below:")
    st.code(traceback.format_exc(), language="python")

# =========================================================================
# ABOUT MY AI
# =========================================================================
st.markdown('<div class="scroll-reveal">', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Model &amp; methodology</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧠 About my AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Verdant AI is a computer-vision model that has learned to recognize what healthy '
    'and diseased tomato leaves look like, the same way you\'d learn to spot a rash by seeing many examples of it.</div>',
    unsafe_allow_html=True,
)

pillars = [
    ("🧬", "How it learned", "Trained on the PlantVillage tomato-leaf dataset — thousands of labeled photos of healthy leaves and nine common diseases/pests."),
    ("⚙️", "Architecture", "Built on MobileNetV2, a lightweight convolutional neural network, fine-tuned specifically to tell tomato diseases apart."),
    ("🔍", "How it decides", "Every uploaded photo is resized to 224×224 pixels, and the model outputs a probability for each of the 10 possible classes — the highest one becomes the diagnosis."),
    ("🎯", "What confidence means", "The percentage you see is how sure the model is about its top guess. Lower confidence usually means a blurry photo, poor lighting, or an unusual case."),
    ("🌱", "Where it helps", "It's built for quick first-look screening in the field or garden — catching common issues early, before they spread."),
    ("⚠️", "Its limits", "It only recognizes the 10 classes it was trained on, works best on a single leaf filling most of the frame, and is an educational aid, not a lab diagnosis."),
]

cols = st.columns(3)
for i, (emoji, title, text) in enumerate(pillars):
    with cols[i % 3]:
        st.markdown(
            f'<div class="ai-pillar fade-in fade-in-{(i % 3) + 1}">'
            f'<div class="pillar-emoji">{emoji}</div><h4>{title}</h4><p>{text}</p></div>',
            unsafe_allow_html=True,
        )
st.write("")
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================================
# INTERACTIVE DISEASE GALLERY
# =========================================================================
st.markdown('<div class="scroll-reveal">', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Reference library</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🍃 Disease gallery</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Hover a card for a quick summary, or click the button under it to open the full guide below.</div>',
    unsafe_allow_html=True,
)

# Build each flip-card as ONE continuous line (no line breaks / indentation)
# so the browser always renders it as HTML instead of an indented code block.
card_pieces = []
for label, info in DISEASE_INFO.items():
    color = SEVERITY_COLOR.get(info["severity"], COL_SAGE)
    name = clean_name(label)
    icon = leaf_svg(info["pattern"], color)
    card_pieces.append(
        f'<div class="flip-card"><div class="flip-inner">'
        f'<div class="flip-front"><div class="icon-wrap">{icon}</div><div class="flip-name">{name}</div></div>'
        f'<div class="flip-back"><b>{name}</b>{info["description"]}</div>'
        f'</div></div>'
    )
cards_html = '<div class="flip-grid">' + "".join(card_pieces) + "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

st.write("")
st.caption("🔍 Click a disease to view full details, symptoms, treatment, and care tips:")
button_cols = st.columns(5)
for i, label in enumerate(DISEASE_INFO.keys()):
    with button_cols[i % 5]:
        if st.button(clean_name(label), key=f"gallery_{label}", use_container_width=True):
            st.session_state.selected_disease = label
            st.rerun()

# ---- Disease detail panel (opens when a disease is clicked, above or below) ----
if st.session_state.selected_disease:
    sel_label = st.session_state.selected_disease
    sel_info = DISEASE_INFO.get(sel_label, {})
    sel_name = clean_name(sel_label)
    sel_color = SEVERITY_COLOR.get(sel_info.get("severity", ""), COL_SAGE)
    sel_icon = leaf_svg(sel_info.get("pattern", "healthy"), sel_color)

    with st.container(border=True):
        top_col, close_col = st.columns([6, 1])
        with top_col:
            st.markdown(f"### 🔍 {sel_name}")
        with close_col:
            if st.button("✕ Close", key="close_detail"):
                st.session_state.selected_disease = None
                st.rerun()

        img_col, text_col = st.columns([1, 2])
        with img_col:
            photo_src, photo_credit = get_disease_photo(sel_label)
            if photo_src:
                try:
                    st.image(photo_src, use_container_width=True, caption=f"📷 {photo_credit}")
                except Exception:
                    st.markdown(
                        f'<div class="diagram-frame"><div class="diagram-icon">{sel_icon}</div></div>'
                        f'<div class="diagram-caption">Diagnostic illustration</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<div class="diagram-frame"><div class="diagram-icon">{sel_icon}</div></div>'
                    f'<div class="diagram-caption">Diagnostic illustration · no reference photo yet</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<span class="detail-badge" style="background:{sel_color};">'
                f'Severity: {sel_info.get("severity","Unknown")}</span>',
                unsafe_allow_html=True,
            )
        with text_col:
            st.markdown(f"**What it is** — {sel_info.get('description','N/A')}")
            st.markdown(f"**Symptoms** — {sel_info.get('symptoms','N/A')}")
            st.markdown(f"**Reasons / cause** — {sel_info.get('cause','N/A')}")

        st.markdown("---")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("**💊 Treatment**")
            st.caption(sel_info.get("remedy", "N/A"))
        with d2:
            st.markdown("**🛡️ Precautions**")
            st.caption(sel_info.get("precautions", "N/A"))
        with d3:
            st.markdown("**🌱 Care tips**")
            st.caption(sel_info.get("care_tips", "N/A"))

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================================
# STATS ROW
# =========================================================================
s1, s2, s3, s4 = st.columns(4)
stats = [
    ("Classes detected", "10"),
    ("Base architecture", "MobileNetV2"),
    ("Input resolution", "224 × 224"),
    ("Dataset", "PlantVillage"),
]
for col, (stat_label, value) in zip([s1, s2, s3, s4], stats):
    with col:
        with st.container(border=True):
            st.markdown(f'<div class="metric-label">{stat_label}</div><div class="metric-value">{value}</div>', unsafe_allow_html=True)

st.write("")

# =========================================================================
# MAIN: UPLOAD + RESULT
# =========================================================================
st.markdown('<div class="scroll-reveal">', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Live analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔬 Run a diagnosis</div>', unsafe_allow_html=True)

col_upload, col_result = st.columns([1, 1.4])

with col_upload:
    with st.container(border=True):
        st.markdown("#### 📤 Upload a leaf photo")
        uploaded_file = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, use_container_width=True, caption="Uploaded image")
        else:
            st.caption("JPG or PNG · a single tomato leaf works best, filling most of the frame.")

with col_result:
    if uploaded_file is not None and model_ready:
        loader_placeholder = st.empty()
        loader_placeholder.markdown(
            f'<div class="leaf-loader-wrap"><svg class="leaf-loader" viewBox="0 0 60 60">'
            f'<circle cx="30" cy="30" r="24" fill="none" stroke="rgba(255,255,255,0.15)" stroke-width="6"/>'
            f'<circle cx="30" cy="30" r="24" fill="none" stroke="{COL_SAGE}" stroke-width="6" '
            f'stroke-linecap="round" stroke-dasharray="80 80"/></svg>'
            f'<div class="leaf-loader-text">Analyzing leaf structure…</div></div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.9)
        label, confidence, all_preds = predict(img, model, class_names)
        loader_placeholder.empty()

        if confidence < CONFIDENCE_THRESHOLD:
            with st.container(border=True):
                st.markdown("### 🚫 Invalid image")
                st.error(
                    f"This doesn't look like a tomato leaf the model can confidently identify "
                    f"(best guess confidence was only {confidence:.1f}%, below the {CONFIDENCE_THRESHOLD:.0f}% threshold)."
                )
                st.markdown(
                    "**Tips for a valid photo:**\n"
                    "- Use a single tomato leaf, filling most of the frame\n"
                    "- Good, even lighting — avoid heavy shadows or glare\n"
                    "- Plain background behind the leaf\n"
                    "- In-focus, not blurry or too far away"
                )
        else:
            info = DISEASE_INFO.get(label, {})
            disease_name = clean_name(label)
            is_healthy = "healthy" in label.lower()
            severity = info.get("severity", "Unknown")
            sev_color = SEVERITY_COLOR.get(severity, "#999")
            result_icon = "✅" if is_healthy else "⚠️"

            with st.container(border=True):
                st.markdown(f"### {result_icon} {disease_name}")
                st.markdown(
                    f'<span class="badge" style="background:{sev_color};">Severity: {severity}</span> '
                    f'<span class="badge" style="background:#dfe7e1;">Confidence: {confidence:.1f}%</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.markdown(f"**What it is** — {info.get('description','N/A')}")
                st.markdown(f"**Symptoms** — {info.get('symptoms','N/A')}")
                st.markdown(f"**Reasons / cause** — {info.get('cause','N/A')}")

            with st.container(border=True):
                st.markdown(f"**💊 Recommended treatment** — {info.get('remedy','N/A')}")
                st.markdown(f"**🛡️ Precautions** — {info.get('precautions','N/A')}")
                st.markdown(f"**🌱 Ongoing care tips** — {info.get('care_tips','N/A')}")

            pdf_bytes = generate_diagnosis_pdf(disease_name, severity, confidence, info)
            st.download_button(
                "📄 Download full diagnosis report (PDF)",
                data=pdf_bytes,
                file_name=f"Verdant_AI_{disease_name.replace(' ', '_')}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            gauge_col, bar_col = st.columns([0.8, 1.2])
            with gauge_col:
                with st.container(border=True):
                    st.markdown(f'<div class="gauge-wrap">{radial_gauge_svg(confidence, sev_color)}</div>', unsafe_allow_html=True)

            with bar_col:
                with st.container(border=True):
                    st.markdown('<div class="metric-label" style="margin-bottom:10px;">FULL CLASS BREAKDOWN</div>', unsafe_allow_html=True)
                    sorted_pairs = sorted(zip(class_names, all_preds), key=lambda x: -x[1])
                    df = pd.DataFrame({
                        "Disease": [clean_name(c) for c, _ in sorted_pairs],
                        "Confidence (%)": [round(p * 100, 2) for _, p in sorted_pairs],
                    }).set_index("Disease")
                    st.bar_chart(df, color=sev_color, height=260)

    elif uploaded_file is not None and not model_ready:
        with st.container(border=True):
            st.markdown("#### Model not loaded")
            st.caption("Add crop_disease_model.h5 and class_names.json next to app.py, then reload.")

    else:
        with st.container(border=True):
            st.markdown("#### 🍃 Awaiting an image")
            st.caption("Upload a tomato leaf photo on the left — your diagnosis, charts, and treatment plan will appear here.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ Verdant AI is an educational diagnostic aid, not a substitute for professional agronomic advice.")

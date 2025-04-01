# === Updated app.py with Phase 1 Features and Layout Styling ===
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from fpdf import FPDF
import tempfile

# ---- Page Config ----
st.set_page_config(page_title="Should I Grade This?", page_icon="📸")
st.markdown("""
    <style>
    .section-header {
        background-color: #e6f0ff;
        color: #000000;
        padding: 0.5em 1em;
        margin-top: 1.5em;
        border-left: 5px solid #0066cc;
        font-size: 1.2em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📸 Should I Grade This?")
st.markdown("Upload a sports card image and get centering, corner sharpness, and surface scores — with a visual heatmap, ROI estimator, and PDF report.")

# ---- Upload ----
uploaded_file = st.file_uploader("Upload a card image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# ---- User Input ----
st.markdown("<div class='section-header'>📝 Card Details</div>", unsafe_allow_html=True)
card_title = st.text_input("Card Title (Optional)", placeholder="e.g. 2023 Topps Chrome J-Rod Refractor")
card_notes = st.text_area("Notes (Optional)", placeholder="e.g. Pulled from hobby box, looks sharp!")

st.markdown("<div class='section-header'>💸 Grading ROI Estimator</div>", unsafe_allow_html=True)
raw_value = st.number_input("Estimated Raw Card Value ($)", min_value=0.0, value=20.0)
grading_cost = st.number_input("Grading Cost ($)", min_value=0.0, value=19.0)
psa9_value = st.number_input("Estimated PSA 9 Value ($)", min_value=0.0, value=35.0)
psa10_value = st.number_input("Estimated PSA 10 Value ($)", min_value=0.0, value=65.0)

# ---- Analysis Functions ----
# ... [functions stay the same]

# ---- Main Flow ----
if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))
    center_score, card_rect = analyze_centering(image_np)
    corner_score = analyze_corners(image_np, card_rect)
    surface_score = analyze_surface(image_np, card_rect)
    heatmap_img = generate_surface_heatmap(image_np, card_rect)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class='section-header'>📊 Scores</div>", unsafe_allow_html=True)
        st.markdown(f"**Centering:** {center_score}/100")
        st.markdown(f"**Corners:** {corner_score}/100")
        st.markdown(f"**Surface:** {surface_score}/100")

        st.markdown("<div class='section-header'>🎯 Instant Grade Probability</div>", unsafe_allow_html=True)
        avg_score = (center_score + corner_score + surface_score) / 3
        if avg_score > 90:
            st.success("Most likely grade: PSA 10 (High confidence)")
        elif avg_score > 80:
            st.info("Most likely grade: PSA 9–10 (Medium confidence)")
        elif avg_score > 70:
            st.warning("Most likely grade: PSA 8–9 (Low confidence)")
        else:
            st.error("Most likely grade: Below PSA 8")

        st.markdown("<div class='section-header'>📈 Grading ROI Estimate</div>", unsafe_allow_html=True)
        expected_profit_9 = psa9_value - grading_cost
        expected_profit_10 = psa10_value - grading_cost
        st.write(f"**Profit if PSA 9:** ${expected_profit_9:.2f}")
        st.write(f"**Profit if PSA 10:** ${expected_profit_10:.2f}")
        if expected_profit_9 < 0 and expected_profit_10 < 10:
            st.warning("Grading may not be worth it based on ROI.")
        else:
            st.success("Could be worth grading depending on actual grade!")

        st.markdown("<div class='section-header'>📤 Export Report</div>", unsafe_allow_html=True)
        pdf_data = create_grading_report(center_score, corner_score, surface_score, card_title, card_notes, image_np, heatmap_img)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_data,
            file_name="grading_report.pdf",
            mime="application/pdf"
        )

    with col2:
        st.markdown("<div class='section-header'>🖼️ Card Preview</div>", unsafe_allow_html=True)
        st.image(image_np, caption="Uploaded Card", use_container_width=True)

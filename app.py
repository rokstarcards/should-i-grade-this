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
st.title("📸 Should I Grade This?")
st.markdown("Upload a sports card image and get centering, corner sharpness, and surface scores — with a visual heatmap, ROI estimator, and PDF report.")

# ---- Upload ----
uploaded_file = st.file_uploader("Upload a card image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# ---- User Input ----
st.subheader("📝 Card Details")
card_title = st.text_input("Card Title (Optional)", placeholder="e.g. 2023 Topps Chrome J-Rod Refractor")
card_notes = st.text_area("Notes (Optional)", placeholder="e.g. Pulled from hobby box, looks sharp!")

st.subheader("💸 Grading ROI Estimator")
raw_value = st.number_input("Estimated Raw Card Value ($)", min_value=0.0, value=20.0)
grading_cost = st.number_input("Grading Cost ($)", min_value=0.0, value=19.0)
psa9_value = st.number_input("Estimated PSA 9 Value ($)", min_value=0.0, value=35.0)
psa10_value = st.number_input("Estimated PSA 10 Value ($)", min_value=0.0, value=65.0)

# ---- Analysis Functions ----
def analyze_centering(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, (0, 0, 0, 0)
    card_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_contour)
    img_center_x, img_center_y = image.shape[1] / 2, image.shape[0] / 2
    card_center_x, card_center_y = x + w / 2, y + h / 2
    off_center_x = abs(img_center_x - card_center_x) / img_center_x
    off_center_y = abs(img_center_y - card_center_y) / img_center_y
    center_score = max(0, 100 - (off_center_x + off_center_y) * 100)
    return round(center_score, 2), (x, y, w, h)

def analyze_corners(image, rect):
    x, y, w, h = rect
    if w == 0 or h == 0:
        return 0
    corner_regions = [
        image[y:y+20, x:x+20], image[y:y+20, x+w-20:x+w],
        image[y+h-20:y+h, x:x+20], image[y+h-20:y+h, x+w-20:x+w]
    ]
    edge_scores = []
    for region in corner_regions:
        if region.size == 0:
            continue
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(region_gray, cv2.CV_64F).var()
        edge_scores.append(laplacian)
    if not edge_scores:
        return 0
    avg_edge_sharpness = np.mean(edge_scores)
    return round(min(100, max(0, avg_edge_sharpness)), 2)

def analyze_surface(image, rect):
    x, y, w, h = rect
    if w == 0 or h == 0:
        return 0
    surface_region = image[y+20:y+h-20, x+20:x+w-20]
    if surface_region.size == 0:
        return 0
    gray = cv2.cvtColor(surface_region, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    overexposed_pixels = np.sum(gray > 240)
    glare_ratio = overexposed_pixels / gray.size
    glare_penalty = min(glare_ratio * 100, 40)
    base_score = min(100, lap_var)
    return round(max(0, base_score - glare_penalty), 2)

def generate_surface_heatmap(image, rect):
    x, y, w, h = rect
    if w == 0 or h == 0:
        return image
    surface_region = image[y+20:y+h-20, x+20:x+w-20]
    gray = cv2.cvtColor(surface_region, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_map = np.abs(laplacian)
    glare_mask = (gray > 240).astype(np.uint8) * 255
    combined = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    combined = cv2.addWeighted(combined, 0.7, glare_mask, 0.3, 0)
    heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
    heatmap_full = image.copy()
    heatmap_full[y+20:y+h-20, x+20:x+w-20] = cv2.addWeighted(surface_region, 0.6, heatmap, 0.4, 0)
    return heatmap_full

def create_grading_report(center, corner, surface, label, notes, original_img, heatmap_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Grading Pre-Check Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    if label:
        pdf.cell(200, 10, txt=f"Card: {label}", ln=True)
    if notes:
        pdf.multi_cell(0, 10, txt=f"Notes: {notes}")
    pdf.cell(200, 10, txt=f"Centering Score: {center}/100", ln=True)
    pdf.cell(200, 10, txt=f"Corner Sharpness Score: {corner}/100", ln=True)
    pdf.cell(200, 10, txt=f"Surface Score: {surface}/100", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_orig:
        Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)).save(tmp_orig.name)
        orig_path = tmp_orig.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_heat:
        Image.fromarray(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)).save(tmp_heat.name)
        heat_path = tmp_heat.name
    pdf.ln(10)
    pdf.cell(200, 10, txt="Original Card Image:", ln=True)
    pdf.image(orig_path, w=150)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Surface Heatmap Overlay:", ln=True)
    pdf.image(heat_path, w=150)
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

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
        st.subheader("📊 Scores")
        st.markdown(f"**Centering:** {center_score}/100")
        st.markdown(f"**Corners:** {corner_score}/100")
        st.markdown(f"**Surface:** {surface_score}/100")

        st.subheader("🎯 Instant Grade Probability")
        avg_score = (center_score + corner_score + surface_score) / 3
        if avg_score > 90:
            st.success("Most likely grade: PSA 10 (High confidence)")
        elif avg_score > 80:
            st.info("Most likely grade: PSA 9–10 (Medium confidence)")
        elif avg_score > 70:
            st.warning("Most likely grade: PSA 8–9 (Low confidence)")
        else:
            st.error("Most likely grade: Below PSA 8")

        st.subheader("📈 Grading ROI Estimate")
        expected_profit_9 = psa9_value - grading_cost
        expected_profit_10 = psa10_value - grading_cost
        st.write(f"**Profit if PSA 9:** ${expected_profit_9:.2f}")
        st.write(f"**Profit if PSA 10:** ${expected_profit_10:.2f}")
        if expected_profit_9 < 0 and expected_profit_10 < 10:
            st.warning("Grading may not be worth it based on ROI.")
        else:
            st.success("Could be worth grading depending on actual grade!")

        st.subheader("📤 Export Report")
        pdf_data = create_grading_report(center_score, corner_score, surface_score, card_title, card_notes, image_np, heatmap_img)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_data,
            file_name="grading_report.pdf",
            mime="application/pdf"
        )

    with col2:
        st.subheader("🖼️ Card Preview")
        st.image(image_np, caption="Uploaded Card", use_container_width=True)

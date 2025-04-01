# === Updated app.py with Phase 1 Features and Layout Styling ===
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

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

# ---- Upload & Card Details Side by Side ----
col_u1, col_u2 = st.columns([1, 1])
with col_u1:
    uploaded_file = st.file_uploader("Upload a card image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    

with col_u2:
    st.markdown("<div class='section-header'>📝 Card Details</div>", unsafe_allow_html=True)
    card_title = st.text_input("Card Title (Optional)", placeholder="e.g. 2023 Topps Chrome J-Rod Refractor")
        

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

    # Extract and analyze each corner region
    corner_regions = [
        image[y:y+25, x:x+25],                 # top-left
        image[y:y+25, x+w-25:x+w],             # top-right
        image[y+h-25:y+h, x:x+25],             # bottom-left
        image[y+h-25:y+h, x+w-25:x+w]          # bottom-right
    ]

    scores = []
    for region in corner_regions:
        if region.size == 0:
            continue
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Apply custom curve: ideal range is ~10–40
        if lap_var < 5:
            score = 20
        elif lap_var < 15:
            score = 40 + (lap_var - 5) * 4  # 40 to 80
        elif lap_var < 40:
            score = 80 + (lap_var - 15) * 0.8  # up to 100
        else:
            score = 95  # flatten out high Lap variance (often over-sharpened or glare)
        scores.append(score)

    return round(np.mean(scores), 2) if scores else 0

def analyze_surface(image, rect):
    x, y, w, h = rect
    if w == 0 or h == 0:
        return 0
    surface_region = image[y+20:y+h-20, x+20:x+w-20]
    if surface_region.size == 0:
        return 0
    gray = cv2.cvtColor(surface_region, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    glare = np.sum(gray > 235)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    scratches = np.sum((np.abs(sobel_x) > 50) | (np.abs(sobel_y) > 50))
    penalty = min((glare + scratches) / gray.size * 100, 60)
    base_score = np.clip((lap_var / 150) * 100, 0, 100)
    surface_score = max(0, base_score - penalty)
    return round(surface_score, 2)

def analyze_edges(image, rect):
    x, y, w, h = rect
    if w == 0 or h == 0:
        return 0
    edge_region = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(edge_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / edges.size
    score = 100 - min(edge_density * 300, 100)  # higher edge density = lower score (jagged/broken)
    return round(score, 2)

def generate_surface_heatmap(image, rect):



# ---- Main Flow ----
if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))
    center_score, card_rect = analyze_centering(image_np)
    corner_score = analyze_corners(image_np, card_rect)
    surface_score = analyze_surface(image_np, card_rect)
    edge_score = analyze_edges(image_np, card_rect)
    heatmap_img = generate_surface_heatmap(image_np, card_rect)

    col1, col2 = st.columns([1, 1])
    with col1:
        avg_score = (center_score + corner_score + surface_score + edge_score) / 4
        if avg_score > 90:
            grade_prediction = "⭐ Most likely grade: PSA 10 ⭐"
            st.markdown(f"<div style='background-color:#28a745;color:white;padding:10px;border-radius:5px;text-align:center;font-weight:bold'>{grade_prediction}</div>", unsafe_allow_html=True)
        elif avg_score > 80:
            grade_prediction = "Most likely grade: PSA 9"
            st.markdown(f"<div style='background-color:#4CAF50;color:white;padding:10px;border-radius:5px;text-align:center;font-weight:bold'>{grade_prediction}</div>", unsafe_allow_html=True)
        else:
            grade_prediction = "Most likely grade: PSA 8 or lower"

        st.markdown("<div class='section-header'>📊 Scores</div>", unsafe_allow_html=True)
        st.markdown(f"**Centering:** {center_score}/100")
        st.markdown(f"**Corners:** {corner_score}/100")
        st.markdown(f"**Surface:** {surface_score}/100")
        st.markdown(f"**Edges:** {edge_score}/100")

                    
        

        st.markdown("<div class='section-header'>📈 Grading ROI Estimator</div>", unsafe_allow_html=True)
        with st.expander("Estimate Grading ROI"):
            raw_value = st.number_input("Estimated Raw Card Value ($)", min_value=0.0, value=20.0)
            grading_cost = st.number_input("Grading Cost ($)", min_value=0.0, value=19.0)
            psa9_value = st.number_input("Estimated PSA 9 Value ($)", min_value=0.0, value=35.0)
            psa10_value = st.number_input("Estimated PSA 10 Value ($)", min_value=0.0, value=65.0)

            expected_profit_9 = psa9_value - grading_cost
            expected_profit_10 = psa10_value - grading_cost
            st.write(f"**Profit if PSA 9:** ${expected_profit_9:.2f}")
            st.write(f"**Profit if PSA 10:** ${expected_profit_10:.2f}")
            if expected_profit_9 < 0 and expected_profit_10 < 10:
                st.warning("Grading may not be worth it based on ROI.")
            else:
                st.success("Could be worth grading depending on actual grade!")

        

    with col2:
        st.markdown("<div class='section-header'>🖼️ Card Preview</div>", unsafe_allow_html=True)
        show_heatmap = st.checkbox("Show surface heatmap overlay", value=False)
        st.image(heatmap_img if show_heatmap else image_np, use_container_width=True)

        

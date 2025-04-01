import streamlit as st
import cv2
import numpy as np
from PIL import Image

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
st.markdown("Upload a sports card image and get centering, corner sharpness, edge cleanliness, and surface flaw scores — with visual overlays and a grading ROI estimator. Optimized for PSA-style decisions.")

# ---- Upload & Card Details ----
col_u1, col_u2 = st.columns([1, 1])
with col_u1:
    uploaded_file = st.file_uploader("Upload a card image (JPG/PNG)", type=["jpg", "jpeg", "png"])

with col_u2:
    card_title = st.text_input("Card Title (Optional)", placeholder="e.g. 2023 Topps Chrome J-Rod Refractor")

# ---- Analysis Functions ----
def analyze_centering(image, rect):
    x, y, w, h = rect
    if w == 0 or h == 0:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    card_contour = max(contours, key=cv2.contourArea)
    card_center_x = x + w / 2
    card_center_y = y + h / 2
    img_center_x, img_center_y = image.shape[1] / 2, image.shape[0] / 2
    off_center_x = abs(img_center_x - card_center_x) / img_center_x
    off_center_y = abs(img_center_y - card_center_y) / img_center_y
    center_score = max(0, 100 - (off_center_x + off_center_y) * 100)
    return round(center_score, 2), (x, y, w, h)

def analyze_corners(image, rect):
    x, y, w, h = rect
    if w == 0 or h == 0:
        return 0
    corner_regions = [
        image[y:y+25, x:x+25], image[y:y+25, x+w-25:x+w],
        image[y+h-25:y+h, x:x+25], image[y+h-25:y+h, x+w-25:x+w]
    ]
    scores = []
    for region in corner_regions:
        if region.size == 0:
            continue
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 5:
            score = 20
        elif lap_var < 15:
            score = 40 + (lap_var - 5) * 4
        elif lap_var < 40:
            score = 80 + (lap_var - 15) * 0.8
        else:
            score = 95
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
    return round(max(0, base_score - penalty), 2)

def analyze_edges(image, rect):
    x, y, w, h = rect
    if w == 0 or h == 0:
        return 0
    edge_region = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(edge_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = np.sum(edges > 0)
    edge_density = edge_pixels / (w * h)
    score = 100 - min(edge_density * 400, 100)
    return round(score, 2)

# ---- Main Logic ----
if uploaded_file:
    # Step 1: Load and prepare the image
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))

    # Step 2: Automatically detect the edges and bounding box
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        card_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(card_contour)
        detected_card_rect = (x, y, w, h)
    else:
        detected_card_rect = (0, 0, 0, 0)  # Default if no edges detected

    # Step 3: Show detected bounding box for the card
    image_with_box = image_np.copy()
    if detected_card_rect != (0, 0, 0, 0):
        cv2.rectangle(image_with_box, (detected_card_rect[0], detected_card_rect[1]), (detected_card_rect[0] + detected_card_rect[2], detected_card_rect[1] + detected_card_rect[3]), (0, 255, 0), 2)

    st.image(image_with_box, caption="Detected Card", use_container_width=True)

    # User manually adjusts the bounding box
    adjusted_rect = st.slider("Adjust card edges (x1, y1, width, height)", 0, 1000, (detected_card_rect[0], detected_card_rect[1], detected_card_rect[2], detected_card_rect[3]), step=1)
    adjusted_x, adjusted_y, adjusted_w, adjusted_h = adjusted_rect

    # Step 4: Recalculate centering score based on adjusted rectangle
    center_score, card_rect = analyze_centering(image_np, (adjusted_x, adjusted_y, adjusted_w, adjusted_h))

    # Display centering score
    st.write(f"Centering Score: {center_score}/100")

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Should I Grade This?", page_icon="📸")
st.title("📸 Should I Grade This?")
st.markdown("Upload a sports card image and get an automated centering and corner sharpness score. This demo uses basic image processing — no AI yet!")

uploaded_file = st.file_uploader("Upload a card image (JPG/PNG)", type=["jpg", "jpeg", "png"])

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
        image[y:y+20, x:x+20],               # Top-left
        image[y:y+20, x+w-20:x+w],           # Top-right
        image[y+h-20:y+h, x:x+20],           # Bottom-left
        image[y+h-20:y+h, x+w-20:x+w]        # Bottom-right
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
    sharpness_score = min(100, max(0, avg_edge_sharpness))  # crude normalization

    return round(sharpness_score, 2)

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))

    st.image(image_np, caption="Uploaded Card", use_column_width=True)

    center_score, card_rect = analyze_centering(image_np)
    corner_score = analyze_corners(image_np, card_rect)

    st.subheader("🔍 Analysis Results")
    st.write(f"**Centering Score:** {center_score}/100")
    st.write(f"**Corner Sharpness Score:** {corner_score}/100")

    if center_score > 85 and corner_score > 80:
        st.success("Looks like a good candidate for grading!")
    else:
        st.warning("Might not be worth grading — check centering and corners.")

    st.markdown("---")
    st.caption("This is an MVP demo using basic image processing with OpenCV. Future versions will include AI-based surface and grading predictions.")

import cv2
import numpy as np
import os
import av
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Absolute path resolution for Streamlit Cloud deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# ─────────────────────────────────────────────
# Filter Functions (Identical to local version)
# ─────────────────────────────────────────────
def anime_filter(frame):
    smooth = frame.copy()
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
    div = 32
    quantized = (smooth // div) * div + div // 2
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255).astype(np.uint8)
    vivid = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(vivid, edges_bgr)

def xray_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, None, iterations=1)
    result = cv2.addWeighted(enhanced, 0.8, edges, 0.5, 0)
    colored = np.zeros_like(frame)
    colored[:, :, 0] = np.clip(result * 1.0, 0, 255).astype(np.uint8)
    colored[:, :, 1] = np.clip(result * 0.8, 0, 255).astype(np.uint8)
    colored[:, :, 2] = np.clip(result * 0.3, 0, 255).astype(np.uint8)
    bright = cv2.GaussianBlur(result, (15, 15), 0)
    glow = np.zeros_like(frame)
    glow[:, :, 0] = np.clip(bright * 0.3, 0, 255).astype(np.uint8)
    glow[:, :, 1] = np.clip(bright * 0.6, 0, 255).astype(np.uint8)
    glow[:, :, 2] = np.clip(bright * 0.1, 0, 255).astype(np.uint8)
    return cv2.add(colored, glow)

def thermal_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def neon_edges_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 20, 80)
    edges1 = cv2.dilate(edges1, None, iterations=1)
    edges2 = cv2.dilate(edges2, None, iterations=1)
    result = np.zeros_like(frame)
    result[:, :, 0] = np.clip(edges1.astype(np.int16) * 1.0, 0, 255).astype(np.uint8)
    result[:, :, 1] = np.clip(edges1.astype(np.int16) * 0.9, 0, 255).astype(np.uint8)
    result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) + edges2.astype(np.int16) * 0.8, 0, 255).astype(np.uint8)
    result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) + edges2.astype(np.int16) * 0.3, 0, 255).astype(np.uint8)
    glow = cv2.GaussianBlur(result, (9, 9), 0)
    result = cv2.add(result, glow)
    dark_bg = cv2.convertScaleAbs(frame, alpha=0.1, beta=0)
    return cv2.add(result, dark_bg)

def pencil_sketch_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256)
    colored = np.zeros_like(frame)
    colored[:, :, 0] = np.clip(sketch * 0.85, 0, 255).astype(np.uint8)
    colored[:, :, 1] = np.clip(sketch * 0.9, 0, 255).astype(np.uint8)
    colored[:, :, 2] = np.clip(sketch * 0.95, 0, 255).astype(np.uint8)
    return colored

def oil_painting_filter(frame):
    result = frame.copy()
    for _ in range(4):
        result = cv2.bilateralFilter(result, 9, 75, 75)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(result, edges_3ch)

def pixel_art_filter(frame):
    pixel_size = 8
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    div = 48
    quantized = (small // div) * div + div // 2
    return cv2.resize(quantized, (w, h), interpolation=cv2.INTER_NEAREST)

def glitch_filter(frame):
    h, w = frame.shape[:2]
    result = frame.copy()
    shift_r = np.random.randint(-15, 15)
    shift_b = np.random.randint(-15, 15)
    result[:, :, 2] = np.roll(frame[:, :, 2], shift_r, axis=1)
    result[:, :, 0] = np.roll(frame[:, :, 0], shift_b, axis=1)
    for _ in range(5):
        y = np.random.randint(0, h)
        thickness = np.random.randint(1, 4)
        shift = np.random.randint(-30, 30)
        result[y:y+thickness] = np.roll(result[y:y+thickness], shift, axis=1)
    if np.random.random() < 0.3:
        bx = np.random.randint(0, max(1, w - 50))
        by = np.random.randint(0, max(1, h - 20))
        bw = np.random.randint(30, 100)
        bh = np.random.randint(5, 20)
        if bx+bw <= w and by+bh <= h:
            result[by:by+bh, bx:bx+bw] = result[by:by+bh, bx:bx+bw][:, :, ::-1]
    return result

FILTERS = {
    "ANIME / CARTOON": anime_filter,
    "X-RAY": xray_filter,
    "THERMAL VISION": thermal_filter,
    "NEON EDGES": neon_edges_filter,
    "PENCIL SKETCH": pencil_sketch_filter,
    "OIL PAINTING": oil_painting_filter,
    "PIXEL ART / 8-BIT": pixel_art_filter,
    "GLITCH": glitch_filter,
}

def order_quad(pts):
    pts = sorted(pts, key=lambda p: p[1])
    top_two = sorted(pts[:2], key=lambda p: p[0])
    bot_two = sorted(pts[2:], key=lambda p: p[0])
    return [top_two[0], top_two[1], bot_two[1], bot_two[0]]

# ─────────────────────────────────────────────
# Streamlit WebRTC Processor
# ─────────────────────────────────────────────
class LiveFilterProcessor(VideoProcessorBase):
    def __init__(self):
        self.filter_type = "ANIME / CARTOON"
        self.enable_region = False
        
        # Initialize Mediapipe via the classic 'solutions' API
        # This completely bypasses the .tasks C++ dynamic ABI errors in Streamlit cloud
        self.mp_hands = mp.solutions.hands
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror for webcam usability
        h, w, _ = img.shape

        # 1. Apply Full Filter
        try:
            filter_fn = FILTERS[self.filter_type]
            filtered = filter_fn(img)
        except Exception:
            filtered = img.copy()

        # 2. Region Magic (Render only inside hand if active)
        active_quad = None
        if self.enable_region and self.hand_detector:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hand_detector.process(img_rgb)
            
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
                finger_pts = []
                for hand_lm in results.multi_hand_landmarks[:2]:
                    finger_pts.append((int(hand_lm.landmark[4].x * w), int(hand_lm.landmark[4].y * h)))
                    finger_pts.append((int(hand_lm.landmark[8].x * w), int(hand_lm.landmark[8].y * h)))
                active_quad = order_quad(finger_pts)

        if active_quad is not None:
            quad_np = np.array(active_quad, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, quad_np, 255)
            mask_3ch = cv2.merge([mask, mask, mask])
            inv_mask_3ch = cv2.bitwise_not(mask_3ch)
            
            dark_bg = cv2.convertScaleAbs(img, alpha=0.35, beta=5)
            result = cv2.bitwise_and(filtered, mask_3ch) + cv2.bitwise_and(dark_bg, inv_mask_3ch)
            
            cv2.polylines(result, [quad_np], True, (0, 255, 255), 3, cv2.LINE_AA)
        else:
            result = filtered

        return av.VideoFrame.from_ndarray(result, format="bgr24")

# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="AR Live Filters", layout="centered")
st.title("🎨 AR Live Video Filters")
st.markdown("This is a live Python OpenCV script running directly in your browser via **Streamlit WebRTC**!")

col1, col2 = st.columns(2)
with col1:
    filter_choice = st.selectbox("Select Filter Effect", list(FILTERS.keys()))
with col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    region_toggle = st.checkbox("🔮 Magic Region Mode (Requires 2 Hands)")
    st.caption("Show both hands (thumb + index) to create a glowing magic window!")

ctx = webrtc_streamer(
    key="live-filters",
    video_processor_factory=LiveFilterProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

if ctx.video_processor:
    ctx.video_processor.filter_type = filter_choice
    ctx.video_processor.enable_region = region_toggle

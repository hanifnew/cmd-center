import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn import Sequential, Module, ModuleList, Conv2d, BatchNorm2d, SiLU, Upsample, MaxPool2d
from ultralytics.nn.modules import C2f, SPPF
try:
    from ultralytics.nn.modules.conv import Conv
except ImportError:
    from ultralytics.nn.modules import Conv

# Add safe globals for model loading
torch.serialization.add_safe_globals([
    DetectionModel,
    Sequential,
    Module,
    ModuleList,
    Conv2d,
    BatchNorm2d,
    SiLU,
    Upsample,
    MaxPool2d,
    C2f,
    SPPF,
    Conv
])

def draw_rounded_rectangle(img, pt1, pt2, color, thickness=2, r=10):
    # Draw a rounded rectangle by combining rectangles and ellipses
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness < 0:
        # Filled rounded rectangle
        overlay = img.copy()
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, -1)
        cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, -1)
        cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, -1)
        cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    else:
        # Outline rounded rectangle
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def draw_label(img, text, topleft, color, font_scale=0.6, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = topleft
    # Draw filled rounded rectangle for label background
    draw_rounded_rectangle(img, (x, y - th - 10), (x + tw + 10, y), color, thickness=-1, r=8)
    # Draw text
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)

# Set page config
st.set_page_config(
    page_title="MBG Command Center",
    page_icon="🎥",
    layout="wide"
)

# Title
st.title("MBG Command Center - Live Detection")

# Create a placeholder for the video feed
video_placeholder = st.empty()

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    try:
        model = YOLO('weights/weights.pt', task='detect')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

# Color map for classes
class_colors = {
    'apron': (0, 200, 0),          # Green
    'no_apron': (0, 0, 200),       # Red
    'gloves': (0, 200, 200),       # Cyan
    'no_gloves': (200, 0, 0),      # Blue
    'hairnet': (200, 200, 0),      # Yellow
    'no_hairnet': (200, 0, 200),   # Magenta
    'with_mask': (0, 255, 0),      # Bright Green
    'without_mask': (0, 0, 255),   # Bright Red
    'rat': (128, 0, 128),          # Purple
    'cockroach': (165, 42, 42),    # Brown
    'lizard': (0, 128, 128)        # Teal
}

# Initialize video capture with webcam
cap = cv2.VideoCapture(0)  # Use default webcam
if not cap.isOpened():
    st.error("Error: Could not open webcam")
    st.stop()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        # Video ended, reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # Resize frame for better performance
    frame = cv2.resize(frame, (1280, 720))  
    
    # Run detection
    results = model.predict(frame, conf=0.4)
    annotated_frame = frame.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls]
            # Skip mask_weared_incorrect class
            if class_name == 'mask_weared_incorrect':
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = class_colors.get(class_name, (200, 200, 200))
            draw_rounded_rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness=3, r=12)
            label = f'{class_name} {conf:.2f}'
            draw_label(annotated_frame, label, (x1, y1), color)
    
    # Display the video feed
    video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
    
    # Add a small delay to prevent overwhelming the system
    time.sleep(0.01)

# Clean up
cap.release() 
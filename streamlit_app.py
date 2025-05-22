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

# Initialize session state for counts
if 'apron_count' not in st.session_state:
    st.session_state.apron_count = 0
if 'cockroach_count' not in st.session_state:
    st.session_state.cockroach_count = 0
if 'gloves_count' not in st.session_state:
    st.session_state.gloves_count = 0
if 'hairnet_count' not in st.session_state:
    st.session_state.hairnet_count = 0
if 'lizard_count' not in st.session_state:
    st.session_state.lizard_count = 0
if 'mask_count' not in st.session_state:
    st.session_state.mask_count = 0
if 'mask_weared_incorrect_count' not in st.session_state:
    st.session_state.mask_weared_incorrect_count = 0
if 'no_apron_count' not in st.session_state:
    st.session_state.no_apron_count = 0
if 'no_gloves_count' not in st.session_state:
    st.session_state.no_gloves_count = 0
if 'no_hairnet_count' not in st.session_state:
    st.session_state.no_hairnet_count = 0
if 'no_mask_count' not in st.session_state:
    st.session_state.no_mask_count = 0
if 'rat_count' not in st.session_state:
    st.session_state.rat_count = 0
if 'with_mask_count' not in st.session_state:
    st.session_state.with_mask_count = 0
if 'without_mask_count' not in st.session_state:
    st.session_state.without_mask_count = 0

# Create columns for metrics
col1, col2, col3, col4 = st.columns(4)

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    try:
        # Patch torch.load to always use weights_only=False
        orig_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return orig_torch_load(*args, **kwargs)
        torch.load = patched_torch_load

        model = YOLO('weights/weights (1).pt', task='detect')

        torch.load = orig_torch_load  # Restore original
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

# Color map for classes
class_colors = {
    'apron': (0, 255, 0),  # Green
    'cockroach': (0, 0, 255),  # Red
    'gloves': (255, 0, 0),  # Blue
    'hairnet': (255, 255, 0),  # Cyan
    'lizard': (0, 255, 255),  # Yellow
    'mask': (128, 0, 128),  # Purple
    'mask_weared_incorrect': (255, 165, 0),  # Orange
    'no_apron': (255, 0, 255),  # Magenta
    'no_gloves': (165, 42, 42),  # Brown
    'no_hairnet': (0, 128, 128),  # Teal
    'no_mask': (128, 128, 0),  # Olive
    'rat': (128, 0, 0),  # Maroon
    'with_mask': (0, 128, 0),  # Dark Green
    'without_mask': (0, 0, 128)  # Dark Blue
}

# Create a placeholder for the video feed
video_placeholder = st.empty()

# Create a placeholder for the metrics
metrics_placeholder = st.empty()

# Initialize video capture with video file
cap = cv2.VideoCapture('videos/test.mp4')  # Make sure to put your video file in the videos directory
if not cap.isOpened():
    st.error("Error: Could not open video file")
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
    apron_count = 0
    cockroach_count = 0
    gloves_count = 0
    hairnet_count = 0
    lizard_count = 0
    mask_count = 0
    mask_weared_incorrect_count = 0
    no_apron_count = 0
    no_gloves_count = 0
    no_hairnet_count = 0
    no_mask_count = 0
    rat_count = 0
    with_mask_count = 0
    without_mask_count = 0
    annotated_frame = frame.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = class_colors.get(class_name, (200, 200, 200))
            draw_rounded_rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness=3, r=12)
            label = f'{class_name} {conf:.2f}'
            draw_label(annotated_frame, label, (x1, y1), color)
            
            # Update counts based on class
            if class_name == 'apron':
                apron_count += 1
            elif class_name == 'cockroach':
                cockroach_count += 1
            elif class_name == 'gloves':
                gloves_count += 1
            elif class_name == 'hairnet':
                hairnet_count += 1
            elif class_name == 'lizard':
                lizard_count += 1
            elif class_name == 'mask':
                mask_count += 1
            elif class_name == 'mask_weared_incorrect':
                mask_weared_incorrect_count += 1
            elif class_name == 'no_apron':
                no_apron_count += 1
            elif class_name == 'no_gloves':
                no_gloves_count += 1
            elif class_name == 'no_hairnet':
                no_hairnet_count += 1
            elif class_name == 'no_mask':
                no_mask_count += 1
            elif class_name == 'rat':
                rat_count += 1
            elif class_name == 'with_mask':
                with_mask_count += 1
            elif class_name == 'without_mask':
                without_mask_count += 1
    
    # Update session state
    st.session_state.apron_count = apron_count
    st.session_state.cockroach_count = cockroach_count
    st.session_state.gloves_count = gloves_count
    st.session_state.hairnet_count = hairnet_count
    st.session_state.lizard_count = lizard_count
    st.session_state.mask_count = mask_count
    st.session_state.mask_weared_incorrect_count = mask_weared_incorrect_count
    st.session_state.no_apron_count = no_apron_count
    st.session_state.no_gloves_count = no_gloves_count
    st.session_state.no_hairnet_count = no_hairnet_count
    st.session_state.no_mask_count = no_mask_count
    st.session_state.rat_count = rat_count
    st.session_state.with_mask_count = with_mask_count
    st.session_state.without_mask_count = without_mask_count
    
    # Display metrics
    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Apron", st.session_state.apron_count)
            st.metric("No Apron", st.session_state.no_apron_count)
            st.metric("Gloves", st.session_state.gloves_count)
            st.metric("No Gloves", st.session_state.no_gloves_count)
        with col2:
            st.metric("Hairnet", st.session_state.hairnet_count)
            st.metric("No Hairnet", st.session_state.no_hairnet_count)
            st.metric("Mask", st.session_state.mask_count)
            st.metric("No Mask", st.session_state.no_mask_count)
        with col3:
            st.metric("With Mask", st.session_state.with_mask_count)
            st.metric("Without Mask", st.session_state.without_mask_count)
            st.metric("Mask Worn Incorrect", st.session_state.mask_weared_incorrect_count)
        with col4:
            st.metric("Rats", st.session_state.rat_count)
            st.metric("Cockroaches", st.session_state.cockroach_count)
            st.metric("Lizards", st.session_state.lizard_count)
    
    # Display the video feed
    video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
    
    # Add a small delay to prevent overwhelming the system
    time.sleep(0.01)

# Clean up
cap.release() 
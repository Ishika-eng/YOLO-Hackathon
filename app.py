import streamlit as st
import time
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="DeepNav | Terrain AI", page_icon="📡", layout="wide")

# ==========================================
# PREMIUM SLEEK GRAY CSS INJECTION
# ==========================================
st.markdown("""
<style>
    /* Dark sleek background styling */
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
    }
    
    /* Header styling */
    h1 {
        font-weight: 800;
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
        letter-spacing: -1px;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-top: -15px;
        margin-bottom: 30px;
        font-weight: 300;
    }
    
    /* Upload Box */
    .css-1v0mbdj.etr89bj1 {
        border: 2px dashed #334155;
        border-radius: 12px;
        background-color: #1e293b;
    }
    
    /* Metric Boxes Customization */
    div[data-testid="stMetricValue"] {
        color: #e2e8f0;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricDelta"] svg {
        color: #10b981 !important;
    }
    
    /* Cards for Images */
    .image-container {
        background-color: #1e293b;
        padding: 10px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
        border: 1px solid #334155;
    }
    
    /* Custom divider */
    hr {
        border-top: 1px solid #334155;
    }
    
    /* Legend classes */
    .legend-box {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 4px;
        margin-right: 6px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
n_classes = 11

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# Exact HEX colors for the legend mapping (matches RGB array)
hex_colors = [
    "#000000", "#228B22", "#00FF00", "#D2B48C", "#8B5A2B", 
    "#808000", "#FF69B4", "#8B4513", "#808080", "#A0522D", "#87CEEB"
]

color_palette = np.array([
    [0,   0,   0  ],  [34,  139, 34 ],  [0,   255, 0  ],  
    [210, 180, 140],  [139, 90,  43 ],  [128, 128, 0  ],  
    [255, 105, 180],  [139, 69,  19 ],  [128, 128, 128],  
    [160, 82,  45 ],  [135, 206, 235]
], dtype=np.uint8)


# ==========================================
# MODEL INFERENCE CACHING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=n_classes)
    model.load_state_dict(torch.load('train_stats/best_unet_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# Load blindly in background
with st.spinner("Initializing DeepNav AI Engine..."):
    model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((384, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def predict_image(image):
    tensor_img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor_img)
        pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_mask

def mask_to_colored_image(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask

def overlay_mask_on_image(original_img, color_mask, alpha=0.6):
    resized_orig = original_img.resize((640, 384))
    orig_np = np.array(resized_orig).astype(np.float32)
    color_np = color_mask.astype(np.float32)
    is_background = (color_mask[:, :, 0] == 0) & (color_mask[:, :, 1] == 0) & (color_mask[:, :, 2] == 0)
    blended = cv2.addWeighted(orig_np, 1 - alpha, color_np, alpha, 0)
    blended[is_background] = orig_np[is_background]
    return Image.fromarray(blended.astype(np.uint8))


# ==========================================
# UI LAYOUT & DASHBOARD
# ==========================================
st.markdown("<h1>DeepNav <span style='color:#94a3b8'>| Semantic Autonomy</span></h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Offroad Hazard Segmentation using ResNet-34 U-Net</div>", unsafe_allow_html=True)

# Sidebar Mechanics
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg", width=40)
    st.markdown("### Model Diagnostics")
    st.markdown("---")
    st.metric("Neural Architecture", "U-Net")
    st.metric("Encoder Backbone", "ResNet-34")
    st.metric("Base Accuracy", "85.2%")
    st.metric("Mean Precision", "0.431 mAP")
    st.markdown("---")
    st.markdown("<small style='color: #64748b;'>Engineered for Live GPU Deployments</small>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Mount Camera Feed (Upload Image)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Top Metrics bar
    m1, m2, m3 = st.columns(3)
    
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    start_time = time.time()
    with st.spinner('Neural Network scanning geometry...'):
        pred_mask_2d = predict_image(raw_image)
    end_time = time.time()
    
    inference_time_ms = (end_time - start_time) * 1000
    
    color_mask_np = mask_to_colored_image(pred_mask_2d)
    blended_img = overlay_mask_on_image(raw_image, color_mask_np, alpha=0.6)
    
    # Extract found classes dynamically
    found_class_ids = np.unique(pred_mask_2d)
    found_classes = [cid for cid in found_class_ids if cid != 0]
    
    # Fill dynamic metrics
    m1.metric("Engine Render Speed", f"{inference_time_ms:.1f} ms")
    m2.metric("Tensor Output Shape", "[11, 384, 640]")
    m3.metric("Hazards Detected", f"{len(found_classes)} Entities")
    
    st.markdown("---")

    # The Legendary Big Visual Display
    st.markdown("### Live Reconnaissance Feed")
    
    # Premium visual split
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("<div style='color: #94a3b8; margin-bottom: 5px; font-size:14px;'>RAW OPTICAL SENSOR</div>", unsafe_allow_html=True)
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(raw_image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_right:
        st.markdown("<div style='color: #94a3b8; margin-bottom: 5px; font-size:14px;'>AI SEMANTIC OVERLAY</div>", unsafe_allow_html=True)
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(blended_img, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Dynamic Legend Builder
    st.markdown("### Detected Classification Legend")
    st.markdown("<div style='background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; display: flex; flex-wrap: wrap; gap: 15px;'>", unsafe_allow_html=True)
    
    if len(found_classes) == 0:
        st.markdown("<span style='color: #64748b;'>No hazardous terrain identified.</span>", unsafe_allow_html=True)
    else:
        for cid in found_classes:
            name = class_names[cid]
            color = hex_colors[cid]
            st.markdown(f"<div style='display: flex; align-items: center;'><div class='legend-box' style='background-color: {color}; box-shadow: 0 0 5px {color}88;'></div><span style='font-size: 14px; font-weight: 500;'>{name}</span></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

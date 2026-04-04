"""
Inference Script for New Images
Runs the trained U-Net model on a directory of raw images (no masks needed).
"""

import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ============================================================================
# Mask Conversion (Must match training exact values)
# ============================================================================

value_map = {
    0: 0,        # Background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    600: 6,      # Flowers
    700: 7,      # Logs
    800: 8,      # Rocks
    7100: 9,     # Landscape
    10000: 10    # Sky
}

n_classes = len(value_map)

# Color mapping for visual output
color_palette = np.array([
    [0,   0,   0  ],  # 0  Background    - black
    [34,  139, 34 ],  # 1  Trees         - forest green
    [0,   255, 0  ],  # 2  Lush Bushes   - lime
    [210, 180, 140],  # 3  Dry Grass     - tan
    [139, 90,  43 ],  # 4  Dry Bushes    - brown
    [128, 128, 0  ],  # 5  Ground Clutter- olive
    [255, 105, 180],  # 6  Flowers       - hot pink
    [139, 69,  19 ],  # 7  Logs          - saddle brown
    [128, 128, 128],  # 8  Rocks         - gray
    [160, 82,  45 ],  # 9  Landscape     - sienna
    [135, 206, 235],  # 10 Sky           - sky blue
], dtype=np.uint8)


def mask_to_color(mask):
    """Convert a class ID mask map to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Main Inference Logic
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run inference on completely new images')
    parser.add_argument('--model_path', type=str, default='train_stats/best_unet_model.pth',
                        help='Path to your trained model weights')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Folder containing your raw new images')
    parser.add_argument('--output_dir', type=str, default='inference_outputs',
                        help='Folder to save the colored mask predictions')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Needs to be exact same as training resolution
    w, h = 640, 384

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading Model from {args.model_path}...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,      # Not downloading weights since we load our own
        in_channels=3,
        classes=n_classes
    )
    
    # Load your physical weights file
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in '{args.input_dir}'.")

    with torch.no_grad():
        for filename in tqdm(image_files, desc="Predicting labels"):
            img_path = os.path.join(args.input_dir, filename)
            
            # Load and Preprocess
            orig_img = Image.open(img_path).convert("RGB")
            
            # Optionally keep original size for inverse resizing later? 
            # We'll just output the 640x384 prediction for now
            tensor_img = transform(orig_img).unsqueeze(0).to(device)

            # Predict
            outputs = model(tensor_img)
            pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Color Map and Save
            colored = mask_to_color(pred_mask)
            
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(args.output_dir, f"{base_name}_prediction.png")
            
            # OpenCV saves in BGR, so we convert RGB back to BGR
            cv2.imwrite(out_path, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

    print(f"\nDone! Colored predictions are saved in: {args.output_dir}")

if __name__ == "__main__":
    main()

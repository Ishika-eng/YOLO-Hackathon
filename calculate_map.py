"""
Calculate Mean Precision (mP) / Semantic mAP
For Semantic Segmentation, true bounding-box mAP doesn't exist, so this computes 
Pixel-wise Mean Average Precision (Mean Precision across all classes).
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

# ============================================================================
# Map configuration
# ============================================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10
}
class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]
n_classes = len(value_map)

# ============================================================================
# Metric Logic
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='train_stats/best_unet_model.pth')
    
    # You asked for the Testing Dataset, so here it is:
    parser.add_argument('--data_dir', type=str, default='Offroad_Segmentation_testImages')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms (must match training exactly)
    w, h = 640, 384
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    print("Loading Model...")
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=n_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    image_dir = os.path.join(args.data_dir, 'Color_Images')
    mask_dir = os.path.join(args.data_dir, 'Segmentation')
    image_files = os.listdir(image_dir)

    print(f"\nComputing Semantic AP (Precision) across {len(image_files)} test images...")

    # Tracking True Positives, False Positives for Semantic mAP
    total_tp = np.zeros(n_classes)
    total_fp = np.zeros(n_classes)

    with torch.no_grad():
        for filename in tqdm(image_files):
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            
            # Skip if mask doesn't exist
            if not os.path.exists(mask_path): continue

            # Preprocess Image
            orig_img = Image.open(img_path).convert("RGB")
            tensor_img = transform(orig_img).unsqueeze(0).to(device)

            # Preprocess Target Mask
            raw_mask = np.array(Image.open(mask_path))
            target_mask = np.zeros_like(raw_mask)
            for k, v in value_map.items():
                target_mask[raw_mask == k] = v
            target_mask = Image.fromarray(target_mask.astype(np.uint8))
            target_mask = np.array(mask_transform(target_mask))

            # Predict
            outputs = model(tensor_img)
            pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

            # Calculate Pixel-wise Precision per class
            for c in range(n_classes):
                pred_inds = (pred_mask == c)
                target_inds = (target_mask == c)
                
                total_tp[c] += np.sum(pred_inds & target_inds)
                total_fp[c] += np.sum(pred_inds & ~target_inds)

    print("\n" + "="*50)
    print("mAP (MEAN PRECISION) RESULTS ON TESTING DATASET")
    print("="*50)
    
    precisions = []
    for c in range(n_classes):
        if total_tp[c] + total_fp[c] == 0:
            precisions.append(float('nan'))
            print(f"  {class_names[c]:<20}: N/A (Never predicted)")
        else:
            p = total_tp[c] / (total_tp[c] + total_fp[c])
            precisions.append(p)
            print(f"  {class_names[c]:<20}: {p:.4f}")
            
    mAP = np.nanmean(precisions)
    print("="*50)
    print(f"Final mAP (Mean Precision): {mAP:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()

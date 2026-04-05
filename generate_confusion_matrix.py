import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
from tqdm import tqdm
import segmentation_models_pytorch as smp
import pandas as pd

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# ============================================================================
# Configuration & Metadata
# ============================================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(class_names)

# ============================================================================
# Utilities
# ============================================================================

def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr

# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Confusion Matrix for Segmentation')
    parser.add_argument('--model_path', type=str, default='train_stats/best_unet_model.pth')
    parser.add_argument('--data_dir', type=str, default='Offroad_Segmentation_Training_Dataset/val')
    parser.add_argument('--output_dir', type=str, default='train_stats/evaluation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model parameters
    w, h = 640, 384
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Model
    print(f"Loading Model from {args.model_path}...")
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=n_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Data Loader
    image_dir = os.path.join(args.data_dir, 'Color_Images')
    mask_dir = os.path.join(args.data_dir, 'Segmentation')
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    # Initialize Confusion Matrix (11x11)
    # confusion_matrix[true_class][pred_class]
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    with torch.no_grad():
        for filename in tqdm(image_files):
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            
            if not os.path.exists(mask_path): continue

            # Preprocess
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            raw_mask = Image.open(mask_path)
            # Use NEAREST to resize mask to match prediction resolution
            # In PIL, Image.NEAREST is 0. 
            raw_mask = raw_mask.resize((w, h), Image.NEAREST)
            target_mask = convert_mask(raw_mask)
            
            # Predict
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # Update Confusion Matrix
            # Fast pixel-wise update using bincount
            y_true = target_mask.flatten()
            y_pred = pred_mask.flatten()
            
            # Combine true and pred into a single index
            # This works like a 2D -> 1D mapping: index = true * n_classes + pred
            combined = y_true.astype(np.int64) * n_classes + y_pred
            
            # We must handle potential out-of-bounds if mask has unknown values
            # but convert_mask should handle it. Just to be safe, clip or filter.
            valid_mask = (y_true < n_classes) & (y_pred < n_classes)
            combined = combined[valid_mask]
            
            counts = np.bincount(combined, minlength=n_classes*n_classes)
            cm += counts.reshape((n_classes, n_classes))

    # Save Raw Confusion Matrix
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    csv_path = os.path.join(args.output_dir, 'confusion_matrix_raw.csv')
    cm_df.to_csv(csv_path)
    print(f"Saved raw confusion matrix to {csv_path}")

    # Plot Normalized Confusion Matrix (Recall per class)
    plt.figure(figsize=(14, 12))
    
    # Normalize by row (True Classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm.astype('float')), where=row_sums!=0)
    
    # Create heatmap
    ax = sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Normalized Confusion Matrix (Recall per Class)', fontsize=16)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to {plot_path}")

if __name__ == "__main__":
    main()

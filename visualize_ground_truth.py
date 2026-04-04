"""
Dataset Ground Truth Visualizer
Picks random images from the dataset and plots them side-by-side with
their ground truth segmentation masks for visual inspection.
"""

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.switch_backend('Agg')

# ============================================================================
# Configuration
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
IMAGES_DIR = os.path.join(DATA_DIR, 'Color_Images')
MASKS_DIR = os.path.join(DATA_DIR, 'Segmentation')
OUTPUT_DIR = os.path.join(script_dir, 'exploration_outputs')

os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_SAMPLES = 5  # Number of random samples to visualize

# ============================================================================
# Class Definitions
# ============================================================================

value_map = {
    0:     0,   # Background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    600:   6,   # Flowers
    700:   7,   # Logs
    800:   8,   # Rocks
    7100:  9,   # Landscape
    10000: 10,  # Sky
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# Color palette mapped 1:1 to class indices
color_palette = [
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
]

# Create normalized colors for matplotlib legend patches
normalized_colors = [(r/255., g/255., b/255.) for (r,g,b) in color_palette]

# ============================================================================
# Helper Functions
# ============================================================================

def convert_mask(mask_arr):
    """Convert raw mask values to class IDs."""
    new_arr = np.zeros_like(mask_arr, dtype=np.uint8)
    for raw_value, class_id in value_map.items():
        new_arr[mask_arr == raw_value] = class_id
    return new_arr

def mask_to_color(mask_arr):
    """Convert class IDs to an RGB image."""
    h, w = mask_arr.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(class_names)):
        color_img[mask_arr == class_id] = color_palette[class_id]
    return color_img

# ============================================================================
# Main Execution
# ============================================================================

def main():
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(MASKS_DIR):
        print(f"Error: Dataset directories not found in {DATA_DIR}")
        return

    # Get matching files
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) == 0:
        print("Error: No images found.")
        return

    # Select random files
    # Note: random.seed() is removed so it picks different images every time you run it
    sample_files = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))

    # Set up matplotlib figure
    fig, axes = plt.subplots(len(sample_files), 2, figsize=(14, 4 * len(sample_files)))
    if len(sample_files) == 1: axes = [axes] # Ensure iterable

    print("\nProcessing samples:")
    for i, file_name in enumerate(sample_files):
        print(f"  [{i+1}/{NUM_SAMPLES}] {file_name}")
        
        # Load Images
        img_path = os.path.join(IMAGES_DIR, file_name)
        mask_path = os.path.join(MASKS_DIR, file_name)
        
        orig_img = np.array(Image.open(img_path).convert("RGB"))
        raw_mask = np.array(Image.open(mask_path))
        
        # Process Mask
        class_mask = convert_mask(raw_mask)
        colored_mask = mask_to_color(class_mask)
        
        # Plot Original
        axes[i][0].imshow(orig_img)
        axes[i][0].set_title(f"Original Image\n({file_name})", fontsize=12)
        axes[i][0].axis('off')
        
        # Plot Mask
        axes[i][1].imshow(colored_mask)
        axes[i][1].set_title(f"Ground Truth Mask\n({file_name})", fontsize=12)
        axes[i][1].axis('off')

    # Shrink layout to make room for legend on the right
    plt.tight_layout(rect=[0, 0, 0.85, 1])  

    # Add floating legend
    patches = [
        mpatches.Patch(color=normalized_colors[i], label=f"{i}: {class_names[i]}")
        for i in range(len(class_names))
    ]
    
    # Place legend outside axes
    fig.legend(handles=patches, loc='center right', title="Classes", fontsize=11, title_fontsize=13)

    plt.suptitle("Dataset Visualization: Source vs Ground Truth", fontsize=18, fontweight='bold', y=1.02)
    
    # Save Plot
    out_path = os.path.join(OUTPUT_DIR, 'ground_truth_samples.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved visualization successfully to '{out_path}'!")

if __name__ == '__main__':
    main()

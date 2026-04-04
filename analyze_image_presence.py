"""
Image-Level Class Presence Analyzer
Counts how many *images* each class appears in, helping identify which classes
might be underrepresented at the image-level (even if they have many pixels).
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

# ============================================================================
# Configuration
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
# Note: Ensure this aligns with where your datasets are actually saved!
MASK_DIR = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train', 'Segmentation')
OUTPUT_DIR = os.path.join(script_dir, 'exploration_outputs')

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ============================================================================
# Core Logic
# ============================================================================

def get_present_classes(mask_path):
    """Load mask and return a list of unique class IDs present in it."""
    arr = np.array(Image.open(mask_path))
    unique_raw_vals = np.unique(arr)
    
    present_ids = []
    for raw_val in unique_raw_vals:
        if raw_val in value_map:
            present_ids.append(value_map[raw_val])
            
    return present_ids

def main():
    if not os.path.exists(MASK_DIR):
        print(f"Error: Mask directory '{MASK_DIR}' does not exist.")
        return

    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(mask_files)
    
    if total_images == 0:
        print(f"Error: No image files found in '{MASK_DIR}'.")
        return

    print(f"\nAnalyzing {total_images} images from {MASK_DIR}...\n")

    # Track how many images each class appears in
    class_presence_count = {class_id: 0 for class_id in range(len(class_names))}

    for filename in tqdm(mask_files, desc="Scanning masks"):
        img_path = os.path.join(MASK_DIR, filename)
        present_classes = get_present_classes(img_path)
        
        for class_id in present_classes:
            class_presence_count[class_id] += 1

    # ========================================================================
    # Prepare and Sort Data
    # ========================================================================
    
    # Create list of tuples: (class_id, class_name, count, percentage)
    results = []
    for class_id, count in class_presence_count.items():
        pct = (count / total_images) * 100
        results.append((class_id, class_names[class_id], count, pct))

    # Sort primarily by count (least frequent first)
    results.sort(key=lambda x: x[2])

    # ========================================================================
    # Print Table Report
    # ========================================================================
    
    print("\n")
    print("=" * 65)
    print(" CLASS PRESENCE PER IMAGE (Sorted by least frequent)")
    print("=" * 65)
    print(f"{'ID':<4} | {'Class Name':<16} | {'Images Present':<16} | {'Percentage':<10}")
    print("-" * 65)
    
    for class_id, name, count, pct in results:
        pct_str = f"{pct:.1f}%"
        # Highlight rare classes (<5%) with an asterisk
        if pct < 5.0:
            name_str = f"{name} (*)"
        else:
            name_str = name
            
        print(f"{class_id:<4} | {name_str:<16} | {count:<16} | {pct_str:<10}")
        
    print("-" * 65)
    print("(*) Denotes rare class present in <5% of images.")
    print("=" * 65)

    # Save textual output as well
    out_txt_path = os.path.join(OUTPUT_DIR, 'class_presence_report.txt')
    with open(out_txt_path, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write(" CLASS PRESENCE PER IMAGE (Sorted by least frequent)\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'ID':<4} | {'Class Name':<16} | {'Images Present':<16} | {'Percentage':<10}\n")
        f.write("-" * 65 + "\n")
        for class_id, name, count, pct in results:
            pct_str = f"{pct:.1f}%"
            name_str = f"{name} (*)" if pct < 5.0 else name
            f.write(f"{class_id:<4} | {name_str:<16} | {count:<16} | {pct_str:<10}\n")
        f.write("-" * 65 + "\n")
        f.write("(*) Denotes rare class present in <5% of images.\n")
        f.write("=" * 65 + "\n")

    # ========================================================================
    # Plot Bar Chart
    # ========================================================================
    
    # Extract data for plotting in sorted order
    sorted_names = [item[1] for item in results]
    sorted_counts = [item[2] for item in results]
    sorted_pcts = [item[3] for item in results]

    # Colors: red for rare (<5%), steelblue for normal
    colors = ['#e74c3c' if pct < 5.0 else '#3498db' for pct in sorted_pcts]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(sorted_names, sorted_counts, color=colors, edgecolor='black')

    # Add text labels on top of bars
    for bar, count, pct in zip(bars, sorted_counts, sorted_pcts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (total_images * 0.01),
                 f"{count}\n({pct:.1f}%)",
                 ha='center', va='bottom', fontsize=9)

    plt.axhline(y=total_images * 0.05, color='red', linestyle='--', alpha=0.5, label='5% Threshold')

    plt.title(f'Class Presence Frequency (Total Images: {total_images})', fontsize=14, fontweight='bold')
    plt.xlabel('Class (sorted least to most frequent)', fontsize=12)
    plt.ylabel('Number of Images containing Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, total_images * 1.15) # Add headroom for labels
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Normal (≥5%)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Rare (<5%)')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    out_png_path = os.path.join(OUTPUT_DIR, 'class_presence_chart.png')
    plt.savefig(out_png_path, dpi=150)
    plt.close()
    
    print(f"\nSaved textual report to '{out_txt_path}'")
    print(f"Saved bar chart to '{out_png_path}'")

if __name__ == '__main__':
    main()

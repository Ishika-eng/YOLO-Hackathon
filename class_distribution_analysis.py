"""
Class Distribution Analysis
Counts pixel frequency per class across the entire training dataset.
Outputs:
  - class_distribution.png  : bar chart of pixel % per class
  - class_distribution.txt  : detailed stats table
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.switch_backend('Agg')

# ============================================================================
# Configuration — adjust paths if needed
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
# Fixed paths: removed '..' because dataset resides in the same directory as script
TRAIN_MASK_DIR  = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train', 'Segmentation')
VAL_MASK_DIR    = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val',   'Segmentation')

OUTPUT_DIR      = os.path.join(script_dir, 'exploration_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Class definitions (must match value_map in train/test scripts)
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

n_classes = len(class_names)  # 11

color_palette = [
    '#000000',  # 0  Background     - black
    '#228B22',  # 1  Trees          - forest green
    '#00FF00',  # 2  Lush Bushes    - lime
    '#D2B48C',  # 3  Dry Grass      - tan
    '#8B5A2B',  # 4  Dry Bushes     - brown
    '#808000',  # 5  Ground Clutter - olive
    '#FF69B4',  # 6  Flowers        - hot pink
    '#8B4513',  # 7  Logs           - saddle brown
    '#808080',  # 8  Rocks          - gray
    '#A0522D',  # 9  Landscape      - sienna
    '#87CEEB',  # 10 Sky            - sky blue
]


# ============================================================================
# Core: convert raw mask to class IDs
# ============================================================================

def convert_mask(mask_path):
    """Load a raw mask and convert pixel values to class IDs (0-10)."""
    arr = np.array(Image.open(mask_path))
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw_val, class_id in value_map.items():
        out[arr == raw_val] = class_id
    return out


# ============================================================================
# Count pixels per class across a directory of masks
# ============================================================================

def count_pixels(mask_dir, split_name):
    """
    Iterate over all masks in mask_dir and accumulate per-class pixel counts.
    Returns:
        pixel_counts  : np.array of shape (n_classes,)
        image_counts  : np.array — number of images each class appears in
        total_pixels  : int
        n_images      : int
    """
    if not os.path.exists(mask_dir):
        print(f"Warning: Directory does not exist: {mask_dir}")
        return np.zeros(n_classes, dtype=np.int64), np.zeros(n_classes, dtype=np.int64), 0, 0
        
    mask_files = sorted(os.listdir(mask_dir))
    n_images   = len(mask_files)
    
    if n_images == 0:
        print(f"Warning: No images found in: {mask_dir}")
        return np.zeros(n_classes, dtype=np.int64), np.zeros(n_classes, dtype=np.int64), 0, 0

    pixel_counts = np.zeros(n_classes, dtype=np.int64)
    image_counts = np.zeros(n_classes, dtype=np.int64)

    print(f"\nCounting pixels in {split_name} split ({n_images} masks)...")
    for fname in tqdm(mask_files, unit='mask'):
        mask = convert_mask(os.path.join(mask_dir, fname))
        for class_id in range(n_classes):
            count = np.sum(mask == class_id)
            pixel_counts[class_id] += count
            if count > 0:
                image_counts[class_id] += 1

    total_pixels = pixel_counts.sum()
    return pixel_counts, image_counts, total_pixels, n_images


# ============================================================================
# Report
# ============================================================================

def print_and_save_report(pixel_counts, image_counts, total_pixels, n_images, split_name, out_path):
    """Print a formatted table and save it to a .txt file."""
    if total_pixels == 0:
        return np.zeros(n_classes)
        
    pct = pixel_counts / total_pixels * 100

    header = f"\n{'='*75}\n  CLASS DISTRIBUTION — {split_name.upper()} SPLIT\n{'='*75}"
    row_fmt = "{:<5} {:<20} {:>14} {:>10} {:>12} {:>10}"
    divider = '-' * 75

    lines = [
        header,
        row_fmt.format('ID', 'Class', 'Pixels', '% Total', 'Images', '% Images'),
        divider,
    ]
    for i in range(n_classes):
        lines.append(row_fmt.format(
            i,
            class_names[i],
            f'{pixel_counts[i]:,}',
            f'{pct[i]:.3f}%',
            f'{image_counts[i]:,}',
            f'{image_counts[i]/n_images*100:.1f}%'
        ))
    lines += [
        divider,
        f"  Total pixels : {total_pixels:,}",
        f"  Total images : {n_images:,}",
        '=' * 75,
    ]

    report = '\n'.join(lines)
    print(report)

    with open(out_path, 'w') as f:
        f.write(report + '\n')
    print(f"\nSaved report to '{out_path}'")

    return pct


# ============================================================================
# Plotting
# ============================================================================

def plot_distribution(train_pct, val_pct, out_path):
    """
    Side-by-side horizontal bar charts for train and val pixel distributions.
    Bars are colour-coded per class. Rare classes (<1%) are highlighted in red.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, pct, title in zip(axes, [train_pct, val_pct], ['Train Split', 'Val Split']):
        if np.sum(pct) == 0:
            continue
            
        y_pos  = np.arange(n_classes)
        bars   = ax.barh(y_pos, pct, color=color_palette, edgecolor='black', linewidth=0.6)

        # Highlight rare classes (<1%)
        for bar, p in zip(bars, pct):
            if p < 1.0:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)

        # Value labels on bars
        for bar, p in zip(bars, pct):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f'{p:.2f}%', va='center', ha='left', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_xlabel('% of Total Pixels', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlim(0, max(pct) * 1.18)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()   # class 0 at top

    # Legend for red-border = rare class
    rare_patch  = mpatches.Patch(facecolor='white', edgecolor='red',  linewidth=2, label='Rare class (<1% pixels)')
    norm_patch  = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=0.6, label='Normal class')
    fig.legend(handles=[norm_patch, rare_patch], loc='lower center', ncol=2, fontsize=10, framealpha=0.9)

    plt.suptitle('Pixel-Level Class Distribution', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution chart to '{out_path}'")


def plot_imbalance_ratio(train_pct, out_path):
    """
    Bar chart showing imbalance ratio relative to the most common class.
    Helps decide class weights for the loss function.
    """
    if np.sum(train_pct) == 0:
        return
        
    dominant = train_pct.max()
    ratios   = dominant / np.where(train_pct > 0, train_pct, np.inf)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(n_classes), ratios, color=color_palette, edgecolor='black', linewidth=0.6)

    # Label bars
    for bar, r in zip(bars, ratios):
        label = f'{r:.1f}x' if r < 1000 else 'N/A'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                label, ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Imbalance Ratio  (dominant / class)', fontsize=11)
    ax.set_title('Class Imbalance Ratio\n(higher = rarer class, needs higher loss weight)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved imbalance ratio chart to '{out_path}'")


def suggest_class_weights(train_pixel_counts, out_path):
    """
    Compute and print inverse-frequency class weights suitable for
    torch.nn.CrossEntropyLoss(weight=...).
    """
    if train_pixel_counts.sum() == 0:
        return
        
    freq  = train_pixel_counts / train_pixel_counts.sum()
    # Inverse frequency, normalised so median weight = 1
    inv   = 1.0 / np.where(freq > 0, freq, np.inf)
    inv   = np.where(np.isinf(inv), 0, inv)
    
    valid_inv = inv[inv > 0]
    if len(valid_inv) > 0:
        weights = inv / np.median(valid_inv)
    else:
        weights = inv

    lines = [
        '\n' + '=' * 55,
        '  SUGGESTED CrossEntropyLoss CLASS WEIGHTS',
        '=' * 55,
        '  (paste into train_segmentation.py)\n',
        '  class_weights = torch.tensor([',
    ]
    for i, (name, w) in enumerate(zip(class_names, weights)):
        comma = ',' if i < n_classes - 1 else ''
        lines.append(f'      {w:.4f}{comma}  # {i}: {name}')
    lines += [
        '  ], dtype=torch.float32).to(device)',
        '  loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)',
        '=' * 55,
    ]

    report = '\n'.join(lines)
    print(report)
    with open(out_path, 'a') as f:
        f.write('\n' + report + '\n')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # --- Train split ---
    train_pixel_counts, train_image_counts, train_total, train_n = count_pixels(
        TRAIN_MASK_DIR, 'train'
    )
    train_pct = print_and_save_report(
        train_pixel_counts, train_image_counts, train_total, train_n,
        split_name='Train',
        out_path=os.path.join(OUTPUT_DIR, 'class_distribution.txt')
    )

    # --- Val split ---
    val_pixel_counts, val_image_counts, val_total, val_n = count_pixels(
        VAL_MASK_DIR, 'val'
    )
    val_pct = print_and_save_report(
        val_pixel_counts, val_image_counts, val_total, val_n,
        split_name='Val',
        out_path=os.path.join(OUTPUT_DIR, 'class_distribution_val.txt')
    )

    # --- Plots ---
    plot_distribution(
        train_pct, val_pct,
        out_path=os.path.join(OUTPUT_DIR, 'class_distribution.png')
    )
    plot_imbalance_ratio(
        train_pct,
        out_path=os.path.join(OUTPUT_DIR, 'class_imbalance_ratio.png')
    )

    # --- Suggested loss weights ---
    suggest_class_weights(
        train_pixel_counts,
        out_path=os.path.join(OUTPUT_DIR, 'class_distribution.txt')
    )

    print(f"\nAll outputs saved to '{OUTPUT_DIR}/'") 

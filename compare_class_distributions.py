"""
Cross-Split Class Distribution Comparison
==========================================
Compares pixel-level class frequencies across train, val, and test splits
for the Offroad Segmentation dataset.

Outputs:
  - comparison_table.csv         : per-class % for all splits
  - split_comparison_grouped.png : grouped bar chart (train vs val vs test)
  - split_comparison_heatmap.png : heatmap of class % per split
  - imbalance_warnings.txt       : flagged classes with >5% drift between splits
"""

import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.switch_backend('Agg')

# ============================================================================
# Configuration
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

SPLIT_DIRS = {
    'Train': os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train', 'Segmentation'),
    'Val':   os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val',   'Segmentation'),
    'Test':  os.path.join(script_dir, 'Offroad_Segmentation_testImages',       'Segmentation'),
}

OUTPUT_DIR = os.path.join(script_dir, 'exploration_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Class definitions (consistent with train/test scripts)
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

n_classes = len(class_names)

color_palette = [
    '#2C2C2C',  # 0  Background     - dark gray
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

# Bar colors for each split (visually distinct)
SPLIT_COLORS = {
    'Train': '#3498db',   # blue
    'Val':   '#e67e22',   # orange
    'Test':  '#2ecc71',   # green
}

# ============================================================================
# Core: convert raw mask → class IDs
# ============================================================================

def convert_mask(mask_path):
    arr = np.array(Image.open(mask_path))
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw_val, class_id in value_map.items():
        out[arr == raw_val] = class_id
    return out


# ============================================================================
# Count pixels per class for a split
# ============================================================================

def count_pixels(mask_dir, split_name):
    if not os.path.exists(mask_dir):
        print(f"  [!] Directory not found: {mask_dir}")
        return None, None, 0, 0

    mask_files = sorted(os.listdir(mask_dir))
    n_images = len(mask_files)

    if n_images == 0:
        print(f"  [!] No masks in: {mask_dir}")
        return None, None, 0, 0

    pixel_counts = np.zeros(n_classes, dtype=np.int64)
    image_counts = np.zeros(n_classes, dtype=np.int64)

    print(f"\n  Scanning {split_name} ({n_images} masks)...")
    for fname in tqdm(mask_files, desc=f"  {split_name}", unit='mask'):
        mask = convert_mask(os.path.join(mask_dir, fname))
        for cid in range(n_classes):
            count = np.sum(mask == cid)
            pixel_counts[cid] += count
            if count > 0:
                image_counts[cid] += 1

    total_pixels = int(pixel_counts.sum())
    return pixel_counts, image_counts, total_pixels, n_images


# ============================================================================
# Compute percentages
# ============================================================================

def compute_percentages(pixel_counts, total_pixels):
    if total_pixels == 0:
        return np.zeros(n_classes)
    return pixel_counts / total_pixels * 100


# ============================================================================
# Print comparison table to console
# ============================================================================

def print_comparison_table(split_data):
    splits = list(split_data.keys())

    print("\n")
    print("=" * 90)
    print("  CROSS-SPLIT CLASS DISTRIBUTION COMPARISON (Pixel %)")
    print("=" * 90)

    # Header
    header = f"{'ID':<4} {'Class':<16}"
    for s in splits:
        header += f" {s+' %':>10}"
    if len(splits) >= 2:
        header += f" {'Delta':>10}"
    print(header)
    print("-" * 90)

    for cid in range(n_classes):
        row = f"{cid:<4} {class_names[cid]:<16}"
        pcts = []
        for s in splits:
            pct = split_data[s]['pct'][cid]
            pcts.append(pct)
            row += f" {pct:>9.3f}%"

        if len(pcts) >= 2:
            delta = max(pcts) - min(pcts)
            flag = " !!" if delta > 5.0 else "  *" if delta > 2.0 else ""
            row += f" {delta:>9.2f}%{flag}"

        print(row)

    print("-" * 90)

    # Summary row
    summary = f"{'':4} {'TOTAL IMAGES':<16}"
    for s in splits:
        summary += f" {split_data[s]['n_images']:>10}"
    print(summary)

    summary2 = f"{'':4} {'TOTAL PIXELS':<16}"
    for s in splits:
        summary2 += f" {split_data[s]['total_pixels']:>10,}"
    print(summary2)
    print("=" * 90)


# ============================================================================
# Save CSV
# ============================================================================

def save_csv(split_data, out_path):
    splits = list(split_data.keys())

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)

        header = ['Class_ID', 'Class_Name']
        for s in splits:
            header += [f'{s}_Pixels', f'{s}_Pct']
        header.append('Max_Delta_Pct')
        writer.writerow(header)

        for cid in range(n_classes):
            row = [cid, class_names[cid]]
            pcts = []
            for s in splits:
                px = int(split_data[s]['pixel_counts'][cid])
                pct = split_data[s]['pct'][cid]
                pcts.append(pct)
                row += [px, round(pct, 4)]
            row.append(round(max(pcts) - min(pcts), 4))
            writer.writerow(row)

        # Summary rows
        writer.writerow([])
        summary = ['', 'Total_Images']
        for s in splits:
            summary += [split_data[s]['n_images'], '']
        writer.writerow(summary)

        summary2 = ['', 'Total_Pixels']
        for s in splits:
            summary2 += [split_data[s]['total_pixels'], '']
        writer.writerow(summary2)

    print(f"\n  [OK] Saved CSV -> {out_path}")


# ============================================================================
# Grouped bar chart
# ============================================================================

def plot_grouped_bars(split_data, out_path):
    splits = list(split_data.keys())
    n_splits = len(splits)

    bar_width = 0.8 / n_splits
    x = np.arange(n_classes)

    fig, ax = plt.subplots(figsize=(16, 8))

    for i, s in enumerate(splits):
        offset = (i - n_splits / 2 + 0.5) * bar_width
        pct = split_data[s]['pct']
        bars = ax.bar(
            x + offset, pct,
            width=bar_width,
            label=f'{s} ({split_data[s]["n_images"]} imgs)',
            color=SPLIT_COLORS.get(s, f'C{i}'),
            edgecolor='black',
            linewidth=0.4,
            alpha=0.85,
        )

        # Value labels (only for non-tiny values to avoid clutter)
        for bar, p in zip(bars, pct):
            if p > 0.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f'{p:.1f}',
                    ha='center', va='bottom',
                    fontsize=7, fontweight='bold',
                )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=10)
    ax.set_ylabel('% of Total Pixels', fontsize=12)
    ax.set_title('Class Distribution: Train vs Val vs Test', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(split_data[s]['pct']) for s in splits) * 1.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved grouped bar chart -> {out_path}")


# ============================================================================
# Heatmap
# ============================================================================

def plot_heatmap(split_data, out_path):
    splits = list(split_data.keys())
    matrix = np.array([split_data[s]['pct'] for s in splits])

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels(splits, fontsize=11)

    # Annotate cells
    for i in range(len(splits)):
        for j in range(n_classes):
            val = matrix[i, j]
            text_color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)

    ax.set_title('Pixel Distribution Heatmap (% per split)', fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, label='% of Pixels', shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved heatmap -> {out_path}")


# ============================================================================
# Log-scale bar chart for rare classes
# ============================================================================

def plot_log_scale(split_data, out_path):
    splits = list(split_data.keys())
    n_splits = len(splits)
    bar_width = 0.8 / n_splits
    x = np.arange(n_classes)

    fig, ax = plt.subplots(figsize=(16, 7))

    for i, s in enumerate(splits):
        offset = (i - n_splits / 2 + 0.5) * bar_width
        pct = split_data[s]['pct']
        pct_safe = np.where(pct > 0, pct, 1e-4)
        ax.bar(
            x + offset, pct_safe,
            width=bar_width,
            label=f'{s}',
            color=SPLIT_COLORS.get(s, f'C{i}'),
            edgecolor='black',
            linewidth=0.4,
            alpha=0.85,
        )

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=10)
    ax.set_ylabel('% of Total Pixels (log scale)', fontsize=12)
    ax.set_title('Class Distribution (Log Scale) — Reveals Rare Classes', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved log-scale chart -> {out_path}")


# ============================================================================
# Imbalance analysis & warnings
# ============================================================================

def analyze_imbalance(split_data, out_path):
    splits = list(split_data.keys())
    warnings = []

    print("\n")
    print("=" * 70)
    print("  IMBALANCE & DRIFT ANALYSIS")
    print("=" * 70)

    for cid in range(n_classes):
        pcts = {s: split_data[s]['pct'][cid] for s in splits}
        max_pct = max(pcts.values())
        min_pct = min(pcts.values())
        delta = max_pct - min_pct

        # Flag 1: Large cross-split drift (>5%)
        if delta > 5.0:
            max_split = max(pcts, key=pcts.get)
            min_split = min(pcts, key=pcts.get)
            msg = (f"  [!] DRIFT  | Class {cid} ({class_names[cid]}): "
                   f"Delta = {delta:.2f}% -- "
                   f"{max_split} ({max_pct:.2f}%) vs {min_split} ({min_pct:.2f}%)")
            print(msg)
            warnings.append(msg)
        elif delta > 2.0:
            max_split = max(pcts, key=pcts.get)
            min_split = min(pcts, key=pcts.get)
            msg = (f"  [i] MINOR  | Class {cid} ({class_names[cid]}): "
                   f"Delta = {delta:.2f}% -- "
                   f"{max_split} ({max_pct:.2f}%) vs {min_split} ({min_pct:.2f}%)")
            print(msg)
            warnings.append(msg)

    # Flag 2: Extremely rare classes (<1% in *any* split)
    print()
    for cid in range(n_classes):
        for s in splits:
            pct = split_data[s]['pct'][cid]
            if 0 < pct < 1.0:
                msg = (f"  [!] RARE   | Class {cid} ({class_names[cid]}): "
                       f"only {pct:.3f}% in {s}")
                print(msg)
                warnings.append(msg)
            elif pct == 0:
                msg = (f"  [X] ABSENT | Class {cid} ({class_names[cid]}): "
                       f"0 pixels in {s}")
                print(msg)
                warnings.append(msg)

    # Flag 3: Classes absent from images
    print()
    for cid in range(n_classes):
        for s in splits:
            img_pct = (split_data[s]['image_counts'][cid] /
                       split_data[s]['n_images'] * 100
                       if split_data[s]['n_images'] > 0 else 0)
            if 0 < img_pct < 5.0:
                msg = (f"  [!] SPARSE | Class {cid} ({class_names[cid]}): "
                       f"appears in only {img_pct:.1f}% of {s} images "
                       f"({split_data[s]['image_counts'][cid]}/{split_data[s]['n_images']})")
                print(msg)
                warnings.append(msg)

    print("=" * 70)

    if not warnings:
        print("  [OK] No significant imbalance or drift detected.")
        warnings.append("No significant imbalance or drift detected.")

    with open(out_path, 'w') as f:
        f.write("IMBALANCE & DRIFT ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        for w in warnings:
            f.write(w.strip() + "\n")
    print(f"\n  [OK] Saved warnings -> {out_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  CROSS-SPLIT CLASS DISTRIBUTION COMPARISON")
    print("  Offroad Segmentation Dataset")
    print("=" * 70)

    split_data = {}

    for split_name, mask_dir in SPLIT_DIRS.items():
        pixel_counts, image_counts, total_pixels, n_images = count_pixels(mask_dir, split_name)

        if pixel_counts is None:
            print(f"\n  Skipping {split_name} (no data)")
            continue

        pct = compute_percentages(pixel_counts, total_pixels)

        split_data[split_name] = {
            'pixel_counts': pixel_counts,
            'image_counts': image_counts,
            'total_pixels': total_pixels,
            'n_images': n_images,
            'pct': pct,
        }

    if len(split_data) < 2:
        print("\n  Error: Need at least 2 splits to compare. Exiting.")
        exit(1)

    # --- Console table ---
    print_comparison_table(split_data)

    # --- CSV ---
    save_csv(split_data, os.path.join(OUTPUT_DIR, 'comparison_table.csv'))

    # --- Plots ---
    plot_grouped_bars(split_data, os.path.join(OUTPUT_DIR, 'split_comparison_grouped.png'))
    plot_heatmap(split_data, os.path.join(OUTPUT_DIR, 'split_comparison_heatmap.png'))
    plot_log_scale(split_data, os.path.join(OUTPUT_DIR, 'split_comparison_logscale.png'))

    # --- Imbalance analysis ---
    analyze_imbalance(split_data, os.path.join(OUTPUT_DIR, 'imbalance_warnings.txt'))

    print(f"\n  All outputs saved to '{OUTPUT_DIR}/'")
    print("  Done!")

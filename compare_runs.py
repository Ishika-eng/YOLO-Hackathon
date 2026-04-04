import matplotlib.pyplot as plt
import numpy as np

# ── Run 1 Data (from previous training session) ──────────────────────────
run1_val_iou = [0.3373, 0.3952, 0.4324, 0.4520, 0.4779, 0.4877, 0.4964, 0.5103,
                0.5103, 0.5103, 0.5103, 0.5103, 0.5103, 0.5103, 0.5103, 0.5103,
                0.5200, 0.5350, 0.5454, 0.5454]
run1_val_loss = [1.148, 1.011, 0.924, 0.911, 0.825, 0.803, 0.786, 0.764,
                 0.750, 0.740, 0.730, 0.720, 0.715, 0.710, 0.705, 0.700,
                 0.695, 0.692, 0.690, 0.688]
run1_val_acc = [0.794, 0.817, 0.815, 0.834, 0.832, 0.833, 0.839, 0.842,
                0.842, 0.845, 0.847, 0.848, 0.849, 0.850, 0.851, 0.852,
                0.852, 0.853, 0.853, 0.853]

# ── Run 2 Data (from evaluation_metrics.txt) ─────────────────────────────
run2_val_iou = [0.2967, 0.3689, 0.4382, 0.4448, 0.4667, 0.4655, 0.4777, 0.4990,
                0.5069, 0.5101, 0.5097, 0.5023, 0.5230, 0.5348, 0.5234, 0.5375,
                0.5115, 0.5399, 0.5433, 0.5387]
run2_val_loss = [1.6498, 1.3369, 1.2233, 1.1553, 1.1223, 1.0933, 1.0538, 1.0202,
                 1.0008, 0.9916, 0.9696, 0.9950, 0.9433, 0.9398, 0.9394, 0.9269,
                 1.1101, 0.9332, 0.9137, 0.9096]
run2_val_acc = [0.7943, 0.8290, 0.8369, 0.8405, 0.8447, 0.8467, 0.8563, 0.8537,
                0.8533, 0.8551, 0.8589, 0.8484, 0.8574, 0.8638, 0.8594, 0.8592,
                0.8565, 0.8590, 0.8620, 0.8617]

run2_train_loss = [2.8292, 1.7210, 1.4804, 1.3456, 1.2857, 1.2420, 1.2010, 1.1572,
                   1.1277, 1.1018, 1.0913, 1.0594, 1.0368, 1.0170, 1.0331, 0.9930,
                   0.9833, 0.9952, 0.9732, 0.9828]

epochs = list(range(1, 21))

# ── Test IoU Comparison Data ─────────────────────────────────────────────
run1_test_classes = {
    'Trees': 0.2962, 'Lush Bushes': 0.0014, 'Dry Grass': 0.3750,
    'Dry Bushes': 0.3259, 'Ground Clutter': 0.0, 'Flowers': 0.0,
    'Logs': 0.0, 'Rocks': 0.0553, 'Landscape': 0.6332, 'Sky': 0.9832
}
run2_test_classes = {
    'Trees': 0.4949, 'Lush Bushes': 0.0, 'Dry Grass': 0.4022,
    'Dry Bushes': 0.3682, 'Ground Clutter': 0.0, 'Flowers': 0.0,
    'Logs': 0.0, 'Rocks': 0.0686, 'Landscape': 0.5863, 'Sky': 0.9855
}

# ── Style Setup ──────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
colors_run1 = '#FF6B6B'
colors_run2 = '#4ECDC4'

# ════════════════════════════════════════════════════════════════════════
# FIGURE 1: Training Metrics Comparison (Val IoU, Val Loss, Val Accuracy)
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Run 1 vs Run 2: Training Metrics Comparison', fontsize=16, fontweight='bold', y=1.02)

# Val IoU
axes[0].plot(epochs, run1_val_iou, color=colors_run1, linewidth=2.5, marker='o', markersize=4, label='Run 1 (ResNet34, 480×256)')
axes[0].plot(epochs, run2_val_iou, color=colors_run2, linewidth=2.5, marker='s', markersize=4, label='Run 2 (ResNet34, 640×384)')
axes[0].set_title('Validation IoU', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('IoU')
axes[0].legend(fontsize=8)
axes[0].set_ylim(0.2, 0.6)

# Val Loss
axes[1].plot(epochs, run1_val_loss, color=colors_run1, linewidth=2.5, marker='o', markersize=4, label='Run 1')
axes[1].plot(epochs, run2_val_loss, color=colors_run2, linewidth=2.5, marker='s', markersize=4, label='Run 2')
axes[1].set_title('Validation Loss', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=8)

# Val Accuracy
axes[2].plot(epochs, run1_val_acc, color=colors_run1, linewidth=2.5, marker='o', markersize=4, label='Run 1')
axes[2].plot(epochs, run2_val_acc, color=colors_run2, linewidth=2.5, marker='s', markersize=4, label='Run 2')
axes[2].set_title('Validation Accuracy', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('train_stats/run1_vs_run2_training.png', dpi=150, bbox_inches='tight')
print("Saved: train_stats/run1_vs_run2_training.png")
plt.close()

# ════════════════════════════════════════════════════════════════════════
# FIGURE 2: Per-Class Test IoU Comparison (Grouped Bar Chart)
# ════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('Run 1 vs Run 2: Per-Class Test IoU Comparison', fontsize=16, fontweight='bold')

class_names = list(run1_test_classes.keys())
run1_vals = list(run1_test_classes.values())
run2_vals = list(run2_test_classes.values())

x = np.arange(len(class_names))
width = 0.35

bars1 = ax.bar(x - width/2, run1_vals, width, label='Run 1 (Mean IoU: 0.3011)', color=colors_run1, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, run2_vals, width, label='Run 2 (Mean IoU: 0.4349)', color=colors_run2, edgecolor='white', linewidth=0.5)

ax.set_ylabel('IoU Score', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.1)

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    if h > 0.01:
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.3f}', ha='center', va='bottom', fontsize=7, color=colors_run1)

for bar in bars2:
    h = bar.get_height()
    if h > 0.01:
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.3f}', ha='center', va='bottom', fontsize=7, color=colors_run2)

plt.tight_layout()
plt.savefig('train_stats/run1_vs_run2_test_iou.png', dpi=150, bbox_inches='tight')
print("Saved: train_stats/run1_vs_run2_test_iou.png")
plt.close()

# ════════════════════════════════════════════════════════════════════════
# FIGURE 3: Summary Dashboard
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Final Results Summary: Run 1 vs Run 2', fontsize=16, fontweight='bold', y=1.02)

# Validation metrics comparison
metrics = ['Val IoU', 'Val Dice', 'Val Accuracy']
run1_final = [0.5457, 0.7492, 0.8531]
run2_final = [0.5433, 0.7326, 0.8617]

x = np.arange(len(metrics))
bars1 = axes[0].bar(x - width/2, run1_final, width, label='Run 1', color=colors_run1, edgecolor='white')
bars2 = axes[0].bar(x + width/2, run2_final, width, label='Run 2', color=colors_run2, edgecolor='white')
axes[0].set_title('Validation Metrics (Best)', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics, fontsize=11)
axes[0].set_ylim(0, 1.0)
axes[0].legend()

for bar in bars1:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Test Mean IoU comparison
test_labels = ['Run 1\n(No Suppression)', 'Run 2\n(With Suppression)']
test_vals = [0.3011, 0.4349]
bar_colors = [colors_run1, colors_run2]
bars = axes[1].bar(test_labels, test_vals, color=bar_colors, width=0.5, edgecolor='white', linewidth=2)
axes[1].set_title('Test Mean IoU', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 0.6)

for bar, val in zip(bars, test_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2., val + 0.015, f'{val:.4f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add improvement arrow
axes[1].annotate('+44.4%', xy=(1, 0.43), fontsize=14, fontweight='bold', color='green',
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('train_stats/run1_vs_run2_summary.png', dpi=150, bbox_inches='tight')
print("Saved: train_stats/run1_vs_run2_summary.png")
plt.close()

print("\nAll comparison charts generated!")

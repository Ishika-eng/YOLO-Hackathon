"""
Segmentation Validation Script
Converted to use U-Net with segmentation_models_pytorch
Evaluates a trained segmentation model on validation data and saves predictions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm
import segmentation_models_pytorch as smp

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
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

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)  # = 11

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


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask, data_id


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=11, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=11, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Visualization Functions
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save a side-by-side comparison of input, ground truth, and prediction."""
    # Denormalize image
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # Convert masks to color
    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    """Save metrics summary to a text file and create bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    # Save text summary
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for i, (name, iou) in enumerate(zip(class_names, results['class_iou'])):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {iou_str}\n")

    print(f"\nSaved evaluation metrics to {filepath}")

    # Create bar chart for per-class IoU
    fig, ax = plt.subplots(figsize=(12, 6))

    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    ax.bar(range(n_classes), valid_iou,
           color=[color_palette[i] / 255 for i in range(n_classes)],
           edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', label='Mean IoU')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")


# ============================================================================
# Main Evaluation Function
# ============================================================================

def main():
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Segmentation prediction/inference script')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'train_stats', 'best_unet_model.pth'),
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(script_dir, 'Offroad_Segmentation_testImages'),
                        help='Path to test dataset directory')
    parser.add_argument('--output_dir', type=str,
                        default='./predictions',
                        help='Directory to save prediction visualizations')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of comparison visualizations to save')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Image dimensions (must perfectly match training dimensions)
    w = 640
    h = 384

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # FIXED: Use NEAREST interpolation for masks to preserve integer class IDs
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    valset = MaskDataset(data_dir=args.data_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(valset)} samples")

    # Load U-Net model
    print(f"Loading U-Net model from {args.model_path}...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Create subdirectories for outputs
    masks_dir = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # Run evaluation and save predictions for ALL images
    print(f"\nRunning evaluation on {len(valset)} images...")

    iou_scores = []
    all_class_iou = []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Processing", unit="batch")
        for batch_idx, (imgs, labels, data_ids) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass
            outputs = model(imgs)

            # ── Test-Time Class Suppression ──────────────────────────
            # Block classes absent/dead in test set:
            # 0: Background, 2: Lush Bushes, 5: Ground Clutter, 6: Flowers, 7: Logs
            outputs[:, [0, 2, 5, 6, 7], :, :] = -float('inf')

            labels_squeezed = labels.squeeze(dim=1).long()
            predicted_masks = torch.argmax(outputs, dim=1)

            # Calculate metrics
            iou, class_iou = compute_iou(outputs, labels_squeezed, num_classes=n_classes)

            iou_scores.append(iou)
            all_class_iou.append(class_iou)

            # Save predictions for every image
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                # Save raw prediction mask (class IDs 0-10)
                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
                pred_img = Image.fromarray(pred_mask)
                pred_img.save(os.path.join(masks_dir, f'{base_name}_pred.png'))

                # Save colored prediction mask (RGB visualization)
                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Save comparison visualization for first N samples
                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_squeezed[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f'sample_{sample_count}_comparison.png'),
                        data_id
                    )

                sample_count += 1

            pbar.set_postfix(iou=f"{iou:.3f}")

    # Aggregate results
    mean_iou = np.nanmean(iou_scores)
    avg_class_iou = np.nanmean(all_class_iou, axis=0)

    results = {
        'mean_iou': mean_iou,
        'class_iou': avg_class_iou
    }

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU: {mean_iou:.4f}")
    print("=" * 50)
    print("\nPer-Class IoU:")
    for name, iou in zip(class_names, avg_class_iou):
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A (class absent)"
        print(f"  {name:<20}: {iou_str}")

    # Save all results
    save_metrics_summary(results, args.output_dir)

    print(f"\nDone! Processed {len(valset)} images.")
    print(f"\nOutputs saved to '{args.output_dir}/'")
    print(f"  masks/           : Raw prediction masks (class IDs 0-10)")
    print(f"  masks_color/     : Colored prediction masks (RGB)")
    print(f"  comparisons/     : Side-by-side comparison images ({args.num_samples} samples)")
    print(f"  evaluation_metrics.txt")
    print(f"  per_class_metrics.png")


if __name__ == "__main__":
    main()
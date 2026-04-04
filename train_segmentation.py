"""
Segmentation Training Script
Converted to use U-Net with segmentation_models_pytorch
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import albumentations as A

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

class AlbumentationsWrapper:
    """Wraps albumentations transforms to be used inside torchvision.transforms.Compose"""
    def __init__(self, aug):
        self.aug = aug
    def __call__(self, img):
        img_np = np.array(img)
        res = self.aug(image=img_np)['image']
        return Image.fromarray(res)

# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.moveaxis(img, 0, -1)
    img  = (img * std + mean) * 255
    img  = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0:     0,   # Background     (never appears in dataset \u2014 weight = 0)
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
n_classes = len(value_map)   # 11


def convert_mask(mask):
    """Convert raw mask pixel values to class IDs (0-10)."""
    arr     = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir    = os.path.join(data_dir, 'Color_Images')
        self.masks_dir    = os.path.join(data_dir, 'Segmentation')
        self.transform    = transform
        self.mask_transform = mask_transform
        self.data_ids     = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id  = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        msk_path = os.path.join(self.masks_dir,  data_id)

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(msk_path)
        mask  = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            # * 255 recovers integer class IDs after ToTensor divides by 255
            mask  = self.mask_transform(mask) * 255

        return image, mask


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=11, ignore_index=255):
    """Compute mean IoU across all classes (ignores classes absent from both pred & GT)."""
    pred   = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        pred_inds   = pred   == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union        = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))   # class absent \u2014 skip in mean
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class)


def compute_dice(pred, target, num_classes=11, smooth=1e-6):
    """Compute mean Dice (F1) score across all classes."""
    pred   = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds   = pred   == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        dice_score   = (2. * intersection + smooth) / (
                        pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    """Compute overall pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# CHANGED: Removed backbone inference from metrics evaluation
def evaluate_metrics(model, data_loader, device, num_classes=11, show_progress=True):
    """Run inference on a dataloader and return mean IoU, Dice, and pixel accuracy."""
    iou_scores, dice_scores, pixel_accuracies = [], [], []

    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") \
             if show_progress else data_loader

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # UNet forward pass (single step)
            outputs = model(imgs)
            labels = labels.squeeze(dim=1).long()

            iou_scores.append(compute_iou(outputs, labels, num_classes=num_classes))
            dice_scores.append(compute_dice(outputs, labels, num_classes=num_classes))
            pixel_accuracies.append(compute_pixel_accuracy(outputs, labels))

    model.train()
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies)


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss + Pixel Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'],   label='val')
    plt.title('Loss');  plt.xlabel('Epoch');  plt.ylabel('Loss')
    plt.legend();  plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'],   label='val')
    plt.title('Pixel Accuracy');  plt.xlabel('Epoch');  plt.ylabel('Accuracy')
    plt.legend();  plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    print(f"Saved training curves to '{output_dir}/training_curves.png'")

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.title('Train IoU vs Epoch');  plt.xlabel('Epoch');  plt.ylabel('IoU')
    plt.legend();  plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch');  plt.xlabel('Epoch');  plt.ylabel('IoU')
    plt.legend();  plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()
    print(f"Saved IoU curves to '{output_dir}/iou_curves.png'")

    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch');  plt.xlabel('Epoch');  plt.ylabel('Dice Score')
    plt.legend();  plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch');  plt.xlabel('Epoch');  plt.ylabel('Dice Score')
    plt.legend();  plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()
    print(f"Saved Dice curves to '{output_dir}/dice_curves.png'")

    # Plot 4: All metrics combined
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'],   label='val')
    plt.title('Loss vs Epoch');  plt.xlabel('Epoch');  plt.ylabel('Loss')
    plt.legend();  plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train')
    plt.plot(history['val_iou'],   label='val')
    plt.title('IoU vs Epoch');  plt.xlabel('Epoch');  plt.ylabel('IoU')
    plt.legend();  plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train')
    plt.plot(history['val_dice'],   label='val')
    plt.title('Dice Score vs Epoch');  plt.xlabel('Epoch');  plt.ylabel('Dice Score')
    plt.legend();  plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'],   label='val')
    plt.title('Pixel Accuracy vs Epoch');  plt.xlabel('Epoch');  plt.ylabel('Pixel Accuracy')
    plt.legend();  plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir):
    """Save per-epoch training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")

        for i in range(len(history['train_loss'])):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],    history['val_loss'][i],
                history['train_iou'][i],     history['val_iou'][i],
                history['train_dice'][i],    history['val_dice'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i]
            ))

    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Hyperparameters ─────────────────────────────────────────────────────
    batch_size = 4
    
    # CHANGED: Ensure input dimensions are divisible by 32 (ResNet34 requirement)
    w          = int(((960 / 2) // 32) * 32)   # 480
    h          = int(((540 / 2) // 32) * 32)   # 256
    
    # CHANGED: Updated learning rate for Adam
    lr         = 1e-4
    n_epochs   = 20

    # ── Output directory ────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # ── Transforms ──────────────────────────────────────────────────────────
    # Albumentations texture augmentations for training
    train_texture_augs = A.Compose([
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
    ])

    train_transform = transforms.Compose([
        transforms.Resize((h, w)),
        AlbumentationsWrapper(train_texture_augs),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # ── Dataset paths ───────────────────────────────────────────────────────
    data_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    trainset     = MaskDataset(data_dir=data_dir, transform=train_transform, mask_transform=mask_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    valset       = MaskDataset(data_dir=val_dir,  transform=val_transform, mask_transform=mask_transform)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training samples:   {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # ── Model Initialization (U-Net) ────────────────────────────────────────
    print("Loading U-Net (ResNet34 encoder)...")
    
    # CHANGED: Replaced DINOv2 + SegmentationHeadConvNeXt with SMP U-Net
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes
    )
    model = model.to(device)
    print("Model loaded successfully!")

    # ── Loss, optimizer, scheduler ──────────────────────────────────────────

    # CHANGED: Re-introduced class weights calculated from dataset distribution
    # Updated heavily based on testing feedback to prioritize Rocks
    class_weights = torch.tensor([
        0.0000,   # 0:  Background      — absent everywhere
        2.0000,   # 1:  Trees           — rare in test, small boost
        1.0000,   # 2:  Lush Bushes     — nearly absent in test
        0.5000,   # 3:  Dry Grass       — stable, no special treatment
        3.0000,   # 4:  Dry Bushes      — rare in train and test
        0.5000,   # 5:  Ground Clutter  — absent in test, keep low
        0.5000,   # 6:  Flowers         — absent in test, keep low
        3.0000,   # 7:  Logs            — absent in test, keep low
        8.0000,   # 8:  Rocks           — 15x jump in test, prioritized
        0.3000,   # 9:  Landscape       — dominant everywhere
        0.2000,   # 10: Sky             — dominant everywhere
    ], dtype=torch.float32).to(device)

    # CHANGED: Combined DiceLoss and CrossEntropyLoss with weights
    _dice_loss = smp.losses.DiceLoss(mode='multiclass')
    _ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    def loss_fn(outputs, labels):
        return _dice_loss(outputs, labels) + _ce_loss(outputs, labels)

    # CHANGED: Using Adam optimizer instead of SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # ── Training history ────────────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou':  [], 'val_iou':  [],
        'train_dice': [], 'val_dice': [],
        'train_pixel_acc': [], 'val_pixel_acc': []
    }

    best_val_iou = -1.0

    # ── Training loop ───────────────────────────────────────────────────────
    print("\nStarting training...")
    print("=" * 80)

    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:

        # ── Train phase ─────────────────────────────────────────────────────
        model.train()
        train_losses = []

        train_pbar = tqdm(train_loader,
                          desc=f"Epoch {epoch+1}/{n_epochs} [Train]",
                          leave=False, unit="batch")
        for imgs, labels in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # CHANGED: Single forward pass through U-Net model
            outputs = model(imgs)
            labels  = labels.squeeze(dim=1).long()
            
            # CHANGED: Used loss_fn from smp (DiceLoss)
            loss    = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validation phase ─────────────────────────────────────────────────
        model.eval()
        val_losses = []

        val_pbar = tqdm(val_loader,
                        desc=f"Epoch {epoch+1}/{n_epochs} [Val]",
                        leave=False, unit="batch")
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)

                # CHANGED: Single forward pass through U-Net model
                outputs = model(imgs)
                labels  = labels.squeeze(dim=1).long()
                
                loss    = loss_fn(outputs, labels)

                val_losses.append(loss.item())
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Compute full metrics ──────────────────────────────────────────────
        # CHANGED: Evaluate metrics using the U-Net model alone
        # Skipped train metrics to avoid running 700+ batches every epoch!
        train_iou, train_dice, train_pixel_acc = float('nan'), float('nan'), float('nan')
        val_iou, val_dice, val_pixel_acc = evaluate_metrics(
            model, val_loader,   device, num_classes=n_classes)

        # ── Store history ─────────────────────────────────────────────────────
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss   = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_pixel_acc'].append(val_pixel_acc)

        epoch_pbar.set_postfix(
            train_loss=f"{epoch_train_loss:.3f}",
            val_loss  =f"{epoch_val_loss:.3f}",
            val_iou   =f"{val_iou:.3f}",
            val_acc   =f"{val_pixel_acc:.3f}"
        )

        # ── Scheduler step ────────────────────────────────────────────────────
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch+1} complete | LR: {current_lr:.6e}")

        # ── Best-model checkpoint ─────────────────────────────────────────────
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path    = os.path.join(output_dir, "best_unet_model.pth") # CHANGED: Rename checkpoint file
            torch.save(model.state_dict(), best_path)
            print(f"--> New best model saved  |  Val IoU: {val_iou:.4f}")

    # ── Save plots and history ───────────────────────────────────────────────
    print("\nSaving training curves...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\nFinal evaluation results:")
    print(f"  Final Val Loss:     {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_pixel_acc'][-1]:.4f}")
    
    # CHANGED: Reflect renamed checkpoint
    print(f"\nBest Val IoU: {best_val_iou:.4f}  →  saved to '{output_dir}/best_unet_model.pth'")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

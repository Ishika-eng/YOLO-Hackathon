# Offroad Segmentation Model (U-Net)

## Summary
This repository contains a semantic segmentation model designed for offroad environments. The model leverages a U-Net architecture with a ResNet34 encoder to classify terrain types and potential obstacles (e.g., Rocks, Trees, Landscape, Sky, etc.). Model training incorporates weighted loss functions and specific data augmentations to address class imbalances and domain shifts.

---

## 1. Environment & Dependency Requirements

Ensure you are using Python 3.8+ and it is highly recommended to use a virtual environment (e.g., `.venv` or conda). 

Install the required packages:

```bash
pip install torch torchvision opencv-python numpy matplotlib Pillow tqdm segmentation-models-pytorch albumentations
```

**Note:** If using a GPU (highly recommended for training), ensure you have installed the correct version of PyTorch with CUDA support matching your system.

---

## 2. Step-by-Step Instructions

### **Training the Model**
To start training the model from scratch using the default configuration:

```bash
python train_segmentation.py
```

* **What it does:** Trains the U-Net model using the data in `Offroad_Segmentation_Training_Dataset` (train and val splits). 
* **Outputs:** As training progresses, the script calculates Loss, mean Intersection over Union (mIoU), Dice Score, and Pixel Accuracy. The best model checkpoint (based on validation IoU) is saved continuously.

### **Evaluating Validation/Test Results**
To compute official performance metrics comprehensively on the validation set:

```bash
python test_segmentation.py --model_path train_stats/best_unet_model.pth --data_dir Offroad_Segmentation_Training_Dataset/val --output_dir ./predictions
```

### **Running Inference on New Images**
To run the model on a folder containing entirely new images (without ground-truth masks):

```bash
python inference.py --input_dir <PATH_TO_YOUR_IMAGES> --output_dir ./inference_outputs
```
*Replace `<PATH_TO_YOUR_IMAGES>` with a folder containing `.png` or `.jpg` images you want to segment.*

---

## 3. Reproducing Final Results

To completely reproduce our best model's results:
1. Ensure the dataset folder `Offroad_Segmentation_Training_Dataset` is correctly located at the project root, containing `train` and `val` directories with nested `Color_Images` and `Segmentation` sub-folders.
2. Execute `python train_segmentation.py`. The built-in pipeline handles class-weighted loss calculations heavily prioritizing rare/challenging classes (e.g., Rocks) using Adam Optimizer with a cosine annealing learning rate scheduler over 20 epochs.
3. Once training completes (checkpoint automatically saved as `train_stats/best_unet_model.pth`), execute `python test_segmentation.py`. This reads your test configuration and calculates final IoU scores.

---

## 4. Expected Outputs & Interpretation

### **A. Training Stage (`train_stats/`)**
* `best_unet_model.pth`: The highest-performing model weights recorded.
* `evaluation_metrics.txt`: A epoch-by-epoch text summary of all metrics.
* `*_curves.png`: Various line graphs showing Training vs. Validation Loss, Accuracy, Dice Score, and mIoU. Used to verify the model didn't overfit.

### **B. Evaluation / Inference Stage (`predictions/` or `inference_outputs/`)**
* `masks/`: Raw generated masks containing pixel ID values corresponding to classes (`0-10`). Necessary for automated metric scripts.
* `masks_color/`: Generated RGB visualisations with standardized colors mappings:
  * **Landscape**: Sienna/Brown
  * **Rocks**: Gray 
  * **Sky**: Sky Blue
  * **Trees**: Forest Green
  * **Lush Bushes**: Lime
* `comparisons/`: (Test-only) Side-by-side composite images comparing `Input Image` | `Ground Truth` | `Prediction` for rapid visual failure analysis.
* `evaluation_metrics.txt`: Gives you the per-class IoU summary, helping you diagnose the hardest classes to predict reliably.

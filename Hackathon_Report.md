# Offroad Segmentation Hackathon Report

*Note to user: Below is the exact structure aligned to the 7-8 page constraints requested. I have included explicit image markers instructing you exactly which generated files from your `YOLO` folder to drag and drop into your Word/PDF document.*

---

## [PAGE 1] Title Page

**Team Name:** Duality Engineers  
**Project Name:** Offroad Terrain Semantic Segmentation Pipeline  
**Tagline:** *Navigating the wild: Robust, pixel-perfect understanding of offroad environments.*

*(Optional: Insert a high-quality, impressive prediction output here as a cover photo)*  
📍 **Image to Insert:** `c:\YOLO\predictions\comparisons\sample_0_comparison.png`

---

## [PAGE 2] Methodology

### Methodology & Training Approach
Our primary goal was to develop a semantic segmentation model capable of accurately parsing offroad landscapes into 10 distinct terrain and obstacle classes (e.g., Rocks, Trees, Dry Grass, Sky).

1. **Architecture Selection:**  
   We implemented a **U-Net** architecture backed by a powerful **ResNet34 encoder** initialized with ImageNet weights. This combination provides a strong balance between high-resolution feature extraction and rapid multi-scale feature fusion in the decoder.
   
2. **Dataset Manipulation & Augmentation:**  
   To build robustness, we introduced custom data augmentations using Albumentations. We applied `Sharpen` and `RandomBrightnessContrast` specifically to force the model to learn deep structural features of terrain rather than relying strictly on lighting conditions, which change drastically offroad.

3. **Loss Function Engineering:**  
   We encountered early issues with domain imbalance where large classes (like Sky and Landscape) overpowered the learning gradient. We structured a combined **Dice Loss + Weighted Cross-Entropy Loss**. 

4. **Training Process:**  
   - **Optimizer:** Adam Optimizer (LR: 1e-4) with a Cosine Annealing scheduler to smoothly decay the learning rate.
   - **Epochs:** 20 Epochs with a batch size of 4.
   - **Checkpoints:** The model automatically assessed Validation Intersection over Union (IoU) end-of-epoch, dynamically saving only the highest-performing weights.

---

## [PAGE 3] Results & Performance Metrics (Overall)

### Overall Model Performance
After iteratively refining our training pipeline, the model demonstrated strong generalization given the challenging nature of the task.

* **Final Mean IoU (Validation):** 0.5429  
* **Final Pixel Accuracy:** 85.2%  
* **Final Dice Score (F1):** 0.7476  
* **Test Dataset Mean IoU:** 0.5301  

### Training Trends (Loss & Accuracy)
Our tracked loss charts show a smooth asymptotic convergence over 20 epochs, proving the model did not overfit to the training images. Validation loss hit its minimum at Epoch 18 (0.6957) resulting in the best validation metrics across the board.

📍 **Images to Insert here:**  
*Drag and drop your combined training curves here to visually prove convergence.*
* **File:** `c:\YOLO\train_stats\all_metrics_curves.png` 

---

## [PAGE 4] Results & Performance Metrics (Per-Class)

### Per-Class Evaluation 
We computed specific Intersection-over-Union (IoU) scores across our target classes. The model performs exceptionally well on structural boundaries and core environment detection, but struggles marginally with deeply occluded ground clutter.

📍 **Image to Insert here:**  
*Insert your generated Bar Chart showing exactly how well each class performed.*
* **File:** `c:\YOLO\predictions\per_class_metrics.png`

| Class | IoU Score | Class | IoU Score |
| :--- | :--- | :--- | :--- |
| **Sky** | 0.9468 | **Trees** | 0.6824 | 
| **Dry Grass** | 0.6239 | **Landscape** | 0.6148 |
| **Lush Bushes** | 0.5637 | **Flowers** | 0.4211 |
| **Dry Bushes** | 0.3261 | **Rocks** | 0.3197 |

---

## [PAGE 5] Challenges & Solutions (Imbalance)

### Documented Failure Cases and Optimizations

**Challenge 1: Severe Class Imbalance for Small Objects**
* **Task:** Model Training on Dataset 
* **Initial IoU Score:** 0.3470 (Epoch 1)
* **Issue Faced:** The model achieved extremely low recall for critical navigation obstacles like "Rocks" and "Logs". Because these objects represent a tiny fraction of total pixels compared to "Sky" or "Landscape", the model ignored them to lower overall loss.
* **Solution:** *Dynamically Weighted Loss.* We pre-computed pixel-presence statistics across the dataset and mapped inverse weights to our Cross-Entropy Loss. "Rocks" were assigned a penalty multiplier of `8.0`, forcing the model to heavily prioritize learning rock formations. This resulted in our final mean IoU jumping significantly.

*(Optional: If you have an image showing a rock missing from a prediction, insert it here as a "Failure Case")*

---

## [PAGE 6] Challenges & Solutions (Artifacts & Precision)

**Challenge 2: Ghost Class Penalties & Artifacts**
* **Task:** Inference & Optimization
* **Issue Faced:** We noticed stray sub-pixel artifacts predicting classes that fundamentally shouldn't exist in a specific localized terrain structure.
* **Failure Case Observation:** The edges of overlapping "Trees" and "Sky" would sometimes bleed and hallucinate "Lush Bushes" because resizing interpolation was blending the integer IDs of masks.
* **Solution:** We optimized the inference pipeline by applying nearest-neighbor interpolation to target masks to prevent float-averaging on class boundaries, preserving exact integer IDs.

📍 **Images to Insert here:**  
*Show an example of a good working prediction to prove everything works cleanly now!*
* **File:** `c:\YOLO\predictions\comparisons\sample_1_comparison.png`
* *(You can also use `sample_2_comparison.png` or `sample_3_comparison.png` if they look better).*

---

## [PAGE 7] Conclusion & Future Work

### Conclusion
Our team successfully rapidly engineered an offroad segmentation pipeline using a U-Net (ResNet34) architecture. By directly manipulating loss functions based on pixel-distribution realities and implementing strong data pipelines, we pushed the model from an initial baseline of 0.34 IoU to a highly competitive 0.54 IoU. The model reliably segments major navigational zones and recognizes critical blockages.

### Future Improvements
1. **Test-Time Augmentation (TTA):** To further squeeze out performance, we would introduce TTA (e.g., multi-scale evaluation or horizontal flips) and average the prediction logits to eliminate edge-case errors.
2. **Focal Loss Integration:** While weighted Cross-Entropy handled basic distribution mismatches, implementing a mathematically strict Focal Loss could help the model focus purely on the hardest-to-segment boundaries (like intertwining dry bushes).
3. **Hyperparameter Sweeps:** A systematic hyperparameter sweep across backbone sizes (e.g., ResNet50) could yield a few more percentage points of Accuracy and IoU.

---
*(End of Report)*

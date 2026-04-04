import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    images_dir = os.path.join(train_dir, 'Color_Images')
    masks_dir = os.path.join(train_dir, 'Segmentation')
    
    output_path = os.path.join(script_dir, 'ENV_SETUP', 'rock_weights.pt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Pre-scanning dataset for Rock class distribution...")
    
    data_ids = os.listdir(images_dir)
    data_ids.sort() # Guarantee identical ordering for train_segmentation.py
    
    weights = []
    rock_count = 0
    
    for filename in tqdm(data_ids, desc="Scanning masks"):
        # We need to map class mapping to raw value.
        # In value_map, Rocks maps 800 -> 8
        # Since PIL loads raw mask, we look for 800.
        mask_path = os.path.join(masks_dir, filename)
        mask = np.array(Image.open(mask_path))
        
        # Check if rock pixel (800) exists
        if np.any(mask == 800):
            weights.append(5.0)  # 5x oversample weight for rocky scenes
            rock_count += 1
        else:
            weights.append(1.0)
            
    print(f"\nScan complete! Found rocks in {rock_count} out of {len(data_ids)} images.")
    print("Saving sampler weights matrix...")
    
    # Save a dictionary mapping exact filename to its weight
    weight_dict = {filename: weight for filename, weight in zip(data_ids, weights)}
    torch.save(weight_dict, output_path)
    print(f"Weights saved to {output_path}")

if __name__ == "__main__":
    main()

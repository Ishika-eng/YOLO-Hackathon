import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_dataset_stats(image_dir, output_txt="dataset_stats.txt", output_png="pixel_distribution.png"):
    """
    Computes the channel-wise mean and standard deviation of images in a dataset.
    Processes images iteratively to avoid excessive memory usage.
    """
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        return

    # Look for common image extensions
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ]

    if not image_files:
        print(f"Error: No images found in '{image_dir}'.")
        return

    print(f"Found {len(image_files)} images. Computing statistics...")

    # Variables for accumulating statistics (using float64 to avoid precision loss/overflow)
    pixel_count = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)

    # Variables for histograms
    bins = 256
    hist_r = np.zeros(bins, dtype=np.uint64)
    hist_g = np.zeros(bins, dtype=np.uint64)
    hist_b = np.zeros(bins, dtype=np.uint64)
    bin_edges = np.linspace(0, 1, bins + 1)

    for idx, filename in enumerate(image_files):
        img_path = os.path.join(image_dir, filename)
        
        try:
            with Image.open(img_path) as img:
                # Force convert to RGB to handle grayscale or RGBA images consistently
                img = img.convert('RGB')
                
                # Convert to numpy array and normalize to [0, 1]
                img_np = np.array(img, dtype=np.float32) / 255.0
                
                # Reshape image array to (N_pixels, 3)
                pixels = img_np.reshape(-1, 3)
                num_pixels = pixels.shape[0]
                
                # Accumulate sums and square sums for mean and std dev calculation
                pixel_count += num_pixels
                channel_sum += np.sum(pixels, axis=0)
                channel_sq_sum += np.sum(pixels ** 2, axis=0)
                
                # Update histograms
                h_r, _ = np.histogram(pixels[:, 0], bins=bin_edges)
                h_g, _ = np.histogram(pixels[:, 1], bins=bin_edges)
                h_b, _ = np.histogram(pixels[:, 2], bins=bin_edges)
                
                hist_r += h_r.astype(np.uint64)
                hist_g += h_g.astype(np.uint64)
                hist_b += h_b.astype(np.uint64)
                
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

        # Progress reporting
        if (idx + 1) % 100 == 0 or (idx + 1) == len(image_files):
            print(f"Processed {idx + 1}/{len(image_files)} images...", end='\r')
    
    print("\nProcessing complete.")

    if pixel_count == 0:
        print("No valid pixels processed.")
        return

    # Calculate Mean and Std
    # E[X] = sum(X) / N
    # Std[X] = sqrt( E[X^2] - (E[X])^2 )
    mean = channel_sum / pixel_count
    std = np.sqrt((channel_sq_sum / pixel_count) - (mean ** 2))

    # ImageNet statistics reference
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    # Calculate absolute differences
    diff_mean = np.abs(mean - imagenet_mean)
    diff_std = np.abs(std - imagenet_std)

    # Format Output String
    output_lines = [
        "Dataset Statistics",
        "="*20,
        f"Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]",
        f"Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]",
        "",
        "ImageNet Comparison",
        "="*20,
        f"ImageNet Mean: [{imagenet_mean[0]:.3f}, {imagenet_mean[1]:.3f}, {imagenet_mean[2]:.3f}]",
        f"ImageNet Std:  [{imagenet_std[0]:.3f}, {imagenet_std[1]:.3f}, {imagenet_std[2]:.3f}]",
        "",
        f"Absolute Mean Diff: [{diff_mean[0]:.4f}, {diff_mean[1]:.4f}, {diff_mean[2]:.4f}]",
        f"Absolute Std Diff:  [{diff_std[0]:.4f}, {diff_std[1]:.4f}, {diff_std[2]:.4f}]"
    ]

    # Check for warnings (difference > 0.05)
    flag_warning = np.any(diff_mean > 0.05) or np.any(diff_std > 0.05)
    if flag_warning:
        output_lines.extend([
            "",
            "WARNING: Dataset distribution differs significantly from ImageNet. "
            "Consider using customized mean and std for model normalization."
        ])

    result_text = "\n".join(output_lines)
    
    # Print to console
    print("\n" + result_text + "\n")

    # Save to text file
    with open(output_txt, "w") as f:
        f.write(result_text)
    print(f"Results saved to {output_txt}")

    # Plot Histograms
    print("Generating histogram plots...")
    plt.figure(figsize=(10, 6))
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram to relative frequency (probability) 
    plt.plot(bin_centers, hist_r / pixel_count, color='red', alpha=0.8, label='R channel')
    plt.plot(bin_centers, hist_g / pixel_count, color='green', alpha=0.8, label='G channel')
    plt.plot(bin_centers, hist_b / pixel_count, color='blue', alpha=0.8, label='B channel')
    
    plt.fill_between(bin_centers, hist_r / pixel_count, color='red', alpha=0.2)
    plt.fill_between(bin_centers, hist_g / pixel_count, color='green', alpha=0.2)
    plt.fill_between(bin_centers, hist_b / pixel_count, color='blue', alpha=0.2)
    
    plt.title("RGB Pixel Value Distribution")
    plt.xlabel("Normalized Pixel Value [0, 1]")
    plt.ylabel("Relative Frequency")
    plt.xlim(0, 1)
    
    # Annotate if distribution is very different from ImageNet
    if flag_warning:
         plt.text(0.5, 0.95, "Warning: Distribution differs from ImageNet",
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform=plt.gca().transAxes,
                  color='red',
                  fontsize=12,
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

    plt.legend()
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram plot saved to {output_png}")

if __name__ == "__main__":
    # Provides CLI capabilities while setting the expected default directory
    parser = argparse.ArgumentParser(description="Compute dataset-specific mean and std for image normalization.")
    parser.add_argument("--image-dir", type=str, default=os.path.join("train", "Color_Images"),
                        help="Path to the directory containing training images")
    parser.add_argument("--output-txt", type=str, default="dataset_stats.txt",
                        help="Output text file to save statistics")
    parser.add_argument("--output-png", type=str, default="pixel_distribution.png",
                        help="Output PNG file for histogram visualization")
    
    args = parser.parse_args()
    
    compute_dataset_stats(
        image_dir=args.image_dir,
        output_txt=args.output_txt,
        output_png=args.output_png
    )

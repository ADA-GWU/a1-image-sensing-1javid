import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import argparse

# Precompute lookup table for fast bit counting
POPCOUNT_TABLE_16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)

def parse_args():
    parser = argparse.ArgumentParser(description='Stereo matching with census transform')
    parser.add_argument('--dataset', type=str, default='dataset1', 
                        help='Dataset to use (dataset1 or dataset2)')
    parser.add_argument('--window', type=int, default=80,
                        help='Census transform window size')
    parser.add_argument('--max-disp', type=int, default=128,
                        help='Maximum disparity to search')
    parser.add_argument('--agg-window', type=int, default=50,
                        help='Cost aggregation window size')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization')
    return parser.parse_args()


def fast_hamming_distance(arr1, arr2):
    """Fast Hamming distance using lookup table for 16-bit chunks"""
    xor_result = np.bitwise_xor(arr1, arr2)
    chunks = xor_result.view(np.uint16)
    counts = POPCOUNT_TABLE_16[chunks]
    return np.sum(counts.reshape(xor_result.shape + (-1,)), axis=-1)

def census_transform(img, window_size=3):
    """
    Census transform - encodes local structure as a bit string.
    Each bit represents if a neighbor is darker than center pixel.
    """
    h, w = img.shape
    offset = window_size // 2
    census = np.zeros((h, w), dtype=np.uint64)
    
    img_padded = np.pad(img, offset, mode='constant', constant_values=0)
    
    bit = 0
    for r in range(window_size):
        for c in range(window_size):
            if r == offset and c == offset:
                continue
            
            neighbor = img_padded[r:r+h, c:c+w]
            center = img_padded[offset:offset+h, offset:offset+w]
            comparison = (neighbor < center).astype(np.uint64) << bit
            census = np.bitwise_or(census, comparison)
            bit += 1
    
    return census

def compute_cost_for_disparity(left_census, right_census, d, width):
    """Compute matching cost for a single disparity level"""
    shifted = np.zeros_like(right_census)
    if d > 0:
        shifted[:, d:] = right_census[:, :width-d]
    else:
        shifted = right_census
    return fast_hamming_distance(left_census, shifted)


def compute_disparity_map(left_img, right_img, window_size=5, max_disp=64, agg_window=5):
    """Main stereo matching pipeline"""
    h, w = left_img.shape
    
    print(f"Computing census transform (window={window_size})...")
    left_census = census_transform(left_img, window_size)
    right_census = census_transform(right_img, window_size)
    
    print(f"Building cost volume (max_disp={max_disp})...")
    cost_volume = Parallel(n_jobs=-1)(
        delayed(compute_cost_for_disparity)(left_census, right_census, d, w)
        for d in range(max_disp)
    )
    cost_volume = np.array(cost_volume)
    
    print(f"Aggregating costs (window={agg_window})...")
    for d in range(max_disp):
        cost_volume[d] = cv2.boxFilter(
            cost_volume[d], -1, (agg_window, agg_window), normalize=False
        )

    disparity = np.argmin(cost_volume, axis=0).astype(np.uint8)
    return disparity

def load_images(dataset):
    """Load left and right stereo images"""
    orig_dir = f'images/originals/{dataset}'
    left_path = os.path.join(orig_dir, 'left.jpeg')
    right_path = os.path.join(orig_dir, 'right.jpeg')
    
    print(f"Loading images from {dataset}...")
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
    if left is None or right is None:
        raise FileNotFoundError(f"Images not found in {orig_dir}")
    
    return left, right

def visualize(left, right, disparity, save_path=None):
    """Show side-by-side comparison and optionally save it"""
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(left, cmap='gray')
    plt.title('Left Image')
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(right, cmap='gray')
    plt.title('Right Image')
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(disparity, cmap='inferno')
    plt.title('Disparity Map')
    plt.colorbar(im3, fraction=0.046, pad=0.04, label='Disparity (pixels)')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        # Load stereo pair
        left, right = load_images(args.dataset)
        
        # Run stereo matching
        disparity = compute_disparity_map(
            left, right,
            window_size=args.window,
            max_disp=args.max_disp,
            agg_window=args.agg_window
        )
        
        # Visualize
        if not args.no_viz:
            disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Create visualization save path
            out_dir = f'images/outputs/{args.dataset}'
            viz_filename = f"{args.dataset}-{args.window}-{args.max_disp}-{args.agg_window}_viz.png"
            viz_path = os.path.join(out_dir, viz_filename)
            
            visualize(left, right, disp_norm, save_path=viz_path)
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
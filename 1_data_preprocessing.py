"""
1_data_preprocessing.py
=======================
BIPA Data Preprocessing Pipeline
- YOLO-based mosquito localization and cropping
- Heuristic BAM (Biological Attention Map) pre-computation
- Dataset preparation for training

Run this script BEFORE training to prepare the dataset.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from tqdm.auto import tqdm
import torch

# Import configuration
from config import *

print("="*70)
print("BIPA DATA PREPROCESSING PIPELINE")
print("="*70)
print_config()

# ====================================================================
# SECTION 1: YOLO PREPROCESSING
# ====================================================================

def run_yolo_preprocessing(csv_in, csv_out, crop_dir, conf_thres=YOLO_CONFIDENCE_THRESHOLD):
    """
    Run YOLOv8 mosquito detection and crop regions of interest.
    Falls back to center crop if YOLO is unavailable or fails.
    
    Args:
        csv_in: Input CSV with 'filepath' and 'label' columns
        csv_out: Output CSV with cropped image paths
        crop_dir: Directory to save cropped images
        conf_thres: YOLO confidence threshold
    """
    os.makedirs(crop_dir, exist_ok=True)
    
    # Try to load YOLO model
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL)
        print(f"✓ Loaded YOLOv8 model: {YOLO_MODEL}")
        use_yolo = True
    except Exception as e:
        print(f"⚠ YOLO unavailable ({e}), using center-crop fallback")
        use_yolo = False
    
    # Load CSV
    df = pd.read_csv(csv_in).copy()
    
    # Fix filepaths (handle relative paths)
    if 'filepath' not in df.columns:
        raise ValueError(f"CSV {csv_in} must contain 'filepath' column")
    
    df['filepath'] = df['filepath'].astype(str).apply(
        lambda p: p.replace("../Dataset_1", DRIVE_ROOT)
                   .replace("../Dataset_2", DRIVE_ROOT)
                   .replace("../Dataset_3", DRIVE_ROOT)
                   .replace("../Dataset_4", DRIVE_ROOT)
    )
    
    out_rows = []
    detection_count = 0
    
    print(f"\nProcessing {len(df)} images...")
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="YOLO Cropping"):
        src_path = row['filepath']
        label = row['label']
        
        try:
            # Load image
            img = Image.open(src_path).convert('RGB')
            w, h = img.size
            
            if use_yolo:
                # Run YOLO detection
                results = model(src_path, conf=conf_thres, verbose=False)
                boxes = results[0].boxes
                
                if boxes is not None and len(boxes) > 0:
                    # Find largest detection (most likely the mosquito)
                    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * \
                            (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
                    largest_idx = torch.argmax(areas)
                    x1, y1, x2, y2 = boxes.xyxy[largest_idx].cpu().numpy().astype(int)
                    
                    # Add padding
                    box_w, box_h = x2 - x1, y2 - y1
                    pad_w = int(box_w * YOLO_PADDING_RATIO)
                    pad_h = int(box_h * YOLO_PADDING_RATIO)
                    
                    x1 = max(0, x1 - pad_w)
                    y1 = max(0, y1 - pad_h)
                    x2 = min(w, x2 + pad_w)
                    y2 = min(h, y2 + pad_h)
                    
                    crop = img.crop((x1, y1, x2, y2))
                    detection_count += 1
                else:
                    # No detection - use center crop
                    side = int(min(w, h) * CENTER_CROP_RATIO)
                    left = (w - side) // 2
                    top = (h - side) // 2
                    crop = img.crop((left, top, left + side, top + side))
            else:
                # YOLO not available - use center crop
                side = int(min(w, h) * CENTER_CROP_RATIO)
                left = (w - side) // 2
                top = (h - side) // 2
                crop = img.crop((left, top, left + side, top + side))
        
        except Exception as e:
            print(f"⚠ Error processing {src_path}: {e}")
            # Fallback: aggressive center crop
            try:
                img = Image.open(src_path).convert('RGB')
                w, h = img.size
                side = int(min(w, h) * 0.8)
                left = (w - side) // 2
                top = (h - side) // 2
                crop = img.crop((left, top, left + side, top + side))
            except:
                print(f"✗ Failed to process {src_path}, skipping...")
                continue
        
        # Resize to standard size
        crop = crop.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        
        # Save cropped image
        save_name = f"{i:06d}_{os.path.basename(src_path)}"
        save_path = os.path.join(crop_dir, save_name)
        crop.save(save_path, quality=95)
        
        out_rows.append({"filepath": save_path, "label": label})
    
    # Save output CSV
    pd.DataFrame(out_rows).to_csv(csv_out, index=False)
    
    print(f"\n✓ Saved cropped dataset: {csv_out}")
    print(f"  Total images: {len(out_rows)}")
    if use_yolo:
        detection_rate = 100 * detection_count / len(df) if len(df) > 0 else 0
        print(f"  YOLO detections: {detection_count}/{len(df)} ({detection_rate:.1f}%)")
    
    return csv_out

# ====================================================================
# SECTION 2: HEURISTIC BAM FILTERS
# ====================================================================

class HeuristicBAMFilters:
    """
    Bio-inspired feature extraction filters for generating heuristic BAMs.
    Implements Gabor filters, Laplacian of Gaussian, and contour detection.
    """
    
    def __init__(self):
        # Pre-compute Gabor kernels
        self.gabor_kernels = []
        for theta in np.arange(0, np.pi, np.pi / GABOR_ORIENTATIONS):
            for frequency in GABOR_FREQUENCIES:
                kernel = np.real(gabor_kernel(frequency, theta=theta))
                self.gabor_kernels.append(kernel)
        
        print(f"✓ Initialized {len(self.gabor_kernels)} Gabor kernels")
    
    def apply_gabor(self, img_gray):
        """Apply Gabor filters for texture/orientation features"""
        responses = []
        for kernel in self.gabor_kernels:
            filtered = ndi.convolve(img_gray, kernel, mode='wrap')
            responses.append(filtered)
        return np.max(responses, axis=0)
    
    def apply_log(self, img_gray, sigma=LOG_SIGMA):
        """Apply Laplacian of Gaussian for edge detection"""
        img_gray_64 = img_gray.astype(np.float64)
        blurred = cv2.GaussianBlur(img_gray_64, (0, 0), sigma)
        log_filter = cv2.Laplacian(blurred, cv2.CV_64F)
        return np.abs(log_filter)
    
    def apply_contours(self, img_gray):
        """Apply Canny edge detection and contour extraction"""
        edges = cv2.Canny(
            (img_gray * 255).astype(np.uint8),
            CANNY_THRESHOLD_LOW,
            CANNY_THRESHOLD_HIGH
        )
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges.astype(np.float32) / 255.0
    
    def generate_heuristic_bam(self, img_pil):
        """
        Generate heuristic BAM using bio-inspired filters.
        
        Args:
            img_pil: PIL Image
            
        Returns:
            numpy array: Normalized BAM [0, 1]
        """
        # Convert to grayscale
        img_gray = np.array(img_pil.convert('L')).astype(np.float32) / 255.0
        
        # Apply filters
        gabor_map = self.apply_gabor(img_gray)
        log_map = self.apply_log(img_gray)
        contour_map = self.apply_contours(img_gray)
        
        # Normalize individual maps
        gabor_map = (gabor_map - gabor_map.min()) / (gabor_map.max() - gabor_map.min() + 1e-8)
        log_map = (log_map - log_map.min()) / (log_map.max() - log_map.min() + 1e-8)
        
        # Weighted aggregation
        M_raw = (BAM_GABOR_WEIGHT * gabor_map + 
                 BAM_LOG_WEIGHT * log_map + 
                 BAM_CONTOUR_WEIGHT * contour_map)
        
        # Final normalization
        M_raw = (M_raw - M_raw.min()) / (M_raw.max() - M_raw.min() + 1e-8)
        
        return M_raw

# ====================================================================
# SECTION 3: BAM PRE-COMPUTATION AND CACHING
# ====================================================================

def precompute_bams(csv_file, cache_dir):
    """
    Pre-compute heuristic BAMs for all images and cache to disk.
    This dramatically speeds up training (10-20x faster).
    
    Args:
        csv_file: CSV with cropped image paths
        cache_dir: Directory to save cached BAMs
        
    Returns:
        dict: {image_path: bam_array}
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if cache already exists
    cache_file = os.path.join(
        cache_dir, 
        os.path.basename(csv_file).replace('.csv', '_bams.pkl')
    )
    
    if os.path.exists(cache_file):
        print(f"\n✓ Loading cached BAMs from {cache_file}")
        with open(cache_file, 'rb') as f:
            bam_dict = pickle.load(f)
        print(f"  Loaded {len(bam_dict)} pre-computed BAMs")
        return bam_dict
    
    # Compute BAMs
    print(f"\nComputing heuristic BAMs for {csv_file}...")
    df = pd.read_csv(csv_file)
    bam_filter = HeuristicBAMFilters()
    bam_dict = {}
    
    start_time = time.time()
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Computing BAMs"):
        img_path = row['filepath']
        
        try:
            img_pil = Image.open(img_path).convert('RGB')
            bam = bam_filter.generate_heuristic_bam(img_pil)
            bam_dict[img_path] = bam
        except Exception as e:
            print(f"⚠ Error computing BAM for {img_path}: {e}")
            # Use zero BAM as fallback
            bam_dict[img_path] = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    
    elapsed = time.time() - start_time
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(bam_dict, f)
    
    print(f"✓ Cached {len(bam_dict)} BAMs to {cache_file}")
    print(f"  Computation time: {elapsed:.2f}s ({elapsed/len(bam_dict)*1000:.2f}ms per image)")
    
    return bam_dict

# ====================================================================
# MAIN PREPROCESSING PIPELINE
# ====================================================================

def main():
    """Run complete preprocessing pipeline"""
    
    print("\n" + "="*70)
    print("STEP 1: YOLO PREPROCESSING")
    print("="*70)
    
    # Process training data
    if not os.path.exists(CROPPED_TRAIN_CSV):
        print("\nProcessing training data...")
        train_crop_dir = os.path.join(CROP_DIR, "train")
        run_yolo_preprocessing(TRAIN_CSV, CROPPED_TRAIN_CSV, train_crop_dir)
    else:
        print(f"\n✓ Training crops already exist: {CROPPED_TRAIN_CSV}")
    
    # Process test data
    if not os.path.exists(CROPPED_TEST_CSV):
        print("\nProcessing test data...")
        test_crop_dir = os.path.join(CROP_DIR, "test")
        run_yolo_preprocessing(TEST_CSV, CROPPED_TEST_CSV, test_crop_dir)
    else:
        print(f"✓ Test crops already exist: {CROPPED_TEST_CSV}")
    
    # Pre-compute BAMs if enabled
    if PRECOMPUTE_HEURISTIC_BAMS and USE_HEURISTIC_BAMS:
        print("\n" + "="*70)
        print("STEP 2: PRE-COMPUTING HEURISTIC BAMs")
        print("="*70)
        print("This is a one-time operation that speeds up training 10-20x")
        
        train_bams = precompute_bams(CROPPED_TRAIN_CSV, BAM_CACHE_DIR)
        test_bams = precompute_bams(CROPPED_TEST_CSV, BAM_CACHE_DIR)
        
        print(f"\n✓ BAM pre-computation complete!")
        print(f"  Training BAMs: {len(train_bams)}")
        print(f"  Test BAMs: {len(test_bams)}")
    else:
        print("\n⊘ BAM pre-computation disabled in config")
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print("\n✅ Ready for training! Run: python 2_model_training.py")
    print("\n📁 Generated files:")
    print(f"   1. {CROPPED_TRAIN_CSV}")
    print(f"   2. {CROPPED_TEST_CSV}")
    if PRECOMPUTE_HEURISTIC_BAMS and USE_HEURISTIC_BAMS:
        print(f"   3. {os.path.join(BAM_CACHE_DIR, '*_bams.pkl')}")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

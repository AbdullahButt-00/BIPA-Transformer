"""
inference.py
============
BIPA Single Image Inference
- Load trained model and predict mosquito species from a single image
- Visualize attention maps
- Support batch inference

Usage:
    python inference.py --image path/to/mosquito.jpg
    python inference.py --image path/to/mosquito.jpg --visualize
    python inference.py --batch path/to/folder/ --output results.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt

# Import configuration
from config import *

# Import model architecture
from importlib import import_module
training_module = import_module('2_model_training')
BIPA = training_module.BIPA
HeuristicBAMFilters = import_module('1_data_preprocessing').HeuristicBAMFilters

# ====================================================================
# INFERENCE CLASS
# ====================================================================

class BIPAInference:
    """BIPA model inference wrapper"""
    
    def __init__(self, model_path=BEST_MODEL_PATH, device=None):
        """
        Initialize inference model
        
        Args:
            model_path: Path to trained model weights
            device: torch device (defaults to CUDA if available)
        """
        self.device = device or DEVICE
        self.model_path = model_path
        
        # Load label mapping (from training)
        try:
            training_module = import_module('2_model_training')
            self.label2idx = training_module.label2idx
            self.idx2label = training_module.idx2label
            self.num_classes = len(self.label2idx)
        except:
            # Fallback: assume 3 classes
            print("⚠ Warning: Could not load label mapping from training, using default")
            self.label2idx = {'Aedes': 0, 'Anopheles': 1, 'Culex': 2}
            self.idx2label = {v: k for k, v in self.label2idx.items()}
            self.num_classes = 3
        
        # Load model
        self.model = self._load_model()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
        
        # BAM generator (if using heuristic)
        if USE_HEURISTIC_BAMS:
            self.bam_generator = HeuristicBAMFilters()
        else:
            self.bam_generator = None
        
        print(f"✓ BIPA model ready for inference")
        print(f"  Model: {model_path}")
        print(f"  Classes: {list(self.label2idx.keys())}")
        print(f"  Device: {self.device}")
    
    def _load_model(self):
        """Load trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        model = BIPA(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_chans=IN_CHANNELS,
            num_classes=self.num_classes,
            embed_dim=EMBED_DIM,
            depth=DEPTH,
            num_heads=NUM_HEADS,
            mlp_ratio=MLP_RATIO,
            drop_rate=0.0  # No dropout during inference
        )
        
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to image file
            
        Returns:
            tuple: (tensor, heuristic_bam, original_pil)
        """
        # Load image
        img_pil = Image.open(image_path).convert('RGB')
        
        # Apply transform
        img_tensor = self.transform(img_pil)
        
        # Generate heuristic BAM if needed
        if self.bam_generator is not None:
            img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
            heuristic_bam = self.bam_generator.generate_heuristic_bam(img_resized)
            heuristic_bam = torch.from_numpy(heuristic_bam).float()
        else:
            heuristic_bam = torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.float32)
        
        return img_tensor, heuristic_bam, img_pil
    
    def predict(self, image_path, return_attention=False):
        """
        Predict mosquito species for a single image
        
        Args:
            image_path: Path to image
            return_attention: Whether to return attention maps
            
        Returns:
            dict: Prediction results
        """
        # Preprocess
        img_tensor, heuristic_bam, img_pil = self.preprocess_image(image_path)
        
        # Add batch dimension
        img_batch = img_tensor.unsqueeze(0).to(self.device)
        heuristic_batch = heuristic_bam.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits, raw_bam, mb = self.model(img_batch, heuristic_batch)
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Prepare results
        results = {
            'predicted_class': self.idx2label[pred_class],
            'predicted_class_idx': pred_class,
            'confidence': confidence,
            'probabilities': {
                self.idx2label[i]: probs[0, i].item()
                for i in range(self.num_classes)
            }
        }
        
        if return_attention:
            results['attention_map'] = raw_bam[0].cpu().numpy()
            results['patch_attention'] = mb[0].cpu().numpy()
            results['heuristic_bam'] = heuristic_bam.cpu().numpy()
        
        return results
    
    def predict_batch(self, image_paths, show_progress=True):
        """
        Predict for multiple images
        
        Args:
            image_paths: List of image paths
            show_progress: Show progress bar
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        
        iterator = tqdm(image_paths, desc="Inference") if show_progress else image_paths
        
        for img_path in iterator:
            try:
                pred = self.predict(img_path, return_attention=False)
                pred['image_path'] = img_path
                results.append(pred)
            except Exception as e:
                print(f"⚠ Error processing {img_path}: {e}")
                results.append({
                    'image_path': img_path,
                    'predicted_class': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with attention maps
        
        Args:
            image_path: Path to image
            save_path: Optional path to save visualization
        """
        # Get prediction with attention
        pred = self.predict(image_path, return_attention=True)
        
        # Load original image
        img_pil = Image.open(image_path).convert('RGB')
        img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_resized) / 255.0
        
        # Prepare attention maps
        attention_map = pred['attention_map']
        attention_map = (attention_map - attention_map.min()) / \
                       (attention_map.max() - attention_map.min() + 1e-8)
        
        heuristic_bam = pred['heuristic_bam']
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heuristic BAM
        axes[1].imshow(heuristic_bam, cmap='hot')
        axes[1].set_title('Heuristic BAM\n(Bio-filters)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Learned attention
        axes[2].imshow(attention_map, cmap='jet')
        axes[2].set_title('Learned Attention\n(A-BAMNet)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Overlay
        axes[3].imshow(img_array)
        axes[3].imshow(attention_map, cmap='jet', alpha=0.5)
        axes[3].set_title(f'Prediction: {pred["predicted_class"]}\n'
                         f'Confidence: {pred["confidence"]*100:.1f}%',
                         fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        plt.show()

# ====================================================================
# COMMAND LINE INTERFACE
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='BIPA Mosquito Species Inference')
    
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Path to folder with images')
    parser.add_argument('--model', type=str, default=BEST_MODEL_PATH,
                       help='Path to model weights')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize attention maps')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV for batch predictions')
    parser.add_argument('--save-viz', type=str,
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Initialize inference
    print("="*70)
    print("BIPA INFERENCE")
    print("="*70)
    
    predictor = BIPAInference(model_path=args.model)
    
    # Single image inference
    if args.image:
        if not os.path.exists(args.image):
            print(f"✗ Error: Image not found: {args.image}")
            sys.exit(1)
        
        print(f"\nProcessing: {args.image}")
        
        if args.visualize:
            predictor.visualize_prediction(args.image, save_path=args.save_viz)
        else:
            result = predictor.predict(args.image)
            
            print(f"\n{'='*70}")
            print(f"PREDICTION RESULTS")
            print(f"{'='*70}")
            print(f"\n🔍 Predicted Species: {result['predicted_class']}")
            print(f"📊 Confidence: {result['confidence']*100:.2f}%")
            print(f"\n📈 Class Probabilities:")
            for class_name, prob in result['probabilities'].items():
                bar = '█' * int(prob * 40)
                print(f"   {class_name:<12} {prob*100:>6.2f}% {bar}")
            print(f"{'='*70}")
    
    # Batch inference
    elif args.batch:
        if not os.path.isdir(args.batch):
            print(f"✗ Error: Directory not found: {args.batch}")
            sys.exit(1)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            os.path.join(args.batch, f)
            for f in os.listdir(args.batch)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        if not image_paths:
            print(f"✗ Error: No images found in {args.batch}")
            sys.exit(1)
        
        print(f"\nFound {len(image_paths)} images in {args.batch}")
        print(f"Running batch inference...")
        
        # Predict
        results = predictor.predict_batch(image_paths)
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        
        print(f"\n✓ Predictions saved: {args.output}")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"BATCH SUMMARY")
        print(f"{'='*70}")
        
        for class_name in predictor.label2idx.keys():
            count = sum(1 for r in results if r.get('predicted_class') == class_name)
            print(f"   {class_name}: {count} images ({count/len(results)*100:.1f}%)")
        
        errors = sum(1 for r in results if r.get('predicted_class') == 'ERROR')
        if errors > 0:
            print(f"   Errors: {errors} images")
        
        print(f"{'='*70}")
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python inference.py --image mosquito.jpg --visualize")
        print("  python inference.py --batch ./test_images/ --output results.csv")

if __name__ == "__main__":
    try:
        from tqdm.auto import tqdm
    except ImportError:
        # Fallback if tqdm not available
        def tqdm(iterable, **kwargs):
            return iterable
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Inference interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Inference error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

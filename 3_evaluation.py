"""
3_evaluation.py
===============
BIPA Model Evaluation Script
- Comprehensive metrics and per-class analysis
- Confusion matrix generation
- BAM attention visualization
- Training history plots

Run this AFTER training: python 3_evaluation.py
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Import configuration and model
from config import *
import sys
sys.path.insert(0, os.path.dirname(__file__))

print("="*70)
print("BIPA MODEL EVALUATION")
print("="*70)
print_config()

# ====================================================================
# IMPORT MODEL ARCHITECTURE (from training script)
# ====================================================================

# We need to import the model classes - copy from 2_model_training.py
from importlib import import_module
training_module = import_module('2_model_training')
BIPA = training_module.BIPA
MosquitoDataset = training_module.MosquitoDataset
label2idx = training_module.label2idx
idx2label = training_module.idx2label
test_loader = training_module.test_loader
test_dataset = training_module.test_dataset

# ====================================================================
# SECTION 1: LOAD BEST MODEL
# ====================================================================

def load_best_model():
    """Load the best model from training"""
    
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"\n✗ Best model not found: {BEST_MODEL_PATH}")
        print("  Please run training first: python 2_model_training.py")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("LOADING BEST MODEL")
    print("="*70)
    
    model = BIPA(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANNELS,
        num_classes=len(label2idx),
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        drop_rate=DROP_RATE
    )
    
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n✓ Model loaded successfully")
    print(f"  Path: {BEST_MODEL_PATH}")
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Device: {DEVICE}")
    
    return model

# ====================================================================
# SECTION 2: DETAILED EVALUATION
# ====================================================================

def detailed_evaluate(model, loader, device):
    """
    Comprehensive evaluation with per-class metrics and inference timing
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    
    print("\n" + "="*70)
    print("RUNNING EVALUATION")
    print("="*70)
    
    with torch.no_grad():
        for images, labels, heuristic_bams, paths in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            heuristic_bams = heuristic_bams.to(device, non_blocking=True)
            
            # Measure inference time per image
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            
            logits, raw_bam, mb = model(images, heuristic_bams)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            
            # Store timing
            batch_time = (t1 - t0) / images.shape[0]
            inference_times.extend([batch_time] * images.shape[0])
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Inference time statistics
    avg_inference_ms = np.mean(inference_times) * 1000
    std_inference_ms = np.std(inference_times) * 1000
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'avg_inference_ms': avg_inference_ms,
        'std_inference_ms': std_inference_ms,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

# ====================================================================
# SECTION 3: RESULTS DISPLAY
# ====================================================================

def display_results(metrics):
    """Display results in paper format"""
    
    print("\n" + "="*70)
    print(f"FINAL TEST RESULTS - Dataset-{ACTIVE_DATASET}")
    print("="*70)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {metrics['precision_macro']:.4f}")
    print(f"   Recall:    {metrics['recall_macro']:.4f}")
    print(f"   F1 Score:  {metrics['f1_macro']:.4f}")
    print(f"   Inference: {metrics['avg_inference_ms']:.2f} ± {metrics['std_inference_ms']:.2f} ms/image")
    
    # Compare with expected results
    expected = EXPECTED_RESULTS.get(ACTIVE_DATASET, {})
    if expected:
        acc_diff = metrics['accuracy']*100 - expected['accuracy']
        f1_diff = metrics['f1_macro'] - expected['f1']
        print(f"\n📈 Comparison with Paper Results:")
        print(f"   Expected Accuracy: {expected['accuracy']:.2f}% | Achieved: {metrics['accuracy']*100:.2f}% "
              f"({'↑' if acc_diff > 0 else '↓'}{abs(acc_diff):.2f}%)")
        print(f"   Expected F1: {expected['f1']:.4f} | Achieved: {metrics['f1_macro']:.4f} "
              f"({'↑' if f1_diff > 0 else '↓'}{abs(f1_diff):.4f})")
    
    print(f"\n📈 Per-Class Performance:")
    print("-" * 70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(label2idx.keys()):
        support = np.sum(metrics['labels'] == i)
        print(f"{class_name:<15} {metrics['precision_per_class'][i]:<12.4f} "
              f"{metrics['recall_per_class'][i]:<12.4f} "
              f"{metrics['f1_per_class'][i]:<12.4f} {support:<10}")
    print("-" * 70)
    
    # Create results dictionary
    results_dict = {
        'Model': 'BIPA',
        'Dataset': f'Dataset-{ACTIVE_DATASET}',
        'Accuracy (%)': round(metrics['accuracy'] * 100, 2),
        'Precision': round(metrics['precision_macro'], 4),
        'Recall': round(metrics['recall_macro'], 4),
        'F1 Score': round(metrics['f1_macro'], 4),
        'Inference (ms)': round(metrics['avg_inference_ms'], 2)
    }
    
    print("\n📋 Results Summary:")
    for key, value in results_dict.items():
        print(f"   {key}: {value}")
    
    return results_dict

# ====================================================================
# SECTION 4: CONFUSION MATRIX
# ====================================================================

def plot_confusion_matrix(metrics):
    """Generate and save confusion matrix"""
    
    print("\n" + "="*70)
    print("GENERATING CONFUSION MATRIX")
    print("="*70)
    
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label2idx.keys()),
                yticklabels=list(label2idx.keys()),
                cbar_kws={'label': 'Count'},
                ax=ax,
                square=True)
    
    ax.set_title(f'Confusion Matrix - BIPA Model\nDataset-{ACTIVE_DATASET}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Add accuracy text
    accuracy_text = f"Overall Accuracy: {metrics['accuracy']*100:.2f}%"
    plt.text(0.5, -0.15, accuracy_text, ha='center', transform=ax.transAxes,
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_dataset{ACTIVE_DATASET}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {cm_path}")
    
    if VERBOSE:
        plt.show()
    plt.close()
    
    # Print confusion matrix analysis
    print("\nConfusion Matrix Analysis:")
    print("-" * 70)
    for i, true_class in enumerate(label2idx.keys()):
        true_count = cm[i].sum()
        correct = cm[i, i]
        accuracy_class = (correct / true_count * 100) if true_count > 0 else 0
        print(f"{true_class:<15} Total: {true_count:>3} | Correct: {correct:>3} | Accuracy: {accuracy_class:>6.2f}%")
    print("-" * 70)
    
    return cm_path

# ====================================================================
# SECTION 5: TRAINING HISTORY PLOTS
# ====================================================================

def plot_training_history(metrics):
    """Plot training history if available"""
    
    history_path = os.path.join(RESULTS_DIR, 'training_history.pkl')
    
    if not os.path.exists(history_path):
        print("\n⚠ Training history not available (model loaded from checkpoint)")
        return None
    
    print("\n" + "="*70)
    print("PLOTTING TRAINING HISTORY")
    print("="*70)
    
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Training Loss
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([1, len(history['train_loss'])])
    
    # Plot 2: Accuracy (Train vs Val)
    train_acc_percent = [x * 100 for x in history['train_acc']]
    val_acc_percent = [x['accuracy'] * 100 for x in history['val_metrics']]
    
    axes[0, 1].plot(epochs_range, train_acc_percent, 'g-', linewidth=2, marker='s',
                    markersize=4, label='Train Accuracy')
    axes[0, 1].plot(epochs_range, val_acc_percent, 'r-', linewidth=2, marker='^',
                    markersize=4, label='Val Accuracy')
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([1, len(history['train_loss'])])
    
    # Plot 3: Validation Metrics
    val_f1 = [x['f1'] * 100 for x in history['val_metrics']]
    val_prec = [x['precision'] * 100 for x in history['val_metrics']]
    val_rec = [x['recall'] * 100 for x in history['val_metrics']]
    
    axes[1, 0].plot(epochs_range, val_f1, 'b-', linewidth=2, marker='o', markersize=4, label='F1 Score')
    axes[1, 0].plot(epochs_range, val_prec, 'g-', linewidth=2, marker='s', markersize=4, label='Precision')
    axes[1, 0].plot(epochs_range, val_rec, 'r-', linewidth=2, marker='^', markersize=4, label='Recall')
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Validation Metrics', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([1, len(history['train_loss'])])
    
    # Plot 4: Best Epoch Indicator
    best_epoch = np.argmax([x['f1'] for x in history['val_metrics']]) + 1
    best_f1_val = max([x['f1'] for x in history['val_metrics']])
    
    axes[1, 1].text(0.5, 0.7, f'Best Model', ha='center', va='center',
                    fontsize=24, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.5, f'Epoch: {best_epoch}', ha='center', va='center',
                    fontsize=18, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.35, f'Val F1: {best_f1_val:.4f}', ha='center', va='center',
                    fontsize=18, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.2, f'Test Acc: {metrics["accuracy"]*100:.2f}%',
                    ha='center', va='center', fontsize=18, transform=axes[1, 1].transAxes, color='green')
    axes[1, 1].axis('off')
    
    plt.suptitle(f'BIPA Training History - Dataset-{ACTIVE_DATASET}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    history_plot_path = os.path.join(RESULTS_DIR, f'training_history_dataset{ACTIVE_DATASET}.png')
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved: {history_plot_path}")
    
    if VERBOSE:
        plt.show()
    plt.close()
    
    return history_plot_path

# ====================================================================
# SECTION 6: BAM ATTENTION VISUALIZATION
# ====================================================================

def visualize_bam_attention(model, dataset, num_samples=VIZ_NUM_SAMPLES):
    """Visualize BAM attention maps with predictions"""
    
    if not SAVE_BAM_VISUALIZATIONS:
        return None
    
    print("\n" + "="*70)
    print("VISUALIZING BAM ATTENTION MAPS")
    print("="*70)
    
    model.eval()
    
    # Select diverse samples (samples per class)
    indices = []
    for class_idx in range(len(label2idx)):
        class_samples = [i for i, (_, label, _, _) in enumerate(dataset) if label == class_idx]
        if len(class_samples) >= VIZ_SAMPLES_PER_CLASS:
            selected = np.random.choice(class_samples, VIZ_SAMPLES_PER_CLASS, replace=False)
            indices.extend(selected)
        elif len(class_samples) > 0:
            indices.extend(class_samples)
    
    # Limit to num_samples
    indices = indices[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(len(indices), 4, figsize=(16, len(indices)*3))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, label, heuristic_bam, path = dataset[idx]
            
            # Unnormalize for display
            img_display = img_tensor.clone()
            mean = torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
            std = torch.tensor(NORMALIZE_STD).view(3, 1, 1)
            img_display = img_display * std + mean
            img_display = img_display.permute(1, 2, 0).cpu().numpy()
            img_display = np.clip(img_display, 0, 1)
            
            # Forward pass
            img_batch = img_tensor.unsqueeze(0).to(DEVICE)
            heuristic_batch = heuristic_bam.unsqueeze(0).to(DEVICE)
            logits, raw_bam, mb = model(img_batch, heuristic_batch)
            
            pred_class = logits.argmax(dim=1).item()
            true_class = label
            confidence = torch.softmax(logits, dim=1)[0, pred_class].item()
            
            # Get BAM
            bam_map = raw_bam[0].cpu().numpy()
            bam_map = (bam_map - bam_map.min()) / (bam_map.max() - bam_map.min() + 1e-8)
            
            # Determine correctness
            is_correct = pred_class == true_class
            border_color = 'green' if is_correct else 'red'
            
            # Plot 1: Original Image
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title(f'Original\nTrue: {idx2label[true_class]}',
                                fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')
            for spine in axes[i, 0].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            
            # Plot 2: Heuristic BAM
            axes[i, 1].imshow(heuristic_bam.cpu().numpy(), cmap='hot')
            axes[i, 1].set_title('Heuristic BAM\n(Bio-filters)', fontsize=10, fontweight='bold')
            axes[i, 1].axis('off')
            
            # Plot 3: Learned BAM
            axes[i, 2].imshow(bam_map, cmap='jet')
            axes[i, 2].set_title('Learned BAM\n(A-BAMNet)', fontsize=10, fontweight='bold')
            axes[i, 2].axis('off')
            
            # Plot 4: Overlay
            axes[i, 3].imshow(img_display)
            axes[i, 3].imshow(bam_map, cmap='jet', alpha=0.5)
            axes[i, 3].set_title(f'Attention Overlay\nPred: {idx2label[pred_class]} ({confidence*100:.1f}%)',
                                fontsize=10, fontweight='bold', color=border_color)
            axes[i, 3].axis('off')
            for spine in axes[i, 3].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
    
    plt.suptitle(f'BIPA Attention Visualization - Dataset-{ACTIVE_DATASET}\n'
                 f'Green border = Correct | Red border = Incorrect',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    bam_path = os.path.join(RESULTS_DIR, f'bam_visualization_dataset{ACTIVE_DATASET}.png')
    plt.savefig(bam_path, dpi=300, bbox_inches='tight')
    print(f"✓ BAM visualization saved: {bam_path}")
    
    if VERBOSE:
        plt.show()
    plt.close()
    
    return bam_path

# ====================================================================
# SECTION 7: SAVE RESULTS
# ====================================================================

def save_results(metrics, results_dict):
    """Save evaluation results to CSV"""
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Overall results
    results_df = pd.DataFrame([results_dict])
    results_csv = os.path.join(RESULTS_DIR, f'bipa_results_dataset{ACTIVE_DATASET}.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Overall results saved: {results_csv}")
    
    # Per-class results
    per_class_results = []
    for i, class_name in enumerate(label2idx.keys()):
        support = np.sum(metrics['labels'] == i)
        per_class_results.append({
            'Class': class_name,
            'Precision': round(metrics['precision_per_class'][i], 4),
            'Recall': round(metrics['recall_per_class'][i], 4),
            'F1-Score': round(metrics['f1_per_class'][i], 4),
            'Support': support
        })
    
    per_class_df = pd.DataFrame(per_class_results)
    per_class_csv = os.path.join(RESULTS_DIR, f'bipa_per_class_results_dataset{ACTIVE_DATASET}.csv')
    per_class_df.to_csv(per_class_csv, index=False)
    print(f"✓ Per-class results saved: {per_class_csv}")
    
    # Classification report
    report = classification_report(
        metrics['labels'], 
        metrics['predictions'],
        target_names=list(label2idx.keys()),
        digits=4
    )
    report_path = os.path.join(RESULTS_DIR, f'classification_report_dataset{ACTIVE_DATASET}.txt')
    with open(report_path, 'w') as f:
        f.write(f"BIPA Classification Report - Dataset-{ACTIVE_DATASET}\n")
        f.write("="*70 + "\n\n")
        f.write(report)
    print(f"✓ Classification report saved: {report_path}")
    
    return results_csv, per_class_csv, report_path

# ====================================================================
# MAIN EVALUATION
# ====================================================================

def main():
    """Main evaluation function"""
    
    # Load model
    model = load_best_model()
    
    # Run evaluation
    metrics = detailed_evaluate(model, test_loader, DEVICE)
    
    # Display results
    results_dict = display_results(metrics)
    
    # Generate visualizations
    cm_path = plot_confusion_matrix(metrics)
    history_path = plot_training_history(metrics)
    bam_path = visualize_bam_attention(model, test_dataset)
    
    # Save results
    results_csv, per_class_csv, report_path = save_results(metrics, results_dict)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Output Files:")
    print(f"   1. Overall results: {results_csv}")
    print(f"   2. Per-class results: {per_class_csv}")
    print(f"   3. Classification report: {report_path}")
    print(f"   4. Confusion matrix: {cm_path}")
    if history_path:
        print(f"   5. Training history: {history_path}")
    if bam_path:
        print(f"   6. BAM visualization: {bam_path}")
    
    print(f"\n🏆 Final Performance:")
    print(f"   Model: BIPA")
    print(f"   Dataset: Dataset-{ACTIVE_DATASET}")
    print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   F1 Score: {metrics['f1_macro']:.4f}")
    print(f"   Inference: {metrics['avg_inference_ms']:.2f} ms/image")
    
    print("\n" + "="*70)
    print("🎉 All evaluations completed successfully!")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

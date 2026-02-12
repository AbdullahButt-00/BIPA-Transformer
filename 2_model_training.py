"""
2_model_training.py
===================
BIPA Model Training Script
- Complete BIPA transformer architecture
- End-to-end training with A-BAMNet
- Periodic checkpointing and best model saving

Run this AFTER preprocessing: python 2_model_training.py
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm

# Import configuration
from config import *

# Set random seeds for reproducibility
if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        if DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

print("="*70)
print("BIPA MODEL TRAINING")
print("="*70)
print_config()

# ====================================================================
# SECTION 1: DATASET AND DATALOADERS
# ====================================================================

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=TRAIN_AUG_HORIZONTAL_FLIP),
    transforms.RandomRotation(TRAIN_AUG_ROTATION),
    transforms.ColorJitter(
        brightness=TRAIN_AUG_BRIGHTNESS,
        contrast=TRAIN_AUG_CONTRAST,
        saturation=TRAIN_AUG_SATURATION
    ),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
])

class MosquitoDataset(Dataset):
    """Fast dataset with pre-computed BAM caching"""
    
    def __init__(self, csv_file, label2idx, transform=None, bam_cache=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label2idx = label2idx
        self.bam_cache = bam_cache
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filepath']
        label_str = self.data.iloc[idx]['label']
        label = self.label2idx[label_str]
        
        # Load image
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img_pil)
        
        # Get pre-computed BAM (instant lookup!)
        if self.bam_cache is not None and img_path in self.bam_cache:
            heuristic_bam = torch.from_numpy(self.bam_cache[img_path]).float()
        else:
            heuristic_bam = torch.zeros((IMG_SIZE, IMG_SIZE), dtype=torch.float32)
        
        return img_tensor, label, heuristic_bam, img_path

def build_label_map(csv_paths):
    """Build shared label-to-index mapping"""
    labels = []
    for p in csv_paths:
        df = pd.read_csv(p)
        labels.extend(df['label'].unique())
    labels = sorted(list(set(labels)))
    return {lab: i for i, lab in enumerate(labels)}

# Load cached BAMs if available
def load_cached_bams(csv_file):
    """Load pre-computed BAMs from cache"""
    cache_file = os.path.join(
        BAM_CACHE_DIR,
        os.path.basename(csv_file).replace('.csv', '_bams.pkl')
    )
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Build label mapping
label2idx = build_label_map([CROPPED_TRAIN_CSV, CROPPED_TEST_CSV])
idx2label = {v: k for k, v in label2idx.items()}
NUM_CLASSES = len(label2idx)

print(f"\nClass mapping: {label2idx}")
print(f"Number of classes: {NUM_CLASSES}")

# Load BAM caches
if PRECOMPUTE_HEURISTIC_BAMS and USE_HEURISTIC_BAMS:
    train_bams = load_cached_bams(CROPPED_TRAIN_CSV)
    test_bams = load_cached_bams(CROPPED_TEST_CSV)
    print(f"\n✓ Loaded pre-computed BAMs")
else:
    train_bams = None
    test_bams = None
    print(f"\n⊘ BAM pre-computation disabled")

# Create datasets
train_dataset = MosquitoDataset(
    CROPPED_TRAIN_CSV, label2idx, transform=train_transform, bam_cache=train_bams
)
test_dataset = MosquitoDataset(
    CROPPED_TEST_CSV, label2idx, transform=test_transform, bam_cache=test_bams
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=NUM_WORKERS, 
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0)
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0)
)

print(f"\n✓ Datasets ready:")
print(f"  Train: {len(train_dataset)} samples")
print(f"  Test: {len(test_dataset)} samples")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Workers: {NUM_WORKERS}")

# ====================================================================
# SECTION 2: BIPA MODEL ARCHITECTURE
# ====================================================================

class PatchEmbed(nn.Module):
    """Micro-patch embedding with positional encoding"""
    
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, 
                 in_chans=IN_CHANNELS, embed_dim=EMBED_DIM):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches_side = img_size // patch_size
        self.num_patches = self.n_patches_side ** 2
        
        # Convolutional projection (equivalent to linear projection of flattened patches)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch projection: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, N) -> (B, N, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        return x

class ABAMNet(nn.Module):
    """
    Adaptive BAM Generator (A-BAMNet)
    Lightweight CNN that learns to generate biological attention maps
    """
    
    def __init__(self, in_ch=IN_CHANNELS, base_ch=ABAM_BASE_CHANNELS):
        super().__init__()
        
        # Determine input channels based on BAM mode
        use_heuristic = (BAM_MODE == "heuristic+learned")
        input_channels = in_ch + 1 if use_heuristic else in_ch
        
        self.use_heuristic = use_heuristic
        
        # Lightweight CNN architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
            nn.BatchNorm2d(base_ch // 2),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(base_ch // 2, 1, 1)
    
    def forward(self, x, heuristic_bam=None):
        # Concatenate heuristic BAM if using hybrid mode
        if self.use_heuristic and heuristic_bam is not None:
            heuristic_bam = heuristic_bam.unsqueeze(1)  # (B, 1, H, W)
            x = torch.cat([x, heuristic_bam], dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out(x)
        
        return x.squeeze(1)  # (B, H, W)

class BIPA_MultiHeadAttention(nn.Module):
    """
    BIPA Multi-Head Attention with BAM modulation
    Implements: Score_BIPA = Score + α·M̂_B
    """
    
    def __init__(self, dim=EMBED_DIM, num_heads=NUM_HEADS, qkv_bias=True, 
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Learnable BAM influence parameter α
        self.alpha = nn.Parameter(torch.ones(1))
    
    def forward(self, x, mb):
        B, L, D = x.shape
        
        # QKV projections
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, L, head_dim)
        
        # Attention scores: Q @ K^T / sqrt(d_k)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, L, L)
        
        # Add BAM bias (additive modulation)
        # mb: (B, N) -> add cls token -> (B, N+1) -> expand to (B, num_heads, L, N+1)
        cls_zero = torch.zeros(B, 1, device=mb.device, dtype=mb.dtype)
        mb_with_cls = torch.cat([cls_zero, mb], dim=1)  # (B, N+1)
        mb_bias = mb_with_cls.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N+1)
        mb_bias = mb_bias.expand(-1, self.num_heads, L, -1)  # (B, num_heads, L, N+1)
        
        # CRITICAL: Additive modulation as per paper
        attn = attn + self.alpha * mb_bias
        
        # Softmax and dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class BIPABlock(nn.Module):
    """
    BIPA Transformer Block
    Structure: LayerNorm -> BIPA Attention -> Residual -> LayerNorm -> MLP -> Residual
    """
    
    def __init__(self, dim=EMBED_DIM, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, 
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BIPA_MultiHeadAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x, mb):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), mb)  # Attention with residual
        x = x + self.mlp(self.norm2(x))       # MLP with residual
        return x

class BIPA(nn.Module):
    """
    Complete BIPA Transformer
    End-to-end learning of biological attention for mosquito classification
    """
    
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANNELS,
                 num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, depth=DEPTH,
                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, drop_rate=DROP_RATE):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Components
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.abam = ABAMNet(in_ch=in_chans, base_ch=ABAM_BASE_CHANNELS)
        
        # BIPA encoder blocks
        self.blocks = nn.ModuleList([
            BIPABlock(embed_dim, num_heads, mlp_ratio, drop_rate, drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, heuristic_bam=None):
        B = x.shape[0]
        
        # Patch embedding
        tokens = self.patch_embed(x)  # (B, N+1, embed_dim)
        tokens = self.pos_drop(tokens)
        
        # Generate learned BAM using A-BAMNet
        raw_bam = self.abam(x, heuristic_bam)  # (B, H, W)
        
        # Convert spatial BAM to patch-level attention weights
        ps = self.patch_size
        bam_patches = raw_bam.unfold(1, ps, ps).unfold(2, ps, ps)  # (B, H/P, W/P, P, P)
        bam_patches = bam_patches.contiguous().view(B, -1, ps * ps)  # (B, N, P²)
        mb = bam_patches.mean(dim=2)  # (B, N) - average over each patch
        mb = torch.sigmoid(mb)  # Normalize to [0, 1]
        
        # Pass through BIPA encoder blocks
        for block in self.blocks:
            tokens = block(tokens, mb)
        
        # Classification head
        cls_token = tokens[:, 0]  # Extract class token
        cls_token = self.norm(cls_token)
        logits = self.head(cls_token)
        
        return logits, raw_bam, mb

# ====================================================================
# SECTION 3: TRAINING FUNCTIONS
# ====================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels, heuristic_bams, paths in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        heuristic_bams = heuristic_bams.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        if USE_AMP and scaler is not None:
            with torch.cuda.amp.autocast():
                logits, raw_bam, mb = model(images, heuristic_bams)
                loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, raw_bam, mb = model(images, heuristic_bams)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc

def evaluate_model(model, loader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for images, labels, heuristic_bams, paths in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            heuristic_bams = heuristic_bams.to(device, non_blocking=True)
            
            # Measure inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            
            logits, raw_bam, mb = model(images, heuristic_bams)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            
            inference_times.append((t1 - t0) / images.shape[0])
            
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_infer_ms = np.mean(inference_times) * 1000
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'infer_ms': avg_infer_ms, 'predictions': all_preds, 'labels': all_labels
    }

def save_checkpoint(model, optimizer, epoch, metrics, filepath, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'img_size': IMG_SIZE,
            'patch_size': PATCH_SIZE,
            'embed_dim': EMBED_DIM,
            'depth': DEPTH,
            'num_heads': NUM_HEADS,
            'num_classes': NUM_CLASSES,
            'bam_mode': BAM_MODE,
        }
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = filepath.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)

# ====================================================================
# SECTION 4: MAIN TRAINING LOOP
# ====================================================================

def main():
    """Main training function"""
    
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    # Create model
    model = BIPA(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        drop_rate=DROP_RATE
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model initialized:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Number of patches: {NUM_PATCHES}")
    
    model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    if LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif LR_SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_metrics': [],
        'learning_rates': []
    }
    
    best_metric = -1.0
    epochs_no_improve = 0
    
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        
        # Evaluate
        val_metrics = evaluate_model(model, test_loader, DEVICE)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_metrics'].append(val_metrics)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Acc: {val_metrics['accuracy']*100:.2f}% | F1: {val_metrics['f1']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
        print(f"Inference: {val_metrics['infer_ms']:.2f} ms/image | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Track best model
        current_metric = val_metrics[EARLY_STOPPING_METRIC] if EARLY_STOPPING_METRIC != "loss" else -train_loss
        
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✓ Best model saved (F1: {val_metrics['f1']:.4f})")
        else:
            epochs_no_improve += 1
        
        # Periodic checkpoint
        if epoch % SAVE_CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, val_metrics, ckpt_path, is_best=is_best)
            print(f"✓ Checkpoint saved: {ckpt_path}")
        
        # Early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠ Early stopping triggered (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break
    
    # Save final model
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    
    # Save history
    history_path = os.path.join(RESULTS_DIR, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n✅ Best validation {EARLY_STOPPING_METRIC}: {best_metric:.4f}")
    print(f"\n📁 Saved files:")
    print(f"   1. Best model: {BEST_MODEL_PATH}")
    print(f"   2. Final model: {FINAL_MODEL_PATH}")
    print(f"   3. Training history: {history_path}")
    print(f"\n🎯 Next step: Run evaluation")
    print(f"   python 3_evaluation.py")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

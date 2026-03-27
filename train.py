import os
import csv
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ================= CONFIG =================
CONFIG = {
    "image_dir": r"C:\Users\SAI GOKUL\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Color_Images",
    "mask_dir": r"C:\Users\SAI GOKUL\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation",
    "save_path": "model.pth",
    "epochs": 60,  # Huge 60-epoch run for max accuracy
    "batch_size": 4,  # DeepLabV3+ with ResNet50 at 512x512 requires VRAM reduction
    "lr": 1e-3,
    "img_size": (512, 512),  # Doubled resolution to capture small rocks/logs
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ================= DATASET =================
class SegmentationDataset(Dataset):
    def __init__(self, images, masks, img_size, id_to_idx=None, augment=False):
        self.images = images
        self.masks = masks
        self.img_size = img_size
        self.augment = augment

        if augment:
            self.transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.2),
                A.GaussNoise(p=0.2),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        # Detect classes
        if id_to_idx is None:
            print("Detecting classes...")
            unique_vals = set()
            for m in tqdm(self.masks[:500]):  # Sample for speed, full scan was done before
                mask = np.array(Image.open(m))
                unique_vals.update(np.unique(mask))
            self.unique_values = sorted(list(unique_vals))
            self.id_to_idx = {val: idx for idx, val in enumerate(self.unique_values)}
        else:
            self.id_to_idx = id_to_idx
            self.unique_values = list(id_to_idx.keys())

        self.num_classes = len(self.id_to_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask_raw = np.array(Image.open(self.masks[idx]))
        
        # ID to Index mapping
        mask = np.zeros_like(mask_raw)
        for val, idx_map in self.id_to_idx.items():
            mask[mask_raw == val] = idx_map

        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long()


# ================= IoU =================
def calculate_iou(preds, labels, num_classes):
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0: continue
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0


# ================= TRAIN =================
def train():
    os.makedirs("train_stats", exist_ok=True)
    images = sorted(glob.glob(os.path.join(CONFIG["image_dir"], "*")))
    masks = sorted(glob.glob(os.path.join(CONFIG["mask_dir"], "*")))

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.1, random_state=42
    )

    train_dataset = SegmentationDataset(train_imgs, train_masks, CONFIG["img_size"], augment=True)
    val_dataset = SegmentationDataset(val_imgs, val_masks, CONFIG["img_size"], id_to_idx=train_dataset.id_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # Ultimate Architecture Optimization: DeepLabV3+ with ResNet50
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=train_dataset.num_classes,
    ).to(CONFIG["device"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["lr"], 
        steps_per_epoch=len(train_loader), 
        epochs=CONFIG["epochs"]
    )
    
    # Aggressive Class Weights to boost rare classes (Dry Grass, Rocks, Logs)
    class_weights = torch.tensor(
        [1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 5.0, 5.0, 1.0, 1.0], 
        dtype=torch.float32
    ).to(CONFIG["device"])
    
    criterion_ce = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion_dice = smp.losses.DiceLoss(mode='multiclass')

    best_iou = 0
    log_path = os.path.join("train_stats", "log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "val_iou", "best_iou", "elapsed_sec"])

    train_start = time.time()
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision Training

    print(f"\n🚀 Starting Ultimate 60-Epoch Training Run (DeepLabV3+ ResNet50 | 512x512)")
    print(f"Goal: Maximize mIoU | Max Epochs: 60 | Batch: {CONFIG['batch_size']}")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(CONFIG["device"]), masks.to(CONFIG["device"])
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(): # Mixed Precision
                outputs = model(imgs)
                loss = 0.5 * criterion_ce(outputs, masks.long()) + 0.5 * criterion_dice(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Evaluation
        model.eval()
        total_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(CONFIG["device"]), masks.to(CONFIG["device"])
                outputs = model(imgs)
                total_iou += calculate_iou(outputs, masks, train_dataset.num_classes)

        mean_iou = total_iou / len(val_loader)
        print(f"Validation mIoU: {mean_iou:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save({
                'model_state': model.state_dict(),
                'id_to_idx': train_dataset.id_to_idx,
                'num_classes': train_dataset.num_classes,
                'unique_values': train_dataset.unique_values,
                'arch': 'DeepLabV3Plus',
                'encoder': 'resnet50'
            }, CONFIG["save_path"])
            print("🌟 Best model milestone hit!")

        elapsed = time.time() - train_start
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, f"{total_loss/len(train_loader):.4f}", f"{mean_iou:.4f}", f"{best_iou:.4f}", f"{elapsed:.1f}"])

    print(f"\n✅ Leap to 0.70 Challenge Completed. Best mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    train()
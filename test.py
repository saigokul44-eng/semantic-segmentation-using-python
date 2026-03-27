import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import ttach as tta

# Configuration
CONFIG = {
    "test_image_dir": r"C:\Users\SAI GOKUL\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Color_Images",
    "test_mask_dir": r"C:\Users\SAI GOKUL\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Segmentation",
    "model_path": "model.pth",
    "img_size": (512, 512),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "test_results"
}

# Class names for reporting (based on dataset documentation)
CLASS_NAMES = [
    'Flowers', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# ============================================================================
# Dataset Logic (Matches Train)
# ============================================================================

class TestSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size, id_to_idx, unique_values=None):
        self.images = sorted(glob.glob(os.path.join(image_dir, "*")))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, "*")))
        self.img_size = img_size
        self.id_to_idx = id_to_idx
        self.unique_values = unique_values if unique_values is not None else sorted(list(id_to_idx.keys()))

        if len(self.images) == 0:
            raise FileNotFoundError(f"No test images found in {image_dir}")
        if len(self.masks) == 0:
            raise FileNotFoundError(f"No test masks found in {mask_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        filename = os.path.basename(img_path)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Preprocessing & Resize
        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        # Map mask values
        mask_array = np.array(mask)
        new_mask = np.zeros_like(mask_array, dtype=np.int64)
        for val, idx_mapping in self.id_to_idx.items():
            new_mask[mask_array == val] = idx_mapping

        # Conversion
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        mask = torch.from_numpy(new_mask).long()

        return image, mask, filename

# ============================================================================
# Metrics calculation
# ============================================================================

def compute_metrics(pred, target, num_classes):
    """Compute mIoU and Pixel Accuracy."""
    pred = torch.argmax(pred, dim=1)
    
    # Pixel Accuracy
    correct = (pred == target).sum().item()
    total_pixels = target.numel()
    pixel_acc = correct / total_pixels

    # IoU per class
    ious = []
    cls_ious = {}
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        intersection = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            iou = intersection / union
            ious.append(iou)
            cls_ious[cls] = iou
        else:
            cls_ious[cls] = float('nan')
    
    miou = np.mean(ious) if ious else 0
    return miou, pixel_acc, cls_ious

# ============================================================================
# Evaluation & Visualization
# ============================================================================

def visualize(image_tensor, mask_tensor, pred_tensor, filename, save_dir, mapping):
    """Create a side-by-side visualization."""
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean) * 255
    img = img.astype(np.uint8)

    # Convert masks to colored versions
    cmap = plt.get_cmap("tab10")
    num_classes = len(mapping)

    def mask_to_color(m):
        colored = np.zeros((*m.shape, 3), dtype=np.uint8)
        for i in range(num_classes):
            colored[m == i] = np.array(cmap(i % 10)[:3]) * 255
        return colored

    gt_colored = mask_to_color(mask_tensor.cpu().numpy())
    pred_colored = mask_to_color(pred_tensor.cpu().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[1].imshow(gt_colored)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_colored)
    axes[2].set_title("Prediction")
    
    for ax in axes: ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_{filename}"))
    plt.close()

def test():
    if not os.path.exists(CONFIG["model_path"]):
        print(f"Model file {CONFIG['model_path']} not found. Please train first.")
        return

    # Load model with metadata
    checkpoint = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    
    # Metadata support for different architectures
    arch = checkpoint.get('arch', 'Unet')
    encoder = checkpoint.get('encoder', 'resnet34')
    num_classes = checkpoint['num_classes']
    id_to_idx = checkpoint['id_to_idx']
    unique_values = checkpoint['unique_values']

    if arch == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        ).to(CONFIG["device"])
    else:
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        ).to(CONFIG["device"])

    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Dataset
    dataset = TestSegmentationDataset(
        CONFIG["test_image_dir"], 
        CONFIG["test_mask_dir"], 
        CONFIG["img_size"],
        id_to_idx,
        unique_values
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    total_miou, total_pix_acc = 0, 0
    all_cls_ious = []
    
    print(f"Evaluating {len(dataset)} images...")

    with torch.no_grad():
        for i, (image, mask, filename) in enumerate(tqdm(loader)):
            image_dev = image.to(CONFIG["device"])
            mask_dev = mask.to(CONFIG["device"])

            # TTA: Flip Left-Right inference
            tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.flip_transform(), merge_mode='mean')
            output = tta_model(image_dev)
            
            miou, pix_acc, cls_ious = compute_metrics(output, mask_dev, num_classes)
            total_miou += miou
            total_pix_acc += pix_acc
            all_cls_ious.append(cls_ious)

            # Save visualization for first few samples
            if i < 10:
                pred = torch.argmax(output, dim=1).squeeze(0)
                visualize(image[0], mask[0], pred, filename[0], CONFIG["output_dir"], id_to_idx)

    avg_miou = total_miou / len(loader)
    avg_pix_acc = total_pix_acc / len(loader)
    
    # Calculate Per-Class IoU average
    final_cls_ious = {}
    for cls in range(num_classes):
        cls_vals = [d[cls] for d in all_cls_ious if not np.isnan(d[cls])]
        final_cls_ious[cls] = np.mean(cls_vals) if cls_vals else 0

    results_text = []
    results_text.append("="*50)
    results_text.append("FINAL EVALUATION RESULTS")
    results_text.append("="*50)
    results_text.append(f"Overall mIoU:     {avg_miou:.4f}")
    results_text.append(f"Pixel Accuracy:   {avg_pix_acc:.4f}")
    results_text.append("-" * 30)
    results_text.append("Per-Class IoU Details:")
    
    # Try to map to real names if possible
    for cls, iou in final_cls_ious.items():
        val = unique_values[cls] if cls < len(unique_values) else f"ID_{cls}"
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class_{cls}"
        results_text.append(f"  [{cls}] {name:<15} (Val {val:>5}): {iou:.4f}")
    
    results_text.append("="*50)
    
    summary = "\n".join(results_text)
    print("\n" + summary)
    
    with open("evaluation_report.txt", "w") as f:
        f.write(summary)
    
    print(f"\nReport saved to: evaluation_report.txt")
    print(f"Visualizations saved to: {CONFIG['output_dir']}/")

if __name__ == "__main__":
    test()


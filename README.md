# COSMOS-2026 Off-Road Segmentation (Team NAG_DEV)

## Overview
This repository contains a **precision real-time semantic segmentation pipeline** for off‑road desert imagery. Developed for the COSMOS-2026 Hackathon, we utilize an ultimate **DeepLabV3+ architecture with a ResNet50 encoder** running at **512x512 high-resolution**.

To solve severe dataset class imbalances (such as extremely rare `Dry Grass` and `Logs` instances), we deployed a heavily customized **hybrid Dice + Explicitly Weighted CrossEntropy Loss**, applying a `5x` penalty multiplier to our minority classes.

## Final 60‑Epoch Ultimate Results
- **Overall mIoU:** `0.4818`
- **Pixel Accuracy:** `82.63%`
- **Per‑class IoU** (10 classes):

| ID | Class Name | IoU | Improvment Notes |
|---|---|---|---|
| 0 | Flowers | 0.5908 | |
| 1 | Trees | 0.4253 | |
| 2 | Lush Bushes | 0.5453 | |
| 3 | Dry Grass | 0.0811 | *(+335% Boost via Class Weighting)* |
| 4 | Dry Bushes | 0.3068 | *(+266% Boost via Class Weighting)* |
| 5 | Ground Clutter | 0.4786 | |
| 6 | Logs | 0.1604 | |
| 7 | Rocks | 0.3048 | |
| 8 | Landscape | 0.5584 | |
| 9 | Sky | 0.9621 | |

*The massive DeepLabV3+ model checkpoint (`model.pth`) is ~ 102 MB and is **ignored by Git** due to 100MB limits (see Download Link below).*

## Repository Structure
```
├── train.py                 # Ultimate 60-epoch training script
├── test.py                  # Evaluation script (outputs to test_results/)
├── generate_pdf_report.py   # Code to generate the 8-page Hackathon PDF
├── Hackathon_Report_NAG_DEV.pdf # Official structured hackathon report
├── evaluation_report.txt    # Plain-text metric outputs
├── README.md                # You are reading this!
├── requirements.txt         # Dependencies
└── test_results/            # Visual comparisons of the model predictions
```

## Quick Start
1. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
2. **Download the Pre-trained Weights**:
   Download `model.pth` from the Google Drive link below and place it in this directory.
3. **Evaluate the model** (generates visual results in `test_results/`):
   ```powershell
   python test.py
   ```
4. **Generate the PDF report**:
   ```powershell
   python generate_pdf_report.py
   ```

## Model Weights Download
[Google Drive link to `model.pth` (102 MB)](https://drive.google.com/file/d/1jusTtOryMSkkMnUv0u8bh7-AWYIMkrgq/view?usp=drive_link)

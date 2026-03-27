# Offroad Segmentation Project

## Overview
This repository contains a **real‑time semantic segmentation pipeline** for off‑road imagery. We use a lightweight **ResNet34‑Unet** model trained with a hybrid **Dice + Focal loss** and an **OneCycleLR** scheduler. The goal is to achieve high accuracy while keeping inference speed > 50 FPS, which is ideal for embedded robotics.

## Final 50‑Epoch Results
- **Overall mIoU:** `0.4364`
- **Pixel Accuracy:** `83.13%`
- **Per‑class IoU** (10 classes):
  | ID | Class Name | IoU |
  |---|---|---|
  | 0 | Background | 0.5530 |
  | 1 | Trees | 0.4096 |
  | 2 | Lush Bushes | 0.5667 |
  | 3 | Dry Grass | 0.0242 |
  | 4 | Dry Bushes | 0.1150 |
  | 5 | Ground Clutter | 0.3211 |
  | 6 | Logs | 0.1392 |
  | 7 | Rocks | 0.2182 |
  | 8 | Landscape | 0.5793 |
  | 9 | Sky | 0.9572 |

The model checkpoint (`model.pth`) is ~ 98 MB and is **ignored by Git** (see `.gitignore`).

## Repository Structure
```
Offroad_Segmentation_Scripts/
│   README.md                # <-- you are reading this file
│   train.py                 # training script (50‑epoch run)
│   test.py                  # evaluation script (uses TTA)
│   generate_pdf_report.py   # creates the PDF report (now with confusion matrix)
│   Offroad_Segmentation_Final_Report.pdf   # final PDF with results
│   evaluation_report.txt    # plain‑text summary of metrics
│   .gitignore               # excludes large files from Git
│   model.pth                # final checkpoint (ignored by Git)
│
├───train_stats/            # CSV log of loss / IoU per epoch
├───test_results/           # visualisations of predictions
├───datasets/               # (optional) dataset folder structure
└───...                     # other helper scripts
```

## Quick Start (Beginner Friendly)
1. **Install dependencies** (run once):
   ```powershell
   pip install -r requirements.txt   # if you have a requirements file
   pip install torch torchvision segmentation-models-pytorch tqdm albumentations fpdf ttach
   ```
2. **Run the training** (the 50‑epoch run is already completed, but you can retrain):
   ```powershell
   python train.py
   ```
   The script will automatically save the best checkpoint to `model.pth`.
3. **Evaluate the model** (generates visual results in `test_results/`):
   ```powershell
   python test.py
   ```
4. **Generate the PDF report** (creates `Offroad_Segmentation_Final_Report.pdf`):
   ```powershell
   python generate_pdf_report.py
   ```
## Hackathon Submission Checklist

**Files you must include in the GitHub repository (these are the only ones judges need to run the code):**

- `train.py` – training script (50‑epoch run, saves `model.pth`).
- `test.py` – evaluation script (produces per‑class IoU and `test_results/` images).
- `generate_pdf_report.py` – creates the final PDF report (includes confusion matrix).
- `requirements.txt` – all Python dependencies.
- `README.md` – this documentation.
- `evaluation_report.txt` – plain‑text summary of final metrics.
- `Offroad_Segmentation_Final_Report.pdf` – polished PDF report.
- `test_results/` – a few representative prediction PNGs (e.g., 5 samples).
- `train_stats/` – `log.csv` with loss/IoU per epoch.

**Large files (ignored by `.gitignore`):**

- `model.pth` – final model checkpoint (~98 MB).
- `Offroad_Segmentation_Complete_Project_v2.zip` – full package (code + checkpoint + data).  
  Upload this zip to Google Drive/OneDrive and add the shareable link below.

### Download Full Package (optional)

[Google Drive link to the full zip (≈ 100 MB)](https://drive.google.com/your‑link‑here)

---

### Quick GitHub upload (for beginners)

1. Create a new repo on GitHub.
2. Click **Add file → Upload files** and select everything **except** `model.pth` and any `*.zip`.
3. Commit with a message like `Initial hackathon submission – 50‑epoch results`.
4. Verify the `.gitignore` prevents large files from being uploaded.

You’re now ready to submit!

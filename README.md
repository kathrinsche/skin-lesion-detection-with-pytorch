# Skin Lesion Image Classifier

Small, reproducible PyTorch project for a first local test and a later realistic training run.

## Approach

- Python + PyTorch + torchvision
- Transfer learning with pretrained ResNet-18
- HAM10000 metadata/images
- Patient-level train/validation/test split
- Configurable image size, batch size, epochs, learning rate, augmentation, freezing and class weighting
- Metrics: accuracy, macro F1, balanced accuracy, sensitivity/recall per class, specificity per class, confusion matrix
- Checkpointing and CSV output

This is a technical research prototype, not a clinical diagnostic system.

## 1. Environment

Recommended: Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

## 2. Dataset

Download HAM10000 from its official/public distribution and place:

```text
data/raw/HAM10000/
├── HAM10000_metadata.csv
└── images/
    ├── ISIC_0024306.jpg
    ├── ...
```

The code expects the metadata columns `lesion_id`, `image_id`, `dx`, and preferably `patient_id`.

The implementation performs the split at patient level when `patient_id` is available. This is important for avoiding leakage between train/validation/test.

## 3. Minimal test

```bash
python src/train.py --config configs/minimal.yaml
python src/evaluate.py --config configs/minimal.yaml --checkpoint checkpoints/best.pt
```

The minimal configuration is deliberately small and is intended to verify the complete pipeline, not to produce scientifically meaningful performance estimates.

## 4. Realistic run

```bash
python src/train.py --config configs/realistic.yaml
python src/evaluate.py --config configs/realistic.yaml --checkpoint checkpoints/best.pt
```

Before a scientific experiment, inspect the dataset composition, class imbalance, split sizes and preprocessing choices.

## 5. Useful extensions

The project is intentionally structured so that these can be added later without changing the basic workflow:

- binary classification
- additional architectures
- repeated patient-level splits / cross-validation
- calibration
- ROC/PR curves
- experiment tracking
- Docker/Secure Processing Environment integration
- external test set

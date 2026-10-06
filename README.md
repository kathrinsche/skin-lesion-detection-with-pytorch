# Skin Lesion Image Classifier

Small, reproducible PyTorch project for a first local test and a later realistic training run.

## Approach

- Python + PyTorch + torchvision
- Transfer learning with pretrained ResNet-18
- HAM10000 metadata/images
- Patient-level train/validation/test split
- Configurable image size, batch size, epochs, learning rate, augmentation, freezing and class weighting
- Metrics: accuracy, macro F1, balanced accuracy, sensitivity/recall per class, specificity per class, confusion matrix
- Checkpointing and CSV/JSON output
- Separation of application code, experiment configuration, input data and output data
- Local execution or execution inside a Singularity container

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

The HAM10000 dataset is stored outside the project directory. The input directory is passed to the application at runtime.

The expected input directory structure is:

```text
<INPUT_DIR>/
├── HAM10000_metadata.csv
└── images/
    ├── ISIC_0024306.jpg
    ├── ISIC_0024307.jpg
    └── ...
```

The code expects the metadata columns `image_id`, `dx`, and preferably `patient_id`.

The implementation performs the split at patient level when `patient_id` is available. This is important for avoiding leakage between train, validation and test data.

The physical location of `<INPUT_DIR>` is not defined in the YAML configuration. This allows the same experiment configuration to be used locally and inside a container.

## 3. Input and output directories

Input and output directories are runtime parameters:

```text
--input-dir <INPUT_DIR>
--output-dir <OUTPUT_DIR>
```

The YAML configuration only contains paths relative to these directories.

For example:

```yaml
data:
  metadata_file: HAM10000_metadata.csv
  image_subdir: images

output:
  checkpoint: checkpoints/best.pt
  metrics: results/training_metrics.csv
  split: results/data_split.csv
```

For an output directory such as:

```text
/home/user/training_runs/run01
```

the application creates:

```text
/home/user/training_runs/run01/
├── checkpoints/
│   └── best.pt
└── results/
    ├── data_split.csv
    ├── training_metrics.csv
    └── test_metrics.json
```

## 4. Minimal local test

The minimal configuration is deliberately small and is intended to verify the complete pipeline, not to produce scientifically meaningful performance estimates.

Training:

```bash
.venv/bin/python src/train.py \
    --config configs/minimal.yaml \
    --input-dir /path/to/HAM10000 \
    --output-dir /path/to/output
```

Evaluation:

```bash
.venv/bin/python src/evaluate.py \
    --config configs/minimal.yaml \
    --input-dir /path/to/HAM10000 \
    --output-dir /path/to/output
```

The evaluation loads the checkpoint specified in `configs/minimal.yaml` relative to `--output-dir`.

## 5. Realistic local run

Training:

```bash
.venv/bin/python src/train.py \
    --config configs/realistic.yaml \
    --input-dir /path/to/HAM10000 \
    --output-dir /path/to/output
```

Evaluation:

```bash
.venv/bin/python src/evaluate.py \
    --config configs/realistic.yaml \
    --input-dir /path/to/HAM10000 \
    --output-dir /path/to/output
```

Before a scientific experiment, inspect the dataset composition, class imbalance, split sizes and preprocessing choices.

## 6. Singularity container

The application can also be built and executed inside a Singularity container.

The application code, Python dependencies and YAML configuration are included in the container image.

Input data and output data are **not** stored in the container. They are provided at runtime using bind mounts:

```text
Host input directory  -> /input  (read-only)
Host output directory -> /output (read-write)
```

This separates the application from the physical storage locations.

### 6.1 Build the container

Run the build command from the project root:

```bash
cd /path/to/skin_lesion_classifier
```

Build the container:

```bash
sudo singularity build container.sif configs/container.def
```

Depending on the Singularity installation and system configuration, building without `sudo` may also be supported.

The resulting container image is:

```text
container.sif
```

The SIF file should normally not be committed to Git. Add the following entry to `.gitignore`:

```gitignore
*.sif
```

### 6.2 Run minimal training inside the container

Assume the host directories are:

```text
Input:
/data/datasets/HAM10000

Output:
/data/training_runs/run01
```

Run:

```bash
singularity run \
    --bind /data/datasets/HAM10000:/input:ro \
    --bind /data/training_runs/run01:/output:rw \
    container.sif \
    --config /app/configs/minimal.yaml
```

Inside the container, the application sees:

```text
/input
/output
```

The actual host paths are therefore independent of the application configuration.

The input dataset is mounted read-only (`ro`). The output directory is mounted read-write (`rw`).

### 6.3 Run realistic training inside the container

```bash
singularity run \
    --bind /data/datasets/HAM10000:/input:ro \
    --bind /data/training_runs/run02:/output:rw \
    container.sif \
    --config /app/configs/realistic.yaml
```

The same container image can therefore be used for minimal and realistic experiments. Only the configuration and runtime directories change.

### 6.4 General container invocation

The general pattern is:

```bash
singularity run \
    --bind <HOST_INPUT_DIR>:/input:ro \
    --bind <HOST_OUTPUT_DIR>:/output:rw \
    container.sif \
    --config /app/configs/<CONFIG>.yaml
```

The container runscript maps the container directories to the Python application as:

```text
--input-dir /input
--output-dir /output
```

Therefore, the host-specific paths do not appear in the YAML configuration or Python code.

## 7. Project structure

```text
skin_lesion_classifier/
├── configs/
│   ├── container.def
│   ├── minimal.yaml
│   └── realistic.yaml
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── model.py
│   └── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

Input datasets, training outputs, checkpoints and container images are kept outside the source repository.

## 8. Useful extensions

The project is intentionally structured so that these can be added later without changing the basic workflow:

- binary classification
- additional architectures
- repeated patient-level splits / cross-validation
- calibration
- ROC/PR curves
- experiment tracking
- Secure Processing Environment integration
- external test set
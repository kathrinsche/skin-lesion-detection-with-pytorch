import argparse
from pathlib import Path
import json
import yaml
import torch
from torch.utils.data import DataLoader

from data import (
    CLASS_NAMES,
    HAM10000Dataset,
    load_metadata,
    make_transforms,
    patient_split,
)
from metrics import classification_metrics
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the input dataset"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for evaluation results"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    df = load_metadata(cfg, input_dir)
    split_df = patient_split(
        df,
        cfg["seed"],
        cfg["data"]["train_fraction"],
        cfg["data"]["val_fraction"],
        cfg["data"]["test_fraction"],
    )
    test_df = split_df[split_df["split"] == "test"].copy()

    dataset = HAM10000Dataset(test_df, make_transforms(cfg, train=False))
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = output_dir / cfg["output"]["checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, CLASS_NAMES)

    print(json.dumps(metrics, indent=2))

    out = output_dir / "results/test_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

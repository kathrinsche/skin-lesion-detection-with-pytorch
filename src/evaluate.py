import argparse
from pathlib import Path
import json
import yaml
import torch
from torch.utils.data import DataLoader

from .data import (
    CLASS_NAMES,
    HAM10000Dataset,
    load_metadata,
    make_transforms,
    patient_split,
)
from .metrics import classification_metrics
from .model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = load_metadata(cfg)
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
    checkpoint = torch.load(args.checkpoint, map_location=device)

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

    out = Path("results/test_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

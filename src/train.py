import argparse
from pathlib import Path
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    CLASS_NAMES,
    HAM10000Dataset,
    class_weights,
    load_metadata,
    make_transforms,
    patient_split,
    seed_everything,
)
from .metrics import classification_metrics
from .model import build_model


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, CLASS_NAMES)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])

    df = load_metadata(cfg)
    split_df = patient_split(
        df,
        cfg["seed"],
        cfg["data"]["train_fraction"],
        cfg["data"]["val_fraction"],
        cfg["data"]["test_fraction"],
    )

    Path(cfg["output"]["split_csv"]).parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(cfg["output"]["split_csv"], index=False)

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()

    train_ds = HAM10000Dataset(train_df, make_transforms(cfg, train=True))
    val_ds = HAM10000Dataset(val_df, make_transforms(cfg, train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(split_df[split_df['split']=='test'])}")

    model = build_model(
        num_classes=len(CLASS_NAMES),
        pretrained=cfg["training"]["pretrained"],
        freeze_backbone=cfg["training"]["freeze_backbone"],
    ).to(device)

    weights = class_weights(train_df).to(device) if cfg["training"]["class_weighting"] else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    history = []
    best_f1 = -1.0
    patience = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}")
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(row)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            patience = 0
            Path(cfg["output"]["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "config": cfg,
                "best_val_macro_f1": best_f1,
            }, cfg["output"]["checkpoint"])
            print("Saved best checkpoint.")
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping.")
                break

    Path(cfg["output"]["metrics_csv"]).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(cfg["output"]["metrics_csv"], index=False)


if __name__ == "__main__":
    main()

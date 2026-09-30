import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_metadata(cfg, input_dir):
    input_dir = Path(input_dir)
    csv_path = input_dir / cfg["data"]["metadata_file"]
    image_dir = input_dir / cfg["data"]["image_subdir"]

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {csv_path}"
        )

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    df = pd.read_csv(csv_path)

    required = {"image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    df["image_path"] = df["image_id"].map(lambda x: str(image_dir / f"{x}.jpg"))
    exists = df["image_path"].apply(os.path.exists)
    my_df = df[exists].copy()
    df = my_df

    if df.empty:
        raise FileNotFoundError("No images found. Check image_dir and image_id filenames.")

    if "patient_id" not in df.columns:
        # Fallback only when the metadata does not contain patient IDs.
        # For a scientific evaluation, obtain patient IDs rather than using this fallback.
        df["patient_id"] = df["image_id"]

    return df


def patient_split(df, seed, train_fraction, val_fraction, test_fraction):
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.")

    groups = df["patient_id"].astype(str)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=groups))
    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    relative_val = val_fraction / (train_fraction + val_fraction)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_idx, val_idx = next(gss2.split(train_val, groups=train_val["patient_id"].astype(str)))

    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        part["split"] = split_name

    return pd.concat([train, val, test], ignore_index=True)


def make_transforms(cfg, train=False):
    size = cfg["data"]["image_size"]
    aug = cfg["augmentation"]

    if train:
        ops = [
            transforms.Resize((size, size)),
        ]
        if aug.get("horizontal_flip", False):
            ops.append(transforms.RandomHorizontalFlip())
        if aug.get("vertical_flip", False):
            ops.append(transforms.RandomVerticalFlip())
        if aug.get("rotation_degrees", 0) > 0:
            ops.append(transforms.RandomRotation(aug["rotation_degrees"]))
        if aug.get("color_jitter", False):
            ops.append(transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02
            ))
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        return transforms.Compose(ops)

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class HAM10000Dataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[row["dx"]]
        return image, label


def class_weights(train_df):
    counts = train_df["dx"].value_counts()
    weights = []
    for name in CLASS_NAMES:
        weights.append(len(train_df) / (len(CLASS_NAMES) * max(counts.get(name, 0), 1)))
    return torch.tensor(weights, dtype=torch.float32)

import os
import csv
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image

NUM_WORKERS = 0 if os.name == "nt" else 4
INPUT_SIZE   = 448

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

SEX_CLASSES       = ["male", "female", "unknown"]
MALE_FORM_CLASSES = ["major", "minor", "intermediate", "unknown"]


def _load_sex_labels(csv_path: str) -> dict[str, dict]:
    """sex_labels.csv → {split/species/filename: {sex, male_form}} 딕셔너리.
    앞 부분(data/final, data/final_segmented 등)을 제거해 전처리 모드에 무관하게 매칭."""
    labels = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parts = Path(row["local_path"]).parts
            key = "/".join(parts[-3:]) if len(parts) >= 3 else Path(row["local_path"]).as_posix()
            labels[key] = {
                "sex":       row["sex"],
                "male_form": row["male_form"],
            }
    return labels


class MultiTaskDataset(Dataset):
    """
    ImageFolder 구조를 읽되 sex/male_form 라벨을 CSV에서 추가로 붙인다.
    unknown 라벨은 -1로 인코딩 → trainer에서 masked loss 적용.
    """

    def __init__(self, data_dir: Path, transform, sex_labels: dict):
        self._base   = datasets.ImageFolder(data_dir, transform=transform)
        self.classes = self._base.classes
        self.targets = self._base.targets
        self._sex_labels = sex_labels
        self.transform   = transform

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        img_path, species_label = self._base.samples[idx]
        img  = Image.open(img_path).convert("RGB")
        img  = self.transform(img)

        # CSV has relative paths (data/final/split/species/file), ImageFolder gives absolute paths
        parts = Path(img_path).parts
        key   = "/".join(parts[-3:]) if len(parts) >= 3 else Path(img_path).as_posix()
        info  = self._sex_labels.get(key, {"sex": "unknown", "male_form": "unknown"})

        sex_idx  = SEX_CLASSES.index(info["sex"]) if info["sex"] in SEX_CLASSES else 2
        form_idx = MALE_FORM_CLASSES.index(info["male_form"]) if info["male_form"] in MALE_FORM_CLASSES else 3

        # unknown → -1 (masked loss에서 무시)
        sex_label  = -1 if info["sex"]       == "unknown" else sex_idx
        form_label = -1 if info["male_form"] == "unknown" else form_idx

        return img, species_label, sex_label, form_label


def get_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    mode: str = "species_only",
    sex_labels_csv: str = "data/labels/sex_labels.csv",
):
    base_path   = Path(data_dir)
    multi_task  = mode in ("multi_task", "sex_only", "form_only")
    sex_labels  = _load_sex_labels(sex_labels_csv) if multi_task and Path(sex_labels_csv).exists() else {}

    def make_loader(split, shuffle):
        transform = TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM
        if multi_task:
            ds = MultiTaskDataset(base_path / split, transform, sex_labels)
        else:
            ds = datasets.ImageFolder(base_path / split, transform)
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = NUM_WORKERS,
            pin_memory  = torch.cuda.is_available(),
        )

    dataloaders = {
        "train": make_loader("train", shuffle=True),
        "val":   make_loader("val",   shuffle=False),
    }
    class_names = dataloaders["train"].dataset.classes
    return dataloaders, class_names


def get_test_dataloader(data_dir: str, batch_size: int = 16):
    test_dataset = datasets.ImageFolder(Path(data_dir) / "test", EVAL_TRANSFORM)
    return DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = torch.cuda.is_available(),
    ), test_dataset.classes

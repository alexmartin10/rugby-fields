"""
Script that builds the yolo dataset when given one the one hand the
positive dataset with correct annotations and on the other hand the 
negative dataset (with or without annotations).
"""

from pathlib import Path
import random
import shutil

# À adapter
POS_IMAGES_DIR = Path("data/yolo_positives/images")
POS_LABELS_DIR = Path("data/yolo_positives/labels")

NEG_IMAGES_DIR = Path("data/yolo_negatives/images")
NEG_LABELS_DIR = Path("data/yolo_negatives/labels")  # peut ne pas exister

OUT_DIR = Path("data/yolo_dataset")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}

random.seed(42)


def collect_pairs(images_dir: Path, labels_dir: Path | None):
    pairs = []

    for image_path in images_dir.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = None
        if labels_dir is not None:
            candidate = labels_dir / f"{image_path.stem}.txt"
            if candidate.exists():
                label_path = candidate

        pairs.append((image_path, label_path))

    return pairs


def split_pairs(pairs):
    pairs = pairs.copy()
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    return {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }


def copy_pairs(pairs, split_name, prefix):
    images_out = OUT_DIR / "images" / split_name
    labels_out = OUT_DIR / "labels" / split_name

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for idx, (image_path, label_path) in enumerate(pairs):
        new_name = f"{prefix}_{idx:05d}"
        new_image_path = images_out / f"{new_name}{image_path.suffix.lower()}"
        new_label_path = labels_out / f"{new_name}.txt"

        shutil.copy2(image_path, new_image_path)

        if label_path is not None:
            shutil.copy2(label_path, new_label_path)
        else:
            new_label_path.write_text("")


def main():
    if OUT_DIR.exists():
        raise FileExistsError(f"{OUT_DIR} existe déjà. Supprime-le ou change OUT_DIR.")

    positives = collect_pairs(POS_IMAGES_DIR, POS_LABELS_DIR)
    negatives = collect_pairs(
        NEG_IMAGES_DIR,
        NEG_LABELS_DIR if NEG_LABELS_DIR.exists() else None
    )

    print(f"Positifs trouvés : {len(positives)}")
    print(f"Négatifs trouvés : {len(negatives)}")

    missing_pos_labels = sum(1 for _, label in positives if label is None)
    print(f"Labels positifs manquants : {missing_pos_labels}")

    pos_splits = split_pairs(positives)
    neg_splits = split_pairs(negatives)

    for split_name in SPLIT_RATIOS:
        copy_pairs(pos_splits[split_name], split_name, "pos")
        copy_pairs(neg_splits[split_name], split_name, "neg")

    for split_name in SPLIT_RATIOS:
        print(
            f"{split_name}: "
            f"{len(pos_splits[split_name])} positifs, "
            f"{len(neg_splits[split_name])} négatifs"
        )


if __name__ == "__main__":
    main()
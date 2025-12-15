import random
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).parent
TRAIN_ZIP = BASE_DIR / "train_data.zip"
TEST_ZIP = BASE_DIR / "test_data.zip"

TRAIN_DIR = BASE_DIR / "train_data"
TEST_DIR = BASE_DIR / "test_data"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def unzip_if_needed(zip_path: Path, target_dir: Path):
    if target_dir.exists():
        print(f"[INFO] {target_dir.name} already exists, skipping unzip.")
        return

    print(f"[INFO] Unzipping {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    nested_dir = target_dir / target_dir.name
    if nested_dir.exists() and nested_dir.is_dir():
        for item in nested_dir.iterdir():
            item.rename(target_dir / item.name)
        nested_dir.rmdir()

    print(f"[INFO] Done unzipping to {target_dir}")


def get_image_paths(root_dir: Path):
    return list(root_dir.glob("*/*.jpg"))


def get_class_names(root_dir: Path):
    return sorted([d.name for d in root_dir.iterdir() if d.is_dir()])


def inspect_image(image_path: Path):
    with Image.open(image_path) as img:
        return img.size, img.mode  # (width, height), channels mode


def plot_class_distribution(counter: Counter, title: str):
    plt.figure(figsize=(12, 4))
    plt.bar(counter.keys(), counter.values())
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution.png")
    plt.close()


def show_samples_per_class(root_dir: Path, samples_per_class=2):
    class_dirs = get_class_names(root_dir)

    plt.figure(figsize=(samples_per_class * 3, len(class_dirs) * 3))
    plot_idx = 1

    for cls in class_dirs:
        images = list((root_dir / cls).glob("*.jpg")) + \
            list((root_dir / cls).glob("*.png"))
        samples = random.sample(images, min(samples_per_class, len(images)))

        for img_path in samples:
            img = Image.open(img_path)
            plt.subplot(len(class_dirs), samples_per_class, plot_idx)
            plt.imshow(img)
            plt.axis("off")
            plt.title(cls)
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "samples_per_class.png")
    plt.close()


if __name__ == "__main__":
    # Unzip datasets
    unzip_if_needed(TRAIN_ZIP, TRAIN_DIR)
    unzip_if_needed(TEST_ZIP, TEST_DIR)

    # Collect paths
    train_images = get_image_paths(TRAIN_DIR)
    test_images = get_image_paths(TEST_DIR)

    print(f"\n[DATASET SIZE]")
    print(f"Train images: {len(train_images)}")
    print(f"Test images:  {len(test_images)}")

    # Classes
    classes = get_class_names(TRAIN_DIR)
    print(f"\n[CLASSES]")
    print(f"Number of classes: {len(classes)}")
    print(classes)

    # Image properties
    sample_size, sample_mode = inspect_image(train_images[0])
    print(f"\n[SAMPLE IMAGE INFO]")
    print(f"Resolution (W x H): {sample_size}")
    print(f"Color mode: {sample_mode}")

    # Class balance
    train_labels = [p.parent.name for p in train_images]
    class_counts = Counter(train_labels)

    print(f"\n[CLASS DISTRIBUTION]")
    for cls, cnt in class_counts.items():
        print(f"{cls}: {cnt}")

    plot_class_distribution(class_counts, "Training set class distribution")

    # Show example images
    print("\n[SHOWING SAMPLE IMAGES]")
    show_samples_per_class(TRAIN_DIR, samples_per_class=2)

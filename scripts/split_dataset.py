"""
Re-splits the YOLOWaste dataset into a class-balanced train/val/test split.

What it does:
1. Gathers every image+label pair currently spread across data/images/{train,val,test}
   and data/labels/{train,val,test}.
2. For each image, determines a "dominant class" = the class with the most
   labeled instances in that image's label file.
3. Uses stratified splitting (based on dominant class) to create new
   train/val/test sets, so each class is proportionally represented in
   every split.
4. Copies files into fresh data/images/{train,val,test} and
   data/labels/{train,val,test} folders (old contents are cleared first).

Run from the project root: A:\\YOLOWaste
    python scripts/split_dataset.py
"""

import os
import shutil
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# ---- CONFIG ----
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def gather_all_pairs():
    """Collect every (image_path, label_path) pair across the existing splits."""
    pairs = []
    for split in SPLITS:
        img_dir = os.path.join(IMAGES_DIR, split)
        lbl_dir = os.path.join(LABELS_DIR, split)
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            name_no_ext = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, name_no_ext + ".txt")
            if os.path.exists(lbl_path):
                pairs.append((img_path, lbl_path))
            else:
                print(f"WARNING: no label found for {img_path}, skipping")
    return pairs


def dominant_class(label_path):
    """Return the class id with the most instances in this label file."""
    counts = Counter()
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id = line.split()[0]
            counts[class_id] += 1
    if not counts:
        return "empty"
    return counts.most_common(1)[0][0]


def main():
    print("Gathering existing image/label pairs...")
    pairs = gather_all_pairs()
    print(f"Found {len(pairs)} total image/label pairs.")

    print("Computing dominant class per image (for stratification)...")
    dominant_classes = [dominant_class(lbl) for _, lbl in pairs]

    class_counts = Counter(dominant_classes)
    print("Dominant-class distribution across the full dataset:")
    for cls, count in class_counts.items():
        print(f"  class {cls}: {count} images")

    # First split off train vs (val+test)
    train_pairs, temp_pairs, train_labels, temp_labels = train_test_split(
        pairs,
        dominant_classes,
        train_size=SPLIT_RATIOS["train"],
        stratify=dominant_classes,
        random_state=RANDOM_SEED,
    )

    # Now split temp into val vs test
    val_ratio_within_temp = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"])
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        train_size=val_ratio_within_temp,
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )

    split_result = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

    for split, split_pairs in split_result.items():
        print(f"{split}: {len(split_pairs)} images")

    # Stage all files into a temp folder FIRST, before touching the real
    # split folders, so we never delete a source file before it's copied.
    staging_dir = os.path.join(DATA_DIR, "_staging")
    staging_img = os.path.join(staging_dir, "images")
    staging_lbl = os.path.join(staging_dir, "labels")
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_img, exist_ok=True)
    os.makedirs(staging_lbl, exist_ok=True)

    print("Staging all files (safe copy before reshuffling folders)...")
    staged_pairs = {}  # split -> list of (staged_img_path, staged_lbl_path)
    for split, split_pairs in split_result.items():
        staged_list = []
        for img_path, lbl_path in split_pairs:
            img_fname = os.path.basename(img_path)
            lbl_fname = os.path.basename(lbl_path)
            staged_img_path = os.path.join(staging_img, img_fname)
            staged_lbl_path = os.path.join(staging_lbl, lbl_fname)
            # Only copy to staging once per unique file
            if not os.path.exists(staged_img_path):
                shutil.copy2(img_path, staged_img_path)
            if not os.path.exists(staged_lbl_path):
                shutil.copy2(lbl_path, staged_lbl_path)
            staged_list.append((staged_img_path, staged_lbl_path))
        staged_pairs[split] = staged_list

    # Now it's safe to clear and recreate the real split folders
    for split in SPLITS:
        img_dir = os.path.join(IMAGES_DIR, split)
        lbl_dir = os.path.join(LABELS_DIR, split)
        for d in [img_dir, lbl_dir]:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    print("Copying staged files into new split folders...")
    for split, split_pairs in staged_pairs.items():
        img_out = os.path.join(IMAGES_DIR, split)
        lbl_out = os.path.join(LABELS_DIR, split)
        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, img_out)
            shutil.copy2(lbl_path, lbl_out)

    # Clean up staging folder
    shutil.rmtree(staging_dir)

    print("Done. New class-balanced split created.")


if __name__ == "__main__":
    main()

import os
import shutil
import random

DATASET_PATH = "dataset"
OUTPUT_PATH = "dataset_split"
CLASSES = ["Closed", "Open", "yawn", "no_yawn"]
def split_data():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_PATH, split, cls), exist_ok=True)

    for cls in CLASSES:
        src = os.path.join(DATASET_PATH, cls)

        if not os.path.exists(src):
            raise FileNotFoundError(f"Folder not found: {src}")

        images = [
            f for f in os.listdir(src)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)

        train_end = int(0.7 * len(images))
        val_end = int(0.85 * len(images))

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for img in train_images:
            shutil.copy(os.path.join(src, img), os.path.join(OUTPUT_PATH, "train", cls))

        for img in val_images:
            shutil.copy(os.path.join(src, img), os.path.join(OUTPUT_PATH, "val", cls))

        for img in test_images:
            shutil.copy(os.path.join(src, img), os.path.join(OUTPUT_PATH, "test", cls))

    print("Dataset split completed successfully.")

if __name__ == "__main__":
    split_data()
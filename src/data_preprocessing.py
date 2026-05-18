# I used this code to split the dataset into train, validation, and test sets with a 70:15:15 ratio. 
# The images were shuffled before splitting, and each class was saved in separate folders for proper model training.

# Import Libraries
# These libraries are used for folder handling, copying files, and random selection
import os
import shutil
import random

# I defined the dataset path, output path for the split dataset, and the class names
DATASET_PATH = "dataset"
OUTPUT_PATH = "dataset_split"
CLASSES = ["Closed", "Open", "yawn", "no_yawn"]

# This function splits the dataset into train, validation, and test folders
def split_data():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_PATH, split, cls), exist_ok=True) # This creates folders for each class inside train, validation, and test directories

    for cls in CLASSES:
        src = os.path.join(DATASET_PATH, cls) # This builds the source path for the current class

        if not os.path.exists(src):
            raise FileNotFoundError(f"Folder not found: {src}")

        images = [
            f for f in os.listdir(src)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
# This randomly mixes the image order before splitting
        random.shuffle(images)

        train_end = int(0.7 * len(images)) #This splits the data into: 70% training, 15% validation, and 15% testing
        val_end = int(0.85 * len(images))

        train_images = images[:train_end] # This divides the shuffled images into three groups
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for img in train_images:
            shutil.copy(os.path.join(src, img), os.path.join(OUTPUT_PATH, "train", cls)) # This copies training images into the train folder

        for img in val_images:
            shutil.copy(os.path.join(src, img), os.path.join(OUTPUT_PATH, "val", cls)) # This copies validation images into the validation folder

        for img in test_images:
            shutil.copy(os.path.join(src, img), os.path.join(OUTPUT_PATH, "test", cls)) # This copies test images into the test folder

    print("Dataset split completed successfully.")

if __name__ == "__main__":
    split_data()
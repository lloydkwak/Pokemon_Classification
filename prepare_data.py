import os
import shutil
import random

# Define source directory where Kaggle data is extracted
# Update this path if your extracted folder name is different
SOURCE_DIR = './PokemonData' 

# Define target directories
BASE_DIR = './dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

# Set train and validation split ratio
SPLIT_RATIO = 0.8
# Set random seed for reproducibility
random.seed(42)

# Create base directories if they do not exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Get all class folders from the source directory
if not os.path.exists(SOURCE_DIR):
    print(f"Error: Source directory '{SOURCE_DIR}' not found.")
    exit(1)

classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for cls in classes:
    # Create class specific directories in train and val folders
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

    cls_dir = os.path.join(SOURCE_DIR, cls)
    
    # Filter only file items to avoid directories
    images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
    
    # Shuffle the image list to ensure random distribution
    random.shuffle(images)

    # Calculate the index to split the dataset
    split_idx = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Copy files to the train directory
    for img in train_images:
        src_path = os.path.join(cls_dir, img)
        dst_path = os.path.join(TRAIN_DIR, cls, img)
        shutil.copy(src_path, dst_path)

    # Copy files to the validation directory
    for img in val_images:
        src_path = os.path.join(cls_dir, img)
        dst_path = os.path.join(VAL_DIR, cls, img)
        shutil.copy(src_path, dst_path)

print("Data preprocessing and splitting completed successfully.")
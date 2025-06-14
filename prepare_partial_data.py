import pandas as pd
import shutil
import os
from tqdm import tqdm

# Paths
metadata_path = "data/HAM10000_metadata.csv"
image_dir = "data/HAM10000_images_part_1"
target_root = "data/train"

# Load metadata
df = pd.read_csv(metadata_path)

# Available images
available_images = set(os.listdir(image_dir))

# Create class-wise folders and copy images
moved = 0
skipped = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_file = row['image_id'] + ".jpg"
    disease = row['dx']
    src = os.path.join(image_dir, image_file)
    dst_dir = os.path.join(target_root, disease)
    dst = os.path.join(dst_dir, image_file)

    if os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, dst)
        moved += 1
    else:
        skipped += 1

print(f"\n Done! {moved} images copied.")
print(f"Skipped {skipped} images (likely from part 2).")
print("Classes created:", os.listdir(target_root))

import os, shutil
from sklearn.model_selection import train_test_split

data_dir = "EuroSAT_RGB"
train_dir = "EuroSAT_RGB_dataset/train"
val_dir = "EuroSAT_RGB_dataset/val"

os.makedirs(train_dir) 
os.makedirs(val_dir)

for imgclass in os.listdir(data_dir):
    path = os.path.join(data_dir, imgclass)
    
    if not os.path.isdir(path):
        continue
    
    images = os.listdir(path)
    train_files, val_files = train_test_split(images, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(train_dir, imgclass), exist_ok=True)
    os.makedirs(os.path.join(val_dir, imgclass), exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(path, f), os.path.join(train_dir, imgclass, f))
    for f in val_files:
        shutil.copy(os.path.join(path, f), os.path.join(val_dir, imgclass, f))

print("EuroSAT Dataset Created!!")


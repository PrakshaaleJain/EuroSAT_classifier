import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data_dir = "EuroSAT_RGB_dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Perform transformation on the EuroSAT images
transform = transforms.Compose([
    transforms.Resize((224, 224)),                                      # resizes the image to 224x224
    transforms.RandomHorizontalFlip(),                                  # performs horizontal flip of the input 
    transforms.ToTensor(),                                              # Convets to Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalizes the channels [R, G, B], [mean, std_dev]
])

# Dataset -----------------------------------------------------------------------------------------------------------------------------------
dataset = datasets.ImageFolder(root= data_dir, transform = transform)
num_classes = len(dataset)
print("No. of classes: ", num_classes)

val_size = int(0.2 * len(dataset)) 
train, val = random_split(dataset, [len(dataset)-val_size, val_size])   # [train_size, val_size]

train_loader = DataLoader(train, batch_size=64, shuffle=True)
val_loader = DataLoader(val, batch_size=64, shuffle=True)


# Model --------------------------------------------------------------------------------------------------------------------------------------
model = models.resnet50(pretrained=True)                            # Using pretrained ResNet50 as recommended by the paper
model.fc = nn.Linear(model.fc.in_features, num_classes)             # chanignthe last layer
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training -------------------------------------------------------------------------------------------------------------------------------------
total_epochs = 1000
train_size = (1 - val_size) * len(dataset)

for epoch in range(total_epochs):
    model.train()
    train_loss = train_correct = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)   
        optimizer.zero_grad()
        outputs = model(imgs)                                       # forward pass
        loss = criterion(outputs, labels)                           # compute loss
        loss.backward()                                             # backward pass        
        optimizer.step()                                            # update parameters

        train_loss += loss.item() * imgs.size(0)                     
        train_correct += (outputs.argmax(1) == labels).sum().item()  

    train_acc = train_correct / train_size                          # computes accuracy

    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_acc = val_correct / val_size

    print(f"Epoch {epoch+1}/{total_epochs} "
          f"Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")       

torch.save(model.state_dict(), "resnet50_eurosat_rgb.pth")
print("Model saved to resnet50_eurosat_rgb.pth!!")


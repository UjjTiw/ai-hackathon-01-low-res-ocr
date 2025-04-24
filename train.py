import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from models.simple_cnn import EfficientNetCustom
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision import transforms

from utils.dataset_utils import LABELS, ImageLabelDataset, set_seed
from utils.training_utils import evaluate_model, predict_test, train_model

# シード固定
set_seed(42)
device = torch.device("cuda")
print("Using device:", device)

# 前処理 - separate transforms for train and validation
transform_train = transforms.Compose([
    transforms.Resize((144, 144)),  # Slightly larger than final size
    transforms.RandomCrop((128, 128)),
    transforms.RandomRotation(3),  # Small rotations to simulate alignment issues
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Slight distortions
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# データ読み込み - No transform yet
full_dataset = ImageLabelDataset("data/train/train/low", transform=None, with_label=True)
labels = [full_dataset[i][1] for i in range(len(full_dataset))]

# データ分割
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

# Apply transforms
full_dataset.transform = transform_train  # This will affect train_dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create a separate dataset for validation with different transform
val_full_dataset = ImageLabelDataset("data/train/train/low", transform=transform_val, with_label=True)
val_dataset = Subset(val_full_dataset, val_idx)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# モデルとオプティマイザ
model = EfficientNetCustom(num_classes=len(LABELS)).to(device)
summary(model, input_size=(1, 128, 128))
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Set up model save directory first
model_save_dir = "checkpoints"
os.makedirs(model_save_dir, exist_ok=True)

best_f1 = 0
patience = 5
patience_counter = 0

# 学習ループ
num_epochs = 10 
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    train_model(model, train_loader, criterion, optimizer, device)
    print("Validation...")
    f1_score = evaluate_model(model, val_loader, device, criterion)
    
    # Learning rate scheduling
    scheduler.step(f1_score)
    
    # Early stopping
    if f1_score > best_f1:
        best_f1 = f1_score
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), os.path.join(model_save_dir, "model_best.pth"))
        print(f"New best model saved with F1: {best_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# === モデル保存 ===
model_filename = "model_final.pth"
model_save_path = os.path.join(model_save_dir, model_filename)
torch.save(model.state_dict(), model_save_path)
print(f"Saved model to {model_save_path}")

# 推論 - Use the best model for inference
print("\nLoading best model for prediction...")
model.load_state_dict(torch.load(os.path.join(model_save_dir, "model_best.pth")))

# Use the same transform as validation for testing
test_dataset = ImageLabelDataset("data/test/low", transform=transform_val, with_label=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
results = predict_test(model, test_loader, device)

# 結果保存
submission_filename = "submission.csv"
submission_df = pd.DataFrame(results, columns=["filename", "pred"])

# ファイル名で数値ソート
submission_df["sort_key"] = submission_df["filename"].str.extract(r"(\d+)").astype(int)
submission_df = submission_df.sort_values("sort_key").drop(columns="sort_key")

os.makedirs("pred", exist_ok=True)
submission_path = os.path.join("pred", submission_filename)
submission_df.to_csv(submission_path, index=False)
print(f"Saved predictions to {submission_path}")
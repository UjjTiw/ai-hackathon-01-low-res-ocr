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

# 前処理
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

# データ読み込み
full_dataset = ImageLabelDataset("data/train/train/low", transform=transform, with_label=True)
labels = [full_dataset[i][1] for i in range(len(full_dataset))]

# データ分割
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# モデルとオプティマイザ
model = EfficientNetCustom(num_classes=len(LABELS)).to(device)
summary(model, input_size=(1, 128, 128))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 学習ループ
for epoch in range(10):
    print(f"\nEpoch {epoch + 1}")
    train_model(model, train_loader, criterion, optimizer, device)
    print("Validation...")
    evaluate_model(model, val_loader, device, criterion)

# === モデル保存 ===
model_save_dir = "checkpoints"
model_filename = "model_final.pth"
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, model_filename)
torch.save(model.state_dict(), model_save_path)
print(f"Saved model to {model_save_path}")

# 推論
print("\nPredicting on test data...")
test_dataset = ImageLabelDataset("data/test/low", transform=transform, with_label=False)
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

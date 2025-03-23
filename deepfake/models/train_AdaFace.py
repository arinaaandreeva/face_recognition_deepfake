"""
Модуль для обучения модели AdaFace с Adaptive Margin Loss.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from PIL import Image
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.utils.utils import load_pretrained_model, load_checkpoint


class PairsDataset(Dataset):
    """Датасет пар изображений, загружаемый из CSV файла (ожидается 5 столбцов)."""

    def __init__(self, csv_file, transform):
        self.data = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader) 
            for row in reader:
                # Ожидаем строку: [pathA, pathB, label, deepfake_a, deepfake_b]
                self.data.append(row)
        self.transform = transform

        # Фильтруем записи: оставляем только те, где оба изображения существуют
        filtered_data = []
        for row in self.data:
            img_a_path = row[0]
            img_b_path = row[1]
            if os.path.exists(img_a_path) and os.path.exists(img_b_path):
                filtered_data.append(row)
            else:
                print(f"Пропускаем отсутствующий файл: {img_a_path} или {img_b_path}")
        self.data = filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img_a_path = row[0]
        img_b_path = row[1]
        label = int(row[2])
        deepfake_a = int(row[3]) if len(row) >= 5 else 0
        deepfake_b = int(row[4]) if len(row) >= 5 else 0

        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b, label, img_a_path, img_b_path, deepfake_a, deepfake_b


class AdaptiveMarginLoss(nn.Module):
    """Adaptive Margin Loss для пар изображений."""

    def __init__(self, s=64, base_margin=0.35, dynamic_scale=0.1):
        super(AdaptiveMarginLoss, self).__init__()
        self.s = s
        self.base_margin = base_margin
        self.dynamic_scale = dynamic_scale
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, cos_sim, labels):
        dynamic_margin = self.base_margin + self.dynamic_scale * (1 - cos_sim)
        logits = self.s * (cos_sim - dynamic_margin * labels.float())
        loss = self.bce(logits, labels.float())
        return loss


def compute_eer(scores, labels):
    """Вычисляет EER на основе roc_curve."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx_eer = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2
    return eer


def evaluate(model, dataloader, device, threshold=0.2):
    """Функция валидации модели."""
    model.eval()
    correct = 0
    total = 0
    scores_all = []
    labels_all = []
    error_counts = {
        "real-real": 0,           # пары с label == 1
        "real-another-real": 0,   # пары с label == 0, deepfake_a == 0 и deepfake_b == 0
        "real-deepfake": 0        # пары с label == 0, где хотя бы один deepfake-флаг равен 1
    }
    with torch.no_grad():
        for batch_idx, (img_a, img_b, labels, paths_a, paths_b,
                          deepfake_a, deepfake_b) in enumerate(
            tqdm(dataloader, desc="Validation", leave=True, dynamic_ncols=True)
        ):
            with torch.amp.autocast('cuda'):
                img_a = img_a.to(device)
                img_b = img_b.to(device)
                labels = labels.to(device)
                emb_a, _ = model(img_a)
                emb_b, _ = model(img_b)
                emb_a = nn.functional.normalize(emb_a, p=2, dim=1)
                emb_b = nn.functional.normalize(emb_b, p=2, dim=1)
                cos_sim = torch.sum(emb_a * emb_b, dim=1)
            preds = (cos_sim >= threshold).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            scores_all.extend(cos_sim.cpu().numpy().tolist())
            labels_all.extend(labels.cpu().numpy().tolist())

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                if true_label == 1:
                    if pred_label != true_label:
                        error_counts["real-real"] += 1
                else:
                    if deepfake_a[i] == 0 and deepfake_b[i] == 0:
                        if pred_label != true_label:
                            error_counts["real-another-real"] += 1
                    else:
                        if pred_label != true_label:
                            error_counts["real-deepfake"] += 1

    accuracy = correct / total if total > 0 else 0
    eer = compute_eer(np.array(scores_all), np.array(labels_all))
    return accuracy, eer, error_counts


def main():
    parser = argparse.ArgumentParser(
        description="Обучение модели AdaFace с CSV-датасетом и Adaptive Margin Loss."
    )
    parser.add_argument("--checkpoint", type=str,
                        default="models/pretrained/ckpt_epoch1_batch43000_acc0.9825_eer0.0165.ckpt",
                        help="Путь к файлу чекпоинта модели.")
    parser.add_argument("--train_csv", type=str,
                        default="data/train/train_pairs.csv",
                        help="Путь к CSV файлу с тренировочными парами.")
    parser.add_argument("--val_csv", type=str,
                        default="data/train/val_pairs.csv",
                        help="Путь к CSV файлу с валидационными парами.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Количество эпох обучения.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Размер батча для обучения.")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Скорость обучения для оптимизатора.")
    parser.add_argument("--s", type=float, default=64,
                        help="Параметр масштабирования для Adaptive Margin Loss.")
    parser.add_argument("--base_margin", type=float, default=0.35,
                        help="Базовый margin для Adaptive Margin Loss.")
    parser.add_argument("--dynamic_scale", type=float, default=0.1,
                        help="Динамический scale для Adaptive Margin Loss.")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay для оптимизатора.")
    parser.add_argument("--architecture", type=str, default="ir_101",
                        help="Архитектура модели: 'ir_101' или другая.")
    args = parser.parse_args()

    wandb.init(
        project="adaface_train",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "s": args.s,
            "base_margin": args.base_margin,
            "dynamic_scale": args.dynamic_scale,
            "weight_decay": args.weight_decay,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(args.architecture).to(device)
    model = load_checkpoint(args.checkpoint, model).to(device)

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = PairsDataset(args.train_csv, transform=transform)
    val_dataset = PairsDataset(args.val_csv, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)

    loss_fn = AdaptiveMarginLoss(s=args.s,
                                 base_margin=args.base_margin,
                                 dynamic_scale=args.dynamic_scale)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        threshold=1e-4,
    )

    num_epochs = args.epochs
    global_batch_counter = -1

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (img_a, img_b, labels, paths_a, paths_b, _, _) in enumerate(
                tqdm(train_dataloader,
                     desc=f"Epoch {epoch}/{num_epochs} Training",
                     leave=True, dynamic_ncols=True)
        ):
            global_batch_counter += 1

            img_a = img_a.to(device)
            img_b = img_b.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                emb_a, _ = model(img_a)
                emb_b, _ = model(img_b)
                emb_a = nn.functional.normalize(emb_a, p=2, dim=1)
                emb_b = nn.functional.normalize(emb_b, p=2, dim=1)
                cos_sim = torch.sum(emb_a * emb_b, dim=1)
                loss = loss_fn(cos_sim, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * img_a.size(0)

        epoch_loss = running_loss / len(train_dataset)
        val_acc, eer, error_counts = evaluate(model, val_dataloader, device, threshold=0.2)
        epoch_log = {
            "epoch": epoch,
            "epoch_loss": epoch_loss,
            "val_accuracy": val_acc,
            "EER": eer,
            "real_another_real": error_counts["real-another-real"],
            "real_deepfake": error_counts["real-deepfake"],
            "real-real": error_counts["real-real"],
            "lr": optimizer.param_groups[0]["lr"]
        }
        print(epoch_log)
        wandb.log(epoch_log)
        print(f"Эпоха {epoch} завершена, lr теперь {optimizer.param_groups[0]['lr']:.2e}")

    # Сохранение финальной модели
    final_model_path = "models/final_model.ckpt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Финальная модель сохранена в {final_model_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
import os
import json
import random
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd

# Устройство для вычислений (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Заданные параметры
META_PATH = "data/train/meta.json"  # Путь к файлу метаданных
IMAGES_ROOT = "data/train/images"  # Путь к папке с изображениями

# Загрузка метаданных
with open(META_PATH, 'r', encoding='utf-8') as f:
    meta = json.load(f)  # meta теперь обычный словарь (dict)

# Словарь для хранения изображений по ID
id_to_images = {}  # ключ: id (строка '000000'), значение: список [(filename, label), ...]

for key, value in meta.items():
    person_id = key.split('/')[0]  # "000000"
    filename = key.split('/')[1]  # "0.jpg"
    if person_id not in id_to_images:
        id_to_images[person_id] = []
    id_to_images[person_id].append((filename, value))

all_ids = list(id_to_images.keys())

# Функция для создания триплетов
def create_triplets(id_to_images):
    triplets = []
    all_ids = list(id_to_images.keys())

    for person_id in all_ids:
        # Получаем все изображения для текущего человека
        imgs = id_to_images[person_id]

        # Выбираем реальные изображения (label == 0)
        real_imgs = [img for img in imgs if img[1] == 0]

        # Если реальных изображений меньше 2, пропускаем
        if len(real_imgs) < 2:
            continue

        # Создаем пары (anchor, positive)
        for anchor, positive in combinations(real_imgs, 2):
            # Выбираем negative из другого класса
            negative_id = random.choice([i for i in all_ids if i != person_id])
            negative_imgs = id_to_images[negative_id]
            negative = random.choice([img for img in negative_imgs if img[1] == 0])

            # Добавляем triplet с полным путем
            anchor_path = os.path.join(person_id, anchor[0])  # Пример: '00000000/0.jpg'
            positive_path = os.path.join(person_id, positive[0])  # Пример: '00000000/1.jpg'
            negative_path = os.path.join(negative_id, negative[0])  # Пример: '00000001/0.jpg'

            triplets.append((anchor_path, positive_path, negative_path))

    return triplets

# Создание триплетов
triplets = create_triplets(id_to_images)
print(f"Создано {len(triplets)} триплетов")

# Модель CNN на основе ResNet18
class FaceCNN(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceCNN, self).__init__()
        self.backbone = models.resnet18(pretrained=True)  # Используем предобученную ResNet18
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size)

    def forward(self, x):
        return self.backbone(x)

# Адаптивная функция потерь
class AdaptiveMarginLoss(nn.Module):
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

# Инициализация модели и оптимизатора
model = FaceCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = AdaptiveMarginLoss(s=64, base_margin=0.35, dynamic_scale=0.1)

# Класс для датасета триплетов
class TripletFaceDataset(Dataset):
    def __init__(self, triplets, root_dir, transform=None):
        self.triplets = triplets
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        # Загрузка изображений
        anchor_img = Image.open(os.path.join(self.root_dir, anchor_path)).convert('RGB')
        positive_img = Image.open(os.path.join(self.root_dir, positive_path)).convert('RGB')
        negative_img = Image.open(os.path.join(self.root_dir, negative_path)).convert('RGB')

        # Применение трансформаций
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Создание Dataset и DataLoader
dataset = TripletFaceDataset(triplets, IMAGES_ROOT, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Обучение модели
epochs = 5
for epoch in tqdm(range(epochs)):
    model.train()
    for anchor, positive, negative in loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # Получение эмбеддингов
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        # Нормализация эмбеддингов
        anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
        positive_emb = F.normalize(positive_emb, p=2, dim=1)
        negative_emb = F.normalize(negative_emb, p=2, dim=1)

        # Вычисление косинусной схожести
        cos_sim_positive = torch.sum(anchor_emb * positive_emb, dim=1)
        cos_sim_negative = torch.sum(anchor_emb * negative_emb, dim=1)

        # Вычисление потерь
        loss_positive = criterion(cos_sim_positive, torch.ones_like(cos_sim_positive))
        loss_negative = criterion(cos_sim_negative, torch.zeros_like(cos_sim_negative))
        loss = loss_positive + loss_negative

        # Обновление модели
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Сохранение модели
torch.save(model.state_dict(), 'CNN_2.pth')
import os
import json
import random
import itertools
import csv
import argparse
from tqdm import tqdm
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as TF
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Устройство для вычислений (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceAligner:
    """Класс для выравнивания лиц на изображениях с использованием модели CVLFaceAlignmentModel."""
    def __init__(self, model_config):
        # Инициализация модели выравнивания
        self.aligner = CVLFaceAlignmentModel(model_config).to(device)
        self.aligner.eval()  # Переводим модель в режим оценки
        # Преобразования для изображений
        self.transform = Compose([
            ToTensor(),  # Преобразуем изображение в тензор
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация
        ])

    def align_face(self, image_path):
        """Выравнивает лицо на изображении."""
        img = Image.open(image_path).convert("RGB")  # Открываем и конвертируем изображение
        input_tensor = self.transform(img).unsqueeze(0).to(device)  # Применяем преобразования
        output = self.aligner(input_tensor)  # Получаем результат от модели
        aligned_tensor = output[0][0]  # Извлекаем выровненное изображение
        aligned_tensor = (aligned_tensor * 0.5 + 0.5).clamp(0, 1)  # Приводим к диапазону [0, 1]
        return TF.to_pil_image(aligned_tensor)  # Конвертируем тензор обратно в изображение

class DatasetProcessor:
    """Класс для обработки датасета и генерации пар изображений."""
    def __init__(self, meta_path, images_root):
        self.meta_path = meta_path  # Путь к файлу метаданных
        self.images_root = images_root  # Путь к папке с изображениями
        self.persons = self._load_meta()  # Загружаем метаданные

    def _load_meta(self):
        """Загружает метаданные из файла meta.json."""
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        persons = {}
        for key, value in meta.items():
            person_id, filename = key.split("/")  # Разделяем ключ на ID человека и имя файла
            if person_id not in persons:
                persons[person_id] = {"real": [], "fake": []}  # Инициализируем списки для реальных и фейковых изображений
            if value == 0:
                persons[person_id]["real"].append(filename)  # Добавляем реальное изображение
            else:
                persons[person_id]["fake"].append(filename)  # Добавляем фейковое изображение
        return persons

    def generate_pairs(self, person_ids):
        """Генерирует пары изображений для заданных ID людей."""
        all_pos, all_neg_rf, all_neg_rar = [], [], []
        for pid in tqdm(person_ids, desc="Generating pairs"):
            pos = self._sample_positive_pairs(pid)  # Генерация позитивных пар
            neg_rf = self._sample_real_fake_pairs(pid)  # Генерация пар real-fake
            neg_rar = self._sample_real_another_real_pairs(pid, person_ids)  # Генерация пар real-another-real
            all_pos.extend(pos)
            all_neg_rf.extend(neg_rf)
            all_neg_rar.extend(neg_rar)
        return all_pos, all_neg_rf, all_neg_rar

    def _sample_positive_pairs(self, person_id):
        """Генерирует позитивные пары (real-real) для одного человека."""
        real_imgs = [os.path.join(self.images_root, person_id, fname) for fname in self.persons[person_id]["real"]]
        return [(a, b, "1", "0", "0") for a, b in itertools.combinations(real_imgs, 2)] if len(real_imgs) >= 2 else []

    def _sample_real_fake_pairs(self, person_id):
        """Генерирует пары real-fake для одного человека."""
        real_imgs = [os.path.join(self.images_root, person_id, fname) for fname in self.persons[person_id]["real"]]
        fake_imgs = [os.path.join(self.images_root, person_id, fname) for fname in self.persons[person_id]["fake"]]
        return [(r, f, "0", "0", "1") for r in real_imgs for f in fake_imgs] if real_imgs and fake_imgs else []

    def _sample_real_another_real_pairs(self, person_id, all_persons):
        """Генерирует пары real-another-real для одного человека."""
        real_imgs = [os.path.join(self.images_root, person_id, fname) for fname in self.persons[person_id]["real"]]
        if not real_imgs:
            return []
        other_persons = [pid for pid in all_persons if pid != person_id and self.persons[pid]["real"]]
        if not other_persons:
            return []
        other_pid = random.choice(other_persons)
        other_real_imgs = [os.path.join(self.images_root, other_pid, fname) for fname in self.persons[other_pid]["real"]]
        return [(random.choice(real_imgs), random.choice(other_real_imgs), "0", "0", "0")] if other_real_imgs else []

def save_pairs(csv_file, pairs):
    """Сохраняет пары изображений в CSV файл."""
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pathA", "pathB", "label", "deepfake_a", "deepfake_b"])  # Заголовок CSV
        writer.writerows(pairs)  # Запись пар

def main(args):
    """Основная функция для обработки данных и генерации пар."""
    processor = DatasetProcessor(args.meta_path, args.images_root)  # Инициализация процессора
    train_persons, val_persons = split_persons(processor.persons, args.train_split)  # Разделение на train и val

    train_pairs = processor.generate_pairs(train_persons)  # Генерация пар для train
    val_pairs = processor.generate_pairs(val_persons)  # Генерация пар для val

    save_pairs(args.output_train, train_pairs)  # Сохранение train пар
    save_pairs(args.output_val, val_pairs)  # Сохранение val пар

    print(f"Train pairs saved to {args.output_train}")
    print(f"Validation pairs saved to {args.output_val}")

def split_persons(persons, train_split):
    """Разделяет людей на train и val наборы."""
    all_persons = sorted(persons.keys())
    random.shuffle(all_persons)  # Перемешиваем для случайного разделения
    split_idx = int(len(all_persons) * train_split)  # Индекс для разделения
    return all_persons[:split_idx], all_persons[split_idx:]

if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Process images and generate pairs for training and validation.")
    parser.add_argument("--meta_path", type=str, default="data/train/meta.json", help="Path to meta.json")
    parser.add_argument("--images_root", type=str, default="data/train/images", help="Path to images directory")
    parser.add_argument("--train_split", type=float, default=0.95, help="Train split ratio")
    parser.add_argument("--output_train", type=str, default="data/train/train_pairs.csv", help="Output path for train pairs")
    parser.add_argument("--output_val", type=str, default="data/train/val_pairs.csv", help="Output path for validation pairs")
    args = parser.parse_args()
    main(args)  # Запуск основной функции
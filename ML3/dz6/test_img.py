from tqdm import tqdm
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image

# Загрузка матриц сверток
with open('algos.csv', 'r') as f:
    matrix1 = list(map(float, f.readline().split(',')))
    matrix2 = list(map(float, f.readline().split(',')))

mat1 = torch.FloatTensor(matrix1).reshape(1, 1, 3, 3)
mat2 = torch.FloatTensor(matrix2).reshape(1, 1, 3, 3)

# Загрузка данных
png_files = list(Path("best_pictures/").glob("*.png"))
txt_files = list(Path("best_pictures/").glob("*.txt"))
big_dict = {}

for filepath in png_files + txt_files:
    number = int(filepath.stem)
    if number not in big_dict:
        big_dict[number] = [None, None]

    if filepath.suffix == '.png':
        img = Image.open(filepath)
        img_tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0)
        big_dict[number][0] = img_tensor
    elif filepath.suffix == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            arr = torch.FloatTensor([list(map(float, x.strip().split())) for x in f.readlines()])
            big_dict[number][1] = arr.unsqueeze(0).unsqueeze(0)


class ConvModel(nn.Module):
    def __init__(self, pos1, mat1, pos2, mat2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, 1, 3, padding=1) for _ in range(3)])

        # Устанавливаем известные свертки
        if pos1 <= 3:
            self.convs[pos1 - 1].weight.data = mat1.clone()
            self.convs[pos1 - 1].bias.data.zero_()
            self.convs[pos1 - 1].weight.requires_grad = False
            self.convs[pos1 - 1].bias.requires_grad = False

        if pos2 <= 3:
            self.convs[pos2 - 1].weight.data = mat2.clone()
            self.convs[pos2 - 1].bias.data.zero_()
            self.convs[pos2 - 1].weight.requires_grad = False
            self.convs[pos2 - 1].bias.requires_grad = False

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


def train_model(model, data_dict, epochs=500):
    """Обучает модель и возвращает средний MSE после обучения"""
    # Находим обучаемые параметры
    trainable_params = []
    for i, conv in enumerate(model.convs):
        if conv.weight.requires_grad:
            trainable_params += list(conv.parameters())

    if not trainable_params:
        return evaluate_model(model, data_dict)

    optimizer = torch.optim.Adam(trainable_params, lr=0.01)

    for epoch in range(epochs):
        total_loss = 0
        for img, target in data_dict.values():
            if img is None or target is None:
                continue

            optimizer.zero_grad()
            pred = model(img)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return evaluate_model(model, data_dict)


def evaluate_model(model, data_dict):
    """Вычисляет средний MSE модели на данных"""
    total_loss = 0.0
    with torch.no_grad():
        for img, target in data_dict.values():
            if img is None or target is None:
                continue
            pred = model(img)
            loss = torch.mean((pred - target) ** 2)
            total_loss += loss.item()
    return total_loss / len(data_dict)


# Все возможные варианты позиций сверток
variants = [
    (1, mat1, 2, mat2),
    (1, mat1, 3, mat2),
    (2, mat1, 1, mat2),
    (2, mat1, 3, mat2),
    (3, mat1, 1, mat2),
    (3, mat1, 2, mat2),
]

results = []

# Обучаем каждую конфигурацию
for variant in tqdm(variants, desc="Training configurations"):
    model = ConvModel(*variant)
    mse = train_model(model, big_dict)
    results.append({
        'variant': variant,
        'model': model,
        'mse': mse
    })

# Находим лучшую модель
best_result = min(results, key=lambda x: x['mse'])
best_model = best_result['model']

print("\nЛучшая конфигурация:")
print(f"Позиции сверток: {best_result['variant'][0]} (mat1) и {best_result['variant'][2]} (mat2)")
print(f"MSE: {best_result['mse']:.6f}")

print("\nЯдра всех трех сверток:")
for i, conv in enumerate(best_model.convs, 1):
    print(f"Свертка {i}:")
    print(conv.weight.data.squeeze().numpy())
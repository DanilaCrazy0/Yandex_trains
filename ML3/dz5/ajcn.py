import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import random
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
# Предполагаем, что df_train - pandas DataFrame
X_train = df_train.drop(columns=['target']).values
y_train = df_train['target'].values.reshape(-1, 1)  # Преобразуем в [N, 1] для BCELoss

# Масштабирование
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Создаем список кортежей (tensor, tensor)
train_data = [
    (torch.FloatTensor(x), torch.FloatTensor(y))
    for x, y in zip(X_train, y_train)
]


# Разделение на train/val
val_size = 0.25
random.shuffle(train_data)
split_idx = int(len(train_data) * (1 - val_size))
train_data, val_data = train_data[:split_idx], train_data[split_idx:]


# Правильный Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

    def __len__(self):
        return len(self.data)


train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)

batch_size = 1288
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Модель
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Dropout(0.3),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Dropout(0.3),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
num_epochs = 5000

best_val_acc = 0.0
print(device)

for epoch in range(num_epochs):
    # print(f'\nEpoch {epoch + 1}/{num_epochs}')
    model.train()
    train_loss = 0.0
    train_correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = (outputs > 0.5).float()
        train_correct += (preds == labels).sum().item()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_dataset)
    train_acc = train_correct / len(train_dataset)

    # Валидация
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds = (outputs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)

    # scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        torch.save(model.state_dict(), 'best_model.pth')

    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}\n')

X_test = scaler.transform(df_test.values)
test_dataset = CustomDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

out = outputs.numpy().flatten()
out = np.where(out >= 0.5, 1, 0)
df_out = pd.Series(out)
df_out.to_csv('answer.csv', header=False, index=False)
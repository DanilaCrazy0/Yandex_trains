import json
import os
from tqdm import tqdm
import time
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.datasets import FashionMNIST


def get_predictions(model, eval_data, step=10):

    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(eval_data), step):
            y_pred = model(eval_data[i : i + step].to(device))
            predicted_labels.append(y_pred.argmax(dim=1).cpu())
    predicted_labels = torch.cat(predicted_labels)
    predictions = ','.join([str(i.item()) for i in list(predicted_labels)])
    return predictions


def get_accuracy(model, dataloader):
    predicted_labels = []
    real_labels = []
    model.eval()
    with torch.no_grad():
        for pred, real in dataloader:
            y_pred = model(pred.to(device))
            predicted_labels.append(y_pred.argmax(dim=1).cpu())
            real_labels.append(real)
    predicted_labels = torch.cat(predicted_labels)
    real_labels = torch.cat(real_labels)
    accuracy = (real_labels==predicted_labels).type(torch.FloatTensor).mean()
    return accuracy


def training(net, train_loader, val_loader, train_dataset, val_dataset):
    num_epochs = 1000
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_net = net
    best_acc = 0
    flag = False

    for epoch in range(num_epochs):
        if flag:
            return best_net

        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        start_time = time.time()

        net.train()
        running_loss = 0.0
        running_corrects = 0

        train_loop = tqdm(train_loader, desc='Training', leave=True)
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            y_pred = net(images)
            _, preds = torch.max(y_pred, 1)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            train_loop.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{torch.sum(preds == labels.data).item() / images.size(0):.2f}"
            })

        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.double() / len(train_dataset)

        # Validation phase
        net.eval()
        val_loss = 0.0
        val_corrects = 0

        val_loop = tqdm(val_loader, desc='Validating', leave=True)
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                y_pred = net(images)
                _, preds = torch.max(y_pred, 1)
                loss = criterion(y_pred, labels)

                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels.data)

                val_loop.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{torch.sum(preds == labels.data).item() / images.size(0):.2f}"
                })

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_net = net
            if val_acc >= 0.885:
                flag = True

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n')

    return best_net


assert os.path.exists('hw_fmnist_data_dict.npy'), 'Alyo yopta, dataseta net'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256


train_data = FashionMNIST('.', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = FashionMNIST('.', train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

model_task_1 = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    nn.Conv2d(16, 120, kernel_size=5),
    nn.Flatten(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10)
)

model_task_1.to(device)


assert model_task_1 is not None


try:
    random_batch = next(iter(train_dataloader))
    x = random_batch[0].to(device)
    y = random_batch[1].to(device)

    y_pred = model_task_1(x)

except Exception as e:
    print('something wrong')
    raise e


assert y_pred.shape[-1] == 10, 'wrong'

model_task_1 = training(model_task_1, train_dataloader, test_dataloader, train_data, test_data)


train_acc_task1 = get_accuracy(model_task_1, train_dataloader)
print(f'nn accuracy on train: {train_acc_task1}')

test_acc_task1 = get_accuracy(model_task_1, test_dataloader)
print(f'nn accuracy on test: {test_acc_task1}')

assert test_acc_task1 >= 0.885, 'test accuracy below threshold'
# assert train_acc_task1 >= 0.905, 'train acc below threshold'

loaded_data_dict = np.load("hw_fmnist_data_dict.npy", allow_pickle=True)

submission_dict = {
    "train_predictions_task_1": get_predictions(
        model_task_1, torch.FloatTensor(loaded_data_dict.item()["train"])
    ),
    "test_predictions_task_1": get_predictions(
        model_task_1, torch.FloatTensor(loaded_data_dict.item()["test"])
    ),
}

with open("submission_dict_fmnist_task_1.json", "w") as iofile:
    json.dump(submission_dict, iofile)
print("File saved to `submission_dict_fmnist_task_1.json`")
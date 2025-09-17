# Definition de la boucle d'apprentissage
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.model import CNN
import evaluate


import torch
import torch.nn as nn
# from ..models.model import CNN
# from . import evaluate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    current_loss = []
    nb_ok = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        nb_ok += (output.argmax(dim=1) == target).float().sum()
        loss = criterion(output, target)
        current_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    current_loss = sum(current_loss)/len(current_loss)
    print(f"Epoch {epoch} - loss: {current_loss:.2f}")
    acc_train = nb_ok/len(train_loader.dataset)
    print(f"Accuracy: {acc_train:.2f}")
    return current_loss, acc_train


model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# Chargement des données
train_data = datasets.MNIST(
    root = '../data',
    train = True,                         
    transform = transforms.ToTensor(), 
    download = True,            
)

test_data = datasets.MNIST(
    root = '../data', 
    train = False, 
    transform = transforms.ToTensor()
)

print(len(train_data), len(test_data))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Apprentissage
train_acc, test_acc = [], []
train_loss, test_loss = [], []

for epoch in range(1, 11):
    train_loss_cur, train_acc_cur = train(model, train_loader, optimizer, criterion, epoch)
    test_loss_cur, test_acc_cur = evaluate.test(model, test_loader, criterion)
    train_acc.append(train_acc_cur)
    test_acc.append(test_acc_cur)
    train_loss.append(train_loss_cur)
    test_loss.append(test_loss_cur)
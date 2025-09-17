# Definition de la boucle d'apprentissage
import torch


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

# Definition de la boucle de test
def test(model, test_loader, criterion):
    model.eval()
    test_loss = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            cur_loss = criterion(output, target)
            test_loss.append(cur_loss.item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss = sum(test_loss)/len(test_loader)
    acc_loss = correct/len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.2f}, Accuracy: {acc_loss:.2f}")
    return test_loss, acc_loss

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Apprentissage
train_acc, test_acc = [], []
train_loss, test_loss = [], []

for epoch in range(1, 11):
    train_loss_cur, train_acc_cur = train(model, train_loader, optimizer, criterion, epoch)
    test_loss_cur, test_acc_cur = test(model, test_loader, criterion)
    train_acc.append(train_acc_cur)
    test_acc.append(test_acc_cur)
    train_loss.append(train_loss_cur)
    test_loss.append(test_loss_cur)
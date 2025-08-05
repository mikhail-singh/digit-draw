import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNNClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data transform
mean, std = 0.1307, 0.3081
train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)),
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)),
])

train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=test_transforms),
                          batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('data', train=False, download=True, transform=test_transforms),
                         batch_size=1000)

# model, loss, optimizer
model = CNNClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training loop
epochs = 7
for epoch in range(epochs):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = loss_fn(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(
        f"Epoch {epoch+1}/{epochs} loss={total_loss/len(train_loader):.3f}"
        f" acc={100*correct/total:.2f}%")

# test
model.eval()
correct = total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Test accuracy: {100*correct/total:.2f}%")

# save weights
torch.save(model.state_dict(), "model.pt")
print("Model weights saved to model.pt")
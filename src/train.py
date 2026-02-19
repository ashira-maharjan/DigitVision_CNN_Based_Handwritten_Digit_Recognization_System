import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
from model import CNN
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model folder
if not os.path.exists("../models"):
    os.makedirs("../models")

# Dataset
transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(
    root='../data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Model
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 50

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}")

# Save Model
torch.save(model.state_dict(), "../models/mnist_cnn.pth")
print("Model Saved Successfully!")

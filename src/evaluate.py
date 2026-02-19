import torch
import torchvision
import torchvision.transforms as transforms
from model import CNN
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

testset = torchvision.datasets.MNIST(
    root='../data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

model = CNN().to(device)
model.load_state_dict(torch.load("../models/mnist_cnn.pth", map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print("Test Accuracy:", acc)

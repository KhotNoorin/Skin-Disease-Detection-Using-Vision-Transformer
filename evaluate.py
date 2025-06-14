import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ViTSkinDiseaseClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.ImageFolder('data/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32)

model = ViTSkinDiseaseClassifier(num_classes=7).to(device)
model.load_state_dict(torch.load("vit_skin_model.pth"))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

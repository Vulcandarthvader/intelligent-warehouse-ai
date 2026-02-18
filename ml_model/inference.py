import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def classify_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model architecture
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 3)

    # Load trained weights
    model.load_state_dict(torch.load("ml_model/model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    classes = ["fragile", "hazardous", "heavy"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]


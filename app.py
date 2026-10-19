
import json
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import transforms

# Choose CPU or NVIDIA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Same CNN architecture used during training
class ButterflyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Load trained weights
model = ButterflyCNN(len(class_names)).to(device)
model.load_state_dict(
    torch.load("models/butterfly_cnn.pth", map_location=device, weights_only=True)
)
model.eval()

# Image preprocessing must match validation preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_butterfly(image):
    if image is None:
        return {}

    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_values, top_indices = torch.topk(
        probabilities, min(3, len(class_names))
    )

    return {
        class_names[index.item()]: float(value.item())
        for value, index in zip(top_values, top_indices)
    }

# Create interactive web interface
demo = gr.Interface(
    fn=predict_butterfly,
    inputs=gr.Image(type="pil", label="Upload a butterfly image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted species"),
    title="🦋 Butterfly Species Classifier",
    description=(
        "Upload a butterfly image to see the model's top 3 predictions. "
        "Predictions may be incorrect; the model's validation accuracy is 52.08%."
    )
)

if __name__ == "__main__":
    demo.launch()

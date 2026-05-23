import torch
from torchvision import transforms
from PIL import Image

from models.cnn_model import ChestXrayCNN


MODEL_PATH = "models/model.pth"
IMG_SIZE = 224

CLASSES = ["NORMAL", "PNEUMONIA"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Image preprocessing
# =========================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# Load model
# =========================

model = ChestXrayCNN(num_classes=2).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()


# =========================
# Tool function
# =========================

def classify_xray_tool(image_path):
    try:
        image = Image.open(image_path).convert("RGB")

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted_class = torch.max(probabilities, 1)

        prediction = CLASSES[predicted_class.item()]
        confidence_score = confidence.item()

        return {
            "prediction": prediction,
            "confidence": round(confidence_score, 4)
        }

    except Exception as e:
        return {
            "error": str(e),
            "prediction": None,
            "confidence": 0.0
        }
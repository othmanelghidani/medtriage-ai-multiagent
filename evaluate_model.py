import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from data_loader import get_data_loaders
from models.cnn_model import ChestXrayCNN


# =========================
# Configuration
# =========================

MODEL_PATH = "models/model.pth"
REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device used:", device)


# =========================
# Load data
# =========================

train_loader, val_loader, test_loader, classes = get_data_loaders(batch_size=32)

print("Classes:", classes)


# =========================
# Load model
# =========================

model = ChestXrayCNN(num_classes=2).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print("Model loaded successfully.")


# =========================
# Evaluation
# =========================

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())


# =========================
# Metrics
# =========================

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average="weighted")
recall = recall_score(all_labels, all_predictions, average="weighted")
f1 = f1_score(all_labels, all_predictions, average="weighted")

print("\nEvaluation Results")
print("------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report")
print("---------------------")
print(classification_report(all_labels, all_predictions, target_names=classes))


# =========================
# Save metrics to text file
# =========================

with open(os.path.join(REPORTS_DIR, "evaluation_metrics.txt"), "w") as f:
    f.write("Evaluation Results\n")
    f.write("------------------\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n\n")
    f.write("Classification Report\n")
    f.write("---------------------\n")
    f.write(classification_report(all_labels, all_predictions, target_names=classes))


# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(all_labels, all_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(values_format="d")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))
plt.close()

print("\nConfusion matrix saved in reports/confusion_matrix.png")
print("Metrics saved in reports/evaluation_metrics.txt")
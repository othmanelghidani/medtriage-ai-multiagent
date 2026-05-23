import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import get_data_loaders
from models.cnn_model import ChestXrayCNN


# =========================
# Configuration
# =========================

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = "models/model.pth"
REPORTS_DIR = "reports"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device used:", device)


# =========================
# Load data
# =========================

train_loader, val_loader, test_loader, classes = get_data_loaders(
    batch_size=BATCH_SIZE
)

print("Classes:", classes)


# =========================
# Model, loss, optimizer
# =========================

model = ChestXrayCNN(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# Training function
# =========================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


# =========================
# Validation function
# =========================

def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total

    return val_loss, val_acc


# =========================
# Training loop
# =========================

os.makedirs("models", exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

best_val_acc = 0.0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion,
        device
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print("Best model saved!")


# =========================
# Save training curves
# =========================

plt.figure()
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig(os.path.join(REPORTS_DIR, "training_loss.png"))
plt.close()

plt.figure()
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig(os.path.join(REPORTS_DIR, "validation_accuracy.png"))
plt.close()


# =========================
# Final message
# =========================

print("\nTraining finished.")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Model saved at: {MODEL_SAVE_PATH}")
print(f"Training loss curve saved at: {REPORTS_DIR}/training_loss.png")
print(f"Validation accuracy curve saved at: {REPORTS_DIR}/validation_accuracy.png")
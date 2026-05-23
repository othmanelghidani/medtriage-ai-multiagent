import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Chemin du dataset
DATASET_PATH = "dataset/chest_xray"

# Taille standard des images
IMG_SIZE = 224

# Transformations pour l'entraînement
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Transformations pour validation et test
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_data_loaders(batch_size=32):
    train_dataset = datasets.ImageFolder(
        root=f"{DATASET_PATH}/train",
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        root=f"{DATASET_PATH}/val",
        transform=test_transforms
    )

    test_dataset = datasets.ImageFolder(
        root=f"{DATASET_PATH}/test",
        transform=test_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, train_dataset.classes
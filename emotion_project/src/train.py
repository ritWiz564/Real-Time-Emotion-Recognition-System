import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EmotionCNN
from config import *


def main():
    # ----------------------------
    # Device setup
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.is_available())
        print("GPU name:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # ----------------------------
    # Data transforms
    # ----------------------------
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # ----------------------------
    # Dataset paths
    # ----------------------------
    train_path = os.path.join(BASE_DIR, "train")
    test_path = os.path.join(BASE_DIR, "test")

    print("Train path:", train_path)
    print("Test path:", test_path)

    # ----------------------------
    # Load datasets
    # ----------------------------
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

    label_counts = Counter(train_dataset.targets)
    num_classes = len(train_dataset.classes)
    total_samples = sum(label_counts.values())

    class_weights = []
    for i in range(num_classes):
        class_weights.append(total_samples / label_counts[i])

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Class names:", train_dataset.classes)
    print("Class counts:", dict(label_counts))
    print("Class weights:", class_weights)

    # ----------------------------
    # Data loaders
    # ----------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # ----------------------------
    # Model, loss, optimizer
    # ----------------------------
    model = EmotionCNN(num_classes=num_classes).to(device)
    print("Model device:", next(model.parameters()).device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(EPOCHS):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if epoch == 0 and batch_idx == 0:
                print("Images device:", images.device)
                print("Labels device:", labels.device)
                if torch.cuda.is_available():
                    print(
                        "GPU memory allocated before forward:",
                        f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
                    )

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if epoch == 0 and batch_idx == 0 and torch.cuda.is_available():
                print(
                    "GPU memory allocated after forward/backward:",
                    f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
                )

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            progress_bar.set_postfix(
                loss=f"{running_loss / (batch_idx + 1):.4f}",
                acc=f"{100 * correct / total:.2f}%"
            )

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # ----------------------------
    # Save model
    # ----------------------------
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print("Model saved to:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

def main():
    # ตรวจสอบ GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # กำหนดค่าพารามิเตอร์
    data_dir = "./animal_dataset"
    img_size = 224
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    patience = 5

    # Data Augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # โหลด Dataset
    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform_val)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4, pin_memory=True)

    # โหลดโมเดล ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(train_data.classes))
    )
    model = model.to(device)

    # Loss และ Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    # Mixed Precision Training
    scaler = GradScaler()

    # เทรนโมเดล
    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_total_loss = 0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_total_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss = val_total_loss / len(val_loader)
        val_acc = 100 * val_correct / len(val_loader.dataset)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # บันทึกโมเดลที่ดีที่สุด
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), "animal.pth")
            print(f"✅ Best model saved with validation loss: {val_loss:.4f}")
        else:
            no_improve += 1

        # Early Stopping
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step(val_loss)

    print("✅ Training complete.")

if __name__ == '__main__':
    main()
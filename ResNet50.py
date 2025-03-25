import pymysql
from datetime import datetime
# from google.colab import files
# files.upload()

def insert_training_log(epoch, start_time, end_time, train_loss, train_accuracy, val_loss, val_accuracy, early_stop):
    conn = pymysql.connect(
        host='',
        port=9999,
        user='avnadmin',
        password='',
        database='cnn_training_log',
        ssl={'ca': '/content/ca.pem'},
        charset='utf8mb4'
    )
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO cnn_training_log 
                (epoch, start_time, end_time, train_loss, train_accuracy, val_loss, val_accuracy, early_stop)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (epoch, start_time, end_time, train_loss, train_accuracy, val_loss, val_accuracy, early_stop))
        conn.commit()
    finally:
        conn.close()

# 데이터 증강 적용
# 12번 학습


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50  
from PIL import Image


data_train_path = "/content/fruit-and-vegetable-image/train"
data_test_path = "/content/fruit-and-vegetable-image/test"
data_val_path = "/content/fruit-and-vegetable-image/validation"

img_width, img_height = 224, 224 
batch_size = 32

# 데이터 증강
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")), 
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(10), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")), 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageFolder(root=data_train_path, transform=train_transform)
val_dataset = ImageFolder(root=data_val_path, transform=test_transform)
test_dataset = ImageFolder(root=data_test_path, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

class_names = train_dataset.classes
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의
model = resnet50(pretrained=True)  
model.fc = nn.Linear(model.fc.in_features, num_classes)  
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5) 

# 학습
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_acc, val_acc, train_loss, val_loss = [], [], [], []

    for epoch in range(epochs):
        start_time = datetime.now()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_train_acc = 100 * correct / total
        epoch_train_loss = running_loss / len(train_loader)
        train_acc.append(epoch_train_acc)
        train_loss.append(epoch_train_loss)

        model.eval()
        correct, total, val_running_loss = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        epoch_val_acc = 100 * correct / total
        epoch_val_loss = val_running_loss / len(val_loader)
        val_acc.append(epoch_val_acc)
        val_loss.append(epoch_val_loss)

        end_time = datetime.now()

        insert_training_log(
            epoch=epoch + 1,
            start_time=start_time,
            end_time=end_time,
            train_loss=epoch_train_loss,
            train_accuracy=epoch_train_acc,
            val_loss=epoch_val_loss,
            val_accuracy=epoch_val_acc,
            early_stop=False
        )

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.2f}%, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_acc:.2f}%")

    return train_acc, val_acc, train_loss, val_loss


epochs = 10
train_acc, val_acc, train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

# 시각화
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, train_acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.title("Accuracy")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs_range, train_loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.title("Loss")
plt.legend()

plt.show()


# test 데이터 평가
def evaluate_model(model, test_loader, criterion):
    model.eval()  
    correct, total, test_loss = 0, 0, 0.0

    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_acc = 100 * correct / total  
    test_loss /= len(test_loader)  

    print(f"\n Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
    return test_acc, test_loss

test_acc, test_loss = evaluate_model(model, test_loader, criterion)


# 예측 결과 시각화
def predict_image(image_path, model):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]


test_folder = "/content/fruit-and-vegetable-image/test"
categories = ["apple", "banana", "carrot", "corn", "onion", "orange", "potato"]

fig, axes = plt.subplots(1, len(categories), figsize=(15, 5))  

for idx, category in enumerate(categories):
    category_path = os.path.join(test_folder, category) 
    if os.path.exists(category_path):  
        images = sorted(os.listdir(category_path))  
        if images:  
            first_image_path = os.path.join(category_path, images[0])  
            prediction = predict_image(first_image_path, model)  

            image = Image.open(first_image_path)
            axes[idx].imshow(image)
            axes[idx].axis("off") 

            if prediction == category:
                title = f" {category}\n(Pred: {prediction})"
                color = "green"
            else:
                title = f" {category}\n(Pred: {prediction})"
                color = "red"

            axes[idx].set_title(title, fontsize=10, color=color)
        else:
            axes[idx].set_title(f" No Image\n{category}", fontsize=10, color="gray")
    else:
        axes[idx].set_title(f" No Folder\n{category}", fontsize=10, color="gray")

plt.tight_layout()
plt.show()
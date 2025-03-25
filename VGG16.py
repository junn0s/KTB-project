# early stop 적용, 20회 학습 patience=3
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


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16
from PIL import Image
from datetime import datetime

data_train_path = "/content/fruit-and-vegetable-image/train"
data_test_path = "/content/fruit-and-vegetable-image/test"
data_val_path = "/content/fruit-and-vegetable-image/validation"

img_width, img_height = 224, 224
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageFolder(root=data_train_path, transform=transform)
val_dataset = ImageFolder(root=data_val_path, transform=transform)
test_dataset = ImageFolder(root=data_test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


class_names = train_dataset.classes
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의
model = vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, patience=3):
    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

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

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        
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

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100 * correct / total

        end_time = datetime.now()

        early_stop_flag = False
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop_flag = True

        insert_training_log(
            epoch=epoch + 1,
            start_time=start_time,
            end_time=end_time,
            train_loss=epoch_train_loss,
            train_accuracy=epoch_train_acc,
            val_loss=epoch_val_loss,
            val_accuracy=epoch_val_acc,
            early_stop=early_stop_flag  
        )
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        if early_stop_flag:  
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        train_acc.append(epoch_train_acc)
        train_loss.append(epoch_train_loss)
        val_acc.append(epoch_val_acc)
        val_loss.append(epoch_val_loss)

    return train_acc, val_acc, train_loss, val_loss


epochs = 20
train_acc, val_acc, train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)



# 시각화
epochs_range = range(len(train_acc))
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



# 예측 시각화
def predict_image(image_path, model):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # VGG16 정규화
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
            first_image_path = os.path.join(category_path, images[0])  # 첫 번째 이미지 선택
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


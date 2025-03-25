# 구글 코랩에서 데이터셋 다운로드 및 준비
# from google.colab import files
# files.upload()  # 'kaggle.json' 파일 업로드

# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# Orange Diseases 데이터셋 다운로드
# !kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition

# # 압축 해제
# !unzip fruit-and-vegetable-image-recognition.zip -d fruit-and-vegetable-image

# DB 연동
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
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, resnet50, mobilenet_v2, shufflenet_v2_x1_0
from PIL import Image
from datetime import datetime


# 25번 학습
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
from torchvision.transforms import ToTensor
from PIL import Image

data_train_path = "/content/fruit-and-vegetable-image/train"
data_test_path = "/content/fruit-and-vegetable-image/test"
data_val_path = "/content/fruit-and-vegetable-image/validation"

img_width, img_height = 180, 180
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root=data_train_path, transform=transform)
val_dataset = ImageFolder(root=data_val_path, transform=transform)
test_dataset = ImageFolder(root=data_test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


class_names = train_dataset.classes

# 다운받은 데이터 예시
def imshow(img, label):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.title(class_names[label])
    plt.axis('off')

plt.figure(figsize=(10, 10))
dataiter = iter(train_loader)
images, labels = next(dataiter)
for i in range(9):
    plt.subplot(3, 3, i + 1)
    imshow(images[i], labels[i])
plt.show()


# 모델 정의
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * (img_width // 8) * (img_height // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25):
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

        train_accuracy = 100 * correct / total
        train_avg_loss = running_loss / len(train_loader)

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

        val_accuracy = 100 * correct / total
        val_avg_loss = val_running_loss / len(val_loader)
        end_time = datetime.now()

        # DB에 저장
        insert_training_log(
            epoch=epoch + 1,
            start_time=start_time,
            end_time=end_time,
            train_loss=train_avg_loss,
            train_accuracy=train_accuracy,
            val_loss=val_avg_loss,
            val_accuracy=val_accuracy,
            early_stop=False  # 나중에 조건 넣을 수 있음
        )

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Val Loss: {val_avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        train_acc.append(train_accuracy)
        train_loss.append(train_avg_loss)
        val_acc.append(val_accuracy)
        val_loss.append(val_avg_loss)

    return train_acc, val_acc, train_loss, val_loss


epochs = 25
train_acc, val_acc, train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)


epochs_range = range(epochs)

# 시각화
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

# 평가
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


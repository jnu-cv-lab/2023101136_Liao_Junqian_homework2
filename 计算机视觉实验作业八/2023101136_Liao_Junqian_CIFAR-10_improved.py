# 环境准备
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", DeprecationWarning)

# 自动创建文件夹
os.makedirs("homework8", exist_ok=True)
os.makedirs("homework8/CIFAR10_improved", exist_ok=True)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch 版本:", torch.__version__)
print("使用设备:", device)

# 加载 CIFAR-10 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

batch_size = 512
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 训练样本图
plt.figure(figsize=(12, 4))
dataiter = iter(train_loader)
images, labels = next(dataiter)
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = images[i].permute(1, 2, 0) * 0.5 + 0.5
    plt.imshow(img)
    plt.title(f"true label: {classes[labels[i]]}")
    plt.axis('off')
plt.suptitle("Training Samples")
plt.savefig("homework8/CIFAR10_improved/training_samples.png", dpi=300)
plt.show()

# 改进后的 CNN 模型（适配 CIFAR-10）
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ===================== Adam 优化器训练 =====================
model = ImprovedCNN().to(device)
print("\n 改进后的 CNN 模型结构")
print(model)

epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, train_accs = [], []
val_losses, val_accs = [], []

print("\n开始训练 (Adam)")
for epoch in range(epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / train_total
    train_acc = 100 * train_correct / train_total
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)

    # 验证
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / val_total
    val_acc = 100 * val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    print(f'第 [{epoch+1}/{epochs}] 轮')
    print(f'训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:.2f}%')
    print(f'验证损失: {avg_val_loss:.4f} | 验证准确率: {val_acc:.2f}%\n')

# Adam 测试
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0
wrong_images, wrong_labels, wrong_preds = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_loss += criterion(outputs, labels).item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        mask = predicted != labels
        wrong_images.append(images[mask].cpu())
        wrong_labels.append(labels[mask].cpu())
        wrong_preds.append(predicted[mask].cpu())

adam_test_acc = 100 * test_correct / test_total
print(f"Adam 测试准确率: {adam_test_acc:.2f}%")

# 测试集预测图
plt.figure(figsize=(12,6))
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)
for i in range(10):
    plt.subplot(2,5,i+1)
    img = images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
    plt.imshow(img)
    plt.title(f"True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}")
    plt.axis('off')
plt.suptitle("Test Predictions (Adam)")
plt.savefig("homework8/CIFAR10_improved/test_predictions.png", dpi=300)
plt.show()

# 训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss', marker='s')
plt.title('Training and Validation Loss (Adam)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accs, label='Train Accuracy', marker='o')
plt.plot(range(1, epochs+1), val_accs, label='Val Accuracy', marker='s')
plt.title('Training and Validation Accuracy (Adam)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("homework8/CIFAR10_improved/training_curves.png", dpi=300)
plt.show()

# 错误样本
wrong_images = torch.cat(wrong_images)
wrong_labels = torch.cat(wrong_labels)
wrong_preds = torch.cat(wrong_preds)
plt.figure(figsize=(12,6))
for i in range(min(10, len(wrong_labels))):
    plt.subplot(2,5,i+1)
    img = wrong_images[i].permute(1, 2, 0) * 0.5 + 0.5
    plt.imshow(img)
    plt.title(f"T:{classes[wrong_labels[i]]} P:{classes[wrong_preds[i]]}")
    plt.axis('off')
plt.suptitle("Misclassified Samples (Adam)")
plt.savefig("homework8/CIFAR10_improved/misclassified_samples.png", dpi=300)
plt.show()

# 统计错误
print("\n最容易错的类别：")
for num, c in Counter(wrong_labels.numpy()).most_common(5):
    print(f"{classes[num]}：{c} 次")

# ===================== SGD 优化器训练 =====================
print("\n==================== SGD 优化器训练 ====================")

model_sgd = ImprovedCNN().to(device)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)

sgd_train_losses = []
sgd_train_accs = []
sgd_val_losses = []
sgd_val_accs = []

for epoch in range(epochs):
    # 训练阶段
    model_sgd.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_sgd.zero_grad()
        outputs = model_sgd(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_sgd.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / train_total
    train_acc = 100 * train_correct / train_total
    sgd_train_losses.append(avg_train_loss)
    sgd_train_accs.append(train_acc)

    # 验证阶段
    model_sgd.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_sgd(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / val_total
    val_acc = 100 * val_correct / val_total
    sgd_val_losses.append(avg_val_loss)
    sgd_val_accs.append(val_acc)

    print(f'第 [{epoch+1}/{epochs}] 轮')
    print(f'训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:.2f}%')
    print(f'验证损失: {avg_val_loss:.4f} | 验证准确率: {val_acc:.2f}%\n')

model_sgd.eval()
sgd_test_correct = 0
sgd_test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_sgd(images)
        _, predicted = torch.max(outputs.data, 1)
        sgd_test_correct += (predicted == labels).sum().item()
        sgd_test_total += labels.size(0)
sgd_test_acc = 100 * sgd_test_correct / sgd_test_total

print("\n" + "="*60)
print("                进阶任务2：优化器对比结果")
print("="*60)
print(f"  Optimizer   |   Learning Rate   |   Test Accuracy")
print(f"    Adam      |      0.001        |      {adam_test_acc:.2f}%")
print(f"    SGD       |      0.01         |      {sgd_test_acc:.2f}%")
print("="*60)
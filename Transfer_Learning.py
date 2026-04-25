import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import urllib.request
import zipfile

# ----------------------
# 1. 自动下载+加载 蚂蚁&蜜蜂小样本数据集
# ----------------------
def get_data_loaders(batch_size=8):
    # 下载并解压小样本数据集
    data_dir = './hymenoptera_data'
    if not os.path.exists(data_dir):
        print("正在下载蚂蚁&蜜蜂小样本数据集...")
        url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
        urllib.request.urlretrieve(url, './hymenoptera_data.zip')
        with zipfile.ZipFile('./hymenoptera_data.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
        os.remove('./hymenoptera_data.zip')

    # 数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 加载数据集（2分类：蚂蚁、蜜蜂）
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

# ----------------------
# 2. 训练函数
# ----------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        total_loss += loss.item() * images.size(0)
    
    avg_train_loss = total_loss / len(train_loader.dataset)
    avg_train_acc = correct / total
    return avg_train_loss, avg_train_acc

# ----------------------
# 3. 验证函数
# ----------------------
def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ----------------------
# 4. 主程序：仅训练分类头
# ----------------------
if __name__ == '__main__':
    # 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    lr = 0.001  # 仅训练分类头，学习率可稍大
    num_epochs = 10

    # 初始化数据加载器
    train_loader, val_loader = get_data_loaders(batch_size)

    # === 仅训练分类头：完全冻结ResNet所有层 ===
    model = models.resnet18(pretrained=True)
    # 冻结全部主干网络参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换分类头为2分类（仅这一层参与训练）
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # 损失函数 + 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    # 指标存储列表
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    # 训练循环
    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, val_loader, device)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'resnet_transfer_only_fc.pth')
    print("Model saved as resnet_transfer_only_fc.pth")

    # ----------------------
    # 可视化绘图
    # ----------------------
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 7))

    plt.plot(epochs, train_loss_list, 'b-', linewidth=2, label='train loss')
    plt.plot(epochs, train_acc_list, 'm--', linewidth=2, label='train acc')
    plt.plot(epochs, val_acc_list, 'g--', linewidth=2, label='val acc')

    plt.xlabel('epoch', fontsize=18)
    plt.xticks(range(2, 11, 2))
    plt.ylim(0, 2.4)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=18)
    plt.title('ResNet Transfer (Only FC Train)', fontsize=16)

    plt.savefig('resnet_only_fc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
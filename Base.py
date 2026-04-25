import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os


# ====================== 1. 数据预处理与加载 ======================
# 数据增强：训练集使用随机裁剪/翻转，验证集仅中心裁剪+归一化
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),  # 随机裁剪到224x224（ResNet输入尺寸）
        transforms.RandomHorizontalFlip(),    # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet预训练的均值
                             [0.229, 0.224, 0.225])  # ImageNet预训练的标准差
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),          # 中心裁剪到224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ---------------------- 加载数据集（二选一） ----------------------
# 选项A：使用CIFAR-10示例数据集（自动下载，10类图像）
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])

# 选项B：使用自己的数据集（按文件夹分类，结构如下）
# data/
#   train/
#     class1/  img1.jpg, img2.jpg...
#     class2/  img1.jpg, img2.jpg...
#   val/
#     class1/  img1.jpg, img2.jpg...
#     class2/  img1.jpg, img2.jpg...
# data_dir = 'path/to/your/data'
# train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
# val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

# 数据加载器
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = train_dataset.classes  # 类别名称
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 优先用GPU


# ====================== 2. 模型训练函数 ======================
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())  # 保存最佳模型权重
    best_acc = 0.0

    # 记录训练过程指标
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch分为训练和验证阶段
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()  # 切换模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # 梯度清零

                # 前向传播（仅训练时计算梯度）
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播+优化（仅训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 学习率调度（仅训练阶段）
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            # 计算epoch级指标
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录指标
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

            # 保存最佳验证准确率的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # 输出训练总时长和最佳结果
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epoch')
    plt.show()

    return model


# ====================== 3. 迁移学习配置（二选一） ======================
# ---------------------- 方式1：特征提取（冻结Backbone，仅训练分类头） ----------------------
print("=== 特征提取模式 ===")
# 加载预训练ResNet18（也可换ResNet50/101等）
model_ft = models.resnet18(pretrained=True)

# 冻结所有Backbone参数（不计算梯度）
for param in model_ft.parameters():
    param.requires_grad = False

# 修改分类头：ResNet18的fc层输入维度为512，输出改为你的类别数
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()  # 分类任务用交叉熵损失
optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)  # 仅优化分类头
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 每7轮学习率×0.1

# 开始训练
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)


# ---------------------- 方式2：微调（训练全部或部分层） ----------------------
print("\n=== 微调模式 ===")
# 重新加载预训练ResNet18
model_ft = models.resnet18(pretrained=True)

# 不冻结参数，全部可训练（也可只冻结前几层）
for param in model_ft.parameters():
    param.requires_grad = True

# 修改分类头
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)  # 学习率设小一点，避免破坏预训练权重
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 开始训练
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
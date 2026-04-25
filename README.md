# 迁移学习
基于 PyTorch 实现的 ResNet 迁移学习图像分类，包含 **特征提取** 和 **微调** 两种模式。
# Transfer Learning
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result

---

## 简介
迁移学习是机器学习与深度学习领域的核心学习范式，核心概念源自传统机器学习泛化思想，正式理论体系由杨强团队系统性建立，旨在**将源领域预训练模型习得的通用特征与先验知识，迁移复用至目标领域任务**，打破传统深度学习模型从零开始训练的局限。传统CNN模型需基于大规模数据随机初始化权重完整训练，存在小样本难收敛、训练成本高、泛化能力弱、算力消耗大等问题；而迁移学习依托ImageNet等大规模通用数据集完成预训练的骨干网络，借助视觉底层特征的通用性，实现知识跨任务、跨数据集迁移，大幅降低下游任务的数据依赖与训练门槛。在图像分类、目标检测等计算机视觉任务中，迁移学习已成为工业界与学术研究的主流方案，广泛适配VGG、ResNet、AlexNet等经典CNN架构，极大推动了小样本图像任务、轻量化模型部署与跨场景视觉应用的落地。

## 核心原理与架构逻辑
迁移学习区别于从零训练的完整学习模式，核心围绕**特征复用 + 选择性参数更新**两大核心逻辑展开，以CNN视觉模型为载体形成标准化应用范式，整体分为源域预训练、参数迁移、目标任务微调三大流程：
- **源域预训练阶段**：在数据量充足、类别丰富的大规模通用数据集（如ImageNet 1000类）上，对骨干网络进行完整训练。模型自动学习边缘、纹理、轮廓、色彩、局部语义等**通用底层视觉特征**，这些特征具备强普适性，不局限于单一特定任务。
- **参数迁移加载**：加载预训练骨干网络的完整权重，替代随机初始化参数。相比随机权重，预训练权重自带成熟视觉提取能力，可快速适配下游图像任务，避免浅层特征重复学习。
- **目标任务适配与分层训练**：结合目标数据集规模与任务差异，划分两种经典迁移策略，同时适配下游分类任务：
  1. 特征提取模式：冻结骨干网络全部卷积层与残差块参数，仅替换并训练末端分类头，全程固定通用特征提取能力，仅适配目标数据集类别分布，适合极小样本场景、快速迭代场景；
  2. 分层微调模式：可以自由冻结和解冻任意位置的卷积层或者全部解冻。我们在这里尝试，冻结网络浅层基础卷积层，**仅解冻训练网络最后一层残差块/卷积模块 + 末端分类头**，在保留通用底层特征的基础上，小幅修正高层语义特征，适配目标数据集差异化特征，兼顾泛化能力与任务适配性，也是本次实验采用的核心方案。

该范式规避了深层网络随机训练易梯度消失、过拟合的缺陷，以极小训练代价完成模型收敛，是后续轻量化模型训练、跨域图像分类、小数据场景建模的关键基础技术。

## 实验适配说明
本次实验基于**ResNet骨干网络**实现迁移学习，下游任务数据集选用CIFAR-10，包含10类通用物体彩色图像，图片尺寸为32×32。
原预训练权重基于ImageNet 224×224高清图像训练，因此实验中对输入图像做尺寸适配、标准化归一化处理；同时严格遵循迁移学习分层训练规则：冻结ResNet浅层layer1~layer3全部参数，仅开放最后一层layer4残差块与全连接分类头的参数更新，在保留预训练通用特征的前提下，适配CIFAR-10低分辨率、小尺寸图像特性，核心迁移学习逻辑与分层微调架构完全保留。

## 数据集
迁移学习的核心是跨数据集知识迁移，本次实验涉及**源域预训练数据集ImageNet**与**目标域任务数据集CIFAR-10**两个标准数据集，分别承担通用特征学习与下游任务验证的角色：

### 源域数据集：ImageNet
ImageNet是计算机视觉领域最具影响力的大规模通用图像数据集，由斯坦福大学李飞飞团队于2009年发布，是目前迁移学习预训练的黄金标准数据集。数据集包含超过1400万张标注彩色图像，覆盖1000个基础物体类别，涵盖动物、植物、交通工具、日常用品等几乎所有常见视觉概念，类别分布均衡且场景丰富。
其庞大的数据量与全面的类别覆盖，能够让模型学习到具有强普适性的底层视觉特征（边缘、纹理、形状）与高层语义特征，是ResNet等经典CNN骨干网络预训练的标准数据集。本次实验使用的ResNet-18预训练权重，正是基于ImageNet 1000类分类任务完整训练得到。

### 目标域数据集：CIFAR-10
本次迁移学习下游任务采用CIFAR-10数据集，是轻量化通用物体分类标准数据集，由Alex Krizhevsky、Vinod Nair与Geoffrey Hinton整理发布。数据集由60000张32×32像素RGB彩色图像组成，涵盖飞机、汽车、鸟类、猫、鹿、狗、蛙类、马、船、卡车共10个互斥基础类别，每个类别包含6000张图像。
数据集划分为50000张训练集与10000张测试集，数据量规模适中，图像分辨率低且包含一定噪声与姿态变化，适合验证迁移学习在中小规模数据集上的优化效果。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

数据集采用二进制压缩存储，减少冗余体积，无需额外转换为高清图片格式；本实验聚焦迁移学习算法与网络微调策略，不做数据集底层存储格式解析，专注模型迁移与特征复用的实验验证。

---


## Introduction
Transfer learning is a core learning paradigm in the fields of machine learning and deep learning. Its theoretical system was systematically established by Qiang Yang’s team. It aims to transfer and reuse the general features and prior knowledge learned by pre-trained models in the source domain to target domain tasks, breaking the limitations of traditional deep learning models trained from scratch. Traditional CNN models require complete training with random initialization based on large-scale data, which suffers from difficult convergence with small samples, high training costs, weak generalization ability and large computing power consumption. Relying on backbone networks pre-trained on large-scale general datasets such as ImageNet, transfer learning realizes cross-task and cross-dataset knowledge transfer through the universality of low-level visual features, greatly reducing data dependence and training thresholds for downstream tasks. In computer vision tasks such as image classification and object detection, transfer learning has become a mainstream solution for industry and academic research, widely adapting to classic CNN architectures such as VGG, ResNet and AlexNet, and greatly promoting the implementation of small-sample visual tasks, lightweight model deployment and cross-scene visual applications.

## Core Principles and Architecture Logic
Different from the complete training mode from scratch, transfer learning is centered on two core logics: **feature reuse and selective parameter updating**, forming a standardized application paradigm with CNN visual models as the carrier. It is divided into three major processes: source domain pre-training, parameter transfer, and target task fine-tuning:
- **Source Domain Pre-training Stage**: The backbone network is fully trained on large-scale general datasets with sufficient data and rich categories (such as ImageNet 1000 classes). The model automatically learns general low-level visual features such as edges, textures, contours, colors and local semantics. These features are highly universal and not limited to a single specific task.
- **Parameter Transfer Loading**: Load the complete weights of the pre-trained backbone network to replace randomly initialized parameters. Compared with random weights, pre-trained weights come with mature visual extraction capabilities, which can quickly adapt to downstream image tasks and avoid repeated learning of shallow features.
- **Target Task Adaptation and Layered Training**: Two classic transfer strategies are divided according to the scale of the target dataset and task differences to adapt to downstream classification tasks:
  1. Feature Extraction Mode: Freeze all convolutional layers and residual block parameters of the backbone network, only replace and train the final classification head, fix the general feature extraction capability throughout the whole process, and only adapt to the category distribution of the target dataset, suitable for small sample scenarios and rapid iteration scenarios;
  2. Layered Fine-tuning Mode: Freeze the shallow basic convolution layers of the network, only unfreeze and train the last residual block/convolution module and the final classification head of the network. On the premise of retaining general low-level features, slightly modify high-level semantic features to adapt to differentiated features of the target dataset, balancing generalization ability and task adaptability, which is also the core scheme adopted in this experiment.

This paradigm avoids the defects of gradient disappearance and overfitting caused by random training of deep networks, and completes model convergence at a very low training cost. It is a key basic technology for subsequent lightweight model training, cross-domain image classification and small data scenario modeling.

## Experimental Adaptation Note
In this experiment, transfer learning is implemented based on the ResNet backbone network, and the downstream task dataset is CIFAR-10, which contains 10 categories of general object color images with an image size of 32×32.
The original pre-trained weights are trained based on 224×224 high-definition images in ImageNet. Therefore, input image size adaptation and standardized normalization are performed in the experiment. At the same time, strictly follow the layered training rules of transfer learning: freeze all parameters of ResNet shallow layer1~layer3, and only open the parameter update of the last layer4 residual block and the fully connected classification head. While retaining the pre-trained general features, it adapts to the characteristics of low-resolution and small-size images in CIFAR-10. The core transfer learning logic and layered fine-tuning architecture are completely retained.

## Dataset
The core of transfer learning is cross-dataset knowledge transfer. This experiment involves two standard datasets: **the source domain pre-training dataset ImageNet** and **the target domain task dataset CIFAR-10**, which respectively undertake the roles of general feature learning and downstream task verification:

### Source Domain Dataset: ImageNet
ImageNet is the most influential large-scale general image dataset in the field of computer vision, released by Fei-Fei Li's team at Stanford University in 2009. It is the gold standard dataset for transfer learning pre-training. The dataset contains more than 14 million annotated color images, covering 1000 basic object categories including almost all common visual concepts such as animals, plants, vehicles and daily necessities, with balanced category distribution and rich scenes.
Its huge data volume and comprehensive category coverage enable the model to learn highly universal low-level visual features (edges, textures, shapes) and high-level semantic features, making it the standard dataset for pre-training classic CNN backbones such as ResNet. The ResNet-18 pre-trained weights used in this experiment are obtained from complete training on the ImageNet 1000-class classification task.

### Target Domain Dataset: CIFAR-10
The downstream task of this transfer learning experiment uses the CIFAR-10 dataset, a standard lightweight general object classification dataset compiled by Alex Krizhevsky, Vinod Nair and Geoffrey Hinton. It consists of 60,000 32×32 pixel RGB color images, covering 10 mutually exclusive basic categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck, with 6000 images per category.
The dataset is divided into 50,000 training images and 10,000 test images. With a moderate data scale, low image resolution and certain noise and pose variations, it is suitable for verifying the optimization effect of transfer learning on medium and small-scale datasets.
The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

The dataset is stored in binary compression to reduce redundant volume, and there is no need for additional conversion to high-definition image formats. This experiment focuses on transfer learning algorithms and network fine-tuning strategies, does not analyze the underlying storage format of the dataset, and focuses on the experimental verification of model transfer and feature reuse.

---
## 原文章 | Original article
1. Pan S J, Yang Q. A survey on transfer learning[J]. IEEE Transactions on knowledge and data engineering, 2009, 22(10): 1345-1359.
2. Yosinski J, Clune J, Bengio Y, et al. How transferable are features in deep neural networks?[C]//Advances in neural information processing systems. 2014.

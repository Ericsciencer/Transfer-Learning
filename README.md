# 迁移学习|Transfer Learning
基于 PyTorch 实现的 ResNet 迁移学习图像分类，包含 **特征提取** 和 **微调** 两种模式。
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result
- Train only the classification layer:
<img width="2480" height="1914" alt="resnet_only_fc_curve" src="https://github.com/user-attachments/assets/8e92568d-e6bc-4d0c-9a8f-7a142b9a8f69" />

- Train only the high-dimensional semantic and classification layers:
<img width="2480" height="1914" alt="resnet_transfer_curve" src="https://github.com/user-attachments/assets/18751230-c629-452c-9c57-dcf7cf0afa82" />

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
本次实验选用ResNet18作为基础骨干网络，放弃样本充足的CIFAR-10数据集，改用**蚂蚁&蜜蜂细粒度小样本数据集**，最大化凸显迁移学习的应用价值。
源域预训练权重基于ImageNet 224×224标准图像训练，因此实验统一将输入图像缩放归一化至224×224尺寸，匹配预训练网络输入规范。
实验设置两组对照训练：第一组固定全部骨干权重，仅训练分类头；第二组冻结浅层layer1~layer3，仅开放layer4最后残差块与分类头参数更新。依托少量数百张训练图像，对比两种迁移策略的收敛速度、训练损失与验证准确率，直观证明预训练特征迁移对小样本分类任务的显著提升效果。

## 数据集
本次迁移学习实验采用**源域通用大数据集 ImageNet** 与 **目标域小样本数据集 Ants & Bees** 双数据集组合，形成大小数据跨域知识迁移，精准体现迁移学习核心优势：

### 源域数据集：ImageNet
ImageNet 是计算机视觉领域公认的大型基准预训练数据集，由斯坦福大学团队构建，包含超1400万张多类RGB标注图像，涵盖1000种不同物体类别，覆盖动物、植物、交通工具、生活用品等丰富自然场景。
凭借海量且多样化的样本分布，ResNet在该数据集训练后，可稳定提取边缘、纹理、色彩、形状等通用基础特征，特征泛化能力强、可迁移性高，是目前所有主流CNN模型预训练的标准数据源，本实验所用ResNet18权重均基于该数据集训练得到。

### 目标域数据集：Ants & Bees
为贴合小样本真实应用场景，实验采用迁移学习经典标准数据集 Ants & Bees，仅包含**蚂蚁、蜜蜂**两个高度相似类别，分类难度高、样本总量极少。
数据集整体仅包含398张图像，其中训练集245张、验证集153张，依靠数百张极小体量样本完成模型训练。在该数据条件下，从零初始化训练会发生严重过拟合，模型完全无法正常分类；而借助迁移学习策略，依托ImageNet预训练知识，模型可快速拟合小众细粒度特征，实现高精度分类。
数据集开源地址：https://download.pytorch.org/tutorial/hymenoptera_data.zip

数据集采用标准文件夹分类存储，结构简洁、无需格式转换，适合小样本迁移学习算法验证与对比实验。

---


# Introduction
Transfer learning is a core learning paradigm in machine learning and deep learning. Its core concept derives from the generalization theory of traditional machine learning, and its formal theoretical system was systematically established by Yang Qiang’s research team. It aims to **transfer and reuse universal features and prior knowledge learned by pre-trained models in the source domain to downstream tasks in the target domain**, breaking the limitations of training traditional deep learning models from scratch. Traditional CNN models require full training with randomly initialized weights based on large-scale datasets, suffering from difficulties in convergence with limited samples, high training costs, weak generalization ability and massive computing resource consumption. In contrast, transfer learning leverages backbone networks pre-trained on large-scale universal datasets such as ImageNet. Relying on the universality of low-level visual features, it realizes cross-task and cross-dataset knowledge transfer, greatly reducing data dependence and training thresholds for downstream tasks. In computer vision tasks including image classification and object detection, transfer learning has become a mainstream solution in both industry and academic research. It is widely compatible with classic CNN architectures such as VGG, ResNet and AlexNet, significantly promoting the implementation of few-shot image tasks, lightweight model deployment and cross-scenario visual applications.

# Core Principles and Architectural Logic
Different from the full learning mode of training from scratch, transfer learning centers on two core logics: **feature reuse and selective parameter updating**. With CNN visual models as the carrier, it forms a standardized application paradigm, consisting of three major processes: source domain pre-training, parameter transfer and target task fine-tuning.
- **Source Domain Pre-training**: The backbone network is fully trained on large-scale universal datasets with sufficient data volume and rich categories (e.g., ImageNet with 1,000 classes). The model automatically learns universal low-level visual features such as edges, textures, contours, colors and local semantics. These features possess strong universality and are not limited to a single specific task.
- **Parameter Transfer Loading**: Load the complete weights of the pre-trained backbone network to replace randomly initialized parameters. Compared with random weights, pre-trained weights come with mature visual feature extraction capabilities, enabling rapid adaptation to downstream image tasks and avoiding repeated learning of shallow features.
- **Target Task Adaptation and Hierarchical Training**: Two classic transfer strategies are defined according to the scale of target datasets and task differences to adapt to downstream classification tasks:
  1. Feature Extraction Mode: Freeze all convolutional layers and residual block parameters of the backbone network, only replacing and training the final classification head. The universal feature extraction capability is fixed throughout the process to adapt only to the category distribution of target datasets, suitable for extremely few-shot scenarios and rapid iteration tasks.
  2. Hierarchical Fine-tuning Mode: Freeze or unfreeze convolutional layers at any position or all layers freely. In this experiment, shallow basic convolutional layers of the network are frozen, and **only the last residual block/convolutional module and the final classification head are unfrozen and trained**. While retaining universal low-level features, high-level semantic features are slightly adjusted to adapt to differentiated features of target datasets, balancing generalization ability and task adaptability. This serves as the core strategy adopted in this experiment.

This paradigm avoids the defects of gradient vanishing and overfitting caused by random training of deep networks, and achieves model convergence at a low training cost. It acts as a fundamental technology for lightweight model training, cross-domain image classification and modeling in small-data scenarios.

# Experimental Adaptation Description
This experiment adopts ResNet18 as the basic backbone network. Instead of the well-sampled CIFAR-10 dataset, the **fine-grained few-shot Ants & Bees dataset** is adopted to fully highlight the application value of transfer learning.
The source domain pre-trained weights are trained on standard 224×224 images from ImageNet. Therefore, all input images in the experiment are uniformly resized and normalized to 224×224 to match the input specifications of the pre-trained network.

Two groups of comparative training are set up in the experiment: the first group fixes all backbone weights and only trains the classification head; the second group freezes shallow layers from layer1 to layer3, and only enables parameter updating for the last residual block of layer4 and the classification head. With only hundreds of training images, the convergence speed, training loss and validation accuracy of the two transfer strategies are compared to intuitively verify the significant improvement of pre-trained feature transfer on few-shot classification tasks.

# Datasets
This transfer learning experiment adopts a dual-dataset combination of **ImageNet (large-scale universal source domain dataset)** and **Ants & Bees (small-sample target domain dataset)**, realizing cross-domain knowledge transfer between large and small datasets and accurately reflecting the core advantages of transfer learning.

## Source Domain Dataset: ImageNet
As a recognized large-scale benchmark pre-training dataset in computer vision, ImageNet was constructed by Stanford University. It contains over 14 million labeled RGB images covering 1,000 object categories, including animals, plants, transportation vehicles, daily necessities and various natural scenes.
Benefiting from its massive and diversified sample distribution, ResNet trained on this dataset can stably extract universal basic features such as edges, textures, colors and shapes with strong generalization and high transferability. It is the standard data source for pre-training all mainstream CNN models at present, and the ResNet18 weights used in this experiment are all derived from training on this dataset.

## Target Domain Dataset: Ants & Bees
To fit the real-world few-shot application scenarios, this experiment adopts Ants & Bees, a classic standard dataset for transfer learning. It contains only two highly similar categories: ants and bees, with high classification difficulty and an extremely small total sample size.
The dataset includes merely 398 images in total, with 245 images in the training set and 153 in the validation set. Model training is completed with only hundreds of samples. Under such data conditions, training with randomly initialized weights from scratch will lead to severe overfitting and failure in normal classification. In contrast, with transfer learning and prior knowledge pre-trained on ImageNet, the model can quickly fit fine-grained features of niche categories and achieve high-precision classification.

Dataset open-source link: https://download.pytorch.org/tutorial/hymenoptera_data.zip

The dataset adopts a standard folder-based classified storage structure with a concise format and no conversion required, making it suitable for algorithm verification and comparative experiments on few-shot transfer learning.


---
## 原文章 | Original article
1. Pan S J, Yang Q. A survey on transfer learning[J]. IEEE Transactions on knowledge and data engineering, 2009, 22(10): 1345-1359.
2. Yosinski J, Clune J, Bengio Y, et al. How transferable are features in deep neural networks?[C]//Advances in neural information processing systems. 2014.

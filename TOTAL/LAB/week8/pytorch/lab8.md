
# 60 分钟 PyTorch 极速入门：从 Tensor 到 CIFAR-10 分类（超详细）

> 作者：南方的狮子先生
> 日期：2025-10  
> 关键词：PyTorch、深度学习、CNN、CIFAR-10、Autograd、CUDA、初学者

---

## 1. 写在最前：为什么选 PyTorch？


**NumPy 只负责“科学计算”，PyTorch 专为“深度学习”而生**——两者目标不同，功能重叠但不冲突。

| 维度 | NumPy | PyTorch |
|---|---|---|
| **核心定位** | 通用多维数组库 | 深度学习自动求导框架 |
| **计算图** | ❌ 无 | ✅ 动态图自动构建 |
| **自动求导** | ❌ 手工推导链式法则 | ✅ 自动反向传播 |
| **GPU 加速** | ❌（需 CuPy 等外挂） | ✅ 原生 `.cuda()` |
| **稀疏/量化/分布式** | ❌ | ✅ 内置多种训练策略 |
| **部署工具链** | ❌ | ✅ TorchScript、ONNX、TensorRT |
| **社区生态** | 科学计算 | 预训练模型、数据集、Hub |

形象比喻：

- **NumPy 像一把瑞士军刀**：削铅笔、开瓶盖都能干，但砍树费力。
- **PyTorch 像电锯**：专为砍树（训练神经网络）设计，插上电（GPU）效率爆表；也能削铅笔，但没必要。

所以：

1. 只做传统数值模拟 → NumPy 足够。
2. 要做深度学习 → 直接用 PyTorch，少踩 90% 的坑。




| 特性 | 一句话总结 |
|---|---|
| **动态图** | 写代码就像写 Python，调试无痛 |
| **GPU 加速** | `.cuda()` 一行搞定，速度飞起 |
| **生态丰富** | torchvision、torchaudio、transformers 全家桶 |

---

## 2. 环境准备（1 分钟）

```bash
# CPU 版
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU 版（以 CUDA 11.8 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

验证安装：

```python
import torch
print(torch.__version__)          # 2.x.x
print(torch.cuda.is_available())  # True 说明 GPU 可用
```

---

## 3. Tensor：NumPy 的超级加强版

| 操作 | NumPy | PyTorch |
|---|---|---|
| 创建矩阵 | `np.zeros((3,3))` | `torch.zeros(3,3)` |
| 矩阵乘法 | `a @ b` | `a @ b` 或 `torch.mm(a,b)` |
| GPU 加速 | ❌ | `a.cuda()` |
| 自动求导 | ❌ | `a.requires_grad=True` |

代码速览：

```python
import torch

# 1. 创建
x = torch.rand(5, 3)          # 均匀分布
y = torch.zeros(5, 3, dtype=torch.long)
z = torch.tensor([[1, 2], [3, 4]])

# 2. 运算
print(x + y)
print(x.add_(y))              # 原地加法（带下划线）

# 3. 切片
print(x[:, 1])

# 4. 改变形状
v = x.view(-1, 8)             # -1 表示自动推断

# 5. 与 NumPy 互转
import numpy as np
np_array = x.numpy()          # Tensor -> ndarray
x2 = torch.from_numpy(np_array)
```

---

## 4. Autograd：自动求导黑科技

核心：只要 `requires_grad=True`，PyTorch 会帮你构建**计算图**，调用 `.backward()` 就能自动求导。

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward()                # 反向传播
print(x.grad)                 # ∂out/∂x = 4.5
```

关闭梯度（推理阶段）：

```python
with torch.no_grad():
    print((x * 2).requires_grad)  # False
```

---

## 5. 搭建你的第一个神经网络（LeNet）

网络结构：  
`3×32×32` → Conv2d(3,6,5) → ReLU → MaxPool(2) → Conv2d(6,16,5) → ReLU → MaxPool(2) → 展平 → 120 → 84 → 10

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
```

---

## 6. 损失函数 & 优化器

| 组件 | 常用选择 |
|---|---|
| 损失函数 | `nn.CrossEntropyLoss()`（分类） |
| 优化器 | `optim.SGD` / `optim.Adam` |

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

---

## 7. 训练循环（万能模板）

```python
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()        # 1. 清零梯度
        outputs = net(inputs)        # 2. 前向
        loss = criterion(outputs, labels)
        loss.backward()              # 3. 反向
        optimizer.step()             # 4. 更新权重

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
```

---

## 8. CIFAR-10 完整实战（含数据加载）

### 8.1 数据准备（torchvision 一键搞定）

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),                      # 0-1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # -1-1
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
```

### 8.2 训练 + 测试

把第 7 步的 `trainloader` 换成 CIFAR-10 即可。  
测试准确率（2 epoch，CPU）≈ **54%**（10 类随机 10%，已学到东西！）

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.0f}%')
```

---

## 9. GPU 加速：2 行代码搞定

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)                 # 模型搬过去
inputs, labels = inputs.to(device), labels.to(device)  # 数据搬过去
```

> 小网络 CPU 也能跑；增大通道数/层数后 GPU 速度优势明显。

---

## 10. 常见问题 & 排坑指南

| 报错/现象 | 解决 |
|---|---|
| `CUDA out of memory` | 减小 `batch_size` |
|  loss 震荡不降 | 调低学习率、加 `BatchNorm` |
| 准确率一直 10% | 忘记 `optimizer.zero_grad()` |
| 图片显示全黑 | 忘记 `img / 2 + 0.5` 反归一化 |

---

## 11. 下一步学什么？

| 方向 | 资源 |
|---|---|
| 更深的 CNN | ResNet、DenseNet（torchvision 现成） |
| 数据增强 | `transforms.RandomCrop`、`RandAugment` |
| 学习率调度 | `torch.optim.lr_scheduler.CosineAnnealingLR` |
| 迁移学习 | 预训练 ImageNet 模型微调 |
| 可视化 | TensorBoard、Netron、Grad-CAM |

---


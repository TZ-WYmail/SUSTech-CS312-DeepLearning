<h1 align="center">🧠 Deep Learning (CS324) - Course Hub</h1>

<p align="center">
  南方科技大学《深度学习》课程一站式资源仓库
  <br />
  <a href="#-about"><strong>探索本仓库 »</strong></a>
  <br />
  <br />
</p>




## 🎯 关于课程

&gt; 登录即可解锁全部功能与服务 🚀

本仓库为 **南方科技大学（SUSTech）计算机科学与工程系**  
由 **张建国教授** 开设的《深度学习 CS324》课程资源汇总。

在这里你可以找到：

* 📘 课程讲义与课件
* 🧪 实践作业与参考实现
* 🎙️ 学生展示模板与往届优秀报告
* 💬 期末复习 Q&A 精选
* 🚀 在线 Demo（AI on Chips）


---


## 🔍 核心主题速览 

| 模块 | 关键词 |
| ---- | ------ |
| 深度强化学习 | 强化学习定义、Bellman 方程、Deep RL、Q-Learning、稳定性问题 |
| 策略与模型 | 基于策略的深度 RL、基于模型的深度 RL |
| 学生自选 | 最新 SOTA 架构（Vision Transformer、Diffusion Model...） |

---




## 🛠️ 编程环境配置指南（Windows） 

&gt; ⚙️ 本课程含 3 个编程作业，需使用 **Python + PyTorch**。  
&gt; 提供两种安装方式：① `pip`（轻量）② `conda`（推荐，隔离性强）。

---

### 1️⃣ 方案 A： pip 安装（适合已有 Python 基础）

| 步骤 | 命令 / 说明 |
| ---- | ----------- |
| ① 安装 Python | 下载 [Python 3.7.x](https://www.python.org/downloads/) 勾选 **Add to PATH** + **Install for all users** |
| ② 换清华源（可选） | `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple` |
| ③ 安装科学栈 | 打开 **CMD** 或 **PowerShell** 依次执行：&lt;br&gt;`pip install numpy matplotlib scipy scikit-learn jupyter` |
| ④ 安装 PyTorch | 访问 [pytorch.org](https://pytorch.org) 选择 **CPU 版本** 生成命令，例如：&lt;br&gt;`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |
| ⑤ 启动 Jupyter | `jupyter notebook` 浏览器自动打开 |

---

### 2️⃣ 方案 B： conda 安装（推荐 · 一键搞定）

| 步骤 | 操作 |
| ---- | ---- |
| ① 安装 Anaconda | 下载 [Anaconda](https://www.anaconda.com/) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) |
| ② 打开终端 | **开始菜单** → **Anaconda Powershell Prompt** |
| ③ 创建隔离环境 | `conda create -n cs324 python=3.7` |
| ④ 激活环境 | `conda activate cs324` |
| ⑤ 安装依赖 | `conda install numpy matplotlib scipy scikit-learn jupyter` |
| ⑥ 安装 PyTorch | `conda install pytorch torchvision torchaudio cpuonly -c pytorch` |
| ⑦ 启动 Jupyter | `jupyter notebook` |

&gt; 💡 **提示**：conda 环境可随时备份 / 分享，命令 `conda env export &gt; environment.yml`

---

## 🚀 VS Code 高效开发配置（可选但强烈推荐）

### 1. 安装 VS Code
1. 官网 [code.visualstudio.com](https://code.visualstudio.com) 下载 Windows 版安装包
2. 一路 **Next** 即可，建议勾选
   - 添加到 PATH
   - 右键菜单“Open with Code”

### 2. 必装插件
| 插件 | 用途 |
| ---- | ---- |
| **Python** | 语法高亮、调试、虚拟环境识别 |
| **Jupyter** | 原生 `.ipynb` 支持，变量查看器 |
| **Pylance** | 超快补全与类型检查 |

&gt; 安装完按 `Ctrl+Shift+P` → 输入 **Python: Select Interpreter** 选择刚创建的 `cs324` 环境即可。

### 3. 快速验证
新建 `hello.py` 写入
```python
import torch, numpy as np, matplotlib.pyplot as plt
print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())

x = np.linspace(0, 2*np.pi)
plt.plot(x, np.sin(x))
plt.title("Hello CS324")
plt.show()


```
右键 → **Run Python File in Terminal** 看到弹出的正弦图即成功！

---

## 📓 VS Code 中使用 Jupyter Notebook

| 功能 | 快捷方式 |
| ---- | -------- |
| 新建 Notebook | `Ctrl+Shift+P` → **Python: Create Blank New Jupyter Notebook** |
| 切换内核 | 右上角 **Kernel** → 选 `cs324` |
| 运行单元 | `Shift+Enter` |
| 变量查看器 | 左侧 **Jupyter 选项卡** → **Variables** |
| 信任 Notebook | 首次打开非本机文件时点击 **Trust** 按钮（防恶意代码） |

---

## 🧪 常见问题速查

| 现象 | 解决 |
| ---- | ---- |
| `pip install` 超时 | 换清华源 / 手机热点 |
| conda 环境丢失 | `conda env list` 查看路径后重新激活 |
| VS Code 找不到解释器 | 手动指定 `...\Anaconda3\envs\cs324\python.exe` |
| Jupyter 无法启动 | 确认当前已激活环境并 `conda install jupyter` |

---

> ✅ **环境检查清单**  
> 在终端依次执行，全部不报错即配置完成：
```bash
python -c "import torch, numpy, matplotlib, sklearn, jupyter; print('All OK')"
```

---



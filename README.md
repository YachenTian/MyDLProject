# MyDLProject

这是一个使用 PyTorch 构建的简单深度学习项目，用于识别手写数字（MNIST 数据集）。

## 项目功能

- 使用多层感知机（MLP）模型
- 自动下载 MNIST 数据
- 支持 GPU / CPU
- 精度可达约 97%
- 可拓展为 CNN、ResNet 等结构

## 快速开始

```bash
# 安装依赖
pip install torch torchvision

# 运行项目
python main.py

目录结构
MyDLProject/
├── main.py        # 主程序：训练 & 测试
├── README.md      # 项目说明
└── data/          # MNIST 数据（自动下载）

# EuroSAT MLP

本项目使用 **手工实现 MLP（支持单隐藏层/双隐藏层）** 完成 EuroSAT 10 分类任务，
不依赖 PyTorch / TensorFlow / JAX 自动微分。

## 已实现功能

- 数据加载与预处理：
  - 从 `EuroSAT_RGB/` 读取图片
  - 按类别分层切分 train/val/test
  - 训练集统计量归一化（channel-wise）
- 模型定义：
  - 自定义 `hidden1_dim` / `hidden2_dim`（`hidden2_dim<=0` 时退化为单隐藏层）
  - 激活函数可切换：`relu` / `sigmoid` / `tanh`
  - 手写前向传播与反向传播梯度
- 训练流程：
  - SGD
  - 学习率调度：`step` 或 `exp`
  - 交叉熵损失
  - L2 正则化（Weight Decay）
  - 梯度裁剪（可选）
  - 数据增强（翻转/旋转/亮度扰动，可开关）
  - 按验证集准确率自动保存最优权重
- 测试评估：
  - 加载最优模型权重
  - 输出测试集准确率
  - 输出混淆矩阵与分类报告
- 超参数查找：
  - 网格搜索 / 随机搜索
  - 记录不同超参数组合性能
- 可视化与分析：
  - 训练曲线
  - 混淆矩阵图（原始/归一化）
  - 第一层权重可视化
  - 错例图片网格与错例 JSON 记录

## 目录结构

```text
EuroSAT_MLP/
├── EuroSAT_RGB/
├── requirements.txt
├── README.md
├── scripts/
│   ├── train.py
│   ├── test.py
│   └── tune.py
└── src/
    ├── data.py
    ├── model.py
    ├── trainer.py
    ├── evaluate.py
    ├── search.py
    ├── visualize.py
    ├── analysis.py
    └── utils.py
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练

```bash
python scripts/train.py \
  --data-dir EuroSAT_RGB \
  --epochs 200 \
  --batch-size 128 \
  --early-stop-patience 20 \
  --hidden1-dim 512 \
  --hidden2-dim 256 \
  --activation relu \
  --learning-rate 0.01 \
  --lr-schedule step \
  --lr-step-size 20 \
  --lr-gamma 0.5 \
  --weight-decay 5e-5 \
  --momentum 0.9 \
  --augment
```

训练输出保存在 `runs/train_时间戳/`，包含：
- `checkpoints/best_model.npz`
- `history.json` / `summary.json`
- 图像：训练曲线、混淆矩阵、第一层权重、错例展示

## 测试

```bash
python scripts/test.py \
  --checkpoint runs/train_xxx/checkpoints/best_model.npz \
  --data-dir EuroSAT_RGB
```

## 超参数搜索

```bash
python scripts/tune.py \
  --data-dir EuroSAT_RGB \
  --mode random \
  --num-trials 20 \
  --epochs 200 \
  --batch-size 128 \
  --early-stop-patience 20 \
  --hidden1-dims 512,768,1024 \
  --hidden2-dims 256,384,512 \
  --activations relu \
  --learning-rates 0.012,0.01,0.008,0.006 \
  --lr-schedule step \
  --lr-step-sizes 15,20,25 \
  --lr-gammas 0.4,0.5,0.6 \
  --weight-decays 1e-5,3e-5,5e-5,1e-4 \
  --momentums 0.85,0.9,0.95 \
  --augment
```

搜索结果保存在 `runs/search_时间戳/search_results.json`。

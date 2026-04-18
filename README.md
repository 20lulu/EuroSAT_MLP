# EuroSAT MLP

本项目使用 **手工实现三层 MLP（输入层-隐藏层-输出层）** 完成 EuroSAT 10 分类任务，
不依赖 PyTorch / TensorFlow / JAX 自动微分。

## 已实现功能

- 数据加载与预处理：
  - 从 `EuroSAT_RGB/` 读取图片
  - 按类别分层切分 train/val/test
  - 训练集统计量归一化（channel-wise）
- 模型定义：
  - 自定义 `hidden_dim`
  - 激活函数可切换：`relu` / `sigmoid` / `tanh`
  - 手写前向传播与反向传播梯度
- 训练流程：
  - SGD
  - 学习率衰减（`lr = lr0 * decay^(epoch-1)`）
  - 交叉熵损失
  - L2 正则化（Weight Decay）
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
  --epochs 100 \
  --batch-size 128 \
  --hidden-dim 512 \
  --activation sigmoid \
  --learning-rate 0.05 \
  --lr-decay 0.98 \
  --weight-decay 5e-4
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
  --hidden-dims 128,256,512 \
  --activations relu,sigmoid,tanh \
  --learning-rates 0.1,0.05,0.03,0.01 \
  --lr-decays 0.99,0.985,0.98 \
  --weight-decays 0.0,1e-4,5e-4,1e-3 \
  --epochs 100 \
  --batch-size 128 \
  --seed 42

```

搜索结果保存在 `runs/search_时间戳/search_results.json`。

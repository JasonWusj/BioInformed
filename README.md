# Biophysics Informed Pathological Regularisation — 2D 最小复现

基于论文 *"Biophysics Informed Pathological Regularisation for Brain Tumour Segmentation"* (arXiv:2403.09136v3) 的 2D 简化复现版本。

---

## 论文核心思想

在标准分割网络（UNet）的训练中，引入**生物物理先验**作为正则化：

1. **Fisher-KPP 反应扩散方程**约束肿瘤细胞密度的空间分布
2. **Neumann 边界条件**约束密度在域边界处零通量

总损失函数：

```
L_total = L_dice + λ₁ · L_PDE + λ₂ · L_BC
```

其中 PDE 损失强制网络学到的特征满足：

```
∂u/∂t = D · ∇²u + ρ · u · (1 - u)
```

- `D` ∈ [0.02, 1.5] mm²/day（扩散系数）
- `ρ` ∈ [0.002, 0.2] /day（增殖率）

---

## 项目结构

```
biophysics_informed_seg/
├── pyproject.toml              # 项目依赖（uv 管理）
├── smoke_test.py               # 环境验证脚本
├── configs/
│   ├── default.yaml            # 训练超参数配置（带生物物理正则化）
│   └── baseline.yaml           # 对照实验配置（仅 Dice loss）
├── data/
│   ├── README.md               # 数据集下载说明
│   └── BraTS2023/              # ← 数据集放这里
│       ├── BraTS-GLI-00000-000/
│       ├── BraTS-GLI-00002-000/
│       └── ...
└── src/
    ├── __init__.py
    ├── model/
    │   ├── unet2d.py           # 2D UNet 骨干网络
    │   └── density_estimator.py # SIREN MLP 密度估计器
    ├── losses.py              # Dice + PDE + BC 损失函数
    ├── dataset.py              # BraTS 2D 切片数据加载（NIfTI）
    ├── dataset_fast.py         # 快速 .npy 数据加载（推荐）
    ├── preprocess.py           # NIfTI → .npy 预处理脚本
    ├── train.py                # 训练脚本
    └── evaluate.py             # 评估脚本（Dice + HD95）
```

---

## 环境搭建

```bash
cd biophysics_informed_seg

# 创建虚拟环境
uv venv

# 安装 PyTorch（CUDA 12.1）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
uv pip install nibabel monai scipy scikit-image pyyaml tqdm tensorboard

# 验证环境
.venv\Scripts\python.exe smoke_test.py
```

---

## 数据集准备

### 下载

1. 注册 Synapse 平台：https://www.synapse.org/
2. 进入 BraTS 2023 挑战页面：https://www.synapse.org/#!Synapse:syn51156910
3. 下载 `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`
4. 解压到 `data/BraTS2023/` 目录下

### 数据集规模

| 项目 | 数值 |
|------|------|
| 病例数 | 1251 |
| 压缩包大小 | ~13 GB |
| 解压后大小 | ~8-10 GB |
| 每个病例文件数 | 5（4 模态 + 1 分割标签） |
| 单个体积尺寸 | 240 × 240 × 155 |
| MRI 模态 | T1, T1ce, T2, FLAIR |

### 目录结构要求

```
data/BraTS2023/
    BraTS-GLI-00000-000/
        BraTS-GLI-00000-000-t1n.nii.gz    # T1 native
        BraTS-GLI-00000-000-t1c.nii.gz    # T1 contrast-enhanced
        BraTS-GLI-00000-000-t2w.nii.gz    # T2 weighted
        BraTS-GLI-00000-000-t2f.nii.gz    # T2 FLAIR
        BraTS-GLI-00000-000-seg.nii.gz    # 分割标签
    BraTS-GLI-00002-000/
        ...
```

---

## 数据预处理（推荐）

将 NIfTI 转为 .npy 可获得 ~50x I/O 加速：

```bash
# 预处理：NIfTI → .npy 切片（一次性，约 10-20 分钟）
.venv\Scripts\python.exe src/preprocess.py --data_dir ./data/BraTS2023 --output_dir ./data/preprocessed
```

预处理后目录结构：
```
data/preprocessed/
├── metadata.npy                              # 切片索引
└── slices/
    ├── BraTS-GLI-00000-000_z045_image.npy   # (4, 128, 128) float32
    ├── BraTS-GLI-00000-000_z045_seg.npy     # (128, 128) int8
    └── ...
```

预处理后存储约 **3-6 GB**（取决于有效切片数）。

在 `configs/default.yaml` 中设置 `use_fast_loader: true` 即可启用快速加载（默认已开启）。

---

## 数据加载流程详解

数据加载在 `src/dataset.py` 中实现，流程如下：

### 1. 病例发现与切片筛选

```
扫描 data_dir 下所有 BraTS-* 文件夹
    → 对每个病例，加载 seg.nii.gz
    → 遍历所有轴向切片 (z=0 到 z=154)
    → 只保留肿瘤像素占比 ≥ 1% 的切片
    → 记录 (case_id, slice_index) 对
```

这一步在 `__init__` 中完成，确保训练时只使用有意义的切片（含肿瘤区域），避免大量空白背景切片浪费训练资源。

### 2. 单个样本加载 (`__getitem__`)

```
输入: (case_id, z_index)

Step 1: 加载 4 个模态的 NIfTI 文件
        → 取第 z 层切片 → 得到 4 张 (240×240) 的 2D 图像

Step 2: Z-score 标准化（逐通道，仅对非零体素）
        → mean/std 计算排除背景（值为 0 的体素）
        → 裁剪到 [-5, 5] 去除异常值

Step 3: 标签重映射
        → BraTS 原始标签: 0(背景), 1(NCR), 2(ED), 4(ET)
        → 重映射为连续值: 0(背景), 1(NCR), 2(ED), 3(ET)

Step 4: Resize 到目标尺寸 (128×128)
        → 图像: 双线性插值 (order=1)
        → 标签: 最近邻插值 (order=0)，保持离散值

Step 5: 数据增强（仅训练集）
        → 随机水平翻转 (p=0.5)
        → 随机垂直翻转 (p=0.5)

Step 6: 转为 One-hot 编码
        → 标签 (128×128) → (4, 128, 128) 的 one-hot 张量
```

### 3. 数据集划分

```python
# 按病例划分（非按切片），避免数据泄露
全部 1251 个病例
    → 70% 训练 (~876 例)
    → 10% 验证 (~125 例)
    → 20% 测试 (~250 例)

# 每个病例约产生 30-60 张有效切片
# 总训练切片数约 26000-52000 张
```

### 4. 数据流示意图

```
NIfTI 3D Volume (240×240×155)
        │
        ▼ 取轴向切片 z
2D Slice (240×240) × 4 modalities
        │
        ▼ Z-score 归一化 + Resize
Normalized Tensor (4, 128, 128)  ←── 模型输入
        │
        ▼ UNet2D forward
Logits (4, 128, 128)  ←── 分割预测
Bottleneck Features (1024, 4, 4)
        │
        ▼ DensityEstimator (SIREN MLP)
u_hat (1, 16, 16)  ←── 肿瘤细胞密度估计
        │
        ▼ 计算损失
L = L_dice(logits, target) + L_PDE(u_hat) + L_BC(u_hat)
```

---

## 运行

### 1. 预处理数据（推荐，一次性）

```bash
.venv\Scripts\python.exe src/preprocess.py --data_dir ./data/BraTS2023 --output_dir ./data/preprocessed
```

### 2. 训练

```bash
# 带生物物理正则化（论文方法）
.venv\Scripts\python.exe src/train.py --config configs/default.yaml

# 纯 Dice loss（baseline 对照）
.venv\Scripts\python.exe src/train.py --config configs/baseline.yaml
```

### 3. 评估

```bash
.venv\Scripts\python.exe src/evaluate.py
```

训练参数在 `configs/default.yaml` 中配置：
- Epochs: 175
- Batch size: 8
- 学习率: 3e-4（余弦退火）
- 优化器: AdamW
- 混合精度训练 (AMP)

评估指标：
- **Dice Similarity Coefficient (DSC)** — 越高越好
- **Hausdorff Distance 95% (HD95)** — 越低越好

按三个临床区域分别报告：
- TC (Tumour Core) = NCR + ET
- WT (Whole Tumour) = NCR + ED + ET
- ET (Enhancing Tumour) = ET

---

## 与原论文的差异

| 项目 | 原论文 (3D) | 本复现 (2D) |
|------|-------------|-------------|
| 输入维度 | 128×128×128 | 128×128 |
| Laplacian 核 | 6-连通 (中心 -6) | 4-连通 (中心 -4) |
| 边界条件 | 6 个面 | 4 条边 |
| Batch size | 1 | 8 |
| 优化器 | Ranger 2020 | AdamW |
| 密度估计器输入 | 16×16×16 | 16×16 |
| 推理方式 | 滑动窗口 + TTA | 直接前向 |

---

## 参考文献

- Zhang et al., "Biophysics Informed Pathological Regularisation for Brain Tumour Segmentation", MICCAI 2024
- Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020
- Lê et al., "Bayesian Personalization of Brain Tumor Growth Model", MICCAI 2015

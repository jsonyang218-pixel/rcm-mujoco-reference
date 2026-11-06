# RCM 视觉伺服学习训练流程

这个目录包含使用 Diffusion Policy 训练 RCM 眼科手术机器人视觉伺服策略的完整流程。

## 目录结构

```
lerobot_training/
├── convert_to_lerobot.py      # 数据转换脚本
├── train_diffusion_policy.py  # 训练脚本
├── train_oct_policy.py        # RGB + 状态 + OCT 导航训练入口
├── eval_policy.py              # 评估脚本
├── requirements.txt            # 依赖
└── README.md                   # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
cd lerobot_training
pip install -r requirements.txt
```

### 2. 采集训练数据

首先使用现有的视觉伺服专家采集数据：

```bash
# 回到项目根目录
cd ..

# 采集单个 episode (测试)
python collect_v9_dataset.py --output-dir runs/v9_motor_dataset --episodes 1

# 采集多个 episodes (完整训练集)
python collect_v9_dataset.py \
    --output-dir runs/v9_motor_training \
    --episodes 50 \
    --random-targets \
    --seed 20261103
```

**注意**: 
- 每个 episode 约需 3-5 分钟
- 50 个 episodes 的完整训练集需要 2-4 小时
- 建议至少采集 30+ 成功的 episodes

### 3. 转换数据格式

```bash
cd lerobot_training

# 转换为 train/val/test 三个独立数据集
python convert_to_lerobot.py \
    --input-dir ../runs/paper_stage1/train \
    --output-dir ../data/lerobot_rcm_train \
    --image-workers 8

python convert_to_lerobot.py \
    --input-dir ../runs/paper_stage1/val \
    --output-dir ../data/lerobot_rcm_val \
    --image-workers 8

python convert_to_lerobot.py \
    --input-dir ../runs/paper_stage1/test \
    --output-dir ../data/lerobot_rcm_test \
    --image-workers 8
```

输出结构:
```
data/lerobot_rcm_train/
├── images/                    # 降采样后的图像
├── data/
│   ├── dataset.npz           # 状态和动作
│   └── image_paths.json      # 图像路径
└── metadata.json             # 元数据
```

### 4. 训练策略

```bash
# OCT-aware 正式训练
python train_oct_policy.py \
    --data-dir ../data/lerobot_rcm_train \
    --val-data-dir ../data/lerobot_rcm_val \
    --output-dir ../outputs/diffusion_policy_oct_v2_normalized \
    --epochs 1000 \
    --batch-size 32 \
    --num-workers 4 \
    --prefetch-factor 2

# 无图像 state+OCT 回归基线
python train_oct_regression.py \
    --data-dir ../data/lerobot_rcm_train \
    --val-data-dir ../data/lerobot_rcm_val \
    --output-dir ../outputs/oct_regression_baseline_v2

# Direct-OCT 比例前馈 + 有界残差回归诊断
python train_oct_guided_regression.py \
    --data-dir ../data/lerobot_rcm_train_recovery_full_224 \
    --val-data-dir ../data/lerobot_rcm_val_recovery_full_224 \
    --output-dir ../outputs/oct_guided_regression_v2_20261105

# 为图像策略生成确定性的 OCT 基动作 sidecar
python prepare_guided_oct_base.py \
    --data-dir ../data/lerobot_rcm_train_recovery_full_224 \
    --data-dir ../data/lerobot_rcm_val_recovery_full_224

# RGB + state + direct-OCT 有界残差扩散策略
python train_oct_policy.py \
    --data-dir ../data/lerobot_rcm_train_recovery_full_224 \
    --val-data-dir ../data/lerobot_rcm_val_recovery_full_224 \
    --output-dir ../outputs/diffusion_policy_v6_guided_oct \
    --guided-oct \
    --init-from ../outputs/diffusion_policy_oct_v4_recovery_weighted/best_model.pth \
    --epochs 4

# 使用 WandB 跟踪实验
python train_oct_policy.py \
    --data-dir ../data/lerobot_rcm_train \
    --val-data-dir ../data/lerobot_rcm_val \
    --output-dir ../outputs/diffusion_policy_oct_wandb \
    --epochs 1000 \
    --batch-size 32 \
    --use-wandb
```

**训练参数说明**:
- `--n-obs-steps`: 观测历史帧数 (默认 2)
- `--n-action-steps`: 预测未来动作步数 (默认 8)
- `--lr`: 学习率 (默认 1e-4)
- `--batch-size`: 批大小 (默认 32)
- `--val-data-dir`: 独立验证集目录；best checkpoint 只按验证集 loss 选择
- `--log-file`: 日志路径，默认 `<output-dir>/training.log`，同时显示到控制台
- `--num-workers`: DataLoader 并行进程数；Windows 普通终端建议 4，沙箱环境若报 WinError 5 则使用 0
- `--thread-workers`: 当 `--num-workers 0` 时用于并行图像读取的线程数
- `--prefetch-factor`: 每个 worker 的预取 batch 数
- `--amp/--no-amp`: CUDA 自动混合精度，默认启用
- `--amp-dtype`: `fp16` 或 `bf16`
- `--num-diffusion-iters`: 默认 100
- `--diffusion-schedule`: 默认 `cosine`
- `--pred-x0-clip`: 默认 4，用于限制归一化动作空间中的采样发散

**预期训练时间**:
- GPU (RTX 3090): ~2-3 小时 (1000 epochs)
- CPU: ~24+ 小时 (不推荐)

### 5. 评估策略

```bash
# 评估训练好的模型
python eval_policy.py \
    --model ../outputs/diffusion_policy/best_model.pth \
    --episodes 10 \
    --output-dir ../outputs/evaluation

# 使用随机目标评估
python eval_policy.py \
    --model ../outputs/diffusion_policy/best_model.pth \
    --episodes 20 \
    --random-targets \
    --output-dir ../outputs/evaluation_random
```

评估结果示例:
```
评估结果汇总:
  成功率: 85.0% (17/20)
  平均步数: 523.5
  平均最终误差: 0.012 mm
```

## 架构说明

### Diffusion Policy 架构

```
输入:
├── 观测历史 (n_obs_steps=2)
│   ├── RGB 图像: [2, 3, 640, 480]
│   └── 机器人状态: [2, 3]  # [A, B, C]

编码:
├── ResNet18 视觉编码器 → [2, 512]
├── MLP 状态编码器 → [2, 64]
└── 观测融合 → [256] (条件向量)

Diffusion 去噪网络:
├── 输入: 噪声动作 + 条件 + 时间步
├── MLP: 512 → 512 → 512
└── 输出: 预测噪声

输出:
└── 未来动作序列: [8, 3]  # 8步 [dA, dB, dC]
```

### 为什么选择 Diffusion Policy?

1. **平滑轨迹**: 医疗机器人需要平滑、无抖动的运动
2. **多模态**: 可处理多个合理解 (例如避障路径)
3. **SOTA 性能**: 在精细操作任务上表现最佳
4. **动作序列**: 预测未来多步减少累积误差

## 数据格式

### 输入数据 (v9_motor_dataset)

```
episode_000000/
├── images/
│   ├── frame_000000.png      # 原始 RGB (1920x1440)
│   └── ...
├── motor_steps.jsonl          # 电机执行记录
├── frame_audit.jsonl          # 帧审计
└── control_report.json        # 控制报告
```

### 转换后格式 (lerobot_rcm)

```python
{
    'observation.image': [2, 3, 224, 224],  # 观测历史
    'observation.state': [2, 3],            # [A, B, C] 状态
    'action': [8, 3],                       # 未来8步增量动作
}
```

## 与视觉伺服基线的对比

| 指标 | 视觉伺服 (专家) | Diffusion Policy (学习) |
|------|----------------|------------------------|
| 成功率 | ~100% | 预期 80-95% |
| 平均步数 | 679 步 | 预期 500-800 步 |
| 最终误差 | 6.03 μm | 预期 <20 μm |
| 需要标定 | ✓ (相机、RCM、针具) | ✗ (端到端学习) |
| 泛化性 | 依赖标定 | 更好 (如果训练集多样) |

**学习方法的优势**:
- 无需精确的几何标定
- 可能发现更优的隐式策略
- 更容易适应新传感器/场景

**学习方法的劣势**:
- 需要大量示范数据
- 性能上限受专家限制
- 需要足够的训练数据覆盖

## 实验建议

### 消融实验

1. **观测窗口大小**
   ```bash
   # 对比 n_obs_steps = 1, 2, 5
   python train_diffusion_policy.py --n-obs-steps 1 --output-dir outputs/obs_1
   python train_diffusion_policy.py --n-obs-steps 2 --output-dir outputs/obs_2
   python train_diffusion_policy.py --n-obs-steps 5 --output-dir outputs/obs_5
   ```

2. **动作预测长度**
   ```bash
   # 对比 n_action_steps = 1, 8, 16
   python train_diffusion_policy.py --n-action-steps 1 --output-dir outputs/act_1
   python train_diffusion_policy.py --n-action-steps 8 --output-dir outputs/act_8
   python train_diffusion_policy.py --n-action-steps 16 --output-dir outputs/act_16
   ```

3. **数据量影响**
   ```bash
   # 采集不同数量的 episodes: 10, 30, 50, 100
   # 分别训练并评估
   ```

### 下一步改进

1. **数据增强**: RGB 颜色抖动、随机裁剪
2. **Domain Randomization**: 训练时随机化眼球位置、光照
3. **DAgger 迭代**: 收集策略失败案例，专家纠正，重新训练
4. **多模态融合**: 添加深度或 OCT 输入
5. **Transformer 架构**: 替换 MLP，更好建模时序

## 故障排除

### 问题 1: 训练损失不下降

**可能原因**:
- 数据不足 (< 20 episodes)
- 学习率过大/过小
- 动作幅度过大导致不稳定

**解决**:
```bash
# 降低学习率
python train_diffusion_policy.py --lr 5e-5

# 增加数据
python collect_v9_dataset.py --episodes 100
```

### 问题 2: 评估时策略抖动

**可能原因**:
- 单步动作模式 (n_action_steps=1)
- 观测历史不足

**解决**:
```bash
# 使用动作序列
python train_diffusion_policy.py --n-action-steps 8

# 增加观测历史
python train_diffusion_policy.py --n-obs-steps 5
```

### 问题 3: 泛化性差

**可能原因**:
- 训练数据多样性不足
- 过拟合单一场景

**解决**:
```bash
# 采集更多随机目标
python collect_v9_dataset.py --random-targets --episodes 100

# 数据增强 (修改 train_diffusion_policy.py 的 transform)
```

## 论文撰写建议

### 关键实验

1. **学习曲线**: 损失 vs epochs
2. **数据效率**: 性能 vs 训练数据量
3. **消融研究**: 各个设计选择的影响
4. **对比基线**: 
   - BC (Behavior Cloning, 简单 MLP)
   - ACT (Action Chunking Transformer)
   - 视觉伺服 (专家)

### 可能的论文角度

1. **端到端学习**: 无需标定的视觉伺服
2. **Diffusion Policy**: 在医疗机器人的首次应用
3. **Sim-to-Real**: 仿真训练 → 实机部署的策略
4. **数据效率**: 少样本学习精细操作

## 引用

如果使用 Diffusion Policy，请引用:

```bibtex
@inproceedings{chi2023diffusionpolicy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023}
}
```

## 联系

如有问题，请参考:
- 项目主 README: `../README.md`
- 数据采集文档: `../MOTOR_DATASET.md`
- MuJoCo 场景文档: `../rcm_scene_v9.py`

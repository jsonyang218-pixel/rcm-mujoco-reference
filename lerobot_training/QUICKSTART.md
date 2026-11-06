# RCM Diffusion Policy 训练 - 快速参考

## 📁 文件清单

```
lerobot_training/
├── convert_to_lerobot.py       # 数据转换 (v9 → LeRobot格式)
├── train_diffusion_policy.py   # Diffusion Policy 训练
├── eval_policy.py               # MuJoCo 仿真评估
├── requirements.txt             # Python 依赖
├── README.md                    # 完整文档
├── run_full_pipeline.ps1        # Windows 一键运行
├── run_full_pipeline.sh         # Linux 一键运行
└── QUICKSTART.md                # 本文件
```

## ⚡ 快速开始

正式训练必须先使用独立 train/val 数据。`run_full_pipeline.*` 是旧版单数据集
自动流程，不用于当前 best-on-val 训练；请按下面的手动步骤执行。

## 📋 手动步骤

### 1. 安装依赖
```bash
cd lerobot_training
pip install -r requirements.txt
```

### 2. 采集数据 (50 episodes, ~2-4小时)
```bash
cd ..
python collect_v9_dataset.py \
    --output-dir runs/v9_motor_training \
    --episodes 50 \
    --random-targets \
    --seed 20261103
```

### 3. 转换格式
```bash
cd lerobot_training
python convert_to_lerobot.py \
    --input-dir ../runs/paper_stage1/train \
    --output-dir ../data/lerobot_rcm_train \
    --image-workers 8
python convert_to_lerobot.py \
    --input-dir ../runs/paper_stage1/val \
    --output-dir ../data/lerobot_rcm_val \
    --image-workers 8
```

### 4. 训练 (GPU: ~2-3小时)
```bash
python train_oct_policy.py \
    --data-dir ../data/lerobot_rcm_train \
    --val-data-dir ../data/lerobot_rcm_val \
    --output-dir ../outputs/diffusion_policy_oct_v2_normalized \
    --epochs 1000 \
    --batch-size 32 \
    --num-workers 4 \
    --prefetch-factor 2
```

### 5. 评估
```bash
python eval_policy.py \
    --model ../outputs/diffusion_policy_oct_v2_normalized/best_model.pth \
    --episodes 10 \
    --output-dir ../outputs/evaluation
```

## 🎯 核心架构

**Diffusion Policy**:
- **输入**: RGB图像 (2帧历史) + 机器人状态 [A, B, C]
- **编码**: ResNet18 (视觉) + MLP (状态)
- **输出**: 未来8步增量动作 [dA, dB, dC]

**为什么选 Diffusion**:
- ✅ 生成平滑轨迹（医疗机器人关键）
- ✅ 当前机器人操作 SOTA
- ✅ 处理多模态（多种合理路径）

## 📊 预期性能

| 指标 | 视觉伺服 (专家) | Diffusion Policy |
|------|----------------|------------------|
| 成功率 | ~100% | 预期 80-95% |
| 最终误差 | 6.03 μm | 预期 <20 μm |
| 需要标定 | ✓ | ✗ (端到端) |

## 🔧 常见问题

### Q: 训练损失不下降？
```bash
# 降低学习率
python train_diffusion_policy.py --lr 5e-5

# 或增加数据
python collect_v9_dataset.py --episodes 100
```

### Q: 评估时策略抖动？
```bash
# 增加动作序列长度
python train_diffusion_policy.py --n-action-steps 16

# 增加观测历史
python train_diffusion_policy.py --n-obs-steps 5
```

### Q: 泛化性差？
```bash
# 采集更多随机目标
python collect_v9_dataset.py --random-targets --episodes 100
```

## 🚀 下一步

### 消融实验
```bash
# 对比不同观测窗口
python train_diffusion_policy.py --n-obs-steps 1 --output-dir outputs/obs_1
python train_diffusion_policy.py --n-obs-steps 5 --output-dir outputs/obs_5

# 对比动作长度
python train_diffusion_policy.py --n-action-steps 1 --output-dir outputs/act_1
python train_diffusion_policy.py --n-action-steps 16 --output-dir outputs/act_16
```

### 改进方向
1. **数据增强**: 颜色抖动、随机裁剪
2. **DAgger迭代**: 收集失败案例 → 专家纠正 → 重训练
3. **多模态融合**: 添加深度/OCT输入
4. **Transformer**: 替换MLP，更好的时序建模

## 📚 进阶阅读

- 完整文档: `README.md`
- 数据格式: `../MOTOR_DATASET.md`
- Diffusion Policy 论文: [RSS 2023]

## 💡 论文建议

**可能的研究角度**:
1. 端到端学习视觉伺服（无需标定）
2. Diffusion Policy在医疗机器人的首次应用
3. Sim-to-Real迁移策略
4. 少样本学习精细操作

**对比基线**:
- ✅ 视觉伺服（已有6.03μm）
- ✅ Behavior Cloning (简单MLP)
- ✅ ACT (Transformer)
- ✅ Diffusion Policy (本项目)

---

**预祝训练顺利！有问题参考 README.md 或项目主文档。**

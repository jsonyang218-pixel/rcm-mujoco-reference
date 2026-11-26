# 使用现有数据快速开始训练

## ✅ 已采集的数据划分

- **train**: `runs/paper_stage1/train/`，200 集
- **val**: `runs/paper_stage1/val/`，40 集
- **test**: `runs/paper_stage1/test/`，40 集
- **格式**: v9_motor_dataset 标准格式
- **状态**: 已采集完成 ✓

## 🚀 直接开始训练（跳过数据采集）

### 1. 安装依赖
```bash
cd lerobot_training
pip install torch torchvision numpy Pillow tqdm
```

### 2. 转换数据
```bash
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

### 3. 训练（~2-3小时 GPU）
```bash
python train_oct_policy.py \
    --data-dir ../data/lerobot_rcm_train \
    --val-data-dir ../data/lerobot_rcm_val \
    --output-dir ../outputs/diffusion_policy_oct_v2_normalized \
    --epochs 1000 \
    --batch-size 32
```

### 4. 评估
```bash
python eval_policy.py \
    --model ../outputs/diffusion_policy_oct_v2_normalized/best_model.pth \
    --episodes 10 \
    --random-targets \
    --output-dir ../outputs/evaluation
```

## 📊 数据统计

train 200 集、136,348 帧；val 40 集、27,290 帧；test 40 集、27,317 帧。

在 `n_obs_steps=2, n_action_steps=8` 下，train/val 分别形成
134,748/26,970 个有效窗口。

正式训练只使用 train，checkpoint 只按 val loss 选择；test 保留给最终评估。

#!/bin/bash
# RCM Diffusion Policy 训练完整流程

set -e  # 遇到错误立即退出

echo "=========================================="
echo "RCM Diffusion Policy 训练流程"
echo "=========================================="

# 配置参数
NUM_EPISODES=50
RANDOM_SEED=20261103
EPOCHS=1000
BATCH_SIZE=32

# 目录设置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="${PROJECT_ROOT}/lerobot_training"
DATA_RAW="${PROJECT_ROOT}/runs/v9_motor_training"
DATA_PROCESSED="${PROJECT_ROOT}/data/lerobot_rcm"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/diffusion_policy"

echo "项目根目录: ${PROJECT_ROOT}"
echo ""

# Step 1: 安装依赖
echo "Step 1/5: 安装依赖..."
cd "${TRAINING_DIR}"
pip install -r requirements.txt -q
echo "✓ 依赖安装完成"
echo ""

# Step 2: 采集数据
echo "Step 2/5: 采集训练数据 (${NUM_EPISODES} episodes)..."
cd "${PROJECT_ROOT}"

if [ -d "${DATA_RAW}" ]; then
    echo "警告: 数据目录已存在: ${DATA_RAW}"
    read -p "是否删除并重新采集? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${DATA_RAW}"
        echo "已删除旧数据"
    else
        echo "跳过数据采集步骤"
        goto_step3=true
    fi
fi

if [ -z "$goto_step3" ]; then
    python collect_v9_dataset.py \
        --output-dir "${DATA_RAW}" \
        --episodes ${NUM_EPISODES} \
        --random-targets \
        --seed ${RANDOM_SEED}

    echo "✓ 数据采集完成"
fi
echo ""

# Step 3: 转换数据格式
echo "Step 3/5: 转换数据格式..."
cd "${TRAINING_DIR}"

if [ -d "${DATA_PROCESSED}" ]; then
    echo "警告: 处理后数据已存在: ${DATA_PROCESSED}"
    rm -rf "${DATA_PROCESSED}"
    echo "已删除旧处理数据"
fi

python convert_to_lerobot.py \
    --input-dir "${DATA_RAW}" \
    --output-dir "${DATA_PROCESSED}"

echo "✓ 数据转换完成"
echo ""

# Step 4: 训练模型
echo "Step 4/5: 训练 Diffusion Policy (${EPOCHS} epochs)..."

python train_diffusion_policy.py \
    --data-dir "${DATA_PROCESSED}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr 1e-4

echo "✓ 训练完成"
echo ""

# Step 5: 评估模型
echo "Step 5/5: 评估策略..."

python eval_policy.py \
    --model "${OUTPUT_DIR}/best_model.pth" \
    --episodes 10 \
    --output-dir "${OUTPUT_DIR}/evaluation"

echo "✓ 评估完成"
echo ""

# 完成
echo "=========================================="
echo "训练流程完成!"
echo "=========================================="
echo ""
echo "结果位置:"
echo "  模型: ${OUTPUT_DIR}/best_model.pth"
echo "  评估: ${OUTPUT_DIR}/evaluation/evaluation_results.json"
echo ""
echo "下一步:"
echo "  1. 查看训练日志: ${OUTPUT_DIR}/"
echo "  2. 可视化结果: cat ${OUTPUT_DIR}/evaluation/evaluation_results.json"
echo "  3. 对比基线: python eval_policy.py --model ... --episodes 20"

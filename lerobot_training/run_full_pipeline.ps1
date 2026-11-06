# RCM Diffusion Policy 训练完整流程 (Windows)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "RCM Diffusion Policy 训练流程" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 配置参数
$NUM_EPISODES = 50
$RANDOM_SEED = 20261103
$EPOCHS = 1000
$BATCH_SIZE = 32

# 目录设置
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$TRAINING_DIR = Join-Path $PROJECT_ROOT "lerobot_training"
$DATA_RAW = Join-Path $PROJECT_ROOT "runs\v9_motor_training"
$DATA_PROCESSED = Join-Path $PROJECT_ROOT "data\lerobot_rcm"
$OUTPUT_DIR = Join-Path $PROJECT_ROOT "outputs\diffusion_policy"

Write-Host "项目根目录: $PROJECT_ROOT"
Write-Host ""

# Step 1: 安装依赖
Write-Host "Step 1/5: 安装依赖..." -ForegroundColor Yellow
Set-Location $TRAINING_DIR
pip install -r requirements.txt -q
Write-Host "✓ 依赖安装完成" -ForegroundColor Green
Write-Host ""

# Step 2: 采集数据
Write-Host "Step 2/5: 采集训练数据 ($NUM_EPISODES episodes)..." -ForegroundColor Yellow
Set-Location $PROJECT_ROOT

if (Test-Path $DATA_RAW) {
    Write-Host "警告: 数据目录已存在: $DATA_RAW" -ForegroundColor Yellow
    $response = Read-Host "是否删除并重新采集? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Remove-Item -Recurse -Force $DATA_RAW
        Write-Host "已删除旧数据" -ForegroundColor Green
    } else {
        Write-Host "跳过数据采集步骤" -ForegroundColor Yellow
        $skipDataCollection = $true
    }
}

if (-not $skipDataCollection) {
    python collect_v9_dataset.py `
        --output-dir $DATA_RAW `
        --episodes $NUM_EPISODES `
        --random-targets `
        --seed $RANDOM_SEED

    Write-Host "✓ 数据采集完成" -ForegroundColor Green
}
Write-Host ""

# Step 3: 转换数据格式
Write-Host "Step 3/5: 转换数据格式..." -ForegroundColor Yellow
Set-Location $TRAINING_DIR

if (Test-Path $DATA_PROCESSED) {
    Write-Host "警告: 处理后数据已存在: $DATA_PROCESSED" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $DATA_PROCESSED
    Write-Host "已删除旧处理数据" -ForegroundColor Green
}

python convert_to_lerobot.py `
    --input-dir $DATA_RAW `
    --output-dir $DATA_PROCESSED

Write-Host "✓ 数据转换完成" -ForegroundColor Green
Write-Host ""

# Step 4: 训练模型
Write-Host "Step 4/5: 训练 Diffusion Policy ($EPOCHS epochs)..." -ForegroundColor Yellow

python train_diffusion_policy.py `
    --data-dir $DATA_PROCESSED `
    --output-dir $OUTPUT_DIR `
    --epochs $EPOCHS `
    --batch-size $BATCH_SIZE `
    --lr 0.0001

Write-Host "✓ 训练完成" -ForegroundColor Green
Write-Host ""

# Step 5: 评估模型
Write-Host "Step 5/5: 评估策略..." -ForegroundColor Yellow

$MODEL_PATH = Join-Path $OUTPUT_DIR "best_model.pth"
$EVAL_DIR = Join-Path $OUTPUT_DIR "evaluation"

python eval_policy.py `
    --model $MODEL_PATH `
    --episodes 10 `
    --output-dir $EVAL_DIR

Write-Host "✓ 评估完成" -ForegroundColor Green
Write-Host ""

# 完成
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "训练流程完成!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "结果位置:" -ForegroundColor Yellow
Write-Host "  模型: $MODEL_PATH"
Write-Host "  评估: $EVAL_DIR\evaluation_results.json"
Write-Host ""
Write-Host "下一步:" -ForegroundColor Yellow
Write-Host "  1. 查看训练日志: $OUTPUT_DIR\"
Write-Host "  2. 可视化结果: Get-Content $EVAL_DIR\evaluation_results.json | ConvertFrom-Json"
Write-Host "  3. 对比基线: python eval_policy.py --model $MODEL_PATH --episodes 20"

# 创建 MuJoCo + 训练环境

## 创建 conda 环境
```bash
# 创建新环境（Python 3.8 或 3.11）
conda create -n mujoco python=3.8 -y

# 激活环境
conda activate mujoco
```

## 安装依赖

### 1. 安装 PyTorch (GPU 版本)
```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 或 CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 或 CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### 2. 安装项目依赖
```bash
# 基础依赖
pip install mujoco==3.3.7
pip install numpy>=1.24
pip install pillow>=9
pip install matplotlib>=3.8
pip install trimesh==5.2.0
pip install scipy>=1.10

# 训练依赖
pip install tqdm
pip install wandb  # 可选
```

### 3. 一键安装（推荐）
```bash
cd /e/30_RESOURCES/32_课程教程/医学/中科院/rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225
pip install -r requirements.txt
pip install -r lerobot_training/requirements.txt
```

## 验证环境
```bash
python -c "
import torch
import mujoco
import numpy
import PIL
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ MuJoCo:', mujoco.__version__)
print('✓ NumPy:', numpy.__version__)
print('✓ Pillow:', PIL.__version__)
"
```

## 完整流程（复制粘贴）
```bash
# 1. 创建并激活环境
conda create -n mujoco python=3.8 -y
conda activate mujoco

# 2. 安装 PyTorch (选择合适的 CUDA 版本)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 进入项目目录
cd /e/30_RESOURCES/32_课程教程/医学/中科院/rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225

# 4. 安装项目依赖
pip install -r requirements.txt
pip install tqdm wandb

# 5. 验证
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 导出环境（可选）
```bash
# 导出环境配置
conda env export > environment.yml

# 从配置恢复环境
conda env create -f environment.yml
```

## 切换环境
```bash
# 激活
conda activate mujoco

# 退出
conda deactivate

# 删除环境（如果需要重建）
conda env remove -n mujoco
```

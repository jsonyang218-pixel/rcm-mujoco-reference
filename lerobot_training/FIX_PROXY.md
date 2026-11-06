# 在 rcm_mujoco conda 环境中安装 PyTorch（绕过代理问题）

## 方法1：使用 pip 安装（推荐，绕过 conda 代理）

```bash
# 激活环境
conda activate rcm_mujoco

# 使用 pip 安装 PyTorch（GPU 版本，CUDA 11.8）
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# 或 CPU 版本
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install tqdm Pillow

# 验证安装
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

## 方法2：配置 conda 代理（如果方法1不行）

```bash
# 临时禁用代理
conda config --set proxy_servers {}

# 然后重试
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

## 方法3：直接从清华镜像安装

```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 安装完成后，运行数据转换

```bash
cd E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\lerobot_training

python convert_to_lerobot.py --input-dir ../runs/paper_stage1/test --output-dir ../data/lerobot_rcm
```

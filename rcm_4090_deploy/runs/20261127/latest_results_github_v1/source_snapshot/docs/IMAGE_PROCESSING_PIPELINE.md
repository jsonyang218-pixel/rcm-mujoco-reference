# 当前图像处理链路

本文只描述一张 RGB 图像从 `eye_internal` 相机拍摄开始，到策略视觉编码器
产生 512 维视觉特征的完整处理过程。当前模型为 v7/v8/v9 guided OCT
Diffusion Policy；v8 全量解冻和 v9 terminal-DAgger 数据训练不改变本文的
图像预处理流程。

## 总览

同一张原始 RGB 图像进入两条独立分支：

```text
eye_internal 相机
  ├─ 感知分支：保持 1920×1440 全分辨率，用于颜色检测和 RGB-D 几何测量
  └─ 策略分支：缩放到 224×224，送入 ResNet18 提取视觉特征
```

感知分支产生的目标掩码、针具掩码、深度图和三维几何结果不会直接作为
策略图像输入。策略网络看到的是一张普通 RGB 图，不是检测框、分割图或
带辅助线的图像。

## 1. 相机输出

当前相机为 `RCMSceneV9` 中的 `eye_internal`。固定相机参数来自
`config.json`：

- 图像宽度：`1920 px`
- 图像高度：`1440 px`
- 垂直视场角：`45°`
- 近裁剪面：`0.1 mm`
- 远裁剪面：`60 mm`

每次相机渲染得到：

```text
shape = [1440, 1920, 3]
dtype = uint8
channel order = RGB
```

相机焦距为：

```text
f = 0.5 × height / tan(fovy / 2)
  = 0.5 × 1440 / tan(45° / 2)
  ≈ 1738.6 px
```

图像中心约为：

```text
cx = 1920 / 2 - 0.5 = 959.5
cy = 1440 / 2 - 0.5 = 719.5
```

严格 RGB-D 传感器使用固定相机位姿和同一可见场景，不读取动态目标真值。

## 2. 相机渲染过滤

RGB 和深度使用相同的相机视角及可见性规则，主要渲染设置为：

```text
geomgroup = [0, 0, 0, 1, 1, 1]
sitegroup = 0
```

因此：

- 隐藏仅用于显示的外壳和辅助几何。
- 保留物理针具、病灶和眼底等可见几何。
- RGB 和深度共享同一可见场景。
- 不使用隐藏几何重渲染或仿真对象 ID。
- 深度是相机光轴方向轴向深度，不是像素射线长度。

## 3. 全分辨率感知分支

原始 RGB 保持 `1920×1440`，先进行颜色规则检测。

该分支不使用神经网络分类器、仿真 geom ID、目标坐标或机器人状态，只使用
固定颜色、饱和度和连通域规则。当前目标是人工橙色标记，因此该检测器是
仿真验证标靶检测器，不是临床病灶识别模型。

### 3.1 橙色目标掩码

目标像素满足：

```text
R > 65
G > 25
R > 1.45 × G
G > 1.35 × B
```

得到二值掩码后：

1. 使用 8 邻域连通域标记。
2. 删除小于 6 像素的区域。
3. 只保留最大的目标连通域。

目标几何阶段还会与允许色域相交：

```text
[180, 35, 0] ≤ RGB ≤ [255, 180, 100]
```

### 3.2 绿色针尖标记掩码

绿色 artificial fiducial 满足：

```text
G > 35
G > 1.5 × R
G > 1.25 × B
```

连通域至少包含 2 像素，只保留最大连通域。

绿色标记不是真实细针尖本身。它只是部分姿态下辅助恢复针尖轴向位置；
没有绿色标记时，系统仍可尝试使用针管端面或实体针尖球面。

### 3.3 针管掩码

镀蓝灰近端针管和浅色远端细针使用低饱和中性色规则：

```text
max(R,G,B) ≥ 18
saturation < 0.38
B ≥ R + 1
G ≥ R - 2
B ≥ G - 2
```

检测结果需要排除目标掩码和绿色标记掩码，并删除小于 3 像素的连通区域。

近端针管进一步使用：

```text
B > 1.18 × R
```

只向邻近像素膨胀一次，避免把被遮挡区域或眼底背景加入针管掩码。剩余
中性色针具像素归为远端细针。

### 3.4 感知分支输出

单张 RGB 最终得到四个全分辨率布尔掩码：

```text
target_mask
apex_mask
proximal_mask
distal_mask
```

这些掩码用于后续 RGB-D 几何测量，但不会作为策略网络输入，也不会叠加到
策略使用的那张 RGB 图像上。

## 4. 训练图像路径

策略图像分支与感知分支相互独立。

### 4.1 原始图像保存

数据采集时，每个控制周期保存原始相机 RGB。图像使用无损 PNG，不保存
引导框、检测掩码或 viewer overlay。

原始尺寸：

```text
1920×1440×3 uint8 RGB
```

### 4.2 第一次缩放：1920×1440 → 640×480

数据转换时执行：

```text
1920×1440 RGB
→ PIL RGB
→ 双线性插值
→ 640×480
→ PNG
```

首次转换后的 metadata 记录：

```text
image.shape = [480, 640, 3]
```

### 4.3 第二次缩放：640×480 → 224×224

训练图像缓存生成器执行：

```text
640×480
→ 双线性插值
→ 224×224
→ PNG
```

最终训练 PNG 为：

```text
[224, 224, 3] uint8 RGB
```

变换不保持 4:3 比例，而是直接拉伸成正方形：

```text
horizontal scale = 224 / 640 = 0.35
vertical scale   = 224 / 480 ≈ 0.4667
```

该变换没有中心裁剪、补边或 letterbox，因此水平方向压缩比例大于垂直方向。

### 4.4 image cache 与 vision feature cache

需要区分两种缓存：

- `image_cache`：磁盘上的图像已经是 `224×224` PNG。
- `vision_feature_cache`：预先计算 ResNet 特征并保存为 `[N, 512]` 张量。

v8 关闭的是 `vision_feature_cache`，因此 ResNet18 参与全量反向传播；但
训练输入仍然是预先保存的 224×224 PNG，而不是原始 1920×1440 图像。

## 5. 训练时单张图片预处理

Dataset 从磁盘读取 224×224 PNG：

```python
Image.open(path)
    .convert("RGB")
```

得到：

```text
PIL RGB [224, 224, 3] uint8
```

### 5.1 颜色增强

当前 v8 训练启用：

```text
ColorJitter(
    brightness=0.10,
    contrast=0.10,
    saturation=0.05
)
```

含义为：

- 亮度随机扰动约 ±10%
- 对比度随机扰动约 ±10%
- 饱和度随机扰动约 ±5%
- 不修改色相
- 不修改像素几何位置

当前没有启用：

- RandomResizedCrop
- RandomRotation
- RandomAffine
- HorizontalFlip
- VerticalFlip

毫米级定位任务不能承受随机平移、旋转或翻转造成的几何变化。

### 5.2 ToTensor

`ToTensor()` 执行：

```text
HWC → CHW
uint8 → float32
[0,255] → [0,1]
```

变换结果：

```text
[224,224,3] uint8 HWC
→ [3,224,224] float32 CHW
```

### 5.3 ImageNet 归一化

每个 RGB 通道分别标准化：

```text
R = (R - 0.485) / 0.229
G = (G - 0.456) / 0.224
B = (B - 0.406) / 0.225
```

最终单张图像：

```text
[3,224,224] float32
```

## 6. 闭环推理时单张图片预处理

闭环运行时不读取训练 PNG，而是在内存中处理新渲染的原始图像：

```text
MuJoCo RGB
[1440,1920,3] uint8
→ Image.fromarray
→ 双线性缩放到 640×480
→ Resize 到 224×224
→ ToTensor
→ [3,224,224] float32
→ ImageNet Normalize
→ [3,224,224] float32
```

推理阶段没有：

- ColorJitter
- 随机裁剪
- 直方图均衡
- 锐化
- 模糊
- 目标框或分割掩码叠加
- guidance overlay
- 深度图拼接
- RGB/BGR 通道交换

训练和推理都保持：

```text
MuJoCo RGB → PIL RGB → PyTorch RGB
```

## 7. 进入 ResNet18

### 7.1 输入形状

单张图像增加 batch 维：

```text
[3,224,224]
→ [1,3,224,224]
```

当前模型使用两帧历史，实际策略输入为：

```text
[1,2,3,224,224]
```

其中：

- `1`：batch size
- `2`：当前帧和上一帧
- `3`：RGB 通道
- `224×224`：空间尺寸

ResNet18 编码器首先将 batch 和时间维展平：

```python
B, T, C, H, W = x.shape
x = x.reshape(B * T, C, H, W)
```

因此两帧分别进入同一个 ResNet18：

```text
[1,2,3,224,224]
→ [2,3,224,224]
```

两帧在这里是独立编码的，没有跨时间卷积或时序注意力。时间关系要到后续
视觉特征与状态特征拼接时才由 MLP 处理。

### 7.2 `[512,7,7]` 三个维度的含义

单张图像进入 ResNet18 后，在最后一个卷积阶段得到：

```text
[1,512,7,7]
```

三个维度的含义为：

```text
[512, 7, 7]
  │    │  └─ 特征图宽度：7 个空间位置
  │    └──── 特征图高度：7 个空间位置
  └───────── 512 个视觉特征通道
```

更准确地说：

- `1`：batch size
- `512`：512 张不同的视觉特征图
- `7×7`：每张特征图保留的粗粒度空间网格

这不是已经压缩后的 512 维向量，也不是把 `512×7×7` 直接展平成 25,088
维输入。当前模型会先保留完整空间图，再使用 Spatial Softmax 压缩。

两帧同时编码时，中间特征图形状为：

```text
[2,512,7,7]
```

其中第一个 `2` 分别对应当前帧和上一帧。

### 7.3 为什么 224×224 会变成 7×7

ResNet18 中包含 5 次空间降采样：

```text
224
→ conv1 stride=2
→ 112
→ maxpool stride=2
→ 56
→ layer2 stride=2
→ 28
→ layer3 stride=2
→ 14
→ layer4 stride=2
→ 7
```

总体空间 stride 为：

```text
224 / 7 = 32
```

最终特征图每移动一个格子，大致对应输入图像上约 `32 px` 的网格步长。
但卷积感受野相互重叠，每个 `7×7` 单元看到的实际输入区域远大于 32×32。

### 7.4 完整 ResNet18 特征尺寸

当前使用 ImageNet 预训练 ResNet18。单张图像的完整尺寸变化为：

| 阶段 | 操作 | 输出尺寸 |
|---|---|---|
| 输入 | RGB | `[1,3,224,224]` |
| conv1 | 7×7 convolution, 64 channels, stride 2 | `[1,64,112,112]` |
| BN + ReLU | 标准化和激活 | `[1,64,112,112]` |
| maxpool | 3×3, stride 2 | `[1,64,56,56]` |
| layer1 | 2 个 BasicBlock | `[1,64,56,56]` |
| layer2 | 2 个 BasicBlock，首次 stride 2 | `[1,128,28,28]` |
| layer3 | 2 个 BasicBlock，首次 stride 2 | `[1,256,14,14]` |
| layer4 | 2 个 BasicBlock，首次 stride 2 | `[1,512,7,7]` |

当前代码删除 torchvision ResNet18 的最后平均池化和分类全连接层：

```python
children = list(resnet.children())
self.encoder = nn.Sequential(*children[:-2])
```

`children[:-2]` 删除的最后两项是：

```text
avgpool
fc
```

因此网络停在：

```text
[1,512,7,7]
```

而不会继续变成：

```text
avgpool → [1,512,1,1]
fc      → [1,1000]
```

### 7.5 BasicBlock 的结构

ResNet18 的 layer1-layer4 由 BasicBlock 组成。一个 BasicBlock 的计算为：

```text
输入
  ├─ Conv3×3 → BatchNorm → ReLU
  ├─ Conv3×3 → BatchNorm
  └─ identity 或 1×1 投影 shortcut
        │
        ▼
     残差相加
        │
        ▼
       ReLU
```

每一层包含两个 BasicBlock：

```text
layer1: 2 个 BasicBlock
layer2: 2 个 BasicBlock
layer3: 2 个 BasicBlock
layer4: 2 个 BasicBlock
```

每次空间尺寸或通道数变化时，shortcut 使用一个 `1×1` 卷积和 stride 2 将
输入调整为与主分支相同的形状。残差连接使梯度可以直接跨过多个卷积层。

### 7.6 512 个通道代表什么

512 个通道可以理解为 512 张不同的视觉响应图。不同通道通常对不同结构
更敏感，例如：

- 边缘
- 针管轮廓
- 细针方向
- 颜色边界
- 病灶区域
- 高光
- 局部纹理
- 空间位置组合

单个通道不能简单解释为“这是针尖”或“这是病灶”。任务相关信息分散在
512 个通道和 7×7 网格中，由 Spatial Softmax 和后面的线性层共同组合。

### 7.7 7×7 空间网格的含义

对单个通道，可以得到一个 `7×7` 响应网格：

```text
[
  [a00, a01, ..., a06],
  [a10, a11, ..., a16],
  ...
  [a60, a61, ..., a66]
]
```

网格左上角对应图像左上区域，右下角对应图像右下区域。但它是粗粒度、
重叠感受野的特征，不是精确像素坐标。

更浅层还保留：

```text
56×56
28×28
14×14
```

细节信息主要存在于这些较浅层。最后的 `7×7` 已经比较粗糙，因此当前
224×224 ResNet18 图像分支会损失一部分细针和微小目标的精细位置信息。

### 7.8 v7 与 v8 的区别

v7：

- 视觉特征来自预计算缓存。
- ResNet18 参数冻结。
- 编码器处于 eval 模式。
- BatchNorm running statistics 不更新。

v8：

- 关闭 vision feature cache。
- ResNet18 全量解冻。
- 图像重新经过 ResNet18 前向和反向传播。
- 卷积核、BatchNorm 参数和 running statistics 都可以更新。

训练时 ResNet18 处于 train 模式；闭环推理时整个策略切换到 eval 模式，
BatchNorm 使用训练得到的 running statistics。

### 7.9 张量规模

单帧最后特征图包含：

```text
512 × 7 × 7 = 25,088 个浮点值
```

两帧包含：

```text
2 × 512 × 7 × 7 = 50,176 个浮点值
```

FP32 下约 `200 KB`。

batch 96、两帧时为：

```text
96 × 2 × 512 × 7 × 7 = 4,816,896 个浮点值
```

FP32 下该层输出约 `18.4 MB`。这只是最后特征图本身，训练时还需要保存
卷积中间激活、梯度和优化器状态，因此实际显存占用更高。

## 8. Spatial Softmax

当前 vision mode 为 `spatial_softmax`。

对 512 个通道中的每一个通道：

1. 将 `7×7=49` 个空间响应做 softmax。
2. 得到该通道的二维空间概率分布。
3. 计算期望横坐标。
4. 计算期望纵坐标。

坐标范围归一化为：

```text
[-1, 1]
```

每个通道输出两个值：

```text
[expected_x, expected_y]
```

512 个通道得到：

```text
512 × 2 = 1024
```

然后通过：

```text
Linear(1024 → 512)
```

得到单帧最终视觉特征：

```text
[512]
```

两帧历史得到：

```text
[2,512]
```

Spatial Softmax 保留每个视觉通道的空间响应中心，比单纯全局平均池化更适合
需要针尖和目标位置信息的操作任务。

## 9. 单张图像最终输出

只考虑图像分支时，一张图从相机到网络的形状变化为：

```text
[1440,1920,3] uint8
→ [480,640,3] uint8
→ [224,224,3] uint8
→ [3,224,224] float32
→ ResNet18
→ [512,7,7]
→ SpatialSoftmax
→ [1024]
→ Linear
→ [512]
```

一张图像最终提供 `512` 维视觉特征。

该特征保留：

- 眼底和病灶区域外观
- 针具和细针视觉形状
- 目标与针尖的图像空间位置
- 颜色和纹理上下文

该特征不直接保留：

- 原始 1920×1440 像素
- 深度图
- 目标三维坐标
- 真值针尖位置
- 检测掩码
- 仿真 geom ID
- 未缩放的亚像素几何关系

高精度三维位置并不来自 224×224 策略图像，而来自同一帧的全分辨率 RGB-D
感知分支。该分支生成 OCT 导航向量和 guided base action，后续再与 512 维
视觉特征融合。

## 10. 完整流程图

```text
MuJoCo eye_internal
        │
        ▼
1920×1440×3 uint8 RGB
        │
        ├───────────────────────────────────────┐
        │                                       │
        ▼                                       ▼
全分辨率颜色检测                         PIL 双线性缩放
target / apex / proximal / distal        640×480
        │                                       │
        ▼                                       ▼
供 RGB-D 几何测量使用                    双线性缩放 224×224
不进入策略网络                                 │
                                               ▼
                                     训练：ColorJitter
                                     推理：无增强
                                               │
                                               ▼
                                           ToTensor
                                      [3,224,224] float32
                                               │
                                               ▼
                                      ImageNet Normalize
                                               │
                                               ▼
                                   ResNet18 [512,7,7]
                                               │
                                               ▼
                                 Spatial Softmax [1024]
                                               │
                                               ▼
                                    Linear [512]
                                               │
                                               ▼
                              单帧策略视觉特征 [512]
```

## 11. 当前限制

- 当前颜色检测依赖人工橙色目标和针具颜色，不是临床病灶分割。
- 策略只看到 224×224 RGB，细针和微小目标的像素信息会明显减少。
- `640×480 → 224×224` 是不保持宽高比的直接拉伸。
- 训练和推理没有在线使用原始 1920×1440 图像输入 ResNet。
- 当前没有曝光补偿、自动白平衡、颜色校准或真实相机畸变处理。
- 当前相机、目标标记和深度均为仿真标定假设，不代表真实显微镜或临床精度。

## 12. 高斯模糊和中心涂黑的适用范围

当前 v7/v8 主策略的训练和普通闭环评估**不使用高斯模糊，也不做中心涂黑**。
它们只使用 ColorJitter、ToTensor 和 ImageNet Normalize，见本文第 5、6 节。

高斯模糊和中心黑色遮挡属于独立的 restricted-perception 消融实验，不是
主策略图像的默认预处理。其调用顺序为：

```text
RGB
→ 可选缩放
→ 可选 GaussianBlur
→ 可选中心黑色矩形
→ ToTensor
→ ImageNet Normalize
```

当前 restricted Stage 1-4 常用参数为：

```text
image resolution          = 112
gaussian blur radius      = 1.5
center occlusion fraction = 0.25
```

中心遮挡以 `min(width, height)` 为基准计算黑色正方形边长：

```text
side = round(min(width, height) × center_occlusion_fraction)
```

因此 `112×112` 图像对应约 `28×28` 的黑色中心区域。一个 factored robustness
场景还会把遮挡比例提高到 `0.45`。中心涂黑使用像素值 `(0,0,0)`，是在图像
上直接绘制，不是仿真场景中的物理遮挡，也不是根据检测结果自动遮挡。

## 相关实现

- 严格 RGB-D 渲染与掩码：[`rcm/strict_rgbd.py`](../rcm/strict_rgbd.py)
- RGB 颜色检测：[`rcm/rgb_features.py`](../rcm/rgb_features.py)
- ResNet18 和 Spatial Softmax：[`lerobot_training/train_diffusion_policy.py`](../lerobot_training/train_diffusion_policy.py)
- 数据转换中的 640×480 缩放：[`lerobot_training/convert_to_lerobot.py`](../lerobot_training/convert_to_lerobot.py)
- 224×224 图像缓存：[`lerobot_training/prepare_image_cache.py`](../lerobot_training/prepare_image_cache.py)
- 闭环图像预处理：[`lerobot_training/eval_policy.py`](../lerobot_training/eval_policy.py)
- 受限感知图像退化：[`restricted_perception/image_ops.py`](../restricted_perception/image_ops.py)

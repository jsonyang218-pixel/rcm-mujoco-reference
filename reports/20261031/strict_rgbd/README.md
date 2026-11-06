# 严格 RGB＋可见深度的 v9 闭环验证

从 A/B/C 零位出发，仅用当前 RGB 检测掩码和同一可见场景的深度，完成 MotorBackend 单目标闭环。共 **682 帧、34.05 秒仿真时间，终点误差 5.867 μm，连续 3 帧编码器停稳**。导航未使用物体 ID 掩码或隐藏几何重渲染的测量；目标逆解被禁止，场景真值仅供日志及独立验收。

- [严格感知 10 项检查](perception_verification.json)
- [完整试采的数据与电机重放](dataset_verification.json)
- [终点与来源摘要](pilot_result.json)
- [配置](config_snapshot.json) · [实际采集源码哈希](collection_protocol.json)

10 项观测检查包括：冻结输入后禁止动态真值 API、分割字段缺失/污染无效、RGB 目标消失后不能从仍存在的深度恢复目标标签、无效深度、深度因果性、相同可见场景的 RGB/深度两次渲染、外部配准数组和禁止目标坐标注入。代表位姿针尖误差均小于 2 μm，目标误差约 1.1 μm。

这是一个橙色人工标靶与绿色针尖标记的受控仿真。零位和 C=8 mm 仍依赖可见绿标；近目标处可用可见后端面重建针尖，目标位姿绿标完全不可见时测试仍通过。不能视为无标记真实针尖、病灶自主识别或真实 OCT 性能验证。详细边界见 [严格 RGB-D 说明](../../../STRICT_RGBD_NAVIGATION.md)。

原始 RGB、状态/动作、完整控制报告、正常深度最终帧、电机区间和实际执行源码快照位于 `runs/v9_strict_rgbd_home_20261031`。本目录只保存精简证据，原始图像不纳入项目下载包。运行中的采集源码已冻结；随后增加了不改变默认测量的目标注入拒绝与参数透传检查，原始包保留采集时的准确源码哈希及快照。

感知改动保留了此前电机模型和控制器速率参数。实际速度/加速度仍存在相对名义门限的超限，验证报告明确标为 false；视觉到达通过不代表这些实际运动约束通过。

复现时在仓库根目录指定新输出目录：

```bash
MUJOCO_GL=egl python collect_v9_dataset.py --output-dir runs/strict_rgbd_reproduction --episodes 1 --max-iterations 1200
python verify_v9_dataset.py --run-dir runs/strict_rgbd_reproduction
```

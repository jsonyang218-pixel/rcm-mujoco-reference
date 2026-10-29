# 实机标定工作流

当前 v9 模型仍使用照片、视频和旧驱动常数。标定工具只读取真实测量并生成审查报告，不会因为运行命令而覆盖模型参数。

## 采集格式

使用 [calibration_samples.template.csv](calibration_samples.template.csv) 建立数据表。每行是一个静态测量点：

| 字段 | 含义 |
|---|---|
| `axis` | `A`、`B`、`C` 或 `SILVER`；`SILVER` 表示银罩相对 B 编码器的实际位移 |
| `encoder_count` | 反馈编码器计数；银罩仍填写 B 编码器计数 |
| `measured_value` | 实测角度或位移 |
| `measured_unit` | A/B 使用 `deg`，C/SILVER 使用 `mm` |
| `repeat` | 同一重复实验的编号，用于重复性统计 |
| `direction` | `static`、`forward`、`backward` 或 `unknown` |
| `timestamp_s` | 可选的采样时间 |

每个轴至少采集零点、正向和反向多个点，并重复三次。B 轴需要覆盖工作区，因为它是连杆非线性映射；银罩需要记录真实行程端点和方向。不要把照片估计值或仿真真值当作实测值写入表格。

如果还采集了坐标系对应点，可使用 JSON：

```json
{
  "metadata": {"operator": "", "date": ""},
  "axis_samples": [
    {"axis": "A", "encoder_count": 0, "measured_value": 0, "measured_unit": "deg", "repeat": 1, "direction": "static"}
  ],
  "pose_samples": [
    {"robot_xyz_mm": [0, 0, 0], "world_xyz_mm": [0, 0, 0]}
  ]
}
```

`pose_samples` 至少需要三个不共线点，用于拟合机器人坐标到世界坐标的刚体变换。RCM 和安装角仍需人工审查，工具不会自动把拟合结果写进模型。

## 运行标定报告

在仓库根目录运行：

```powershell
python mujoco_model/calibrate_model.py `
  --input path\to\real_measurements.csv `
  --output mujoco_model\outputs\calibration_report.json
```

先生成空模板：

```powershell
python mujoco_model/calibrate_model.py `
  --write-template mujoco_model\calibration_samples_2026-10-29.csv
```

报告包含每个轴的拟合系数、RMSE、最大残差、方向计数、重复性统计、B 轴相对连杆公式的修正量以及可选坐标系拟合。`A` 和 `C` 的一次拟合可以作为配置候选；`B`、`SILVER`、安装变换和 RCM 必须先人工审查。

## 应用 A/C 候选值

只有确认报告和原始数据无误后，才显式使用：

```powershell
python mujoco_model/calibrate_model.py `
  --input path\to\real_measurements.csv `
  --output mujoco_model\outputs\calibration_report.json `
  --apply-config `
  --config config.json
```

此操作只更新根目录 `config.json` 中 A/C 的计数比例、零点和方向，并创建 `config.json.before_calibration` 备份；不会更新 B 反解、安装角、RCM、银罩传动或 MuJoCo XML。应用后必须重新运行根目录和 `mujoco_model/` 下的验证脚本，并将命令、报告路径和结果追加到 `TEST_LOG.md`。

## 推荐采集顺序

1. 记录复位时 A/B/C/E 编码器和重复回零误差。
2. 逐轴低速采集静态点，再采集正反向运动点。
3. 用量具或外部跟踪设备记录 A/B 角度、C/银罩位移。
4. 采集至少三个机器人坐标到世界坐标的对应点。
5. 运行报告，检查残差、方向和重复性，再决定是否应用 A/C 候选值。
6. 用真实轨迹回放检查 RCM、针尖位置、限位和停止误差。

所有结果都必须标明“实机测量”或“仿真/照片估计”，避免把软件内部误差写成硬件精度。

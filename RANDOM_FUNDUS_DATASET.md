# 随机眼底目标数据集

`random_fundus_dataset.py` 在编译后的眼球后部曲面上生成可复现的随机目标集合。目标位于归一化后部坐标 `z >= 0.9`，并按毫米距离执行最小间隔检查。目标 JSON、曲面验证和每个 episode 的 seed 都会保存。

只生成目标文件：

```powershell
.\.venv\Scripts\python.exe random_fundus_dataset.py `
  --output-dir runs\历史记录\05_random_datasets\targets_v1 `
  --episodes 100 20 20 `
  --targets-per-episode 3 `
  --seed 20261019
```

生成并采集三种导航方法：

```powershell
.\.venv\Scripts\python.exe random_fundus_dataset.py `
  --output-dir runs\历史记录\05_random_datasets\dataset_v1 `
  --episodes 100 20 20 `
  --targets-per-episode 3 `
  --seed 20261019 `
  --collect
```

任务输入模式有两种。默认 `visual_current_target` 的 task 文本对不同目标集合保持一致，不含目标 ID、坐标、顺序或 seed；`registered_coordinate_prior` 会将仿真生成的机器人坐标写入 task 文本，代表“假设已有配准坐标先验”的导航任务，不代表已测得 OCT 坐标。模式由 `dataset_index.json`、各运行的 `experiment_protocol.json` 与 `dataset/manifest.json` 共同记录，例如：

```powershell
.\.venv\Scripts\python.exe random_fundus_dataset.py `
  --output-dir runs\历史记录\05_random_datasets\coord_prior_pilot `
  --episodes 1 0 0 --targets-per-episode 3 --seed 20261019 `
  --task-regime registered_coordinate_prior --collect
```

`visual_current_target` 只限制学习策略的输入；示教专家仍可使用仿真分割/深度和目标姿态等信息。眼底视觉伺服在本实现中使用已知目标姿态校准特征，并用解析回撤/转移到目标附近，RGB 在这些转移段只记录、不反馈，最后沿 C 轴接近才使用视觉深度。因此这是混合先验专家示范，不是全程纯单目策略示范。RRT/PRM 的 RGB-D 建图和控制先验也记录在协议文件中。

无图像消融输入包只移除 RGB，保留相同的 state 与 task；动作监督逐帧对齐。项目当前只导出配对数据，尚未实现策略训练、推理或消融评估。完整逐帧 NPZ/JSONL 中的 waypoint ID、phase、visibility、时间戳、访问顺序和场景坐标是审计标签，不应拼入策略输入。必须先按目标集合划分数据，避免同组目标、方法或随机种子跨越训练/验证/测试集。

目标数量可以改为任意正整数。例如 5 个目标：

```powershell
.\.venv\Scripts\python.exe random_fundus_dataset.py `
  --output-dir runs\历史记录\05_random_datasets\five_targets `
  --episodes 10 2 2 `
  --targets-per-episode 5 `
  --collect --task-regime visual_current_target
```

每个采集 episode 的 `run_reports.json`、`trajectory.csv`、`visual_feedback.json` 和 `dataset/` 位于 `episodes/<split>/episode_XXXXXX/`。总索引是 `dataset_index.json`。视觉建图失败时仍保存目标和日志，不会用场景真值补齐缺失测量。

默认 RGB/state/action 采样频率为 20 Hz（间隔 0.05 秒），可用 `--sampling-frequency-hz` 更改。阶段/规划边界可能额外写入事件帧，时间戳始终原样保存。

每帧另记录标靶可见像素、隐藏针体后重渲染得到的无遮挡像素、由两者差值计算的被针遮挡比例、针尖/针体像素数及遮挡状态。标签来自当前姿态的 MuJoCo 几何分割；隐藏针体重渲染仅用于离线诊断标签，不作为控制输入，也不会改写 RGB 图像。此版本不加入感知噪声，实际几何遮挡自然产生，不通过图像贴片伪造。数据格式在 `dataset/manifest.json` 中定义，完整逐帧字段位于 JSONL/NPZ。

目标成功率按一次目标访问是否经历过某种可见条件分层，成功要求到达验收通过且该目标停针约束通过。一个目标访问可以同时计入“可见”和“针体遮挡”组，例如接近前可见、接近时被针遮挡；其他不可见与未知状态也会单独记录。分母是目标访问次数，不是图像帧数。每个目标访问及整次运行都会汇总停针验证，运行通过还要求所有目标成功并停稳、最终指令速度为零。

可针对遮挡专门采集一个小视觉目标 pilot：

```powershell
.\.venv\Scripts\python.exe random_fundus_dataset.py `
  --output-dir runs\历史记录\05_random_datasets\pilot_n5_visual_current_target_next `
  --episodes 1 0 0 `
  --targets-per-episode 5 `
  --seed 20261021 `
  --collect --methods visual_servo RRT PRM --planner-seeds 7 `
  --sampling-frequency-hz 20 --task-regime visual_current_target
```

该命令使用新输出目录，不会覆盖旧 pilot；目标随机种子相同则可复现同一组五个点。跑完检查各 episode 的 `report.json`/`dataset/manifest.json` 中逐帧遮挡统计，并运行 `verify_waypoint_comparison.py --run-dir <episode目录>`。已有 `pilot_n5_occlusion_v2` 的 task 文本实际包含五个目标坐标，虽然旧协议未写 task regime，故必须按 `registered_coordinate_prior` 归档；它的视觉伺服、RRT、PRM 均 PASS 不能作为无坐标视觉模型或自主识别/排序的证据。旧结果保持不变，新实验请写入新的输出目录。

## 无目标坐标 task pilot（2026-10-19）

新运行位于 `runs/历史记录/05_random_datasets/pilot_n5_visual_current_target_v1/`：1 个训练 episode、5 个仿真眼底目标、seed `20261021`，视觉伺服和 RRT/PRM 各运行一次（RRT/PRM seed `7`）。三种导航轨迹均 PASS，全部帧为 **5,692**，名义采样率 **20 Hz**、最大记录间隔 **0.05 s**。详细帧级和方法报告在 `episodes/train/episode_000000/`，字段/轨迹审计结果在 `episodes/train/episode_000000/dataset_verification.json`。

| 方法 | 结果 | 帧数 | 仿真运动时间 (s) | 最大到达误差 (μm) | 部分遮挡帧 | 完全遮挡帧 | 最大针体遮挡比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 视觉伺服 | PASS | 1,580 | 78.739 | 5.30 | 604 | 0 | 65.7% |
| RRT (seed 7) | PASS | 2,056 | 102.750 | 14.60 | 979 | 0 | 54.3% |
| PRM (seed 7) | PASS | 2,056 | 102.750 | 14.60 | 979 | 0 | 54.3% |

新 pilot 通过策略输入字段白名单、坐标泄漏拒绝、动作对齐、图片重渲染及遮挡标签审计。此处“部分遮挡”由 MuJoCo 针体几何分割和离线隐藏针体诊断渲染计算；RGB 未贴遮挡图层。本组没有完全遮挡帧，因此不能据此证明完全失去目标时可恢复。一个目标集合也不是可训练的 train/validation/test 划分；当前无策略训练、图像消融或自主多病灶排序结果。专家额外信息和多目标范围参见每次采集的 `experiment_protocol.json`。

可参考的开源项目：

- [MuJoCo](https://github.com/google-deepmind/mujoco)：物理/运动学仿真与渲染。
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)：MJCF 机器人资产组织。
- [dm_control](https://github.com/google-deepmind/dm_control)：任务、episode 和随机种子组织。
- [robosuite](https://github.com/ARISE-Initiative/robosuite)：机器人相机观测与数据采集。
- [LeRobot](https://github.com/huggingface/lerobot)：图像、状态、动作和 episode 数据格式。
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)：批量随机场景和训练/验证/测试划分。
- [GitHub 眼科手术机器人检索](https://github.com/search?q=ophthalmic+surgical+robot&type=repositories)。
- [GitHub OCT 视觉伺服检索](https://github.com/search?q=OCT+visual+servoing&type=repositories)。

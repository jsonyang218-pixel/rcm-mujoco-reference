# Gate 2：OCT 偏差归因（2a）

固定五目标×4稳态帧×3渲染变体；诊断状态：completed。未运行 Gate 2b 闭环。
每帧通过禁止动态真值接口的 FrozenCalibration 重新测量；真值只用于误差评估。世界xyz、相机xyz、像素uv和光轴深度分量见 oct_bias_breakdown.json。

|渲染变体|有效帧|目标误差中位数 µm|OCT误差中位数 µm|OCT最大 µm|
|---|---|---|---|---|
|original|20|9.514390270433701|9.317192738089464|9.767217542014155|
|single_sample|20|1.0603360125836998|1.0635020078468609|1.2030208496617656|
|corrected|20|0.07300603413105938|0.07653267493221294|0.11476366626478683|

原渲染为4重采样、28切片/16层；single_sample仅关闭多重采样；corrected再提高至112切片/64层。
前一对照定位深度 resolve/覆盖与像素中心射线的不一致；后一对照定位低精度球面网格与已标定理想球面的差别。相机标定、RGB检测器和球心拟合公式不变，无经验真值偏移补偿。
SyntheticOCTProvider仅提供可见表面深度；目标估计器本来就在拟合半径0.15mm球心，并非误将表面当球心。单纯掩码腐蚀未移除主偏差。
跨四个确定性稳态姿态的均值/标准差记录在JSON；无法据此估计真实传感器随机噪声或证明理论上限。

## 历史存储观察

{
  "dataset": "data/plan2_plus_original_train_224_20261119",
  "dataset_sha256": "9d80523f519faf4e00bbaef3d30e29dec1401181f3dcddd438c5087c05eb54db",
  "metadata_sha256": "7f02f265acadf965a9dc0de29b2058cfccc240ea39d06dc88050179d0ae2698d",
  "total_frames": 319697,
  "skipped": {
    "invalid_oct": 7558,
    "missing_target": 13161
  },
  "groups": {
    "Original": {
      "count": 214136,
      "mean_xyz_um": [
        2.0554795848133836,
        -9.964163922416239,
        0.5007801169774784
      ],
      "std_xyz_um": [
        1.0191358910868054,
        3.6589327896756743,
        5.044774505688767
      ],
      "median_norm_um": 9.944304738033686,
      "max_norm_um": 430.7344910101882
    },
    "Plan2": {
      "count": 84842,
      "mean_xyz_um": [
        2.614526414037085,
        -9.350503034372306,
        0.33834138164122973
      ],
      "std_xyz_um": [
        0.40631637323081343,
        1.1713830044487077,
        2.060081317650875
      ],
      "median_norm_um": 9.56919639547099,
      "max_norm_um": 184.31856516643583
    }
  },
  "near_50um_groups": {
    "Original": {
      "count": 19282,
      "mean_xyz_um": [
        2.6352887648302925,
        -8.9817810875683,
        0.37670418644091924
      ],
      "std_xyz_um": [
        0.470147487996242,
        1.2840746429485927,
        0.3665712133499281
      ],
      "median_norm_um": 9.510453797124768,
      "max_norm_um": 11.24150971573359
    },
    "Plan2": {
      "count": 55918,
      "mean_xyz_um": [
        2.694649076677107,
        -9.172037256091341,
        0.3109164532276194
      ],
      "std_xyz_um": [
        0.2697525568729142,
        0.3102314320557714,
        0.36263344645488366
      ],
      "median_norm_um": 9.502314459455578,
      "max_norm_um": 11.398239648129676
    }
  },
  "resolved_target_episodes": 1001,
  "total_episodes": 1039,
  "method": "Every valid stored OCT vector with an independently resolved target minus FK truth. Merge offsets and exact terminal audits take precedence over raw ancestry. Unresolved targets are explicitly skipped. State stored as float32 deg/deg/mm; reconstruction precision is finite. No images regenerated or frozen data changed.",
  "action_label_scope": "Stored OCT observation bias is audited; action-label corruption is not inferred solely from observation bias."
}

上述检查仅覆盖目标可独立追溯的 Original 与 Plan2 有效观察；未追溯部分明确跳过，不能外推为全量审计。保留原始标签/数据；OCT观察带偏差不等同于已经证明动作标签以同样数值损坏。

## 尚未完成

Gate 2b 依赖 Gate 1 通过并冻结执行协议，目前跳过。因此没有 RGB-D 教师终值、成功率或 Oracle 配对改善结果；离线帧改善不能替代闭环验收。Phase 3 和全部训练暂停。
完整帧/NPZ证据：`runs/20261120/motor_perception_gate_v1/perception`。仅为仿真结果。

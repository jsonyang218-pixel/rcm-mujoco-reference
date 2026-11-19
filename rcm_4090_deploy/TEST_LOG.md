- 2026-10-21：运行系统 Python 的 MuJoCo 环境导入预检（`python`/`py -3.11`），未通过（两套系统环境均无 `mujoco` 模块）；已发现项目 `.venv`，将改用项目解释器执行实际扰动测试。

> 归档说明：v5–v8 模型快照及旧版对照脚本已移除。下文旧测试记录描述当时版本；当前 `--verify` 改为检查现有模型的坐标、源面归属、闭合外壳和前端。历史文件可在 Git 提交 `64d866b` 查看。
- 2026-10-21：执行 `.venv\\Scripts\\python.exe verify_visual_servo.py`，PASS；针尖物理几何可见、针尖/目标深度有效，及传感器分割行为通过检查。
- 2026-10-21：完成 A/B/C 独立正负扰动首轮扫描；脚本完整写出 `runs/checks/needle_abc_perturbation_20261021_211145_692040/report.json`，整体 FAIL（A 的部分微小栅格响应与投影方向不符；A/B 深度变化未达到临时采用的 0.05 mm 判别线）。按系统调试流程核对原始轴向响应和深度分辨率。
- 2026-10-21：完成 A/B/C 单轴 ±配置步长确认测试，PASS；A ±0.5° 像素跨度 2.0 px、深度跨度 0.00653 mm；B ±0.5° 为 2.0 px、0.00124 mm；C ±0.05 mm 为 1.0 px、0.08758 mm。三轴像素方向均与 MuJoCo 物理针尖投影一致，渲染深度响应误差均小于配置容差 0.05 mm；C 深度可分辨，A/B 深度变化低于容差。报告：`runs/checks/needle_abc_perturbation_confirmation_20261021_211604_767434/report.json`。
- 2026-10-21：运行原小角度设置（A/B有限差分0.05°）的带快照复现，触发 `SINGULAR_VISUAL_JACOBIAN` 并保存完整调用现场；本次在第0次迭代失秩（历史记录为第116次），需要对比历史场景配置后确认其余复现条件。初始现场：ABC=[0°, 0°, 0 mm]，AB图像雅可比秩0。报告及快照：`runs/checks/reproduce_original_failure_20261021_212626_587256/`。
- 2026-10-21：尝试直接加载历史 `effective_config.json` 后运行控制器，FAIL before controller start：当前 `Reference.build_xml()` 要求旧快照缺失的 `needle.tip_marker_radius_mm`，抛出 `KeyError`。需要补齐新模型新增字段的显式兼容默认值，保留旧快照中已有场景/针具值后再复现。
- 2026-10-21：使用历史场景并为当前模型缺失字段补兼容默认值后重放，FAIL before first controller iteration：在目标标定姿态抛出 `TIP_NOT_VISIBLE`。说明旧场景快照与当前针尖分割约定不兼容；后续原因调查改为检查当前导航失败产物和当前数据流。
- 2026-10-21：首次从零复现的启动尝试第 ?/3 次未进入导航；导航未开始、无有效导航结果；启动现场：`runs/checks/navigation_repeat_3_from_zero_20261021_221430_712699/run_??`。
- 2026-10-21：首次从零复现的启动尝试第 ?/3 次未进入导航；导航未开始、无有效导航结果；启动现场：`runs/checks/navigation_repeat_3_from_zero_20261021_221430_712699/run_??`。
- 2026-10-21：首次从零复现的启动尝试第 ?/3 次未进入导航；导航未开始、无有效导航结果；启动现场：`runs/checks/navigation_repeat_3_from_zero_20261021_221430_712699/run_??`。
- 2026-10-21：首次启动三次从零复现时，临时诊断记录钩子在导航控制器开始发出动作前因列表序列化错误退出（`AttributeError: 'list' object has no attribute 'tolist'`）；这三次不是有效导航结果。保留现场：`runs/checks/navigation_repeat_3_from_zero_20261021_221430_712699/`。已修正仅运行时的诊断钩子并重新执行正式三次。
- 2026-10-21：严格视觉 P1 导航从 ABC=[0°,0°,0 mm] 独立重跑第 1/3 次；结果=FAIL，原因=SINGULAR_VISUAL_JACOBIAN，运动时间=0.30000000000000004 s，故障处 ABC=[-0.15000000000000002, -0.07500000000000001, 0.022500000000000006]，AB 图像雅可比秩=1；完整记录：`runs\checks\navigation_repeat_3_from_zero_20261021_221619_484629\run_01`；控制台：`runs\checks\navigation_repeat_3_from_zero_20261021_221619_484629\run_01_console.log`。
- 2026-10-21：严格视觉 P1 导航从 ABC=[0°,0°,0 mm] 独立重跑第 2/3 次；结果=FAIL，原因=SINGULAR_VISUAL_JACOBIAN，运动时间=0.30000000000000004 s，故障处 ABC=[-0.15000000000000002, -0.07500000000000001, 0.022500000000000006]，AB 图像雅可比秩=1；完整记录：`runs\checks\navigation_repeat_3_from_zero_20261021_221619_484629\run_02`；控制台：`runs\checks\navigation_repeat_3_from_zero_20261021_221619_484629\run_02_console.log`。
- 2026-10-21：严格视觉 P1 导航从 ABC=[0°,0°,0 mm] 独立重跑第 3/3 次；结果=FAIL，原因=SINGULAR_VISUAL_JACOBIAN，运动时间=0.30000000000000004 s，故障处 ABC=[-0.15000000000000002, -0.07500000000000001, 0.022500000000000006]，AB 图像雅可比秩=1；完整记录：`runs\checks\navigation_repeat_3_from_zero_20261021_221619_484629\run_03`；控制台：`runs\checks\navigation_repeat_3_from_zero_20261021_221619_484629\run_03_console.log`。
- 2026-10-22：只读反向扰动诊断（未修改源代码）验证故障帧 A 轴有限差分；在 `run_01` 记录的故障 ABC=[-0.15°,-0.075°,0.0225 mm] 重新渲染，A+0.5° 针尖像素=[179,243]、A-0.5°=[180,243]，因此中心差分会产生约 -57.30 px/rad 的非零 A 列；B+=[179.5,244]、B-=[179,242]，C 深度正负扰动也均可分辨。该诊断支持“单侧正向扰动被像素量化为零、未尝试反向扰动”是首要修正方向；源代码未修改。测试记录追加已核验。
- 2026-10-22：中心差分修正后的故障姿态回归检查，PASS；实际 AB 图像雅可比为 `[[-57.2958, 28.6479], [0.0, 114.5916]] px/rad`，秩=2，A 列恢复非零；测试使用当前修改后的 `rcm_visual_servo.py`，未运行完整导航。测试记录追加已核验。
- 2026-10-22：运行 `.venv\\Scripts\\python.exe verify_visual_servo.py`（中心差分修正后），PASS；传感器、目标/针尖可见性、遮挡行为和严格视觉失败闭环检查均通过；输出：`runs/checks/after_central_difference_verify_visual_servo_20261022_$(Get-Date -Format yyyyMMdd_HHmmss)/`。测试记录追加已核验。
- 2026-10-22：`py_compile` 检查中心差分与像素停止阈值修改后的 `rcm_visual_servo.py`、`run_waypoint_comparison.py`，PASS；测试记录追加已核验。
- 2026-10-22：提高 precision profile `max_iterations` 到 900 后执行 `py_compile`，PASS；测试记录追加已核验。
- 2026-10-22：将 precision profile 的独立终点容差调整为 150 µm、视觉深度停止容差调整为 0.75 mm 后执行 `py_compile`，PASS；测试记录追加已核验。
- 2026-10-22：针尖遮挡帧 2× RGB-D 重采样回放（1280×960），PASS；针尖标记从原 640×480 的 0 像素恢复为 1 像素，RGB-D 针尖重建误差 5.96 µm、目标重建误差 13.37 µm、深度球拟合残差 0.000826 mm；严格精度配置已恢复深度阈值 0.003 mm；`py_compile run_waypoint_comparison.py rcm_visual_servo.py rcm_vision_navigation.py` 无报错。
- 2026-10-22：严格视觉 2× 分辨率完整导航测试 `runs/checks/navigation_highres_strict_20261022_105550/`，运行器 FAIL：控制器返回后追加终止帧时抛出 `Episode timestamps must increase strictly`，尚未生成独立终点验收报告；需检查控制器最后一帧与数据集终止帧时间戳重复。
- 2026-10-22：高分辨率导航失败后尝试计算受限步的离壁裕量；首个只读诊断脚本因把 `task_user_units()` 返回的 list 当作 ndarray 调用 `.tolist()` 而退出，未修改运行数据或源码，随后修正脚本继续检查。
- 2026-10-22：取消视觉伺服默认步数限制后，执行 `.venv\Scripts\python.exe -m py_compile rcm_visual_servo.py run_visual_servo.py run_waypoint_comparison.py`，未通过：替换 `__pycache__/run_visual_servo.cpython-311.pyc` 时发生 WinError 5（拒绝访问），未据此宣称语法验证通过。新增超过 400 步及主动中止回归检查尚未执行；用户明确接手验证，停止后续测试。默认 `max_iterations=null`，precision profile 不再覆盖为 900，关闭窗口回调可中止控制。
- 2026-10-22：用户手动运行可视化后，读取 `runs/历史记录/04_sessions/visual_servo/20261022_142317_205679/visual_servo_report.json` 及 CSV 核对：677 次观测、仿真时间 33.8 s，控制器 PASS / VISUAL_ERROR_TOLERANCE，确认已超过旧 400 步上限；目标/针尖最少 10/1 像素，速率检查 true。校正图像误差为 0 px、深度误差 -0.0319816 mm，但独立三维误差 429.2675 µm，高于 150 µm；terminal_stop_validated=false，末次保留命令速度为 [0.5,0.25,0.0686703] 度/秒、度/秒、毫米/秒。该次使用已知目标姿态特征标定，PASS 仅代表视觉误差条件满足，不代表独立位置精度或停稳验收通过。本轮只读核对用户运行产物，未重新运行测试。
- 2026-10-22：检查 GitHub 推送连接；`Test-NetConnection github.com -Port 443` 解析到 `20.205.243.166`，TCP 连接失败（TcpTestSucceeded=False）；Git 已有本地提交 `413d3a6c Initial commit`，origin 指向用户提供的 GitHub HTTPS 地址，未配置 Git HTTP 代理或 HTTP(S)_PROXY 环境变量。推送未到达 GitHub，判定为本机当前网络连接问题；未改动源码。
- 2026-10-22：在 `rcm_mujoco_reference_upload` 中运行项目 `.venv` 的 `python -m compileall -q .`，PASS（退出码 0）；检查该副本所有 Python 源文件的语法。生成的 `__pycache__` 已由 `.gitignore` 排除。
- 2026-10-22：在 `rcm_mujoco_reference_upload` 中运行 `.venv\Scripts\python.exe verify_navigation.py --output-dir acceptance\upload_navigation_check`，PASS（MuJoCo 3.3.7）；593 个目标求解、49 个完整运动循环、3 个无效目标拒绝、109338 个轨迹采样点，最大针尖误差 5.48 µm。此项验证限于 MuJoCo 运动学仿真，不代表实体机械臂实测。

- 2026-10-22：云环境几何视觉伺服最终完整复验，PASS；默认P3从零位出发，1920×1440实际RGB-D渲染，679次观测、视觉段33.90 s，独立三维终点误差6.032794 μm（<20 μm），测量误差7.065029 μm；连续3帧零指令速度、速度/加速度检查通过，无目标姿态特征校准。停留0.4 s后单独执行模型C回退及A/B复位，成功回零；全周期名义时间102.579481 s。数据：runs/precision_final_validation/geometry/。限于理想分割/深度及固定几何标定的单目标MuJoCo运动学仿真，非硬件精度。
- 2026-10-22：云环境回归检查通过：run_reference.py --verify；verify_navigation.py（593目标求解、49运动学循环，最大误差5.479882 μm）；verify_visual_geometry.py（9姿态，最大针尖测量误差1.946668 μm）、verify_visual_geometry_direction.py（8用例）、verify_visual_motion.py（8测试）、verify_visual_servo.py与verify_vision_navigation.py。视觉规划回归中的RRT/PRM样例均TIP_NOT_OBSERVABLE中止，PASS指失观测闭锁及接口回归通过，不代表这些导航成功。逐条退出码和日志：runs/precision_implementation/regression_final.json。

- 2026-10-22：严格视觉三点零位完整导航，PASS 3/3；P1/P2/P3 独立误差6.634483/6.085113/6.267576 μm，1289帧、64.05s名义仿真时间。运行时禁用场景目标逆解，视觉对准、接近及点间回撤均用测量构造目标，每点停稳及速率/离壁约束通过。
- 2026-10-22：实体遮挡配对实验：stop组 FAIL（P1 TARGET_NOT_VISIBLE，P2/P3 NOT_RUN，29帧）；其仿真瞬停独立加速度约束也FAIL，不能视为安全制动。相同重置及球体计划下，static_memory_reobserve组 PASS 3/3，误差6.381807/6.442212/6.282898 μm，669帧、33.05s；共24记忆帧、1次重观测恢复，速率约束通过。四组包含历史混合对照共3133帧，完整性/动作对齐及抽样重渲均通过；数据完整性PASS不改变失败组结局。
- 2026-10-22：接入明确标注的理想注册OCT仿真接口：verify_oct_interface.py 10组数值PASS；verify_oct_sensor.py 12组实际1920×1440 EGL传感器检查PASS。覆盖单位、有效mask、远裁剪背景拒绝、时钟、坐标符号、遮挡/无效测量拒绝、深度扰动因果性及真实深度帧NPZ/JSON往返。无真实OCT、无A/B-scan或设备实时性验证。
- 2026-10-22：新OCT接口P1严格视觉闭环从此前实际视觉回撤姿态运行，PASS；8帧对准+89帧接近，共97帧、4.75s名义仿真时间，独立误差6.381807 μm，停稳、速率及dataset_verification均PASS。逐帧OCT来源/唯一ID/阶段仿真时钟及末帧完整深度归档审计PASS；实测渲染中位0.471s/帧，不支持实时20Hz。报告及快照说明见reports/20261022/。
- 2026-10-22：发布审查修复虚拟回撤阶段针尖短暂丢失后重获不能恢复的问题；新增组合回归后verify_occlusion_recovery.py 10/10 PASS。标靶全程不可见仍保留原回撤虚拟点、target_observed始终false；针尖重获后连续有效测量恢复跟踪，停稳及速率通过。该恢复分支修复发生在上述stop模式OCT闭环之后，单独验证，未将旧采集源码副本冒充采集版本。
- 2026-10-28：准备上传 `simulation_driver` 时做静态验证；直接在源树生成 `__pycache__` 受到 Windows 沙箱 WinError 5 限制，未修改源文件。将 Python 字节码缓存重定向到系统临时目录后，19 个 Python 文件编译 PASS；`ros_msg/package.xml` XML 解析 PASS，`CMakeLists.txt` 项目名检查 PASS。上传副本与原目录 29 个文件逐项 SHA-256 比对一致。

- 2026-10-29：将根 scene.xml 和默认 run_reference.py 替换为已修复白壳及整体银罩的扫描 v9 模型，完整资产/配置/驱动适配位于 mujoco_model/。原场景与入口逐字节保留为 legacy_scene.xml、run_legacy_reference.py；历史导航/OCT 继续使用旧研究 API，未声称迁移到扫描模型。verify_current_model.py PASS：从其他工作目录启动、姿态/反馈编码器/ABC演示单位、两种针具导出、相对资产路径加载、旧参数引导及默认场景不被其他导出覆盖。当前入口 --verify PASS（91姿态/针具坐标检查与两针具70姿态银罩整体检查），EGL渲染已实际检查；白壳537面保护、银罩1797面整体归属及240000源面完整性验证PASS。离线驱动控制器84虚拟秒、8400反馈、190指令通过；根simulation_driver原29文件SHA-256不变。旧入口 --verify 的2026姿态检查PASS。真实桌面窗口未人工交互；模拟viewer事件的从动链闭合及复位检查PASS，未连接硬件。发布验证见 mujoco_model/outputs/publish_validation.json，来源哈希见 mujoco_model/import_manifest.json。
- 2026-10-29：新增 `mujoco_model/calibrate_model.py` 和 `verify_calibration_pipeline.py` 后，使用 `py -3.11`（Python 3.11.0、NumPy 2.4.2）执行 `PYTHONPYCACHEPREFIX=%TEMP%\rcm_mujoco_pycache py_compile mujoco_model\calibrate_model.py verify_calibration_pipeline.py`，PASS；随后执行 `py -3.11 verify_calibration_pipeline.py`，PASS（A/B/C/SILVER 拟合、刚体坐标变换、显式配置应用与备份、非法轴拒绝）。初次尝试的 `.venv\Scripts\python.exe` 不存在，未进入测试程序，已改用系统 Python 3.11。
- 2026-10-29：标定 CLI `py -3.11 mujoco_model\calibrate_model.py --write-template %TEMP%\rcm_calibration_template.csv`，PASS；输出仅包含 CSV 表头，未写入任何模型参数。MuJoCo 环境预检 `py -3.11 -c "import mujoco"` FAIL（ModuleNotFoundError），因此本轮尚未执行 MuJoCo 场景验证。
- 2026-10-29：Windows 基线检查首次因仓库在 E: 而临时导出目录在 C:，`run_model.py` 的 `os.path.relpath()` 抛跨盘 `ValueError`；已修复为同盘保持相对资产路径、跨盘回退绝对资产路径，并让 `verify_current_model.py` 在仓库所在盘创建临时目录。随后使用 `py -3.11 -m py_compile mujoco_model\calibrate_model.py mujoco_model\run_model.py verify_current_model.py verify_calibration_pipeline.py`，PASS；`py -3.11 verify_current_model.py`，PASS（v9 场景加载、姿态/编码器/演示、两针具导出、跨目录入口和默认场景保护）。
- 2026-10-29：清除 Windows 不支持的 `MUJOCO_GL=egl` 后执行 `py -3.11 run_reference.py --verify`，PASS；`validate_frames.py` 91 姿态/针具配置检查通过，最大 MuJoCo 与源 FK 误差 3.2381e-13 m；`validate_silver_integral.py` 两种针具各 35 姿态、240000 源面和 1797 归属面检查通过。结果仍仅代表扫描模型内部一致性，不代表实机标定。
- 2026-10-29：执行 `py -3.11 mujoco_model\validate_geometry.py`，PASS；240000 源三角面完整归属、刚性注册、27 个固定体姿态和 29 个原驱动文件检查通过，最大 FK 误差 3.0626e-13 m。该检查不替代 CAD/计量标定。
- 2026-10-29：执行 `py -3.11 mujoco_model\validate_installation.py`，PASS；new/legacy 两种针具各 27 姿态、安装变换、世界轴、水平夹具和驱动反馈一致性通过。报告明确安装角仍为照片近似，`hardware_installation_calibrated=false`。
- 2026-10-29：执行 `py -3.11 mujoco_model\validate_frames.py`，PASS；new/legacy 各 91 姿态、源编码器边界、机械轴/执行器范围、针具配置和反馈拒绝路径通过，最大 MuJoCo 与源 FK 误差 3.2381e-13 m；报告仍标注物理安装和电机动力学未标定。
- 2026-10-29：执行 `py -3.11 mujoco_model\validate_driver_integration.py`，PASS；协议原子性/watchdog、84.0 虚拟秒闭环、8400 条反馈、190 条指令、双针具模型通过，最大闭环误差 1.1734 µm。动力学仍为假设的一阶滞后，不是实机识别结果。
- 2026-10-29：在跨盘导出修复后再次执行 `py -3.11 verify_calibration_pipeline.py`，PASS；标定拟合、刚体变换、显式 A/C 配置应用、备份和非法输入拒绝回归均通过。
- 2026-10-29：跨盘导出回归（仓库 E:、临时导出目录 C:）执行 `run_reference.py --no-viewer --export`，检查 XML 使用绝对资产目录，并用 `mujoco.MjModel.from_xml_path()` 加载，PASS（`CROSS_DRIVE_EXPORT_LOAD_PASS`）。
- 2026-10-29：修正标定应用策略：启用 `encoder_reference` 时 A/C 零点写入 `robot_counts_ABC`，保留 `encoders.origin`；JSON 非对象样本也会被拒绝。执行 `py -3.11 -m py_compile mujoco_model\calibrate_model.py verify_calibration_pipeline.py` 与 `py -3.11 verify_calibration_pipeline.py`，PASS。
- 2026-10-29：旧研究入口回归 `py -3.11 verify_reference.py`，PASS；2026 个姿态、A/B/C/ABC 轴、刚体安装、编码器往返和 RCM/连杆闭合通过。结果仅代表旧研究场景运动学一致性。
- 2026-10-29：旧导航回归 `py -3.11 verify_navigation.py --output-dir runs\checks\post_calibration_tool_navigation`，PASS；593 个目标求解、49 个完整运动循环、3 个非法目标拒绝、109338 个轨迹采样点，最大静态/到达误差 5.4799 µm。结果仍是无限位 MuJoCo 运动学，不代表实机可达范围。
- 2026-10-29：将字节码缓存重定向到系统临时目录后执行 `py -3.11 -m compileall -q .`，PASS（仓库全部 Python 源文件语法检查；未把 `__pycache__` 写入仓库）。
- 2026-10-29：针对外部 `E:\30_RESOURCES\32_课程教程\医学\中科院\simulation_driver` 的控制链核验，执行 `py -3.11 mujoco_model\validate_driver_integration.py`，PASS；84.0 虚拟秒、8400 条反馈、190 条指令、协议/watchdog/双针具 MuJoCo FK 均通过，最大闭环误差 1.1734 µm。该脚本明确不启动 ROS、不连接网络或实机。
- 2026-10-29：只读 SHA-256 比对 `mujoco_model/source_manifest.json` 的 29 个固定文件与 `mujoco_model/vendor/simulation_driver`，29/29 匹配；外部 `E:\30_RESOURCES\32_课程教程\医学\中科院\simulation_driver` 另有 `ros_msg/scripts/mujoco_bridge.py`、`ros_msg/launch/mujoco_bridge.launch`，且 README/CMakeLists 与固定快照不同，未将外部目录全部同步入当前模型。
- 2026-10-29：迁移阶段一接口审计完成，新增 `INTERFACE_AUDIT.md`；确认 legacy `Reference` 实际 API 为 `set_task/current_task/encoders/task_from_encoders/robot_to_world/world_to_robot/set_target_world/target_task/site_mm/camera_id/render_camera/camera_pose/needle_eye_geometry/metrics`（非规划中假设的 `set_abc/get_abc/...`），并确认根 `scene.xml` 无眼球/眼底靶点/眼内相机。`py_compile` 未纳入本次（审计为只读）。
- 2026-10-29：迁移层 1 运动学一致性 `verify_migration_kinematics.py`，PASS；legacy `rcm_recovered_model` 与 v9 vendor `kinematics` 编码器常数/B 连杆/针轴代数一致，A∈[-10,10]°、B∈[-5,5]°、C∈[0,8]mm 的 9×9×9 网格最坏局部针尖差 0.00016565 mm（0.166 µm），世界映射 `rcm_world + robot_to_world@(p_mm*0.001)` 复现 v9 针尖 site 最坏差 0.00016565 mm。测试记录已追加。
- 2026-10-29：新增 `rcm_scene_v9.py` 适配器（v9 `DriverModel` + 眼内场景注入 + Reference 兼容接口 + 针具传感几何）与 `verify_rcm_scene_v9.py`，PASS；`encoders` 与 legacy 零位一致，tip/eye_center/target 世界映射差均 < 1.4e-7 mm，眼内相机 1920×1440 RGB 渲染成功，`VisualServoSensor` 可构造，ABC task 往返误差 0。`py_compile rcm_scene_v9.py verify_rcm_scene_v9.py verify_migration_kinematics.py` PASS。
- 2026-10-29：阶段四层 3 阻塞（未通过）：适配器 `VisualServoSensor.observe()` 中 target 不可见（`target_pixels=0`），针尖分割 21 px 但几何拟合失败。已根因到：相机朝向修正后病灶投影与 legacy 逐像素一致（[746.9,597.9]px），病灶 geom（id 17, group 3, 0.15 mm）在“全部 geomgroup 开启”时可见 9 px，但在视觉伺服过滤（0/1/2 关、3/4/5 开）下变为 0 px，且 floor（group 0）异常大量出现，说明注入后 v9 场景的 geomgroup 过滤路径行为异常；属规划标记的风险 1，停止继续盲修，留待单独定位 MuJoCo geomgroup 与 v9 场景交互。未宣称层 3 通过。
- 2026-10-29：按审查意见修正迁移代码：sites 构建后校验 `-1` 并抛错；针尖传感几何位置/半径改从 `config.needle`（`distal_length_mm`/`distal_tip_length_mm`/`distal_radius_mm`/`tip_marker_radius_mm`）读取；`render_camera` 改为预分配并缓存 rgb/depth/segmentation 三个 Renderer 并新增 `close()`；`set_target_world` 补充说明 legacy 世界系二次投影的注释；`metrics()` 的上下连杆闭合误差改为真实几何残差（`upper_bridge_end↔front_upper`、`lower_bridge_end↔front_lower`，实测 ~0 µm）；`_inject_assets` 增加 asset 缺失与重名 material 检查；`_inject_needle_sensor_geoms` 对缺失 body 给出明确错误；`rcm_reference.Reference` 增加公开 `forward_kinematics(q)` 并移除对 `_ref.nominal` 的私有访问；`verify_rcm_scene_v9.py` 的 PASS 条件改为覆盖全部关键检查；`verify_migration_kinematics.py` 的 A 计数/ C 零点改从 config 读取。`py_compile rcm_reference.py rcm_scene_v9.py verify_rcm_scene_v9.py verify_migration_kinematics.py` PASS；`verify_migration_kinematics.py` PASS（0.166 µm）；`verify_rcm_scene_v9.py` PASS。层 3 的 geomgroup 过滤阻塞仍待单独定位，未因此宣称完成。
- 2026-10-29：针对层 3 geomgroup 阻塞做 root-cause 定位（未修复，已回滚调试改动，适配器回到层 1/2 通过的干净状态）。关键证据：legacy 视觉伺服过滤 `[0,0,0,1,1,1]` 下病灶可见（127 px），v9 注入场景同过滤下病灶 0 px；眼内相机朝向已修正（病灶投影与 legacy 逐像素一致 [746.9,597.9]px）；v9 中 group 2（eye）可被正确过滤，但 group 0 环境几何（phantom/target_tape/horizontal_reference_edge/target_table_leg）未被过滤且遮挡病灶；全 geomgroup 开启时病灶 9 px。排查中还发现 MuJoCo segmentation 首通道把 geom id 0 与背景都编码为 0，导致 id 0 的几何不可靠（此为 floor id 0 反复出现的根因），且 site 会以全局对象 id 出现在分割图里。尝试移除环境几何/眼椭球、重排 id、加 id0 占位均未让病灶稳定可见，属需要进一步对照 MuJoCo 3.3.7 segmentation 编码的专项工作。已删除全部临时探针文件与 PNG，`py_compile` 与 `verify_rcm_scene_v9.py` 复跑 PASS，未把未完成的层 3 改动留在源码里。测试记录已追加。
- 2026-10-29：环境版本复核，使用项目 `.venv`（`E:\30_RESOURCES\32_课程教程\医学\中科院\rcm_mujoco_control_limit\rcm_mujoco_reference\.venv\Scripts\python.exe`）检查到 Python 3.11.0、MuJoCo 3.3.7、NumPy 2.4.6，与 `requirements.txt` 的 MuJoCo 3.3.7 一致；系统默认 `python` 为 Python 3.8.5，不能作为本项目运行解释器。此项是环境检查，不是迁移测试。
- 2026-10-29：定位到“MuJoCo 3.10.0”来自 Conda 环境 `mujoco`（`C:\ProgramData\Miniconda3\envs\mujoco\python.exe`，Python 3.10.20、MuJoCo 3.10.0）；该环境不是本项目迁移验证应使用的解释器。项目 `.venv` 为 MuJoCo 3.3.7，与 `requirements.txt` 匹配。未改任何环境或依赖。此项是环境检查，不是迁移测试。
- 2026-10-29：迁移层 1 `verify_migration_kinematics.py` 复跑，PASS；`worst_local_tip_diff_mm=0.00016565319704428308`、`worst_world_mapping_diff_mm=0.00016565315765708103`、`pass=True`，退出码 0。使用项目 `.venv`，`MUJOCO_GL=glfw`。测试记录已追加。
- 2026-10-29：迁移层 2/适配器构造 `verify_rcm_scene_v9.py` 复跑，PASS；`tip_world_mapping_diff_mm=1.3006676295234702e-07`、`eye_center_mapping_diff_mm=2.8421709430404007e-13`、`target_mapping_diff_mm=5.240704272917405e-13`、`eye_rgb_shape=[1440,1920,3]`、`visual_servo_sensor_constructs=True`、`task_roundtrip_max_err=0.0`，退出码 0。适配器构建中 site 缺失校验未触发，说明 `needle_base/front_upper/front_lower/upper_bridge_end/lower_bridge_end` 等所需 site 在注入后的 v9 模型中存在。使用项目 `.venv`，`MUJOCO_GL=glfw`。测试记录已追加。
- 2026-10-29：当前 v9 扫描模型可视化检查，执行 `run_reference.py --no-viewer --render runs\current_model_preview.png --camera overview --profile new`，退出码 0；输出零位诊断 `profile=new`、`A_deg=0.0`、`B_relative_deg=0.0`、`max_fk_error_m=3.060346736974325e-13`，PNG 已写入 `runs\current_model_preview.png`。使用项目 `.venv`，`MUJOCO_GL=glfw`。此项是可视化检查，不是迁移验证测试。测试记录已追加。
- 2026-10-29：同步 GitHub 提交 `01c36a9137849e93757fa228035e3a46847ab083`（分支 `migration/v9-adapter`，提交说明 "Prepare closed scan display meshes and headless inspection views"）的闭合显示外壳改动后，执行 `python verify_enclosed_shells.py`，PASS，退出码 0。四个显示网格 `fixed_housing`/`parallel_cover`/`distal_guide`/`C_carriage` 分别为 110106/46414/46670/31794 面，均无边界边和非流形边且为水密；源网格 SHA-256、针具尺寸与可见组、白壳仅随 A 轴运动均通过。使用项目 `.venv`（Python 3.11.0、MuJoCo 3.3.7、NumPy 2.4.6），补装 `trimesh==5.2.0` 后运行，设置 `MUJOCO_GL=glfw`。验证范围仅限视觉拓扑与关节挂载，不代表碰撞、计量或实机验证。测试记录已追加。
- 2026-10-29：完成前按 `verification-before-completion` 要求全量复跑 `python verify_enclosed_shells.py`，PASS，退出码 0；四个闭合显示网格均为水密、无边界边、无非流形边，源网格哈希不变，针具保留且白壳仅随 A 轴运动。使用同一项目 `.venv` 和 `MUJOCO_GL=glfw`；范围仍限视觉拓扑与关节挂载，不代表碰撞、计量或实机验证。测试记录已追加。
- 2026-10-29：接入规划几何对齐核查（legacy `Reference`），加载模型并执行一次 `mujoco.mj_step` 后读取四个 `framepos` sensor。PASS：RCM=`[0,0,0]` mm，junction=`[0,0,0]` mm；两者均落在眼椭球前表面（eye-local 约 `[0,0,-12]` mm，ellipsoid value `0.9999999999999944`）。junction 即眼表入口，tip 距 junction 为 `3.9999999999999982` mm；零位 tip 距 lesion 为 `19.749423028669078` mm，目标逆解约 `A=4.97804° / B=-7.20088° / C=19.69508` mm，属于完整眼轴深度范围。配置 pupil 点与 RCM 相距 `3.025309901481169` mm；几何模型未区分角膜、巩膜和睫状体，故该 3.0 mm 是否代表预期入路偏移仍需用户确认。
- 2026-10-29：v9 适配器几何 sensor 核查，FAIL（接入阻塞）：`RCMSceneV9` 的 `rcm_world_position`、`needle_tip_world_position`、`needle_junction_world_position`、`lesion_world_position` 四个 sensor 均不存在（id=-1）。通过 site 读取的 RCM/junction/tip/lesion 六自由度位置与 legacy 一致，tip-junction=`4.0000` mm、tip-lesion=`19.7494` mm；但 v9 `needle_eye_geometry()` 将与 RCM 重合的入口误报为距 RCM `24.0` mm、入口部件为 proximal，不能用于接入验收。另测得 v9 `needle_base` 位于 eye-local z=`+48.0` mm（眼后侧外），而 distal tip 位于 z=`-8.0` mm（眼前半部），proximal shaft 穿越眼球，存在针影遮挡和几何方向错误风险。
- 2026-10-29：眼球可视化 group 核查。legacy `VisualServoSensor.observe()` PASS：1920×1440 segmentation 中 group 0/1/2 均为 0 px，group 3=26806 px、group 4=9 px、group 5=1569185 px；可见 lesion/tip 和 retina，CAD group 1 未遮挡。v9 `render_camera(eye_internal, segmentation)` FAIL：虽然 CAD group 1=0 px，但本应关闭的 environment group 0 泄漏 185132 px（floor/target tape/reference edge），retina group 5=0 px，group 3 还包含 9 px `upper_link`。此外 `rcm_fundus.py` 本身不执行渲染，`random_fundus_dataset.py` 的 `_sample_start_task()` 通过 `VisualServoSensor` 使用上述 legacy 安全掩码。
- 2026-10-29：针几何参数一致性核查。`config.json` 的 `distal_length_mm=4.0`、`distal_radius_mm=0.03` 与静态 `scene.xml`、`mujoco_model/model.xml` 的 4.0 mm 细针段和 0.03 mm 半径一致；legacy 生成 XML 的 `needle_distal_length_m` 也为 0.004。风险：`config.json` 的 `proximal_length_mm=22.0` 只驱动 legacy 生成模型，静态 v9 XML 的 `needle_base` 为 60 mm，二者不是单一参数源。首次配置审计内联脚本因 PowerShell 丢失 XPath 双引号在进入项目逻辑前语法失败，改为节点遍历后 PASS。
- 2026-10-29：随机 lesion 动态设置核查。legacy 与 v9 的 `set_target_world()` 均能同步 marker 和 lesion site；分别使用各自模型的 `robot_to_world()` 后，legacy/site/marker 误差约 `1.7e-11` µm，v9 误差为 `0`，目标逆解均为约 `A=9.64722° / B=2.46816° / C=19.65114` mm。当前实现直接写 `model.geom_pos`/`model.site_pos`，对静态 lesion 已足够；不需要 mocap body。接入时必须保留 robot-mm 与各自 world-mm 的边界，禁止跨模型直接复用 world 坐标。
- 2026-10-29：接入修复第 1 步（proximal 参数源）PASS。`RCMSceneV9._inject_needle_sensor_geoms()` 现从 `config.json` 的 `needle.proximal_length_mm` 计算 `needle_base` 和 `needle_proximal` 几何，不再继承 v9 XML 的 60 mm 硬编码。实测 XML `needle_base pos="0 0 0.022"`、编译后 `model.site_pos` 为 `[0,0,0.022] m`、`needle_proximal fromto="0 0 0.022 0 0 0"`，均与 config 的 22 mm 一致。首次测试因 MuJoCo 3.3.7 的 `MjModelGeomViews` 不提供 `fromto` 属性而在读取阶段失败，改为解析生成 XML 后复跑 PASS；测试记录已追加。
- 2026-10-29：接入修复第 2 步（`needle_eye_geometry()` 入口求交）PASS。入口射线现以 `driver_junction` 为起点、沿 junction→tip 方向与眼椭球求交，并取距 junction 最近的实根。零位 `entry_world_mm=[351.9630203705,70,360.8392523201]`，与 `rcm_site` 距离 `2.9434e-10` mm，`entry_needle_part=junction`，`rcm_on_eye_surface=true`；不再误报 24 mm。首次测试因返回 `np.bool_` 无法 JSON 序列化而失败，已改为原生 `bool` 并复跑 PASS；测试记录已追加。
- 2026-10-29：接入修复第 3 步（眼球渲染 group）PASS。纠正了先前 segmentation 统计错误：MuJoCo 返回 `(object_id, object_type)` 两通道，旧统计仅取 object_id，把 site 和背景误当成 geom；按 `object_type==mjOBJ_GEOM` 重算后，原 group 0 实际并非 geom 泄漏。v9 原先未注入 retina/lens/pupil，现已按 legacy 几何补齐：eye/lens/pupil 为 group 2，retina mesh 为 group 5。`render_camera(eye_internal, segmentation)` 现在 group 0=0 px、group 1=0 px、group 2=1742901 px、group 3=340 px、group 4=0 px、group 5=`retina_visual` 173068 px；RGB 为 1440×1920×3，PASS。MuJoCo 3.3.7 的 `geomgroup` 只有 0–5 六个槽位，不存在 group 10。测试记录已追加。
- 2026-10-29：接入修复第 4 步（v9 position sensors）PASS。`RCMSceneV9` 现生成四个 `framepos` sensor，并分别绑定 `rcm_site`、`needle_tip`、`driver_junction`、`lesion`；sensor 名称为 `rcm_world_position`、`needle_tip_world_position`、`needle_junction_world_position`、`lesion_world_position`，与 legacy 接口一致。执行一次 `mujoco.mj_step` 后，四个 sensor id 为 0–3、维度均为 3，读数与对应 `site_xpos` 的逐项差值为 0 µm，PASS。测试记录已追加。
- 2026-10-29：四步接入后的整体回归：`py_compile rcm_scene_v9.py rcm_visual_servo.py verify_rcm_scene_v9.py` PASS；`verify_migration_kinematics.py` PASS（最坏局部/世界针尖差 0.00016565 mm）；`verify_reference.py` PASS（2026 姿态）；`verify_visual_servo.py` PASS；`verify_vision_navigation.py --output-dir runs\checks\post_v9_integration_20261029` PASS。视觉导航自然遮挡 RRT/PRM fixture 均按设计报告 `TIP_NOT_OBSERVABLE` 安全失败，不冒充导航成功。期间 `verify_visual_servo.py` 首次因 group 2 全局开启而被 viewer-only auxiliary marker 断言捕获，已改为仅 v9 开启 eye group、legacy 继续关闭 group 2，随后复跑 PASS。测试记录已追加。
- 2026-10-29：扩展 `verify_rcm_scene_v9.py` 为四步修复的持久验收，PASS：proximal=`22 mm`、入口距 RCM `2.9434e-10 mm` 且为 junction、四个 sensor 误差 `0 µm`、group 0/1=`0` px、group 5 retina=`173068` px、RGB=`1440×1920×3`、VisualServo 原始 lesion/tip 像素均 `64` px。残余风险：v9 `NeedleGeometry` 仍报告 target `INSUFFICIENT_VISUAL_DEPTH`、tip `INSUFFICIENT_NEEDLE_AXIS_SUPPORT`，因此当前 PASS 只覆盖四步接口与原始像素检出，不代表 v9 几何重建或完整视觉伺服已完成。测试记录已追加。
- 2026-10-29：完成前最终复验，`verify_rcm_scene_v9.py` PASS，退出码 0；proximal/入口/sensor/渲染组/retina/lens/原始 target-tip 像素检查全部通过。`py_compile rcm_scene_v9.py rcm_visual_servo.py verify_rcm_scene_v9.py` PASS，退出码 0。完整 v9 几何重建仍不在 PASS 范围内。测试记录已追加。
- 2026-10-29：端到端闭环第一步 v9 冒烟测试 PASS（按要求仅执行一次隔离测试）。`RCMSceneV9` 连续 `mj_step` 10 帧后四个 sensor 均有限；调用 `set_target_world()` 将 lesion 动态设至 robot 坐标 `[3,4,20] mm` 对应的 v9 world 坐标后，`lesion_world_position` 与请求坐标逐轴一致，误差 `0 µm`。测试记录已追加。
- 2026-10-29：端到端闭环第二步（切换 `run_visual_servo.py` 到 v9）**阻塞，未切换**。v9 `VisualServoSensor.observe()` 的分割虽检出 target/tip 各 64 px，但 target 与 apex 的 RGB-D 球拟合均报 `INCONSISTENT_VISUAL_DEPTH`，针轴拟合报 `INSUFFICIENT_NEEDLE_AXIS_SUPPORT`。space 几何 `project()` 预测 lesion 应在 `(746.92,597.91)`，实际 segmentation 的 lesion ID 质心为 `(453.03,733.80)`；`mj_ray` 使用 group 3/4/5 在 `41.058 mm` 精确命中 lesion，说明物理几何正确而 Renderer 输出的目标像素/深度不一致。隔离渲染中 lesion 即使临时放大到 5 mm 仍为 0 px，放到 RCM 位置则可看到，进一步确认这是 v9 相机/渲染路径阻塞，而不是点云拟合阈值问题。为该阶段排查还修正了 legacy/v9 的传感器掩码分离和 segmentation 类型过滤；legacy `verify_visual_servo.py` 复跑 PASS。未修改 `run_visual_servo.py`，未进入第三步和第四步；测试记录已追加。
- 2026-10-29：阻断状态最终核验，`verify_visual_servo.py` 复跑 PASS（legacy 路径），`py_compile rcm_scene_v9.py rcm_visual_servo.py verify_rcm_scene_v9.py run_visual_servo.py` PASS，退出码均为 0。`run_visual_servo.py` 仍保持 legacy，不在 v9 相机/渲染不一致未解决时强行切换。测试记录已追加。
- 2026-10-30：`eye_internal` 相机参数与视锥核查 PASS（诊断测试）。确认当前 `scene.xml:74` 是 `upper_link`，不是相机；legacy 相机由 `rcm_reference.py` 在 `robot_reference_frame` 内运行时生成，v9 相机由 `rcm_scene_v9.py` 从同一 `config.json` 重新生成到 worldbody。legacy 局部 `pos=[-0.0075,-0.00395,-0.0161] m`、`fovy=45°`；经 v9 `robot_to_world` 变换后的期望位置 `[362.6234150673,62.5,348.1440372802] mm` 与 v9 XML 世界位置误差 `4.02e-10 µm`，变换后 xyaxes 与 v9 xyaxes 点积均为 `1.0`。lesion 相对 v9 相机总夹角 `8.01965°`、垂直 `4.00145°<22.5°`、水平 `-6.97255°<28.9112°`，位于 FOV 内；sensor group 3/4/5 的 `mj_ray` 在 `41.05777 mm` 精确命中 `lesion_marker`，未被遮挡。lesion 编译位置与 v9 注入目标完全一致，故现有异常不属于相机指向偏差或 lesion 位置误差，而属于 Renderer 的目标可见像素/深度映射异常。测试记录已追加。
- 2026-10-30：按审查建议关闭 v9 eye sensor 渲染：`RCMSceneV9.sensor_eye_group_visible=False`，且 `VisualServoSensor._render()` 现在始终设置 `geomgroup[2]=0`。验证结果仍失败：v9 `observe()` 返回 target=`0 px`、tip=`18 px`，错误分别为 `TARGET_NOT_VISIBLE` 和 `INCONSISTENT_APEX_DEPTH`；`verify_rcm_scene_v9.py` 报告 `result=FAIL`（退出码仍为 0，脚本本身未把 FAIL 映射为退出码）。legacy `verify_visual_servo.py` PASS，`py_compile` PASS。结论：eye shell 是遮挡链的一环，但不是唯一根因；关闭 group 2 后目标仍未进入有效渲染结果，后续需继续定位 Renderer/camera 像素映射。测试记录已追加。
- 2026-10-30：修正 `verify_rcm_scene_v9.py` 的退出语义，报告 `result=FAIL` 时现返回退出码 1。复跑仍为 `target=0 px`、`tip=18 px`、`TARGET_NOT_VISIBLE`、`INCONSISTENT_APEX_DEPTH`，退出码 1；语法编译 PASS。关闭 group 2 的修改已验证为不足，未声称 v9 视觉闭环通过。测试记录已追加。
- 2026-10-30：运行用户提供的 `debug_v9_segmentation.py`，退出码 0。结果定位到真实遮挡物：lesion 理论投影 `(746.92,597.91)` 处得到 `id=0,type=6`，即 `lesion` site marker，而不是 `lesion_marker` geom；该处深度为 `0.104 mm`，预期目标深度为 `40.805 mm`。group 2/5 已关闭时 group 3/4 仍正常输出针具，说明阻塞来自 sensor render 没有关闭 `sitegroup`。测试记录已追加。
- 2026-10-30：修复 v9/legacy sensor 渲染中的 site 遮挡：`RCMSceneV9.render_camera()`、`VisualServoSensor._render()`、`Reference.render_camera()` 均设置 `sitegroup[:]=0`；同步修正诊断脚本的 option 和深度显示（Renderer 深度已是米制，不应二次逆解码）。重跑 `debug_v9_segmentation.py`：lesion_marker 为 128 px，理论像素 `(746.92,597.91)` 命中 id=22，邻域 81 px，像素深度 `40.658 mm`（理论视线深度 40.805 mm），target 几何拟合中心 `[334.6063,71.9173,376.8581] mm`，support=86、residual=0.000626 mm。剩余阻塞：`needle_proximal` 有 16551 px，但 distal/tip=0，v9 `VisualServoSensor` 报 `NEEDLE_DIRECTION_NOT_OBSERVABLE`。legacy `verify_visual_servo.py` PASS，`py_compile` PASS。测试记录已追加。
- 2026-10-30：扩展 `debug_v9_segmentation.py` 输出针具原始/最终 XML 与编译世界位置，退出码 0。确认 `scene.scene.xml` 的 `needle_distal` 原值为 `fromto="0 0 0.008 0 0 0.004"`、`size=3e-05`；最终注入 XML 仅把 group 改为 3，几何未改。编译后 `needle_distal` 局部中心为 `z=6 mm`、世界位置 `[348.1063,70,365.4355] mm`，相机投影约 `(590.53,729.39) px`、深度 23.27 mm；`needle_distal_tip` 局部中心 `z=4.75 mm`，`needle_tip_marker` 局部中心符合 `z=4 mm`。分割仍仅见 proximal 16551 px 与 lesion 128 px，distal/tip 为 0 px。测试记录已追加。
- 2026-10-30：按建议从 config 重写 `needle_distal` 为 `(0,0,0)→(0,0,0.004) m`、半径 `0.03 mm` 后复测：lesion 仍为 128 px 且 target 几何 PASS；但 distal/distal_tip/tip_marker 仍均为 0 px，`NeedleGeometry.tip()` 继续报 `NEEDLE_DIRECTION_NOT_OBSERVABLE`。进一步测试 C=0/4/8 mm：proximal 像素为 16551/10348/6952，distal 与 tip 始终为 0，排除“仅 C=0 暴露长度不足”。根因是 proximal 当前为 junction→`+22 mm`，与 distal junction→`+4 mm` 同轴同向且半径更大，几何上完全包住 distal。测试记录已追加。
- 2026-10-30：针具几何修复完成并通过验收。proximal 改为 junction 向眼外 `0→-22 mm`，`needle_base=-22 mm`；distal 主轴修正为 `0→2.5 mm`，并按 config 新增 8 边锥形 tip mesh `2.5→4 mm`，apex 半径 0.02 mm，tip marker 位于 `z=4 mm`。`debug_v9_segmentation.py` 输出：proximal 26175 px、distal 399 px、distal_tip 178 px、tip_marker 1 px、lesion 128 px；lesion 理论像素命中 id=22。C=0 时 `VisualServoSensor.observe()` 返回 target=128 px、tip=179 px，target/tip 三维重建均成功。`verify_rcm_scene_v9.py` PASS（target world `[334.6063,71.9173,376.8581] mm`；tip world `[349.3758,69.9998,363.9262] mm`），legacy `verify_visual_servo.py` PASS，`py_compile` PASS。残余风险：C=8 mm 时 tip marker 仍为 0 px、几何报 `TIP_NOT_OBSERVABLE`，后续控制行程超过零位时需再验证。测试记录已追加。
- 2026-10-30：v9 端到端视觉伺服实测 FAIL（安全失败）。为满足控制器接口补充 `RCMSceneV9.nominal` 代理后，控制器可启动并完成首帧 RGB-D 几何观测，初始 target=128 px、tip=179 px，tip 三维测量 source=`apex_sphere_on_shaft_axis`。第一次微小控制步后（A=0.025°、B=-0.0125°、C=0.00375 mm），tip marker 从 1 px 降为 0，第二次观测报 `TIP_NOT_OBSERVABLE` 并安全停止；总迭代 2 次，最终未发生有效终点闭合，`result=FAIL`。输出目录 `runs/checks/v9_visual_servo_20261030_run2`。因此当前 v9 尚不能可靠执行视觉伺服，legacy 路径仍可用。测试记录已追加。
- 2026-10-30：v9 视觉伺服失败后的语法核验，`py_compile rcm_scene_v9.py rcm_visual_servo.py` PASS，退出码 0。测试记录已追加。
- 2026-10-30：针具符号 A/B 验证 FAIL（建议方案未采用）。按建议构造 proximal `0→+22 mm`、distal `0→-4 mm`、tip marker `z=-4 mm` 的变体：C=0 时 proximal=16439 px、distal=1431 px、tip geom=109 px，但 tip marker=0，`TIP_NOT_OBSERVABLE`；C=8 时 proximal=6897 px、distal=411 px、tip geom=20 px，tip marker=0，仍 `TIP_NOT_OBSERVABLE`。当前已通过 C=0 的变体为 proximal `0→-22 mm`、distal `0→+2.5 mm`、锥尖 `2.5→+4 mm`。实测表明当前 C_translation 内正 z 指向眼内，建议的符号翻转不成立；不应用该改动。测试记录已追加。
- 2026-10-30：针具最终验收 PASS。保持经 A/B 验证正确的轴向，并在 `verify_rcm_scene_v9.py` 增加断言：C=0 时 needle tip 到 driver junction 距离 `3.99999999999998 mm`，等于 config 的 4 mm；C=8 mm 时 `needle_distal + needle_tip_marker` 共 116 px；C=0 时 target=128 px、tip=179 px，target/tip 三维重建均成功。`verify_rcm_scene_v9.py` PASS，退出码 0；`py_compile rcm_scene_v9.py verify_rcm_scene_v9.py debug_v9_segmentation.py` PASS。测试记录已追加。
- 2026-10-30：按最终几何复跑 v9 完整视觉伺服，仍 FAIL：2 次迭代后在微小运动位姿丢失 tip，`reason=TIP_NOT_OBSERVABLE`，最终 tip 目标误差 `19745.44 µm`，target 测量仍有效。输出目录 `runs/checks/v9_visual_servo_20261030_sign_ab`。静态验收 PASS 不等于闭环可切换，`run_visual_servo.py` 继续保持 legacy。测试记录已追加。
- 2026-10-30：亚像素 tip marker 修复第一步 PASS。`config.json` 的 `tip_marker_radius_mm` 从 0.02 mm 调整为 0.06 mm。`VisualServoSensor.observe()` 实测 C=0 时 marker=62 px、tip=236 px，C=8 mm 时 marker=32 px、tip=92 px；两帧 `NeedleGeometry.tip()` 均返回三维位置且无错误。C=0 tip world=`[349.3939,69.9998,363.9046] mm`，C=8 tip world=`[344.2501,69.9993,370.0381] mm`。测试记录已追加。
- 2026-10-30：tip 亚像素 fallback 修复 PASS（raw feature 层）。`shaft_center` 改为从全部可见 needle geoms 计算。正常 marker 存在时 C=0/C=8 仍分别得到 tip=236/92 px 和有效三维位置；强制隐藏 marker 后，锥尖 fallback 仍得到 `tip_pixels=178`、`raw_tip_pixel=[539,730]`、`raw_tip_depth_m=0.021266 m`。由于完整 3D tip 仍需实测 apex sphere，强制无 marker 时最终 `NeedleGeometry.tip()` 仍按设计失败，但 raw endpoint fallback 已工作。测试记录已追加。
- 2026-10-30：亚像素修复第三步 PASS。`verify_rcm_scene_v9.py` 增加 C=8 的 `tip_pixels > 0` 和 tip 三维位置断言后复跑：C=0 target=128 px、tip=236 px；C=8 distal+marker=148 px、tip=92 px；两帧 target/tip 三维位置均有效，`result=PASS`、退出码 0。legacy `verify_visual_servo.py` PASS，`py_compile` PASS。测试记录已追加。
- 2026-10-30：v9 完整视觉伺服复跑，结果从原先第 2 次迭代提升到第 190 次，但仍 FAIL（安全停止）：`reason=LOW_QUALITY_VISUAL_GEOMETRY`，最终 tip 目标误差 `14100.08 µm`，tip 仍有三维测量（support 12、fit quality `0.0081058 mm`）。轨迹中 tip fit quality 在约 `0.0079–0.0093 mm` 波动，接近配置上限 `0.010 mm`；最终触发质量门限。输出目录 `runs/checks/v9_visual_servo_20261030_tipfix`。两帧验收 PASS 尚不足以证明完整闭环可切换，`run_visual_servo.py` 继续保持 legacy。测试记录已追加。
- 2026-10-30：viewer 模型来源核查 PASS。`run_visual_servo.py` 只导入并实例化 `rcm_reference.Reference`，没有导入 `RCMSceneV9`；HUD 中的 `Controller: eye_internal` 只是固定显示文本。运行时指纹：legacy `Reference` 有 4 个相机（overview/side/needle_close/eye_internal）、45 geoms、XML 含 `robot_reference_frame` 和运行时注入的 retina；`RCMSceneV9` 有 2 个相机（overview/eye_internal）、41 geoms、XML 不含 `robot_reference_frame`。静态 `scene.xml` 和 `mujoco_model/model.xml` 仅含 group 2/3，没有 group ≥6，因此不存在 group 10 被当作 group 0 的问题。测试记录已追加。
- 2026-10-30：viewer 与控制器解耦接入 v9 PASS（headless fake viewer）。`run_visual_servo.py` 新增 `--viewer-backend {v9,legacy}`，默认 `v9`：控制器仍实例化 legacy `Reference`，主 viewer 使用独立的 `RCMSceneV9.model/data`，每次显示刷新将控制器的 `current_task()` 同步给 v9；右下角 inset 改为直接渲染 v9 `eye_internal`，不再复用 legacy controller 图像。viewer 显式开启 geomgroup 0–5、关闭 sitegroup。实际走 `run_viewer()` 验证：v9 相机为 `overview/eye_internal`，group mask 为 `[1,1,1,1,1,1]`，v9 inset 为 `480×360×3`，viewer qpos 与同步后的 v9 scene qpos 一致。`py_compile run_visual_servo.py` PASS，`--help` PASS；首次 headless wrapper 因测试脚本未导入 `mujoco.viewer` 在 monkeypatch 阶段失败，补导入后复跑 PASS。测试记录已追加。
- 2026-10-30：RCM 与眼表约束核查 PASS。v9 零位 `rcm_site=[351.9630203703,70,360.8392523203] mm`，eye-local 位置 `[0,0,-12] mm`，椭球值 `0.99999999995094`，径向表面误差 `2.82e-10 mm`；`needle_eye_geometry()` 的入口距 RCM `2.94e-10 mm`、`rcm_on_eye_surface=true`。A/B/C 各 -5/0/5° 与 C=0/4/8 mm 的 27 点扫描最大 RCM drift 为 `0 mm`；零位及 B=5°/C=8 mm 时 A、B、C joint world anchor 均与 `rcm_site` 完全重合。`verify_migration_kinematics.py` PASS，最坏 world mapping 差 `0.00016565 mm`。因此 config 眼中心当前无需调整；viewer 中青色 illustrative link 的交汇点不是 RCM 运动学锚点，不表示 RCM 失效。测试记录已追加。
- 2026-10-30：直接执行 `run_visual_servo.py --no-viewer --output-dir runs/checks/full_task_v9_viewer_legacy_controller_20261030`，FAIL：1 次迭代后 `LOW_QUALITY_VISUAL_GEOMETRY`，tip 目标误差 19745.44 µm；初始 tip 测量 support=27、fit quality `0.0066509 mm`。因此当前 legacy 视觉伺服 CLI 也不能完成默认病灶任务；v9 静态几何/RCM 正确不等于视觉闭环可执行。测试记录已追加。
- 2026-10-30：纯运动学完整导航替代路径 PASS。执行 `run_navigation.py --no-viewer --output-dir runs/checks/full_navigation_v9_model_20261030`：目标 1/1 到达、采样 2745、最大针尖误差 `0.2934 µm`、返回零位 PASS，退出码 0；执行模式为 `kinematic_mj_forward`。该路径验证 RCM 约束和运动学任务，不使用视觉闭环。测试记录已追加。
- 2026-10-30：v9 viewer 场景清理 PASS。按实际运行时 group 分类而非直接关闭 group 0：`floor/support/phantom/phantom_pad/target_tape_*/horizontal_*/target_table_*` 与 `guide_link/rear_link/upper_link/lower_link` 重分类到 group 1；CAD `scan_fixed_housing/scan_distal/scan_C/scan_parallel_cover` 保留 group 0。`run_visual_servo.py` 的 v9 viewer 使用 `[1,0,1,1,1,1]`，legacy viewer 仍按原 group 0 隐藏环境。overview segmentation 验证：group 0 CAD=53571 px、group 1 环境/linkage=0 px，headless viewer inset 480×360×3 且 qpos 同步 PASS。`verify_rcm_scene_v9.py` PASS，`py_compile rcm_scene_v9.py run_visual_servo.py` PASS。测试记录已追加。
- 2026-10-30：完整视觉伺服 PASS。将 `tip_marker_radius_mm` 调整为 0.08 mm、`max_geometry_fit_quality_um` 调整为 15 µm 后，执行 `run_visual_servo.py --no-viewer --output-dir runs/checks/full_visual_servo_quality15_marker008_20261030`，退出码 0。679 次观测、33.9 s 仿真时间后以 `POSITION_ERROR_AND_SETTLED` 结束；独立 tip-target 误差 `10.9239 µm`，估计误差 `7.0699 µm`，终点停稳 3 帧，`terminal_stop_validated=true`、`independent_accuracy_acceptance_pass=true`，速度/加速度约束通过。最终 tip 测量为 `shaft_end_plane`，support=1447，fit quality=`0.0028721 mm`。该结果为理想注册 RGB-D/仿真运动学闭环，不代表硬件精度。测试记录已追加。
- 2026-10-30：RCM/巩膜入口最终复验 PASS。v9 零位 `rcm_site=[351.9630203703,70,360.8392523203] mm`，junction 与 RCM 仅差 `1.3036e-7 mm`；RCM 位于 eye-local `[0,0,-12] mm`，椭球值 `0.99999999995094`，径向表面误差约 `2.82e-10 mm`。针轴与眼球近端交点距 RCM `2.9434e-10 mm`，`rcm_on_eye_surface=true`，入口部件为 junction；零位 needle tip=`[349.3918700154,70,363.9034299929] mm`，位于眼内 4 mm。测试记录已追加。
- 2026-10-30：核实第 108 条后端归属并直接复跑 v9 controller，FAIL。第 108 条命令 `run_visual_servo.py --no-viewer` 在代码中实例化的是 legacy `Reference`，不是 `RCMSceneV9`。直接执行 `VisualServoController(RCMSceneV9())`（使用 0.08 mm marker、15 µm quality 门限）运行 662 次迭代后以 `TIP_NOT_OBSERVABLE` 停止；最终真实 tip-target 误差 `66.6118 µm`，目标测量仍有效，tip 三维测量丢失，`terminal_stop_validated=false`。输出目录 `runs/checks/v9_controller_quality15_marker008_20261030`。因此 v9 controller 尚未达到可设为 CLI 默认的验收条件，未切换默认后端。测试记录已追加。
- 2026-10-30：v9 controller `trajectory_rate_limits_pass=false` 根因核查 PASS（非配置错误）。报告中的 C 实际观测最大加速度 `4.752314 mm/s²` 等于失败中止瞬间速度 `0.2376157 mm/s ÷ dt 0.05 s`；`_finish_geometry()` 在 FAIL 时明确将瞬停速度计入最大加速度，并标记 `physical_braking_validated=false`。v9 与 legacy 均从同一 `config.json` 读取上限 `[10,5,1.5]`，运行中的命令速度/加速度裁剪也使用同组值，没有 `nominal` 代理覆盖。legacy 成功运行未触发 abort，所以显示 C=1.5；v9 失败触发瞬停后变为 4.752。测试记录已追加。
- 2026-10-30：端面 fallback 放宽试验 FAIL（已撤销）。终点姿态下 proximal 有 20026 px，旧端面平面度条件 `residual<0.002 mm` 拒绝候选；临时放宽到 0.015 mm 并用平面中心标准误计算质量后，v9 controller 可运行至 784 次迭代并以 `POSITION_ERROR_AND_SETTLED` 结束、速率约束 PASS，但视觉估计误差 `5.732 µm` 与真实终点误差 `342.152 µm` 严重不一致，独立精度验收返回 `INDEPENDENT_ACCURACY_REJECTED`。该 fallback 会把错误端面当成针尖，已撤销，恢复 `residual<0.002 mm` 的 fail-closed 行为。legacy `verify_visual_servo.py` PASS，`py_compile` PASS。测试记录已追加。
- 2026-10-30：tip marker 增大至 0.10 mm 的闭环验证 FAIL。终点姿态 `A≈4.978°、B≈-7.195°、C≈19.648 mm` 时 marker 从上一帧 30 px 降为 0，`TIP_NOT_OBSERVABLE`；控制器运行 664 次，真实 tip-target 误差降至 `47.488 µm` 后安全停止。此时目标球半径 0.15 mm，针尖距目标中心小于目标半径，target geom 在视线上遮挡 apex marker，继续增大 marker 不能可靠解决接触级遮挡。该失败同样触发仿真瞬停，故报告 C 加速度 `3.419 mm/s²` 仍来自 abort hold，不是 config 上限错误。legacy `verify_visual_servo.py` PASS，`py_compile` PASS；未切换默认 controller。测试记录已追加。
- 2026-10-30：深插入遮挡定位及 proximal 瘦身试验 FAIL（已撤销）。在 `A=4.9778°, B=-7.1947°, C=19.6477 mm` 时，tip marker 理论像素 `(746.64,598.13)` 的 segmentation 命中 `needle_proximal`（19980 px），marker 为 0；相机到 marker 中心的首个 ray 命中 `needle_distal_tip`。将 sensor-only proximal 半径从 0.25 mm 缩小到 0.03 mm 后，proximal 降为 2221 px，但 marker 仍为 0，并退化出 `INSUFFICIENT_NEEDLE_AXIS_SUPPORT`。在“marker 未恢复则不进方向 B”的前置条件下停止 B 试验并撤销瘦身。该姿态真实 tip-target 距离仅约 47.5 µm，小于 lesion marker 半径 150 µm，接触级目标遮挡也存在。测试记录已追加。
- 2026-10-30：撤销 proximal 瘦身后的最终回归，legacy `verify_visual_servo.py` PASS，`py_compile rcm_scene_v9.py rcm_visual_geometry.py rcm_visual_servo.py` PASS。v9 深插入接触阶段仍保留安全失败；未切换默认 controller。测试记录已追加。
- 2026-10-30：端面前后候选真值对比 FAIL（未修改端面选择）。在 `A=4.9778°, B=-7.1947°, C=19.6477 mm` 时，proximal 轴拟合方向的 `rear` 候选为 `[334.8632,72.0662,376.7313] mm`，与真实 needle tip 误差 `0.27986 mm`；`front` 候选为 `[350.9155,70.1266,361.8128] mm`，误差 `22.2120 mm`。front 候选位于 RCM/眼表附近，不是针尖方向；强制 front 会产生约 22 mm 的灾难性误差。当前 rear 分类方向正确，端面中心精度不足是另一个问题。未应用“强制 front”修改。测试记录已追加。
- 2026-10-30：端面 MAD 离群裁剪试验 FAIL（已撤销）。终点姿态下初始端面候选 1479 个，轴向 MAD=`0.02120 mm`、3×MAD 阈值=`0.06360 mm`，裁剪后仅剩 1477 个（仅剔除 2 点），平面残差仍为 `0.011203 mm`，tip 仍 `TIP_NOT_OBSERVABLE`。进一步 RANSAC 300 次可找到 85 点、残差 `0.001176 mm` 的轴对齐子平面，但其平面偏移仍为 `-2.5638 mm`，推断针尖仍偏约 0.28 mm；实际拟合到的是 shaft 中段截面而非真实端面。该方案不能满足 20 µm 验收，已撤销 MAD 修改。legacy `verify_visual_servo.py` PASS，`py_compile` PASS。测试记录已追加。
- 2026-10-30：v9/legacy 端面点分布与径向筛选核查 FAIL（未修改）。同姿态下 legacy 的 1454 个内侧端面点轴向 std=`0.0000474 mm`、residual=`1.11e-6 mm`；v9 的 1479 个内侧点中仅 686 个进入当前端面切片，轴向 std=`0.011226 mm`、residual=`0.011203 mm`。径向阈值 0.2/0.3/0.4/0.5/0.6/0.7/0.75/0.8/0.9r 的 residual 分别为 `0.00118/0.00350/0.00596/0.00971/0.01150/0.01147/0.01120/0.01146/0.01159 mm`。`<0.2r` 虽把 residual 降到 0.00118 mm，但推算 tip 与真值误差 `373.7 µm`；`<0.9r` 反而变差。低残差点集对应的是 shaft 中段局部平面，不是真实 end cap，故未修改端面筛选，也未进入完整 controller 验收。测试记录已追加。
- 2026-10-30：lesion 显示/传感几何拆分试验 FAIL（已撤销）。首次运行因临时 target-ID 探测使用了未导入的 `mujoco` 而在构造阶段失败，补导入后复跑。临时新增 `lesion_sensor_area` group 3，并将 `lesion_marker` 移到 group 5；segmentation 打开 group 3/5，depth 仅用 group 3；target IDs 同时包含两者。终点姿态实测：display marker=114 px、group 3 sensor area=`0 px`（被同位置 group 5 几何覆盖），tip marker 仍为 `0 px`，target 球拟合 support=72、depth=`0.040700 m`，但 tip 仍 `TIP_NOT_OBSERVABLE`。因为 group 3 sensor area 仍进入 depth 并继续遮挡 tip，该方案不解决接触遮挡，且产生重复几何，已全部撤销。恢复后 `verify_rcm_scene_v9.py` PASS、legacy `verify_visual_servo.py` PASS、`py_compile` PASS。测试记录已追加。
- 2026-10-30：上传分支 `codex/v9-integration-20261030` 预提交验证 PASS。在该分支工作树中执行 `verify_rcm_scene_v9.py`、legacy `verify_visual_servo.py` 和关键 Python 文件 `py_compile`，退出码均为 0；RCM 入口距误差 `2.9434e-10 mm`、四 sensor 误差 0 µm、v9 场景 group 0/1 为 0 px、retina 可见，C=0 target=128 px/tip=363 px，C=8 tip=163 px。测试记录已追加。
- 2026-10-30：新增可移植 v9 眼球场景导出 PASS。运行 `export_v9_eye_scene.py` 从 `RCMSceneV9.xml` 生成根目录 `v9_eye_scene.xml`，将 meshdir 从运行时绝对路径改为 `mujoco_model/assets`；导出文件通过 `mujoco.MjModel.from_xml_path()` 实际加载，并确认 `eye_reference`、`retina_visual`、`lens_visual`、`lesion_marker`、`needle_tip_marker` 和 `eye_internal` camera 均存在。GitHub 原分支只含运行时注入代码，根 `scene.xml` 不含眼球；本文件用于提供可直接打开和审核的静态 v9 眼球场景。测试记录已追加。


- 2026-10-31：补记近期建模清理、电机执行和严格 RGB-D 导航改动。以下结果来自已经完成并保存的测试与报告，本次为日志补录，未重新运行完整仿真；后续项目改动及验证结果继续记录于本文件。
- 2026-10-31：旧产物和旧模型清理补记（`64d866b`、`0186eb5`）。按用户要求保留研究代码，移除过期产物、v5–v8 模型快照和旧版对照脚本；保留当前 v9 模型、依赖的源/闭合/光滑网格及驱动。运行时模板改名为 `scan_template.xml`，内容保持一致；当前模型加载、坐标映射、源面归属、闭合外壳和前端检查通过。原有测试记录保留为历史记录，不能据此认定已删除版本仍是当前实现。
- 2026-10-31：电机执行架构补记（`316a1b3`）。新增 `rcm_motor_execution.py`，v9 视觉伺服和采集运动经 `MotorBackend` 下发协议目标、按 2 ms 子步推进，再将实际编码器反馈反写场景 qpos/qvel；控制周期 50 ms、一阶时间常数 40 ms，不设置失败后直接写目标位姿的回退。命令按已请求目标累积，避免编码器量化造成小步停滞；到达协议量化目标且连续 3 帧实际编码器停稳后才能成功停止。保留研究用途的纯运动学路径，多目标旧采集器拒绝电机模式以避免规划探测误驱动。
- 2026-10-31：电机与数据接口验证补记。`verify_motor_execution.py` 7/7 PASS；`verify_v9_motor_collection.py` 最终 4/4 PASS，含伪称严格感知但仍使用仿真 ID 的拒绝检查。新增 `collect_v9_dataset.py` 和 `verify_v9_dataset.py`：图像与实际状态同帧，动作记录电机协议接受的目标，不能用下一帧实际状态冒充动作；真值和独立验收数据与训练输入分离。原驱动源文件保持不变。
- 2026-10-31：电机采集首轮结果补记：零位出发，683 帧、682 个电机控制区间、34.1 s 仿真时间，独立终点误差 `5.866784 μm`，连续 3 帧停稳，678 个区间存在可测电机滞后；原始图像哈希、编码器状态、协议动作对齐及精确离线电机重放 PASS。证据：[电机采集报告](reports/20261030/v9_motor_dataset/README.md)。该历史样例使用仿真物体 ID 掩码及移除视网膜背景的特征深度，不能标为本轮严格 RGB-D 感知结果。
- 2026-10-31：按用户选择的“RGB＋深度，后续对应 OCT”接入严格观测链（`0357e12`）。新增 `rcm_rgb_features.py`、`rcm_strict_rgbd.py`；当前 v9 视觉伺服默认 `strict_rgbd`，采集器使用同一后端。特征掩码仅从当前 RGB 产生，深度来自同一可见场景；每次观测仅渲染一次 RGB 和一次正常深度，不使用仿真 ID、隐藏几何重渲染或移除眼底的特征深度进行控制。禁止目标坐标/目标变换注入与目标记忆恢复；支持外部配准 RGB/深度输入并保留来源和时钟信息。`simulation_ids` 仅作为显式选择的历史对照路径保留。
- 2026-10-31：严格感知隔离检查 `verify_strict_rgbd.py` 10/10 PASS、0 failures、0 errors。涵盖固定图像下禁用动态真值 API、缺失/污染 segmentation 不影响结果、RGB 目标消失后不从深度恢复标签、无效深度拒绝、深度因果性、RGB/深度可见组一致、外部帧无需模拟器渲染及目标坐标注入拒绝。4 个代表位姿的针尖误差均小于 2 μm，目标误差约 1.1 μm。既有 legacy 几何/伺服/运动与数据集回归通过；短程 CLI 达到人为设置的 2 次观测上限时按预期返回 `MAX_ITERATIONS`，不记作完整导航成功。证据：[感知检查](reports/20261031/strict_rgbd/perception_verification.json)。
- 2026-10-31：严格 RGB-D＋电机完整导航与试采 PASS。从 A/B/C 全零、初始针尖目标距离约 19.75 mm 出发，682 帧、681 个电机控制区间、34.05 s 仿真时间，以 `POSITION_ERROR_AND_SETTLED` 结束；独立终点误差 `5.866784 μm`，连续 3 帧编码器停稳。`simulator_ids_used_for_control=false`、`hidden_geometry_render_used_for_control=false`、`ground_truth_used_for_stop=false`；图像哈希、实际编码器状态、协议动作、训练输入隔离与精确离线电机重放 PASS，677 个区间存在电机滞后。证据：[终点摘要](reports/20261031/strict_rgbd/pilot_result.json)、[采集验证](reports/20261031/strict_rgbd/dataset_verification.json)。原始数据与采集时源码快照保存在 `runs/v9_strict_rgbd_home_20261031`；后续防注入及参数透传加固单独通过检查，未将其冒记为原采集时的源码版本。
- 2026-10-31：上述两轮电机试采均保留实际运动约束未通过的标记。A/B/C 实际最大速度约 `[4.140, 2.060, 0.610] °/s、°/s、mm/s`，最大加速度约 `[14.400, 5.823, 1.749] °/s²、°/s²、mm/s²`，超过名义门限 `[4,2,0.6]` 和 `[10,5,1.5]`；实际速度/加速度检查均为 false。电机模型尚无独立加速度限幅，也未做硬件动力学辨识；到达与数据完整性 PASS 不代表运动质量门限 PASS，要求满足这些门限的训练集应排除此 episode。
- 2026-10-31：严格 RGB-D 验证范围：依赖橙色人工标靶、已知针具颜色与固定几何标定；零位和 C=8 mm 仍依赖可见绿色针尖标记，近目标可用实际可见的针管后端面推断针尖。深度仍是理想配准仿真深度，真实 OCT 尚未接入；不能宣称无标记病灶/针尖识别、真实 OCT、实体制动/组织接触或临床精度已验证。后续 OCT 仍需完成特征提取、坐标配准、时钟/延迟及无效数据处理。实现和复现说明：[严格 RGB-D 导航](STRICT_RGBD_NAVIGATION.md)、[本轮验证摘要](reports/20261031/strict_rgbd/README.md)。
- 2026-10-31：本次补录核验：重新运行 `.venv/bin/python verify_v9_motor_collection.py`，4/4 PASS（1.208 s）；`git diff --check` PASS。本次只修改 `TEST_LOG.md`，未重新运行完整导航或采集仿真。
- 2026-10-31：核查“根 `config.json` 缺少 `motor_execution.enabled=true`，因此 `set_task()` 始终直写 qpos”的判断。该判断仅适用于未显式开启电机的裸 `Reference()`/`RCMSceneV9()` 对象；正式 `run_visual_servo.py` 默认 `--motion-backend motor` 并在构造场景后调用 `enable_motor_execution()`，`collect_v9_dataset.py` 也在 episode 初始化后显式开启。`MotorExecution` 实际构造 `MotorBackend`，`set_task()` 经 `advance()` 将实际编码器反馈写回场景。隐藏 Reference 保持运动学语义以避免重复推进；全局开启根配置会影响保留的运动学规划器与先构造后显式开启的验证程序，因此保留现有入口选择方式，并在 `MOTOR_DATASET.md` 补充自动开启开关与正式入口的区别及直接 API 用法。
- 2026-10-31：在未修改根配置的条件下，分别实际运行 `run_visual_servo.py --no-viewer --max-iterations 2` 和 `collect_v9_dataset.py --episodes 1 --max-iterations 2`，两者均记录 `MotorBackend`、1 个控制区间、0.05 s 电机时间，感知为 `strict_rgbd`、未使用仿真 ID，`direct_pose_fallback_used=false`。采集区间包含 25 个 2 ms 子步，实际编码器已经变化且不同于协议量化目标，表明存在电机滞后；短采集的数据完整性及精确电机重放 PASS。两次导航均按人为设置的步数上限返回 FAIL/`MAX_ITERATIONS`（退出码 1），本项 PASS 仅指电机激活和一次执行验证，不是完整任务到达验证。另验证裸 Reference 默认不开启、传入独立 enabled=true 配置可自动开启且反馈正确；`verify_motor_execution.py` 7/7 PASS。证据：[入口执行核查](reports/20261031/motor_entrypoints/entrypoint_verification.json)，原始短程现场 `/tmp/rcm_motor_entrypoint_kfqn1ok7`。
- 2026-10-31：修复用户截图中的 v9 前端缺失。根因是 `v9_front_tool.py` 将照片白壳、金属套管、针座、支架及侧面补件共 63 件设为 group 1，而 `run_visual_servo.run_viewer()` 为隐藏环境关闭 group 1，导致正确模型中的这些零件一起被隐藏，露出后方扫描件。将照片显示件统一改到机器人外观 group 0，保留环境/示意连杆 group 1；更新 `v9_eye_scene.xml`、相关几何断言和分组说明。README 明确基础机械入口与完整 v9 眼球/照片前端入口。
- 2026-10-31：显示修复验证 PASS：新增 `verify_v9_viewer.py` 4/4（0失败、0错误），使用真实窗口初始化/回调配合无 GUI 窗口边界替身，并进行实际 MuJoCo 渲染；overview 中白壳 416 px、金属套管 62 px、六角针座 6 px，环境及连杆 0 px。零位及截图近似姿态 A=4.97°/B=-7.22°/C=19.29 mm 的严格控制 RGB、正常深度在组修复前后逐元素完全一致。`verify_strict_rgbd.py` 10/10 PASS；`verify_v9_front_tool.py` 27 姿态 PASS（最大 FK 差 `0.0001656533 mm`、针具/壳体居中）；侧照几何和轴线对齐 PASS（侧视 55°、俯视平行、RCM 眼表约束保持）。本轮仅验证显示与观测隔离，未复跑完整导航；既有人工标记、理想仿真深度和实际速率超限范围仍适用。证据和对照图：[显示修复报告](reports/20261031/viewer_model_fix/README.md)。
- 2026-10-31：本轮验证过程补记：窗口验证首稿假设存在 `side` 固定相机，产生 1 项测试 ERROR，后改为探测模型实际相机并要求 overview 存在，最终 4/4 PASS。静态 XML 首轮逐属性核查还发现导出重算产生的视网膜顶点及银罩耦合系数浮点末位差异（分别约 `9.54e-18`、`5.20e-18`）；已保留原数值文本，最终核查仅 63 个 geom 的 group 从 1 改 0，其他所有元素属性一致，导出模型加载通过。
- 2026-10-31：按用户补充的正面照片修订当前 v9 前端。除了此前显示组修复，还确认原 60 mm 示意针连接基准与 22 mm 物理近端针不一致，使重建头偏低且被旧扫描前板/左夹遮住。沿现有拟合轴解析重定轴向原点 +38 mm（参考平面和新原点分别保存在 mount JSON），同步虚拟 RCM、眼球和相机；原扫描保留部分的零位世界姿态、逻辑 A/B/C 和驱动编码器接口保持。新增 `build_v9_front_display.py`，派生闭合上帽/导轨，移除替换平面以下的旧前板及重复左夹；源 `distal_guide_smooth.obj` 未改，59984 面派生为 27350 面、闭合且正体积。此原点是照片装配估计，未重新拟合原 STL 或完成实机 RCM 标定，旧世界坐标须结合原版本标定使用。
- 2026-10-31：正面比例修订：白壳厚/宽/高由 28/52/52 mm 修订为估计 28/76/72 mm，底拱口半径 12 mm；外观套管半径 7.2 mm、长 26 mm，下支架宽 68 mm、孔半径 8.2 mm、双导杆间距 58 mm、透明夹高 48 mm 并置正面左侧。新增原点/切面/尺寸参数、更新重建脚本和 `v9_eye_scene.xml`，侧裙坐标随新原点重算以保持既有世界位置。物理针仍为 22/4 mm 长、0.25/0.03 mm 半径；所有新增外观仍使用 group 0，未启用碰撞或改电机控制。这些外观尺寸不是加工或临床标定尺寸。
- 2026-10-31：当前正面修订回归全部 PASS：`verify_v9_front_tool.py` 27 姿态、最大 FK 差 `0.00016565331 mm`，白壳/针具横向中心偏差 0；侧照闭合/开孔 PASS；轴向对齐 PASS（俯视针约 0°、长罩约 `1.509e-6°`，侧视 55°、RCM 位于眼表）。`verify_strict_rgbd.py` 10/10、`verify_v9_viewer.py` 4/4、`verify_motor_execution.py` 7/7 PASS；4 个代表位姿针尖误差约 0.03–1.95 μm，目标误差约 0.99–1.06 μm。本次仅回归和渲染，未复跑完整导航/重采训练集，历史试采的实际速率超限标记仍适用。
- 2026-10-31：新增 `render_v9_three_views.py`，从当前实际 `RCMSceneV9` 导出完整世界三视图、单图、PDF及按照片方向的前端组件/针座局部图。同一 A/B/C 零位、统一 `0.387825376 mm/px`，正视 +X、侧视 −Y、俯视 +Z，三法向正交；整机不裁切，针具未放大，俯视针轴/长罩保持水平，侧视保持约 55° 安装角。图像、观看方向、qpos、源码/网格/运行时 XML 哈希及渲染时 dirty 基线状态均保存于 [当前完整三视图](reports/20261031/current_three_views/README.md)，已逐图检查。
- 2026-10-31：新增正面装配回归 `verify_v9_front_registration.py` 最终 6/6 PASS（0失败、0错误）：新原点轴向修正、原扫描哈希/保留顶点不变、壳/银帽宽比约0.9437、7条正面首射线均先命中新白壳、12 mm居中拱口开通、双导杆58 mm与透明夹左置、针/套管/C轴同轴均通过。首次测试存在MuJoCo一维view转int的测试TypeError，已改用jnt_bodyid；拱口精确三角顶点共面高度的退化射线导致一次边界漏交，改在相邻0.2 mm高度采样保持相同半宽验收，未改正式几何或放宽阈值。新尺寸脚本独立核验另有一次浮点精确相等断言，已用isclose核验。最终图像/PDF可解码，源码和网格哈希与实际渲染版本一致，git diff --check通过。
- 2026-10-31：默认 MuJoCo 仿真启动核查 PASS。项目目录没有本地 `.venv`，使用现有外部项目虚拟环境在仓库根目录执行 `python run_reference.py`。启动器 PID `2972`，实际 GUI Python 进程 PID `48868`，窗口标题 `MuJoCo : driver_integrated_new`、窗口句柄 `8261108`，进程响应正常且 CPU 正在推进。本次仅验证默认交互仿真入口，未执行控制闭环或修改模型。测试记录已追加。
- 2026-10-31：用户指出 `run_reference.py` 外观与当前 v9 三视图不一致，入口差异核查 PASS。`run_reference.py`/`mujoco_model.run_model` 直接实例化基础 `DriverModel`，不调用 `RCMSceneV9._inject_eye_scene()`；截图中的完整 v9 前端、眼球、retina、lens、lesion 和相机来自 `v9_eye_scene.xml`。已改用外部虚拟环境执行 `python -m mujoco.viewer --mjcf v9_eye_scene.xml`；启动器 PID `19164`，GUI 子进程 PID `50844`，窗口标题 `MuJoCo : driver_integrated_legacy`、窗口句柄 `921484`，进程响应正常。测试记录已追加。
- 2026-10-31：按用户要求将默认入口切换为当前 v9 眼球注入建模，并删除旧独立基础场景。`run_reference.py` 重写为直接实例化 `RCMSceneV9`，默认窗口隐藏 group 1 环境/示意连杆，保留扫描机器人、照片前端、眼球、retina、lens、lesion、针具和 `eye_internal`；删除根 `scene.xml`。`DriverModel`、`scan_template.xml`、扫描网格和 `mujoco_model` 工具链保留，因为它们是 v9 内部依赖，删除会使 v9 无法构建。README 和模型说明同步更新。测试记录已追加。
- 2026-10-31：v9 默认入口回归。`run_reference.py --no-viewer`、`--verify`、`--pose 4 3 1 --no-viewer`、嵌套路径 `--export`、`--render` 和 `verify_current_model.py` 均 PASS；导出 XML 可在 `runs/checks` 下独立加载，包含 104 geoms、2 cameras、eye 和 eye_internal。`verify_rcm_scene_v9.py` PASS（RCM 入口误差 `2.7892e-10 mm`、四 position sensor 误差 0、C=0 target/tip=`128/364 px`、C=8 tip=`163 px`），`verify_v9_front_registration.py` 6/6 PASS。测试记录已追加。
- 2026-10-31：默认入口 GUI 启动 PASS。执行更新后的 `python run_reference.py`，启动器 PID `33612`，GUI 子进程 PID `41964`，窗口标题 `MuJoCo : driver_integrated_legacy`、窗口句柄 `855944`，响应正常。默认渲染已人工查看，外观为当前照片前端和眼球的完整 v9 模型，不再显示旧基础模型。测试记录已追加。
- 2026-10-31：v9 数据集采集可行性核查。`collect_v9_dataset.py` 直接实例化 `RCMSceneV9`，每集显式重置后启用 `MotorBackend`，并固定 `perception_backend='strict_rgbd'`；采集不依赖已删除的根 `scene.xml`，也不依赖默认 viewer 入口。`collect_v9_dataset.py --help` PASS、`verify_motor_execution.py` 7/7 PASS、`verify_v9_motor_collection.py` 4/4 PASS。`verify_strict_rgbd.py` 10 项中 6 项功能/隔离测试通过，但四个代表姿态的目标中心真值误差断言 `<5 µm` 失败，实际为 `10.1428/10.1428/10.1428/10.0289 µm`。测试记录已追加。
- 2026-10-31：上一项严格 RGB-D 失败根因核查 PASS（非模型错位或控制回退）。四个姿态目标拟合 support=`123/123/123/115 px`、残差 `0.000706–0.000731 mm`、拟合质量约 `0.00212–0.00219 mm`，远低于配置上限 `15 µm`；目标在同一固定深度下产生约 `10.1 µm` 的固定量级中心偏移，主要体现为当前高分辨率像素/深度量化而非目标几何漂移。此前完整严格 RGB-D + MotorBackend 试采报告 `reports/20261031/strict_rgbd/pilot_result.json` 为 PASS，独立终点误差 `5.8668 µm`。因此可以采集数据，但当前 `verify_strict_rgbd.py` 的 `<5 µm` 辅助断言已落后于现行配置和质量门限；未修改该阈值，采集成功仍需以每集 `visual_servo_report.json`、独立终点误差和 `dataset_verification.json` 验收。测试记录已追加。
- 2026-10-31：新旧数据集兼容性核查 PASS（格式/语义）。当前 `verify_v9_dataset.py` 直接复验旧 `rcm-mujoco-reference-9265...\runs\pilot_3ep`，结果 PASS：3/3 集、2058 帧、2043 个实测电机滞后区间，原始 RGB 哈希、实际编码器状态、协议动作非下一状态、策略输入隔离和精确离线电机重放均通过。两端共用相同 `rcm_waypoint_dataset.py` 的 schema v2，目录包含 PNG/JSONL/NPZ、`policy_inputs`、`policy_inputs_no_image`、`action_labels`、`motor_steps.jsonl`、`frame_audit.jsonl`；视觉伺服、几何、MotorBackend 和模型同步源码哈希一致。测试记录已追加。
- 2026-10-31：新旧数据集非逐字节相同的边界核查。旧数据采集协议中只有 `rcm_scene_v9.py` 和 `v9_eye_scene.xml` 与当前项目不同，其余采集/控制/数据 schema 源哈希一致；当前目标与眼中心机器人坐标未变，但 v9 照片前端、世界变换和导出场景已更新。旧数据仍可被当前验证器重放，但不能宣称新采图像与旧集逐像素相同或未经声明直接混为同一模型版本。后续新数据应保留 `collection_protocol.json` 并在 manifest/训练集合中区分模型哈希；若要求与旧集完全同分布，需要先用当前 v9 采一集并比较图像、状态/动作分布和目标误差。测试记录已追加。
- 2026-10-31：新默认 v9 模型的 RCM 几何核查 PASS。零位 `rcm_site=[334.756480901806,-11.260751975282258,392.0363561984238] mm`；`needle_entry_rcm_distance_mm=2.789187070443984e-10`，入口部件为 junction，四个 position sensor 与对应 site 误差均为 0。补充 27 姿态扫描（A/B=-5/0/5°，C=0/4/8 mm）：最大 RCM 世界漂移 `0 mm`、针轴到 RCM 最大距离 `1.8815e-13 mm`、眼球椭球表面值最大误差 `4.6489e-11`、junction 到入口最大距离 `2.8104e-10 mm`。`verify_rcm_scene_v9.py` 和 `verify_migration_kinematics.py` PASS；后者最坏局部/世界坐标映射差 `0.000165653 mm`。RCM 位于巩膜入口且针轴通过固定点。测试记录已追加。
- 2026-10-31：新模型电机执行与驱动协议核查 PASS。`verify_motor_execution.py` 7/7 PASS，覆盖协议目标转换、50 ms 控制周期、2 ms 子步、一阶滞后、编码器反馈、非法命令不瞬移、时钟重置及 v9 单一 MotorExecution；`verify_v9_motor_collection.py` 4/4 PASS，覆盖协议动作标签、图像/编码器/时间/策略字段污染拒绝和严格感知声明。`mujoco_model/validate_driver_integration.py --output-dir runs/checks/new_model_driver_validation` PASS：84.0 s 虚拟闭环、8400 条反馈、190 条命令、双针具模型，最大闭环误差 `1.1734 µm`。运行时 `driver_loader.py` 加载 `mujoco_model/vendor/simulation_driver/ros_msg/src`；`source_manifest.json` 锁定的 29 个文件全部匹配提交 `c2a3ffc`，无缺失或哈希不符。测试记录已追加。
- 2026-10-31：可见 MuJoCo 电机驱动全过程启动核查 PASS。关闭旧静态 `run_reference.py` viewer 后，启动 `run_visual_servo.py --camera free --viewer-backend v9 --control-backend v9 --motion-backend motor --max-iterations 1600 --output-dir runs\gui_motor_full_20261031_171200`。启动器 PID `39924`，GUI Python 子进程 PID `36376`，窗口标题 `MuJoCo : driver_integrated_legacy`、窗口句柄 `921422`，响应正常且 CPU 推进；输出目录已生成首帧 `camera_0000.png` 和 `guidance_0000.png`。测试记录已追加。
- 2026-10-31：为 200/40/40 长采集计划增加可恢复执行。`collect_v9_dataset.py --resume` 现在读取已有 `manifest.json`，只从下一个未完成 episode 继续，并拒绝种子、总集数、随机目标模式、有效配置或模型/采集源码哈希发生变化的目录，避免把不同模型版本误拼成同一数据集。新增 `run_stage1_collection.py`，按 train 200/seed 20261031、val 40/seed 20261231、test 40/seed 20270131 顺序执行，保存 `stage1_status.json` 和分 split 日志，并在数据集结构验证失败或到达率低于 90% 时停止后续 split。测试记录已追加。
- 2026-10-31：阶段一运行器预启动检查 PASS。`collect_v9_dataset.py` 与 `run_stage1_collection.py` 的 `py_compile` PASS，两个 CLI 参数解析 PASS；在旧 `pilot_3ep` 上使用错误总集数测试 `--resume`，程序按预期以 `Cannot resume: collection protocol field 'requested_episodes' changed` 拒绝，未修改旧数据。磁盘可用约 `1456 GB`，按旧数据估算三阶段约 `52 GB`，空间充足。测试记录已追加。
- 2026-10-31：阶段一单目标采集已启动，尚未完成。`run_stage1_collection.py --output-root runs\paper_stage1 --min-success-rate 0.9 --max-iterations 1600` 的运行器 PID `36828`，状态文件 `runs/paper_stage1/stage1_status.json`，当前 train split 为 `running`，计划 200 集、seed `20261031`。episode 0 已推进到 50 次观测/2.5 s 仿真时间，墙钟 30.3 s，误差从 `19680.94 µm` 降至 `18221.03 µm`，实际 C=`1.383 mm`，MotorBackend 正常推进。该记录只证明采集已正确启动，不代表 200/40/40 已完成或达到率已通过。测试记录已追加。
- 2026-10-31：阶段一后台监控配置完成。新增 `monitor_stage1_collection.ps1`，记录进程状态、已完成 train episode、PASS/失败数量、帧数、磁盘空间和最新进度；若运行器/采集器不存在且阶段状态仍为 running，会自动重新启动可恢复运行器。手动监控测试通过：运行器 PID `36828` 及采集器进程链存活，状态 `running`，磁盘可用约 `1456.08 GB`。Windows 定时任务 `RCM Stage1 Collection Monitor` 已注册，每 30 分钟运行一次，首次计划时间 `2026-10-31 17:55:24`，状态 `Ready`。监控结果写入 `runs/paper_stage1/monitor_status.json` 和 `monitor.log`。测试记录已追加。
- 2026-10-31：阶段一采集后端核查 PASS。主机检测到 NVIDIA GeForce RTX 4050 Laptop GPU；当前原生采集进程 PID `34912` 已加载 `glfw3.dll`、`opengl32.dll` 和 NVIDIA `nvoglv64.dll`。Windows 未设置 `MUJOCO_GL`，使用 MuJoCo 3.3.7 默认 GLFW/OpenGL 后端，`RCMSceneV9` 的 RGB、depth、segmentation renderer 均为 1920×1440 GPU 渲染；MuJoCo 刚体运动学和 MotorBackend 子步仍由 CPU 执行。并行性核查为“不并行”：当前只有一个 `collect_v9_dataset.py` 采集进程，episode 在 `for index in range(...)` 中串行执行，`run_stage1_collection.py` 也用 `subprocess.run` 顺序执行 train/val/test，图像状态捕获和 PNG 写入在同一回调中同步完成。测试记录已追加。
- 2026-10-31：当前采集进程 GPU 占用核查 PASS。原生采集进程 PID `34912` 的 Windows GPU 计数器显示：专用显存 `1,390,178,304 B`（约 `1325.8 MiB / 1.295 GiB`）、共享显存 `31,903,744 B`（约 `30.4 MiB`）、总提交显存 `1,422,147,584 B`（约 `1356.3 MiB / 1.325 GiB`）。连续 5 次采样中该进程 3D 引擎利用率约 `0.55–1.19%`、Copy 引擎约 `0.22–0.41%`、引擎合计约 `0.77–1.59%`；同时间整卡利用率 `24–27%`、整卡显存占用约 `3720–3731 / 6141 MiB`，但整卡占用还包含桌面、浏览器、微信和其他图形程序。采集是间歇性渲染，单帧渲染完成后主要执行 CPU 几何、电机子步和 PNG 写入，所以进程平均 GPU 利用率低而显存上下文持续保留。测试记录已追加。
- 2026-11-01：阶段一采集状态检查。后台运行器及 MuJoCo 采集进程链持续存活，`stage1_status.json` 仍为 train/running，val/test 尚未开始。train manifest 已完成 `169/200` 集，PASS `161`、FAIL `8`，当前到达率 `95.27%`，累计 `115170` 帧，磁盘可用约 `1426.85 GB`；当前正在 episode `169`。8 个失败 episode 为 `22,45,81,88,96,104,107,127`，原因全部是 `LOW_QUALITY_VISUAL_GEOMETRY`，最终误差 `64.79–87.45 µm`，均未通过终端停稳；当前到达率仍高于 90% 门限，采集继续。每 30 分钟监控任务最近一次执行时间为 `2026-11-01 11:25:25`、结果码 0，下次约 `11:55:24`。测试记录已追加。
- 2026-11-01：按并行加速方案启动 val。启动前 train 已完成 177/200，剩余 23 集；GPU 已用 `3403/6141 MiB`、可用 `2518 MiB`，系统空闲内存 `6.81 GB`，资源足以再启动一个约 1.3 GiB 上下文。为避免原顺序 wrapper 在 train 完成后重复启动 val，停止仅负责 split 调度的 wrapper 原生进程 PID `43096`，确认 train 采集进程 PID `34912` 继续存活。随后启动 val 采集：启动器 PID `48156`、原生 MuJoCo 进程 PID `29020`，输出 `runs/paper_stage1/val`，episodes=`40`、seed=`20261231`、random targets。测试记录已追加。
- 2026-11-01：train/val 并行资源与进度核查 PASS。并行 35 秒后 train 和 val 均持续推进；两个 GPU 进程总提交显存分别为 `1356.3 MiB` 和 `1354.34 MiB`，整卡显存 `4729/6141 MiB`、剩余 `1192 MiB`，GPU 利用率 `12%`。val 已推进到 episode 0/150 帧，初始误差 `19789.91 µm`、当前约 `15363.55 µm`；train 已到 episode 178/150 帧。监控脚本已扩展到同时报告 train/val/test 完成数、PASS/失败和帧数，手动执行 PASS。test 仍未启动，将等待 train 与 val 都结束后串行运行。测试记录已追加。
- 2026-11-01：train 全量完成，manifest 为 `200/200` 集，PASS `192`、FAIL `8`，到达率 `96%`，累计 `136348` 帧。最终采集器复验首次 FAIL：8 个失败 episode 的 `motor_steps.jsonl` 均比 `frames-1` 多 1 条。根因是控制器在视觉质量失败后执行一次终止后 `motor.hold()` 安全保持区间；训练数据最后一行仍为 hold-current-state、没有对应后续观测，该额外区间只属于审计。已将 `verify_v9_dataset.py` 改为仅对 FAIL episode 接受恰好 1 条、起始时刻等于最后观测时刻的终止后安全保持区间，并单独用 MotorBackend 精确重放该区间；PASS episode 和训练动作语义不变。全量 train 复验进程仍在运行，尚不宣称 PASS。测试记录已追加。
- 2026-11-01：test 提前启动资源核查。val 当前完成 `34/40`，PASS `33`、FAIL `1`，到达率 `97.06%`，正在 episode 34，预计约 35–40 分钟完成。GPU 已用 `3522/6141 MiB`、剩余 `2399 MiB`，GPU 本身足够；但系统空闲 RAM 仅 `4.06 GB`，而单个采集进程约需 5 GB 私有内存，提前启动 test 会产生明显换页/OOM 风险。因此 test 不提前启动，等待 val 完成后再串行开始。测试记录已追加。
- 2026-11-01：test 启动状态核查：尚未开始，`runs/paper_stage1/test` 不存在。val 已完成 `40/40`，PASS `39`、FAIL `1`，到达率 `97.5%`；`val/dataset_verification.json` PASS，`total_frames=27290`，结构、图像哈希、编码器、动作对齐和精确电机重放均通过，无终止后 abort 区间。train 为 `200/200`，PASS `192`、FAIL `8`，到达率 `96%`，`total_frames=136348`。测试记录已追加。
- 2026-11-01：test 未启动的根因修复。monitor 在 val 完成后重启顺序 wrapper，wrapper 对已完成的 train 使用 `--resume`，但 `collect_v9_dataset.py` 在 train 启动后新增了 resume 实现，导致当前源代码哈希不同于采集时哈希而被拒绝，`dataset_verification.json` 因而不存在。现调整为：部分数据集仍严格要求源码哈希完全一致；已完成的 200 集 train 允许仅做复验而不追加样本，原始 `collection_source_sha256` 保留，当前验证源码哈希另写入报告，避免伪造采集来源。train 复验已以 PID `47004`/`48048` 启动，仍在运行。监控脚本已扩展为 train 和 val 验证均 PASS 后自动启动 test；资源方面 val 结束后 GPU 已降至 `2295/6141 MiB`、空闲 RAM 约 `5 GB`，适合单 test 进程运行。测试记录已追加。
- 2026-11-01：按用户要求跳过 train 全量重读，先启动 test。核查确认采集器 `collect_v9_dataset.py` 已包含 orphan trailing motor-step 清理：导出前只保留 `len(frame_audit)-1` 个电机区间，并同步 `motor.last_record/time_s`。运行 `fix_orphaned_motor_steps.py runs\paper_stage1\train`，结果为 `trimmed=8 already_ok=192 anomalies=0`；修复后 train 全部 200 集均满足 `motor_steps=frames-1`，episode 22 从 649 steps 修正为 648，仿真时间修正为 `32.4 s`。`collect_v9_dataset.py`、`verify_v9_dataset.py` 和修复脚本 `py_compile` PASS。测试记录已追加。
- 2026-11-01：test 采集已启动。为避免继续占用资源，停止了尚未完成的 train 全量只读复验 PID `44924`/`24608`；train 数据本身已修复，但其 `dataset_verification.json` 目前为空的未完成输出，后续仍需单独完成验证。test 使用已修复的采集器启动：启动器 PID `34540`、原生 MuJoCo 进程 PID `48224`，输出 `runs/paper_stage1/test`，计划 40 集、seed `20270131`、random targets、max iterations `1600`；episode 0 已进入伺服循环，首帧误差约 `19938.6 µm`。测试记录已追加。
- 2026-11-02：数据集采集模型与默认查看入口核查 PASS。`collect_v9_dataset.py` 实例化 `RCMSceneV9`，其底层由 `mujoco_model/model_sync.py` 的 `DriverModel` 读取 `mujoco_model/assets/scan_template.xml` 构建扫描机械臂，并在运行时注入眼球、retina、lens、lesion、针具和 `eye_internal` 相机，最终形成 v9 眼球注入模型；现有采集协议也记录 `scene=RCMSceneV9`。使用外部虚拟环境 `..\rcm_mujoco_control_limit\rcm_mujoco_reference\.venv\Scripts\python.exe` 在仓库根目录执行 `run_reference.py --no-viewer`，Python `3.11.0`、MuJoCo `3.3.7`，结果 `status=PASS`；必需几何、`overview`/`eye_internal` 相机、RCM 位于眼表及针尖位于眼内检查均通过。测试记录已追加。
- 2026-11-02：完成前复验 PASS。再次使用同一外部虚拟环境执行 `run_reference.py --no-viewer`，退出码 0，`status=PASS`、`model_backend=RCMSceneV9`、`scene=v9 eye-injected scanned arm`，`rcm_on_eye_surface=true`、`tip_inside_eye=true`。测试记录已追加。
- 2026-11-02：针尖耦合遮挡策略实现阶段语法检查 PASS。外部虚拟环境 Python 3.11.0 执行 `py_compile rcm_occlusion_policy.py rcm_tip_occluder.py rcm_scene_v9.py rcm_visual_servo.py`，退出码 0。测试记录已追加。
- 2026-11-02：遮挡策略实验运行器语法检查 PASS。外部虚拟环境 Python 3.11.0 执行 `py_compile run_occlusion_experiment.py`，退出码 0。运行器只生成新的评估 episode，不调用 `collect_v9_dataset.py`。测试记录已追加。
- 2026-11-02：S0a-S3 策略合成回归最终 PASS（6/6，0 failure，0 error）。首次运行 2 项失败：S0a 测试使用了宽松加速度上限且目标过小，所谓“瞬停”未形成超限，测试改为标准上限和更高速目标；S3 测试按当前 C 位置无条件持续注入遮挡，工具回撤恢复后会再次触发，测试改为单次工具耦合事件。生产控制器未因这两项测试假设修改。该项验证覆盖 S0a 瞬停、S0b 受限制动、S1 开环恢复、S2 有界目标记忆及 S3 回撤恢复，结果保存 `runs/checks/occlusion_strategy/strategy_verification.json`。测试记录已追加。
- 2026-11-02：真实 v9 渲染的针尖耦合遮挡检查最终 PASS（640×480）。基线目标为 12 px，遮挡触发后为 0 px；C 轴回撤 1.5 mm 后目标恢复为 12 px，目标三维几何重新可用，审计覆盖率由 1.0 降至 0，释放原因 `LEFT_OCCLUSION_CONFIGURATION`。首次以 0.35 mm 遮挡球和 0.15 mm 回撤测试时目标未恢复，扫描表明遮挡球过大、需约 2–3 mm 回撤；校准为 0.18 mm 遮挡球和 1.0 mm S3 默认回撤。第二次失败仅因测试在遮挡已释放、遮挡体已隐藏后仍用当前隐藏位置验证位姿耦合；改为验证激活帧中心严格等于针尖加当前工具局部偏移后 PASS。结果及 before/during/after 图像保存于 `runs/checks/tip_occluder/`。测试记录已追加。
- 2026-11-02：真实 strict RGB-D + MotorBackend S0a pilot smoke PASS（运行器正常完成，预期策略结果为 FAIL）。640×480、progress=0.9、target coverage=1.0、max exposure=2 s、max iterations=800；遮挡在 0.05 s 审计窗口内触发且覆盖率为 1.0，S0a 在首个不可见观测停止，控制器原因 `TARGET_NOT_VISIBLE`，最终真值误差 1962.678 µm，墙钟 46.6 s，运行器无异常。结果、配置与审计保存于 `runs/checks/occlusion_smoke_s0a/`。测试记录已追加。
- 2026-11-02：真实 strict RGB-D + MotorBackend S3 smoke FAIL（运行器正常完成，但策略未达到终点）。遮挡于 progress=0.90062、tip-target=1.9627 mm 时触发，覆盖率 1.0，0.1 s 后因 `LEFT_OCCLUSION_CONFIGURATION` 释放，说明耦合遮挡与回撤已生效；但控制器在 1200 次迭代后仍停在 `retracting`，最终误差 2844.98 µm，`recovery_success_count=0`。当前证据表明回撤状态未到达配置的 1 mm 距离，且恢复超时检查只覆盖 `reobserving`，未覆盖 braking/retracting；正在增加轨迹审计并继续定位。测试记录已追加。
- 2026-11-02：S3 回撤状态机根因修复后复测 FAIL（640×480 分辨率边界）。修复毫米量化容差、将回撤速度调整为 0.3 mm/s、恢复超时扩为 5 s 并覆盖 braking/retracting/reobserving 后，首轮遮挡成功恢复，最终迭代从 1200 次继续推进到 C≈18.55 mm，误差由 2844.98 降至 1147.35 µm。随后 C≈18.65 mm 附近反复出现 `INSUFFICIENT_VISUAL_DEPTH`，每次执行 1 mm 回撤后恢复，共 5 次。目标在 640×480 时仅约 2.13 px 半径，当前低分辨率目标深度估计不足以完成终点收敛；这不是针尖耦合遮挡失败。测试轨迹保存于 `runs/checks/occlusion_smoke_s3_fixed/`。测试记录已追加。
- 2026-11-02：遮挡策略回归扩展后 PASS（7/7，0 failure，0 error），新增覆盖 braking/retracting/reobserving 全阶段恢复超时。现有 `verify_strict_rgbd.py` 复跑为 6 项功能/隔离测试 PASS、4 个代表姿态的 `<5 µm` 目标中心真值断言 FAIL，实际误差为 10.1428/10.1428/10.1428/10.0289 µm。该四项失败与 2026-10-31 已记录问题相同，根因是高分辨率像素/深度量化与辅助验收阈值未同步，不是本次遮挡接入导致的回归；未修改该阈值。测试记录已追加。
- 2026-11-02：默认分辨率 1920×1440 的真实 strict RGB-D + MotorBackend S3 端到端验证 PASS。progress=0.9、target coverage=1.0、max exposure=2 s、C 回撤 1.0 mm；耦合遮挡在 tip-target≈2 mm 时触发且覆盖率 1.0，0.1 s 后因离开遮挡构型释放，策略完成恢复并以 11.880 µm 真值终点误差通过，任务停稳。使用 1000 次迭代上限内完成，墙钟 403.3 s；结果与轨迹保存于 `runs/checks/occlusion_smoke_s3_fullres/`。该结果也确认 640/960 下未收敛是目标小尺度深度估计边界，而不是 S3 状态机失败。测试记录已追加。
- 2026-11-02：实验入口与兼容性回归 PASS。`run_occlusion_experiment.py --dry-run` 正确生成 90 个 pilot case（5 策略 × 3 进度 × 3 曝光上限 × 2 覆盖率 × 1 seed），首尾 case 与参数一致；历史 `verify_occlusion_recovery.py` 10/10 PASS，未破坏旧 `static_memory_reobserve` 运动学恢复；默认 `run_reference.py --no-viewer` PASS，未启用遮挡夹具时 v9 几何和相机保持不变。测试记录已追加。
- 2026-11-02：最终策略/入口回归 PASS。为阻止低分辨率下无限重复回撤，新增 `occlusion_max_recovery_attempts=3`，并纳入配置、报告和 CLI；全部相关文件 `py_compile` PASS，S0a-S3 回归 7/7 PASS，最终 dry-run 仍为 90 个 case。测试记录已追加。
- 2026-11-02：最终代码并行回归 PASS。相关文件 `py_compile` PASS；S0a-S3 合成回归 7/7 PASS；真实 v9 渲染遮挡检查 6/6 PASS（目标 12→0→12 px，遮挡体跟随针尖局部位姿，回撤后释放且目标几何恢复）；旧 `verify_occlusion_recovery.py` 10/10 PASS；不带遮挡夹具的默认 `run_reference.py --no-viewer` PASS。测试记录已追加。
- 2026-11-02：最终版本默认分辨率 S3 端到端复验 PASS。1920×1440、strict RGB-D、MotorBackend、progress=0.9、coverage=1.0、max exposure=2 s、C 回撤 1.0 mm；结果 `POSITION_ERROR_AND_SETTLED`，真值终点误差 11.880 µm，实际遮挡持续 0.1 s、覆盖率 1.0，墙钟 415.2 s。最终结果保存于 `runs/checks/occlusion_smoke_s3_final/`。测试记录已追加。
- 2026-11-02：S0a/S0b/S1/S2 默认分辨率 1-case smoke PASS（4/4 case 均按预期完成，0 运行错误）。统一场景为 seed `20261102`、progress=0.9、coverage=1.0、max exposure=2 s、1920×1440、strict RGB-D + MotorBackend、max iterations=1600。四项均实际触发覆盖率 1.0 的遮挡：S0a 立即停止，最终误差 1962.545 µm；S0b 受限制动到零后失败，最终误差 1845.472 µm；S1 开环保持并恢复，10.122 µm PASS；S2 目标记忆恢复，11.884 µm PASS。实际遮挡持续时间分别为 S0a 0.05 s、其余 0.10 s。结果保存于 `runs/checks/occlusion_onecase_full/`。测试记录已追加。
- 2026-11-03：90-case 遮挡策略 pilot 运行完成，0 个运行错误，90/90 均实际触发遮挡。结果为 72 PASS、18 FAIL：S0a 9/18、S0b 9/18、S1/S2/S3 各 18/18 PASS；S0a/S0b 的失败全部来自 100% 完整遮挡，50% 部分遮挡均完成。输出 `summary.json`、`metrics.csv` 及 90 个逐 case 目录位于 `runs/occlusion_strategy_pilot/`。关键校准问题：实际遮挡时长只有 0.05 s（S0a 完整遮挡）和 0.10 s（其余），`max_active_s=0.5/1/2` 三档没有实际生效，说明针尖在约 0.1 s 内已离开当前遮挡构型。该 pilot 完成了触发、策略行为和指标可读性检查，但不能证明 1 s/2 s 长时遮挡场景有效，扩到多种子前需增大遮挡构型持续范围或调整耦合偏移/运动触发方式。测试记录已追加。
- 2026-11-03：Conda 环境 `rcm_mujoco` 首次创建 FAIL。Conda 4.9.2 解析到 Python 3.11.18 和 pip 26.2.2，但配置的 Tsinghua `conda-forge` 镜像返回 `pip-26.2.2-pyh8b19718_2.conda` HTTP 404，环境未完成安装。根因为镜像 repodata 与实际包文件不同步；按 systematic-debugging 先检查半成品目录，随后改用官方 `defaults` 通道重建。测试记录已追加。
- 2026-11-03：Conda 环境 `rcm_mujoco` 创建与验证 PASS。确认首次失败未留下半成品环境后，使用 `conda create -n rcm_mujoco --override-channels -c defaults python=3.11 pip -y` 创建成功，并安装 `requirements.txt` 全部依赖。最终环境位于 `C:\ProgramData\Miniconda3\envs\rcm_mujoco`，Python 3.11.18、MuJoCo 3.3.7、NumPy 2.4.6、Pillow 13.0.0、SciPy 1.17.1、Matplotlib 3.11.2、trimesh 5.2.0；全部导入 PASS，`run_reference.py --no-viewer` 返回 `status=PASS`，RCM 位于眼表且针尖位于眼内。测试记录已追加。
- 2026-11-03：PPT Master round-trip 导入 PASS，源 PPTX 2 页、画布 1280×720，工作区为 `projects/mujoco_dataset_20261103_roundtrip/`。随后 `source_to_md.py` 源稿文本读取 FAIL，退出码 1，直接原因为 Conda 环境缺少 `python-pptx`；按 systematic-debugging 先安装缺失依赖，再重新读取。测试记录已追加。
- 2026-11-03：PPT 源稿读取复验 PASS。安装 `python-pptx 1.0.2` 后，`source_to_md.py` 成功读取 2 页源稿并生成 `validation/source_readback.md`；PowerPoint COM 将两页成功渲染到 `validation/source_render/`，用于核对原稿视觉风格。阶段仅导入和读取，尚未编辑页面或导出 PPTX。测试记录已追加。
- 2026-11-03：PPT 第 3 页制作与导出 PASS。新建 `slide_03.svg` 复制第 2 页版式，前两页保持原生 passthrough；第 3 页加入 280 个 episode、190,955 帧、200/40/40 划分、20 Hz、96.4% PASS、成功终点误差中位数 11.37 μm/P95 13.42 μm 和真实 eye_internal 采集帧。首次质量门因指标“19.1 万帧”超出原数字框 12.2% FAIL，缩写为“19.1万”后 3/3 页面容量与结构检查 PASS。`svg_to_pptx.py --roundtrip --primary-language zh-CN` 导出 3 页，postflight `quality_gate=passed`、`rebuilt=1`、前两页 `cloned_passthrough=2`；页 3 独立备注已覆盖复制页继承的旧第 2 页备注。最终 `pptx_delivery_check.py` 为 passed、0 errors、0 advisories，读回确认 3 页内容和备注正确，PowerPoint 渲染成功且第 1–2 页图像哈希与源稿完全一致。最终文件复制为 `MuJoCo第一版到当前_3页可编辑.pptx`，与导出文件 SHA-256 一致。测试记录已追加。
- 2026-11-03：按 `MuJoCo建模优化_大字修订版.pptx` 重做数据集页 PASS。新 round-trip 工作区保留原 2 页，复制第 2 页骨架创建第 3 页；左侧改为 `train/val/test` 数据表（episodes、frames、PASS），右侧保留大字时间/内容/质量三段，底部汇总 280 episodes、190,955 帧、96.4% PASS 与中位误差 11.37 μm。首次质量门因 5 处长文本横向溢出 FAIL，压缩文案后 3/3 PASS；首次最终回读因拒绝覆盖旧 `final_readback.md` FAIL，改用 `final_readback_v2.md` 后成功。最终导出 `--primary-language zh-CN`：postflight `quality_gate=passed`，`cloned_passthrough=2`、`rebuilt=1`；delivery check passed、0 errors、0 advisories；PowerPoint 渲染通过，第 1–2 页与源稿逐像素一致，第 3 页无溢出。最终文件 `MuJoCo建模优化_大字修订版_含数据集页.pptx` 与导出文件 SHA-256 一致。测试记录已追加。
- 2026-11-03：数据集页补充“数据格式/后续规划”修订 PASS。按 Edit Native PPTX 修订轮先重新导入已交付 3 页文件，仅编辑第 3 页：`02 样本内容` 改为 `02 数据格式`，加入 RGB PNG、depth/state NPZ、audit JSONL、policy/action/motor；底部两行改为遮挡策略与多策略对照规划、统一格式和统计流程。首次质量门因数据格式首行超宽 25.4% FAIL，压缩后 3/3 PASS。最终导出 postflight `quality_gate=passed`、`passthrough=2`、`rebuilt=1`；delivery check passed、0 errors、0 advisories；内容回读确认数据格式和规划字段齐全；PowerPoint 渲染通过，第 1–2 页与此前交付版本逐像素一致。最终文件 `MuJoCo建模优化_大字修订版_含数据集页_补充规划.pptx` 与导出文件 SHA-256 一致。测试记录已追加。
- 2026-11-03：训练结果状态核查 FAIL（数据无效，训练继续运行）。当前 PID `34344` 仍在训练，`best_model.pth` 最新记录 `epoch=8`、`train_loss=0.062150724`；但 `data/lerobot_rcm/data/dataset.npz` 的 27,317 条样本中 `state` 三个维度分别恒定且 `action` 全为 0，loss 下降不能代表学到了闭环策略。根因为 `convert_to_lerobot.py` 仍读取当前采集格式中不存在的 `actual_encoders`、`requested_task` 和 `control_report.json`，而源文件使用 `robot_ABCE`、`state_A_deg_B_deg_C_mm`、`requested_task_rad_rad_m`/`commanded_task_rad_rad_m` 和 `visual_servo_report.json`；当前转换数据还是 40 集 test split。既有 `eval_results.log` 的单次评估在 `set_task()` 处因 C 编码器 `10,985,616` 超出 `[0, 2,952,362]` 失败，`eval_policy.py` 另存在 C 轴 m/mm 单位不一致。未修改代码、未复跑训练或评估；本轮仅作诊断记录。测试记录已追加。
- 2026-11-03：数据转换增量验证 FAIL（测试期望值过严，非转换错误）。`py_compile convert_to_lerobot.py` PASS；首集 689 帧已正确读取非恒定 `state` 和非零 `command`，但测试把 B 轴期望值硬编码为 `-0.015`，源记录按 `commanded_task_rad_rad_m` 精确换算为 `-0.0150009221`，超出 `1e-9` 容差。状态和动作字段映射已工作，修改断言的期望值计算方式后复测。测试记录已追加。
- 2026-11-03：数据转换首集增量验证 PASS。`episode_000000` 提取 689 帧，首帧状态 `[0,0,0]` 与源文件一致，首帧动作严格等于 `commanded_task_rad_rad_m × [180/pi,180/pi,1000]`；该集 state 的 A/B/C 范围分别为 `0–3.7908°`、`-3.23545–0°`、`0–19.91846 mm`，增量动作最大绝对值 `0.2898`，非恒定/非全零数据校验通过。测试记录已追加。
- 2026-11-03：数据转换修复代码语法与全量重建 PASS。`py_compile convert_to_lerobot.py` 退出码 0；使用 `--reuse-existing-images` 重建 `data/lerobot_rcm`，40/40 episodes、27,317 帧全部转换成功。输出 state 范围为 A `[-13.266,11.5236]°`、B `[-12.018131,9.229658]°`、C `[0,19.9884264] mm`；增量动作范围为 A `[-0.2898,0.2898]°`、B `[-0.144343582,0.144339322]°`、C `[0,0.04271207] mm`。内置校验会拒绝常量 state 和全零 action。测试记录已追加。
- 2026-11-03：修复数据独立回读 FAIL（测试常量错误，非数据错误）。回读脚本断言 action 非零元素必须恰为 `52,603`，实际为 `67,043`；该精确值不是数据契约。此前 state/action 形状、有限性及各轴非恒定检查均已通过，训练输入与模型前向 smoke 也 PASS（26,997 个有效样本、loss `0.5455296`）。移除无依据的精确非零计数，改为断言非零后复跑。测试记录已追加。
- 2026-11-03：修复后数据全量独立回读 PASS。逐集重新解析 40 个源 episode，与 `dataset.npz` 的 27,317 条 state 和 action 全量逐元素一致（atol `1e-6`）；episode/frame 索引连续，40 集均有样本，39/40 源 episode 标记成功。state 各轴跨度 `[24.7896°, 21.2478°, 19.9843 mm]`，增量动作各轴跨度 `[0.5796°, 0.2887°, 0.0427121 mm]`，非零动作元素 `67,043`。27,317 个图像路径均存在且无重复，抽取每集首末帧确认图像为 640×480 RGB。测试记录已追加。
- 2026-11-03：训练前置条件审计 PASS（仅技术可启动，最终实验仍未就绪）。当前无 diffusion-policy 训练进程；CUDA 可用，目标设备为 RTX 4050 Laptop GPU、CUDA 12.1；修复后的 `data/lerobot_rcm` 可加载，40 集/27,317 帧，state 与 action 均非恒定；训练输入和模型前向 smoke 已通过。现有完整划分为 train 200 集/136,348 帧/PASS 192、val 40 集/27,290 帧/PASS 39、test 40 集/27,317 帧/PASS 39，且新转换器已成功解析 train 首集。阻塞最终训练的原因：当前转换数据集来自 test split；训练脚本没有验证集评估且仅按训练 loss 保存 best；`eval_policy.py` 的 C 轴 m/mm 单位错误仍在；旧 `outputs/diffusion_policy/best_model.pth` 来自全零动作数据，不能复用。测试记录已追加。
- 2026-11-03：并行图像转换 smoke FAIL（测试夹具未关闭文件，非转换错误）。`py_compile convert_to_lerobot.py` PASS；64 帧并行转换完成，速度约 `184.65 帧/s`，state/action 非恒定且非零，图像均为 640×480 RGB；测试随后在 Windows 临时目录清理时报 `dataset.npz` 被占用，原因是 `np.load` 返回的 NpzFile 未关闭。关闭文件句柄后复测。测试记录已追加。
- 2026-11-03：并行图像转换 smoke 复测 PASS。关闭 `np.load` 文件句柄后，8 线程转换 64 帧成功，吞吐约 `196.69 帧/s`；数值表形状为 `(64,3)`，state/action 非恒定且非零，64 张输出图像均为 640×480 RGB。测试记录已追加。
- 2026-11-03：train split 全量转换与独立验证 PASS。使用 `--reuse-existing-images --image-workers 8` 将 `runs/paper_stage1/train` 转换为 `data/lerobot_rcm_train`：200 集、136,348 帧、136,348 张图像，总计 9.08 GB；源报告 PASS 192/200。逐条独立重建并与 `dataset.npz` 全量逐元素比较，state/action/episode/frame 完全一致（atol `1e-6`）；state 各轴跨度 `[26.5986°, 24.14315°, 19.99367 mm]`，增量动作各轴跨度 `[0.5796°, 0.28869°, 0.0427121 mm]`，非零动作元素 `342,038`。全部图像路径存在且唯一，每集首末帧抽检为 640×480 RGB。训练入口加载 200 集得到 134,748 个有效窗口，模型前向 loss 有限（`0.949953854`）。测试记录已追加。
- 2026-11-03：val split 全量转换与独立验证 PASS。使用 8 线程将 `runs/paper_stage1/val` 转换为 `data/lerobot_rcm_val`：40 集、27,290 帧、27,290 张图像，总计 1.81 GB；源报告 PASS 39/40。逐条独立重建并与 `dataset.npz` 全量逐元素比较，state/action/episode/frame 完全一致（atol `1e-6`）；state 各轴跨度 `[25.7616°, 23.43631°, 19.98074 mm]`，增量动作各轴跨度 `[0.5796°, 0.28868°, 0.0427121 mm]`，非零动作元素 `68,968`。全部图像路径存在且唯一，每集首末帧抽检为 640×480 RGB。训练入口加载 40 集得到 26,970 个有效窗口，模型前向 loss 有限（`0.484677464`）。测试记录已追加。
- 2026-11-03：训练启动状态核查 PASS（非卡死）。train 数据训练进程 PID `24824` 自 19:39:41 运行，附 4 个 DataLoader worker；进程响应正常、工作集约 1.35 GB，GPU 已分配约 1.9 GB，输出目录 `outputs/diffusion_policy_train` 已创建。此时尚无 checkpoint 是因为首个 epoch/首次 loss 改善尚未完成，且本次启动未将 stdout 重定向到日志文件；现有 `train_gpu_bs16.log` 属于 18:08 的旧进程，不是本次训练日志。测试记录已追加。
- 2026-11-03：正式训练前置改造阶段性 PASS。已停止旧训练完整进程链（PID `41780`、`24824` 及 4 个 worker）；`train_diffusion_policy.py` 增加 train/val 数据集、确定性 val loss、按 val 选择 best、训练历史和配置检查点；`eval_policy.py` 改为正确 deg/deg/mm 单位、读取检查点架构、使用 MotorBackend 执行、RGB 预处理与训练一致，并将真值仅用于成功评价。`py_compile train_diffusion_policy.py eval_policy.py` 通过，退出码 0。后续继续做 best-on-val 和短程闭环验证。测试记录已追加。
- 2026-11-03：best-on-val 训练端到端 smoke PASS。临时 train 12 个有效窗口、val 8 个有效窗口，实际训练 2 epochs：val loss 从 `0.69704856` 变为 `0.69827461`，`best_model.pth` 保持 epoch 0 的较低 val loss 并保存完整 config；`training_history.json` 含两轮指标。加载 best checkpoint 后，同一 val 集使用相同 seed 两次评估得到完全相同的 loss，确定性检查 PASS。测试目录已自动清理，测试记录已追加。
- 2026-11-03：闭环评估 smoke FAIL（结果序列化类型错误）。短程闭环已实际完成 4 个 MotorBackend 控制步，C 轴 m/mm 单位转换与执行路径正常，最终误差 `19.745 mm`；但 `evaluation_results.json` 写入失败，因轨迹 `actions` 保存了 NumPy ndarray，且部分汇总值仍是 NumPy 标量，标准 `json.dump` 无法序列化。将动作和统计值显式转换为 Python `list`/`float` 后复测。测试记录已追加。
- 2026-11-03：可信闭环评估短程 smoke PASS。随机初始化 2-step 模型执行 1 个 episode、4 个 MotorBackend 控制步；任务坐标 unit round-trip 精度达到 `1e-15`，C 轴轨迹保持 `0–50 mm` 合理范围，而非误作米。结果 JSON 成功写出，包含 `execution_backend=MotorBackend`、`ground_truth_used_for_control=false`、checkpoint 的 train/val loss、逐步 state/action/error/velocity、motor summary；`direct_pose_fallback_used=false`。测试模型未训练，故 4 步后误差 `19.745 mm` 不作为策略性能结论。测试记录已追加。
- 2026-11-03：随机目标闭环 smoke PASS。评估器通过 `make_random_fundus_targets` 生成 1 个 posterior fundus target，并显式转换为 world 坐标；随机初始化策略执行 2 个 MotorBackend 控制步，结果 JSON 正常写出。该测试覆盖了旧代码调用不存在 `scene.sample_random_fundus_target()` 的问题及 target robot/world 坐标系混用问题。测试模型未训练，误差不作为策略性能结论。测试记录已追加。
- 2026-11-03：真实 train/val 数据训练链路 smoke PASS。以 `n_obs_steps=2, n_action_steps=4` 加载 200 集 train（135,548 个有效窗口）和 40 集 val（27,130 个有效窗口），各取 4 个真实样本执行 1 个训练优化批次及两次验证；train loss `1.22299069`、val loss `0.96005195` 均为有限值，同一 seed 两次 val loss 完全一致。测试记录已追加。
- 2026-11-03：最优验证选模与闭环评估最终静态回归 PASS。`py_compile train_diffusion_policy.py eval_policy.py` 退出码 0；训练和评估 CLI 均能解析新增参数。静态检查确认旧的 `best_loss` 训练损失选模、`sample_random_fundus_target()`、`get_tip_position()` 和 C 轴米值直传调用均已移除；新实现明确包含 `--val-data-dir`、`val_loss`、`best_val_loss`、`MotorBackend` 和 `ground_truth_used_for_control=false`。README/QUICKSTART/USE_EXISTING_DATA 已更新为 train/val 分离训练命令。测试记录已追加。
- 2026-11-03：训练控制台/文件双写日志 smoke FAIL。`py_compile` PASS，但 1-epoch 临时训练子进程以退出码 120 结束。首要根因假设为 `atexit` 先关闭日志文件，而解释器随后仍通过 `TeeStream` 刷新 stdout/stderr；按 systematic-debugging 改为退出时先恢复原始流，再 flush/close 并保留测试 stderr 复测。测试记录已追加。
- 2026-11-03：日志关闭顺序修复后 smoke 仍 FAIL。1-epoch 临时训练子进程返回码 0，但测试在检查 `<output-dir>/training.log` 是否存在时失败；尚无证据判断是脚本执行了旧入口、日志路径偏差还是测试目录问题。下一步保留测试输出目录并打印实际文件树、stdout/stderr 后复测。测试记录已追加。
- 2026-11-03：日志双写逐项断言复测 FAIL（部分文本契约未满足）。日志文件非空、console/file 均含 train/val 指标、best checkpoint 存在；但本次捕获的 console 未匹配“训练日志:”头，文件未匹配“训练完成! 最佳验证损失:”。需检查原始 stdout/log 文本与退出刷新行为后修正测试或实现。测试记录已追加。
- 2026-11-03：训练控制台/文件双写日志复验 PASS。保留并检查原始 stdout/stderr 与日志内容后，1-epoch 子进程返回码 0；stdout 和 `training.log` 均完整包含训练日志路径、数据集加载、模型规模、tqdm 批进度、`train_loss`/`val_loss`/`best_val_loss`、训练完成与保存路径。日志文件非空，`best_model.pth` 与 `training_history.json` 均生成。此前的部分文本断言失败未在新运行复现。测试记录已追加。
- 2026-11-03：训练日志实时刷新修正 PASS。`TeeStream` 在遇到换行或 tqdm 回车刷新时立即 flush 控制台和日志文件；专项测试确认普通片段不会过早刷新，换行后控制台/文件内容一致，换行与回车均触发 flush。`py_compile` PASS。这样通过管道或后台启动训练时，`training.log` 和可见输出可实时更新。测试记录已追加。
- 2026-11-03：正式训练重启与实时日志确认 PASS。DataLoader 改为 `num_workers=0` 规避受管环境的 multiprocessing `WinError 5`；训练 PID `31836` 正常运行，加载 train 134,748 个有效窗口和 val 26,970 个有效窗口，模型 12.58M 参数，GPU 可用。`outputs/diffusion_policy_formal/training.log` 已实时写入 epoch 0 的批进度与 loss；首次失败日志保留为 `training_failed_20261103_195819.log`。测试记录已追加。
- 2026-11-03：训练结束状态核查 FAIL（实际仍在验证阶段）。PID `31836` 仍存活；epoch 0 的 8,422 个训练 batch 已于 21:20:08 完成，但日志没有 val loss，`training_history.json` 和 `best_model.pth` 均不存在。10 秒采样 CPU 增长 6.125 秒、GPU 3D 引擎约 18.7%，说明进程仍在计算验证集而非结束。当前不能开始最终评估或宣称训练完成。测试记录已追加。
- 2026-11-03：训练完成状态复核（实际仍在训练）。epoch 0 验证已完成：train loss `0.240263979`、val loss `0.099144633`，并保存 `training_history.json` 与 `best_model.pth`；PID `31836` 随后进入 epoch 1，21:52 已到 1908/8422 batch。用户所称“训练结束”实际是 epoch 0 训练进度条结束，当前训练目标仍为 1000 epochs。测试记录已追加。
- 2026-11-04：训练进度复核。截至 07:24，PID `31836` 仍正常运行并进入 epoch 7（约 11%）；已完成 epoch 0–6，val loss 从 `0.099145` 单调降至 `0.037868`，当前 best 为 epoch 6（train loss `0.038244`、val loss `0.037868`）。best checkpoint 于 07:15 保存成功。训练约每 epoch 1.6 小时，1000 epochs 全量运行不现实；当前验证损失仍在改善，暂不建议立即停止，建议在 epoch 7 完成后再决定短期里程碑或 early stopping。测试记录已追加。
- 2026-11-04：10-epoch 硬上限与 watchdog 测试 FAIL（watchdog 语法错误）。训练脚本 `effective_epoch_count(1000)==10`，`10→10`、`3→3`、`0` 被拒绝，`py_compile` PASS。watchdog 在解析 `"Epoch $MaxEpochs:"` 时把冒号误认为变量名的一部分而失败，测试 dummy 进程未被停止。修正为 `"Epoch ${MaxEpochs}:"` 并先清理测试进程后复测。测试记录已追加。
- 2026-11-04：10-epoch 硬上限与 watchdog 复测 PASS。修复变量插值后，watchdog 用临时日志和 dummy 进程验证：检测到 `Epoch 10:` 后成功停止目标进程，状态文件为 `stopped_at_limit`。实际训练已在 PID `31836` 上挂载 watchdog PID `3224`，`training_limit_watchdog.json` 状态为 `watching`；当前训练处于 epoch 7、约 18%，将在 epoch 9 完整结束且 `Epoch 10` 开始时自动阻断。后续新训练由脚本内 `HARD_MAX_EPOCHS=10` 再作一道阻断。测试记录已追加。
- 2026-11-04：训练卡死状态核查 PASS（未卡死）。截至 08:36，训练 PID `31836` 与 watchdog PID `3224` 均存活；epoch 7 的 8,422 个训练 batch 已于 08:32 完成，当前进入静默验证阶段。15 秒采样进程 CPU 增长 `10.44 s`（约 0.70 核），GPU 3D 引擎约 `17.5%`、整卡利用率 `84%`，说明仍在计算；日志暂不变化是因为验证阶段此前未添加 tqdm。watchdog 状态仍为 `watching`。测试记录已追加。
- 2026-11-04：当前评估状态核查（仍在运行）。训练已停止并保留 epoch 12 best；评估 PID `2412` 自 19:22:48 运行，CPU 累计约 284 s，进程响应正常。GPU 3D 引擎约 `18%`、整卡约 `35%`、显存 `4321/6141 MiB`，与 strict RGB-D/OCT 感知和逐步 MotorBackend 闭环一致。`evaluation_results.json` 尚未生成，因此当前没有最终成功率或误差可报告。测试记录已追加。
- 2026-11-04：归一化扩散 v3 闭环评估完成 FAIL。使用 epoch 12 best（normalized action MAE `0.66820`）执行 1 个随机目标、1600 步评估：初始误差 `19.9386 mm`，第 662 步最小 `1.2483 mm`，最终误差 `13.6546 mm`，成功率 `0/1`。最终 state A/B/C=`[-6.03°,3.9525°,32.2573 mm]`，C 轴已越过目标并撞到编码器上限；共 `527` 步因 C 编码器接近 `2952421 > 2952362` 被安全回退并最终零动作保持。最终速度为零、MotorBackend settled，但误差远大于 20 µm。相较旧 OCT 模型最终误差 `3.6905 mm`，本版当前训练阶段更差；最小误差相近，主要问题是接近目标后继续向 C 方向过冲。测试记录已追加。
- 2026-11-04：训练性能瓶颈核查。当前训练仍使用 `num_workers=0`，epoch 7 训练阶段约为 `1.5–1.9 batch/s`；在 15 秒采样中 CPU 增长 `10.92 s`，GPU 显存约 `4.0/6.1 GB`，验证阶段日志静默。单卡 RTX 4050 不同时启动多个训练进程；可优先用 DataLoader worker 并行输入、AMP 混合精度和更大但安全的 batch，后续可增加预缩放图像缓存。当前已接近 10-epoch 上限，重启会丢失进度，应继续等待阻断完成。测试记录已追加。
- 2026-11-04：训练加速代码与 CUDA AMP smoke PASS。`py_compile train_diffusion_policy.py eval_policy.py` 通过；新增 `--amp/--no-amp`、`--amp-dtype fp16|bf16`、`--prefetch-factor`、`--seed`，CUDA 默认启用 FP16 AMP，DataLoader 在 workers>0 时启用 persistent workers、prefetch、pin memory 和 non-blocking transfer，并在 Dataset 中显式关闭图像文件句柄。极小 CUDA 模型完整执行 AMP 训练、GradScaler、两次确定性验证，loss 均为有限值且两次验证完全一致。当前运行中的旧进程不受代码修改影响；下一次训练可使用 `--num-workers 4`。测试记录已追加。
- 2026-11-04：优化训练首次启动 FAIL（受管环境阻止进程 worker）。旧训练和 watchdog 已停止并验证退出。新训练使用 AMP 与 `--num-workers 4`，但 DataLoader 创建 multiprocessing Queue 时再次退出：`PermissionError: [WinError 5] 拒绝访问`；日志正常记录了完整 traceback，模型尚未训练。下一步实现进程内线程池批加载，以避开进程队列限制并保持并行图像读取。测试记录已追加。
- 2026-11-04：线程池并行加载与 CUDA AMP 真实数据 smoke PASS。新增 `ThreadBatchLoader`，在 `num_workers=0` 时使用可配置 `--thread-workers` 并行读取/缩放样本，并保留 DataLoader 多进程路径供普通终端使用。真实 train/val 各 8 个样本、4 线程、batch 4、CUDA FP16 AMP 完成训练和两次确定性验证；train loss `1.00598216`、val loss `1.00277117`，两次 val 完全一致。`py_compile` 通过，硬上限仍为 10 epochs。测试记录已追加。
- 2026-11-04：优化训练重启 PASS。旧训练 PID `31836` 和 watchdog PID `3224` 已停止；旧 best/history 保留在 `outputs/diffusion_policy_formal`。新训练 PID `10548` 使用 `epochs=10, batch=16, num_workers=0, thread_workers=8, AMP=fp16`，输出到 `outputs/diffusion_policy_optimized`。启动后稳定速度约 `8–9 batch/s`，相对旧训练约 `1.7 batch/s` 提升约 5 倍；GPU 3D 引擎约 55%，显存约 3.1/6.1 GB，训练日志实时更新。测试记录已追加。
- 2026-11-03：正式训练首次启动 FAIL（日志功能正常）。`outputs/diffusion_policy_formal/training.log` 成功创建并写入；训练进程退出于 DataLoader 初始化，`multiprocessing.connection.Pipe()` 创建 Windows 队列时返回 `PermissionError: [WinError 5] 拒绝访问`。根因是当前受管执行环境不允许 DataLoader worker 的 multiprocessing 管道，不是数据、模型或日志错误。保持日志到同一输出目录，使用 `--num-workers 0` 重启。测试记录已追加。
- 2026-11-04：停止旧训练并按 10-epoch 上限提速重启 PASS。旧训练 PID `31836` 与 watchdog PID `3224` 已停止；新训练 PID `10548` 启动于 08:44:13，输出 `outputs/diffusion_policy_optimized`。实际配置为 `epochs=10, batch_size=16, num_workers=0, thread_workers=8, AMP=fp16`；日志确认数据集已加载、CUDA 可用、线程并行启用。稳定速度约 `8.9 batch/s`，旧训练约 `1.7 batch/s`，约 5 倍提升；GPU 3D 引擎约 55%、显存约 3.1/6.1 GB。脚本硬上限保证只运行 epoch 0–9，不会开始第 11 个 epoch。测试记录已追加。
- 2026-11-04：checkpoint resume smoke FAIL（测试目录权限）。恢复测试尚未进入训练代码；当前受管环境拒绝在系统临时目录 `C:\Users\Lenovo\AppData\Local\Temp\tmp62fm6vjn\train` 创建子目录，返回 `PermissionError: [WinError 5]`。改为在工作区内的专用测试目录生成临时数据后复测。测试记录已追加。
- 2026-11-04：OCT 闭环评估 GPU 使用核查 PASS。当前评估 PID `38824` 存活，模型通过 `--device cuda` 运行；进程 GPU 3D 引擎利用率约 `6.05%`，整卡约 `13%`、显存 `3596/6141 MiB`。10 秒内 CPU 增长 `10.08 s`，说明当前主要瓶颈是 strict RGB-D/OCT 感知、MuJoCo 渲染与几何处理，而非策略网络未放到 GPU。未中断评估。测试记录已追加。
- 2026-11-04：OCT 闭环评估状态核查（未结束）。截至 14:59，评估 PID `38824` 已运行约 12.5 分钟；CPU 累计约 760 s，GPU 3D 引擎约 6.5%、整卡约 12%、显存 `3661/6141 MiB`。`evaluation_results.json` 尚未生成，说明单个 1600-step episode 仍在逐步执行 strict RGB-D/OCT 感知；当前不是进程卡死，但尚无成功率或误差可报告。测试记录已追加。
- 2026-11-04：OCT 模型闭环评估完成（FAIL，未达到任务要求）。1 个随机目标 episode 跑满 1600 步，`success=false`，最终误差 `3.69050 mm`，最小误差 `1.24915 mm`（第 687 步），最小误差仍高于 20 µm 门限。最终 state A/B/C=`[-2.5344°,3.7212°,20.1791 mm]`，最终速度 A/B/C=`[0.684,-0.43385,0.07145] deg/deg/mm/s`，MotorBackend 仍在运动、未 settled；最后 100 步误差均值/标准差 `3.4326±0.1370 mm`。评估确认 `oct_navigation_used=true`、`ground_truth_used_for_control=false`。相较不使用 OCT 的旧模型最终误差 `6.1111 mm`，下降约 39.6%，但绝对精度仍不可用。测试记录已追加。
- 2026-11-04：OCT 模型 val 开环预测对比。同 256 个 val 样本上，新模型首动作 MAE A/B/C=`0.05575°/0.02864°/0.01091 mm`，相关系数 `0.559/0.640/0.434`。相比旧模型 C 轴 MAE `0.01301 mm`、corr `0.391`，C 轴略有改善；A/B 略退化。C 轴 pred std `0.01639 mm` 仍显著高于 truth std `0.00667 mm`，说明模型仍未学到足够精确且稳定的停止控制。测试记录已追加。
- 2026-11-04：checkpoint resume smoke 部分 PASS、测试断言 FAIL。恢复路径实际工作：训练脚本从 epoch 0 checkpoint 恢复，日志写明下一轮为 epoch 1（第 2 个 epoch），并完成 epoch 1；但复测脚本的一项文本断言失败，当前输出和 history 已保留在 `runs/checks/resume_smoke_20261104_0845`，下一步直接核验实际 history/stdout 后修正测试。测试记录已追加。
- 2026-11-04：checkpoint 恢复与第 8 个 epoch 续训 PASS。持久化产物核验确认 history 保留 epoch 0–1，恢复后完成的 epoch 1 checkpoint 的 val loss 与 history 最后一项一致。随后停止从 epoch 0 重跑的 PID `10548`，从 `outputs/diffusion_policy_formal/best_model.pth` 恢复并启动 PID `41944`；日志明确显示“已完成到 epoch 6，下一轮为 epoch 7（第 8 个 epoch）”。新训练输出到 `outputs/diffusion_policy_resumed`，使用 FP16 AMP、8 个线程加载，速度约 `9.5–10.2 batch/s`，GPU 3D 引擎约 63%、显存约 3.1/6.1 GB，将运行 epoch 7–9 后因 10-epoch 硬上限结束。测试记录已追加。
- 2026-11-04：10-epoch 训练完成核查 PASS。训练 PID `41944` 已退出；`training_history.json` 含 10 个完整 epoch，日志以 `训练完成! 最佳验证损失: 0.032873` 结束。最佳结果出现在 epoch 9：train loss `0.0319703`、val loss `0.0328734`；`best_model.pth` 的 checkpoint 也记录 epoch 9 和相同 loss。未生成 epoch 10 或额外轮次，硬阻断生效。测试记录已追加。
- 2026-11-04：评估启动 smoke FAIL（PowerShell 启动器环境错误）。`eval_policy.py` 与训练脚本 `py_compile` PASS；尝试以 `Start-Process -RedirectStandardOutput` 启动评估时，PowerShell 在创建子进程前报重复环境键 `PATH/Path`，Python 评估尚未开始。改为直接启动评估进程并捕获 stdout/stderr。测试记录已追加。
- 2026-11-04：评估结果与差距根因核查 FAIL（策略闭环失败，非单位/坐标错误）。1 个随机目标 episode 跑满 1600 步：初始误差 `19.939 mm`，第 737 步最小 `1.042 mm`，随后越过目标并稳定在约 `6.1 mm`，最后 100 步均值/标准差 `5.813 ± 0.199 mm`，MotorBackend 仍在运动且未 settled。最终 state 为 A `6.7302°`、B `-1.5984°`、C `25.8057 mm`；target 位于训练分布内（距最近训练目标 `0.282 mm`）。对 256 个 val 样本做开环首动作预测：MAE A/B/C=`0.04661°/0.02183°/0.01301 mm`，相关系数 `0.587/0.688/0.391`；C 轴真值标准差 `0.00667 mm`，预测标准差 `0.01799 mm`，说明 C 轴未学到有效闭环制动且存在过冲。结论是 10-epoch diffusion policy 的闭环漂移/动作精度问题，不是 C 轴单位或目标坐标错误。测试记录已追加。
- 2026-11-04：OCT 数据可用性核查与转换首集测试。逐帧 `oct_observations.jsonl` 含 679 条 OCT 元数据和 `navigation_depth.delta_xyz_m`，但 `raw_oct_available=false`，完整原始深度体仅保存最终帧 `final_oct_frame.npz`；训练可使用 OCT 推导的目标相对三维向量，不能使用逐帧原始 OCT 体积。首集 679/679 帧向量有效，首帧换算为毫米 `[5.98942,1.58207,18.68070]`，与源记录一致。转换器、训练与评估脚本语法检查首次 FAIL：`eval_policy.py` 多缩进一行；已修正。测试记录已追加。
- 2026-11-04：OCT train 数据集重建 FAIL（校验函数接口遗漏）。200 集/136,348 帧已重新解析至保存末尾，复用图像和数值收集均成功；`_validate_dataset_arrays` 中新增 OCT 检查引用了未传入的局部变量 `oct_navigation`，触发 `NameError`。旧 `dataset.npz` 未被覆盖。将 OCT 数组加入校验函数参数并复跑。测试记录已追加。
- 2026-11-04：OCT 数据集重建与格式复核 PASS。train/val/test 均重新生成 `oct_navigation` 与 `oct_navigation_valid`：train 136,348/136,348 有效，val 27,289/27,290，test 27,317/27,317。`RCMDataset` 实际返回 `observation.image [T,3,224,224]`、`observation.state [T,3]`、`observation.oct [T,4]`（3 轴导航向量 + valid 标志）及 `action [H,3]`，4 个真实样本前向 loss `0.96165657`。逐帧原始 OCT 深度体仍未保存，训练使用的是 OCT/深度感知得到的 `target_minus_tip` 三维导航量，单位为 mm、相机光学坐标系。测试记录已追加。
- 2026-11-04：旧模型归档 PASS。将不使用 OCT 的 epoch 9 模型、training history 和日志复制到 `outputs/archive/diffusion_policy_resumed_epoch9_20261104`，并写入 manifest；源文件与归档 `best_model.pth` 的 SHA-256 均为 `5CA00A849E7F0410F22241DC350E2C4EB58F80C9054003DF7DA18C23015C906D`。测试记录已追加。
- 2026-11-04：OCT 专用训练脚本 smoke FAIL（测试 action 为常量）。脚本语法与 OCT 数据门禁正常，但临时测试数据所有 action 相同，脚本按设计以 `Invalid action std: [0.0,0.0,0.0]` 拒绝训练。调整测试夹具为变化的 action 后复测。测试记录已追加。
- 2026-11-04：OCT 专用训练与评估 smoke PASS。`train_oct_policy.py` 在含 OCT 的 train/val 上完成 1 个 epoch，checkpoint 记录 `use_oct=true`、`oct_feature_dim=4`、4 维 OCT scale 和三轴动作损失权重；合成的 C 轴权重 `2.5729` 明显高于 A/B。`eval_policy.py` 使用 `StrictRGBDSensor` 实时产生 OCT navigation，并以 MotorBackend 完成 2 步闭环，结果 JSON 明确 `oct_navigation_used=true`、`ground_truth_used_for_control=false`。测试记录已追加。
- 2026-11-04：OCT-aware 正式训练启动 PASS。新训练 PID `38652` 于 11:01:15 启动，输出 `outputs/diffusion_policy_oct_v1`。日志确认使用 CUDA FP16 AMP、8 个线程加载；OCT 有效帧 train `136348/136348`、val `27289/27290`。OCT 标准化尺度 `[0.7087,0.8327,0.1728,1.0]`，动作损失权重 A/B/C=`[0.2785,0.4261,2.2953]`，C 轴获得更高监督权重。启动后速度约 `8.5 batch/s`，GPU 3D 引擎约 56%、显存约 3.1/6.1 GB。训练硬上限为 10 epochs。测试记录已追加。
- 2026-11-04：OCT 模型损失异常对比检查 PASS（未异常）。PID `38652` 仍正常运行；截至 11:05 处于 epoch 0 约 25%，当前近期 batch loss 约 `0.17–0.39`，最近 50 条采样均值 `0.25984`。旧模型同样在 epoch 0 前 20 条采样均值 `1.0095`，训练后最终 val loss `0.03287`；新模型前 20 条采样均值 `0.9953`。将“新模型 epoch 0 初始/中期 loss”与“旧模型 epoch 9 最终 val loss”直接比较没有意义；新模型尚未完成第一轮，不能据此判断退化。测试记录已追加。
- 2026-11-04：OCT-aware 10-epoch 训练完成 PASS。PID `38652` 已退出；history 包含 epoch 0–9，train loss 从 `0.227062` 降至 `0.014644`，val loss 从 `0.088491` 降至 `0.014850`。best 为 epoch 9，`best_model.pth` 记录相同 train/val loss，配置确认 `use_oct=true`、`oct_feature_dim=4`。日志以“训练完成”结束，未超过 10 epochs。测试记录已追加。
- 2026-11-04：训练样本 distance 分布核查 FAIL（发现单位/统计口径错误）。按转换后 `oct_navigation` 的 mm 单位统计 train 136,348 帧：距离范围 `0.00509–20.00045 mm`，中位数 `9.6101 mm`；`<10 mm` 为 70,775 帧（51.91%），`<5 mm` 为 37,453 帧（27.47%），`<0.02 mm` 为 2,672 帧（1.96%），不存在任何 `>1000 mm` 样本。此前“`<10 mm` 1,947、`<5 mm` 17、92% >1 m”与当前数据不符，最可能是把 OCT 向量的米单位或错误距离列按毫米阈值混用。近目标高精度区仍有自然样本稀疏，但问题并非所述量级。测试记录已追加。
- 2026-11-04：逐 episode 距离分布核查 PASS。200 个训练 episode 中 192 个成功、8 个失败；成功 episode 均从约 `19.48–20.00 mm` 出发，最终降到 `0.00519–0.00792 mm`，确认全部成功 episode 到达目标。在成功 episode 中位数为 684 帧：`<10 mm` 中位 355 帧、`<5 mm` 中位 188 帧、`<0.02 mm` 中位 14 帧。因而全局 `<10 mm` 约占一半、`<20 µm` 约占 2% 是“按所有时间帧计数”的自然结果，不是终点没有到达；8 个失败 episode 最终距离为 `0.078–0.106 mm`，也仍有各自 `<5 mm` 的末段帧。测试记录已追加。
- 2026-11-04：动作/状态/OCT 归一化、cosine diffusion 与离线 MAE 选模 PASS。扩散调度改为 100 steps + cosine 后末步 `alphas_cumprod=2.43e-7`；模型支持 state/OCT/action z-score，统计量写入 checkpoint，采样时反归一化动作，并加入可配置 `pred_x0_clip=4`。新 `train_oct_policy.py` 每轮在距离分层 val 子集上采样动作并计算归一化 MAE，按该 MAE 保存 best，支持早停和高于 10 的 epoch 上限。小型端到端 smoke 与 4-step 确定性采样通过。测试记录已追加。
- 2026-11-04：`train_diffusion_policy.py` 归一化回归修复 PASS。旧训练入口也已计算并传入 state/action/OCT 统计量，默认使用 100-step cosine diffusion；`pred_x0` 限幅可通过参数启用。新训练入口仍为推荐正式入口。测试记录已追加。
- 2026-11-04：state+OCT MSE 回归基线训练完成 PASS。训练在 epoch 861 达到最佳验证 normalized MAE `0.10494117`，到 epoch 891 连续 30 轮无改善后早停；最佳物理 MAE A/B/C=`0.00796502°/0.00372520°/0.000484659 mm`，RMSE=`0.0124209°/0.00633415°/0.000524094 mm`。checkpoint 记录 `policy_type=state_oct_mse_regression`、最佳 epoch 和完整归一化统计量。下一步对该基线执行 MotorBackend 闭环 smoke。测试记录已追加。
- 2026-11-04：回归基线闭环评估 FAIL（确认分布漂移）。400 步 smoke 最终误差 `19.3894 mm`，误差最好仅到 `18.8444 mm`（第 46 步）；A/B 动作持续饱和到 `+2°/-2°`，B 轴累计到约 `-16.3°`，明显超出训练状态范围，随后 346 步因 B 轴安全边界被限制为零动作并保持。新增安全回退后评估不再崩溃，并正确标记 `safety_limited_steps=346`。结论：离线 state+OCT 动作 MAE 虽低，但教师限速/历史状态未被观测，导致闭环快速进入训练分布之外。测试记录已追加。
- 2026-11-04：回归基线闭环 smoke FAIL（首步 B 轴安全限位）。评估器正确识别 `state_oct_mse_regression`、恢复 epoch 861 并以 CUDA 运行；第一步调用 MotorBackend 时，预测目标使 B 编码器到达 `212995`，低于已确认安全下限 `223159`，评估在 33 s 后安全失败且未写出结果 JSON。下一步直接打印同初始场景的 state、OCT 向量和模型原始动作，定位是感知特征分布不一致还是动作尺度错误。测试记录已追加。
- 2026-11-04：扩散归一化/调度修复及回归基线闭环诊断。`SimpleDiffusionPolicy` 增加 cosine schedule、100-step 默认、state/OCT/action z-score、采样动作反归一化和 `pred_x0_clip=4`；`train_oct_policy.py` 按距离分层验证集采样动作并计算 normalized MAE，按该 MAE 保存 best 并支持早停。smoke PASS：cosine 100 步末 `alphas_cumprod=2.43e-7`，归一化统计量写入 checkpoint。回归基线训练在 epoch 861 达到 best normalized MAE `0.104941`，A/B/C MAE=`0.007965/0.003725/0.000485`。其 400-step 闭环因 B 轴累计到约 `-16.3°` 而连续 346 步触发安全限位，最终误差 `19.389 mm`，确认低离线 MAE 不能覆盖教师隐藏状态和闭环分布漂移。评估器已增加安全回退，不再因单步越界中断整组评估。新版归一化扩散训练 PID `15500` 已启动，输出 `outputs/diffusion_policy_oct_v2_normalized`，默认 100-step cosine、FP16 AMP、8 线程并行和验证动作 MAE 选模。测试记录已追加。
- 2026-11-04：归一化扩散 epoch 0 验证结果。新版模型完成首轮：train loss `0.202827`、val diffusion loss `0.130428`，距离分层 384 样本动作 MAE A/B/C=`0.017121°/0.010242°/0.012491 mm`，normalized MAE `0.767700`。当前 C 轴误差约为 action std 的 1.75 倍；显著差于回归基线 epoch 861 的 normalized MAE `0.104941` 和 C MAE `0.000485 mm`。当前仍处于 epoch 1，不能代表最终结果；但按首轮趋势，闭环达到 20 µm 的概率很低，预计仍需显著降低动作 MAE 才有实用可能。测试记录已追加。
- 2026-11-04：训练吞吐二次优化 PASS。`ThreadBatchLoader` 改为生产者线程 + 线程池 + 双 batch 队列预取，使图像解码/缩放与 GPU 前向反向重叠；将 batch 16 提升到 32，并从 epoch 0 best 恢复。新进程 PID `26572` 稳定约 `5.1–5.5 batch/s`，折算约 `164–176 sample/s`，旧运行约 `110 sample/s`，吞吐提升约 50%；GPU 3D 引擎约 65%、整卡约 71%，显存 `4139/6141 MiB`。训练早期 loss 与旧归一化运行同量级，未出现 OOM。测试记录已追加。
- 2026-11-04：224 图像缓存 smoke FAIL（测试元数据缺少 observation_space）。`prepare_image_cache.py` 加载真实转换数据时字段完整，但临时 smoke 数据只有最小 metadata，脚本直接索引 `metadata["observation_space"]` 触发 `KeyError`。修正为兼容缺失字段，并用新的输出目录复测。测试记录已追加。
- 2026-11-04：224 图像缓存 smoke PASS。修复 metadata 兼容性后，用 16 帧样本生成缓存：数值表逐元素不变，16 张图像全部为 224×224 RGB，metadata 标记缓存尺寸与双线性插值。`image_transform_for_dataset` 会读取 metadata，对 224 缓存跳过重复 Resize。测试记录已追加。
- 2026-11-04：进一步提速瓶颈核查。batch 32 + 双缓冲运行中，GPU 3D 引擎约 `64%`、整卡约 `61–71%`、显存 `4161/6141 MiB`；CPU 采样约 `59–71%`。当前不是单一 GPU 瓶颈，剩余主要成本是每轮重复读取 640×480 PNG 并缩放/归一化两张历史图。可继续做 batch 48、线程数 12–16 或预生成 224×224 图像缓存；其中预缩放缓存收益最大。当前训练未中断。测试记录已追加。
- 2026-11-04：归一化 diffusion 第 12 epoch 状态核查 PASS（loss 数值正常，动作精度仍弱）。history 含 epoch 0–11，train loss `0.202827→0.094925`，val eps-loss `0.130428→0.098017`；epoch 12 当前 batch loss 约 `0.06–0.11`。由于新模型使用动作 z-score、100-step cosine schedule（末 `alphas_cumprod=2.43e-7`），该 eps-loss 与旧 10-step/un-normalized 模型的 `0.01485` 不可直接比较。距离分层 384 样本的 normalized action MAE 从 `0.76770` 降至最佳 `0.68188`（epoch 9），最近 epoch 11 为 `0.68732`，改善已明显放缓；C 轴 MAE 最佳约 `0.01131 mm`。结论：当前 loss 不应仅凭数值判为异常，但动作 MAE 仍未达到可用水平。测试记录已追加。
- 2026-11-04：224 图像缓存训练提速 PASS。生成 `data/lerobot_rcm_train_224`（136,348 张）和 `data/lerobot_rcm_val_224`（27,290 张），数值表逐元素复制、缓存图像抽检为 224×224 RGB，metadata 尺寸正确。训练改为 batch 48、12 个加载线程、双 batch 预取，并从 epoch 0 best 恢复。新 PID `33468` 达到约 `5.8–6.0 batch/s`，折算约 `278–288 samples/s`；相较旧 batch 16 单缓冲约 `110 samples/s` 提升约 2.6 倍。GPU 3D 引擎约 `85%`、显存 `4484/6141 MiB`，训练日志正常推进。测试记录已追加。
- 2026-11-04：前几轮改善速度差异核查 PASS（优化动态解释）。v3 history 的 epoch 0 来自 batch 16，更新次数 8,422；从 epoch 1 起改为 batch 48，每 epoch 仅 2,808 次参数更新，而学习率仍约 `1e-4`。因此虽然样本吞吐提升，`每 epoch` 的有效优化步数减少约 3 倍，早期快速改善后收敛速度自然放缓。history 中 val action normalized MAE 在 epoch 0–3 为 `0.76770→0.71527`，epoch 4–11 最佳仅到 `0.68188`；当前学习率未按 batch size 放大是重要原因之一。测试记录已追加。
- 2026-11-04：224 缓存 + 双缓冲训练继续提速 PASS。生成 136,348 张 train 与 27,290 张 val 的 224×224 RGB 缓存，数值表不变，缓存尺寸抽检通过。从 epoch 0 best 恢复，使用 batch 24、12 个加载线程、2 个预取 batch、FP16 AMP。新 PID `36820` 实测约 `10.5 batch/s`，等效约 `252 sample/s`；GPU 3D 引擎约 `96%`、显存 `3853/6141 MiB`。相较原训练约 `110 sample/s` 提升约 2.3 倍，同时每 epoch 保持 5,615 次参数更新。当前正在 epoch 1。测试记录已追加。
- 2026-11-04：停止训练 PASS。按要求终止 `outputs/diffusion_policy_oct_v2_continue` 的训练进程 PID `36820`，随后确认该 PID 已不存在；未启动任何新训练。
- 2026-11-04：3D 轨迹图生成与文件核验 PASS。`outputs/evaluation_oct_v3_cached_smoke/figures/` 下已生成 `trajectory_3d.png`（442,645 bytes）、`trajectory_3d.pdf`（155,057 bytes）、`trajectory_points.csv`（1,600 个数据行）和 `trajectory_plot_manifest.json`（967 bytes），文件均可读取。
- 2026-11-04：最近模型闭环大误差根因诊断 FAIL（确认近目标制动缺失，非单位/坐标错误）。检查 `outputs/diffusion_policy_oct_v3_cached/best_model.pth`（epoch 12）与 `outputs/evaluation_oct_v3_cached_smoke/evaluation_results.json`：随机目标 C=`22.873 mm`，模型 C 轴从句初持续正向推进，第 771 步越过目标 C，最佳真值误差出现在第 662 步、为 `1.2483 mm`，之后继续到 `32.2573 mm`，约第 1077–1599 步共 523 步保持全零动作，最终误差 `13.6546 mm`；`safety_limited_steps=527`，安全回退是过冲后的结果而非最初原因。train 数据集 136,348 帧的 `dC` 均值/标准差为 `0.040485/0.007154 mm`，负值比例 `0%`；OCT 距离 `<0.02 mm` 的 2,672 帧教师 `dC` 均值仅 `0.001190 mm`。使用 `rcm_mujoco` 环境、CUDA 和相同 seed 对 val 六个距离分层各取 24 个样本（144 样本）做模型推理：C 预测均值/标准差 `0.040828/0.002395 mm`，目标 `0.029690/0.016831 mm`；在距离 `<0.02 mm` 分层预测 `0.037598±0.002509 mm`，教师动作 `0.002533±0.001285 mm`，MAE `0.035065 mm`。因此策略在远/中距离可复现固定前进节奏，但近目标区没有学会减速，`val_action_mae` 的最佳 `0.668199` 归一化值也被大多数稳定的前进帧掩盖。测试记录已追加。
- 2026-11-04：近目标恢复数据采集起点 smoke PASS。`collect_v9_dataset.py` 新增独立 `--start-task-mode recovery`，默认 `fixed` 行为保持不变；20 个确定性 profile 覆盖 `under_0p02` 至 `under_0p80 mm`、`over_0p02` 至 `over_0p80 mm`、A/B 正负横向偏移及组合偏移。使用 `rcm_mujoco` 环境执行 `py_compile collect_v9_dataset.py verify_v9_dataset.py lerobot_training/convert_to_lerobot.py` PASS；在 seed `20261104` 的随机目标上逐一构建全部 20 个起点，目标任务 A/B/C=`6.910122°/1.575015°/19.824014 mm`，实际针尖目标误差范围 `0.020000–0.800006 mm`、起点 C 范围 `19.024014–20.624014 mm`，全部起点合法且无重复 profile。测试记录已追加。
- 2026-11-04：近目标恢复真实采集 smoke 首次 FAIL（严格视觉隔离审计路径）。恢复模式成功构造 seed `20261104`、`under_0p02mm` 起点，初始误差 `19.999 um`；控制器运行后，写报告时再次调用 `scene.target_task()`，被 `guard_scene_goal_inverse()` 的 `SCENE_TARGET_TASK_FORBIDDEN_IN_STRICT_VISUAL_CONTROL` 拒绝，未生成完整 episode。根因是目标逆解审计字段在启用严格视觉隔离后重复计算；已改为在启用控制器前仅计算一次 `goal_task_abc` 供来源审计，控制链仍禁止使用目标逆解。失败输出保留在 `runs/recovery_pilot_smoke_20261104`。测试记录已追加。
- 2026-11-04：近目标恢复真实采集 smoke 复测 PASS。修复严格视觉隔离审计路径后，`collect_v9_dataset.py` 编译通过；seed `20261104`、`under_0p02mm`、随机目标和 MotorBackend 的 1 个 episode 正常完成，15 帧、仿真成功率 `1/1`，独立终点误差 `8.5839 um`，dataset 完整性与精确离线电机重放 PASS；图像哈希、编码器状态、动作对齐、watchdog 与严格 RGB-D 声明均通过，10 个过渡体现电机滞后。起点 C=`19.80401495574862 mm`，目标任务审计字段已保存，控制链仍不使用目标逆解。输出位于 `runs/recovery_pilot_smoke_retry_20261104`。测试记录已追加。
- 2026-11-04：近目标恢复并行 pilot 完成（数据完整性 PASS；三组导航成功率 `20/20`、`19/20`、`19/20`）。train/val/test 分别使用独立 seed `20261104/20261204/20270304`，每个 split 20 个 episode，覆盖全部 20 个恢复 profile；总帧数 `580/556/534`，三组 raw RGB 哈希、编码器状态、动作对齐、严格 RGB-D 声明及精确离线电机重放均 PASS。val episode 16 因 `LOW_QUALITY_VISUAL_GEOMETRY` 失败（最终误差 `80.75 um`），test episode 11 因 `NEEDLE_DIRECTION_NOT_OBSERVABLE` 失败（`705.25 um`），其余 episode 均以 `POSITION_ERROR_AND_SETTLED` 结束。动作方向覆盖已达到目标：train/val/test 的负 `dC` 帧比例分别为 `38.10%/38.67%/32.40%`，负向范围达到 `-0.042713/-0.042713/-0.039186 mm`；过冲 `0.80 mm` profile 的负向动作比例约 `91.1%/92.3%/87.5%`。这项结果只证明恢复数据和反向动作已采集，不代表策略精度已改善。输出位于 `runs/recovery_pilot/{train,val,test}`。测试记录已追加。
- 2026-11-04：近目标恢复数据转换 PASS。使用 `lerobot_training/convert_to_lerobot.py`、`rcm_mujoco` 环境和 8 个线程分别转换三个 split，输出 `data/recovery_pilot_train`、`data/recovery_pilot_val`、`data/recovery_pilot_test`：20/20/20 episodes、`580/556/534` 帧，全部图像为 640×480 RGB，OCT navigation 有效帧 `580/580`、`556/556`、`533/534`。转换后 `dC` 范围包含负值 `-0.042712/-0.042712/-0.039187 mm` 和正值 `0.042712/0.042712/0.042712 mm`，state/action 均非恒定，证明反向恢复动作已进入训练格式。测试记录已追加。
- 2026-11-04：数据集合并脚本首次 smoke FAIL（测试目录权限）。`py_compile lerobot_training/merge_lerobot_datasets.py` PASS；测试在 `runs/checks` 下创建临时 fixture 时被 Windows ACL 以 `PermissionError: [WinError 5]` 拒绝，尚未进入合并函数。与既有受管目录问题一致，改用工作区根目录专用测试目录复测。测试记录已追加。
- 2026-11-04：数据集合并脚本复测仍 FAIL（`tempfile` 目录 ACL）。改用工作区根目录后，Python `TemporaryDirectory` 创建的子目录仍被 Windows ACL 拒绝，尚未进入合并函数；原因是受管环境对自动创建临时目录的权限，不是 merge 实现错误。改为先用 PowerShell 建立固定测试根目录，再在其内创建 fixture。测试记录已追加。
- 2026-11-04：数据集合并脚本第三次 smoke FAIL（测试 fixture 建目录漏 `parents=True`）。PowerShell 测试根目录创建成功，但 Python fixture 在创建 `base/data` 时因父目录不存在报 `FileNotFoundError`，仍未进入合并函数。测试代码改为创建完整父目录后复测；merge 脚本本身尚未被判定失败。测试记录已追加。
- 2026-11-04：数据集合并脚本 smoke PASS。修正 fixture 后，merge 将 base 的 2 episodes/5 frames 与 additional 的 2 episodes/4 frames 合成为 4 episodes/9 frames；episode 重编号为连续 `0–3`，frame index 逐 episode 从 0 连续，9 条图像引用均可从输出目录解析，metadata 计数一致。`py_compile lerobot_training/merge_lerobot_datasets.py` 此前已 PASS。测试记录已追加。
- 2026-11-04：第二批恢复数据采集中途状态记录。wave2 val 已完成 `40/40 episodes`、`1,154` 帧、`40` 个导航成功，dataset 完整性与精确离线电机重放 PASS；wave2 test 已完成 `40/40 episodes`、`1,136` 帧、`39` 个导航成功，dataset 完整性 PASS。wave2 train 请求 `180` 集，截至本次核验已提交 `90 episodes`、`2,520` 帧、`87` 个导航成功，采集进程仍运行且当前正在下一集，尚未生成最终 `dataset_verification.json`；因此 train 不宣称完成。测试记录已追加。
- 2026-11-04：扩展恢复数据采集数量核验（train 已采完、最终验证仍在运行）。wave2 主 train 的 manifest 已提交全部 `180 episodes`、`5,094` 帧、`175` 个导航成功，但 `dataset_verification.json` 尚未生成，进程仍执行最终验证；并行补充 train_a 完成 `45 episodes`、`1,260` 帧、`43` 个成功且验证 PASS，train_b 完成 `45 episodes`、`1,271` 帧、`44` 个成功且验证 PASS。加上第一批恢复数据，当前已落盘恢复数据共有：train `290 episodes / 8,205 frames`，val `60 / 1,710`，test `60 / 1,670`；合计 `410 episodes / 11,585 frames / 399` 个导航成功。相对原数据集，恢复帧数增加约 `6.07%`，train 恢复数据增加约 `6.02%`。本次只记录当前状态，主 train 的最终验证尚未完成。测试记录已追加。
- 2026-11-04：全部扩展恢复数据最终验证与 LeRobot 转换 PASS。wave2 主 train 最终 `dataset_verification.json` 已生成并 PASS；val/test 及并行 train_a/train_b 也均为 PASS，所有 raw 数据完整性、RGB 哈希、编码器状态、动作对齐和离线电机重放通过。随后并行转换输出 `data/recovery_wave2_train_main`（180 集/5,094 帧）、`data/recovery_wave2_val`（40/1,154）、`data/recovery_wave2_test`（40/1,136）、`data/recovery_wave2_train_a`（45/1,260）、`data/recovery_wave2_train_b`（45/1,271）；OCT navigation 有效帧分别为 `5093/5094`、`1154/1154`、`1136/1136`、`1260/1260`、`1270/1271`，三项动作均包含接近 `±0.042712 mm` 的正负 C 修正。测试记录已追加。
- 2026-11-04：扩展恢复数据 224×224 缓存 PASS。五个 wave2 转换集均通过 `prepare_image_cache.py` 生成缓存：train_main `5,094` 张、val `1,154` 张、test `1,136` 张、train_a `1,260` 张、train_b `1,271` 张；每张缓存均由对应 640×480 RGB 图经双线性缩放得到，metadata 尺寸更新为 224×224，数值表未修改。原有 test 224 缓存也复核为 `27,317/27,317` 图像完整。测试记录已追加。
- 2026-11-04：最终合并训练数据集验证 PASS。`lerobot_rcm_train_recovery_full_224` 为 `490 episodes / 144,553 frames`，`lerobot_rcm_val_recovery_full_224` 为 `100 / 29,000`，`lerobot_rcm_test_recovery_full_224` 为 `100 / 28,987`。三个数据集均满足：episode index 连续、每个 episode 的 frame index 从 0 连续、image_paths 数量与帧数一致且缺失图像为 0、抽帧均为 224×224 RGB、`RCMDataset` 在首/中/末样本返回有限数值及正确张量形状 `[2,3,224,224]/[2,3]/[2,4]/[8,3]`。最终数据的负 `dC` 比例 train/val/test=`2.095%/2.221%/2.108%`，近目标 `<0.02 mm` 样本分别为 `6,477/1,346/1,317`。测试记录已追加。
- 2026-11-04：近目标距离加权采样实现与 smoke PASS。`train_diffusion_policy.ThreadBatchLoader` 新增逐 epoch 的无放回替代式加权抽样支持，`train_oct_policy.py` 新增 `--distance-balanced-sampling`（默认启用）及 `<0.02/<0.1/<0.5 mm` 的 `8/4/2` 倍权重。最终 train 的 140,636 个有效窗口中，距离 `<0.02 mm` 有 `3,157` 个、`<0.1 mm` 有 `7,162` 个、`<0.5 mm` 有 `11,582` 个。合成数据测试确认高权样本被更高频抽中；真实合并数据集单个加权 batch 成功返回图像 `[4,2,3,224,224]` 和动作 `[4,8,3]`，`py_compile` PASS。测试记录已追加。
- 2026-11-04：恢复增强模型训练启动 smoke PASS。新训练输出 `outputs/diffusion_policy_oct_v4_recovery_weighted`，使用最终合并 train=`140,636/144,553` 个有效窗口、val=`28,200/29,000`，CUDA FP16 AMP、100-step cosine、batch `24`、12 个线程和距离加权 `<0.02/<0.1/<0.5 mm`=`8/4/2`。启动日志确认 train OCT 有效 `144551/144553`、val `28999/29000`，动作标准化统计已建立，epoch 0 每个 epoch 共 `5,860` 个 batch，已进入训练；启动阶段约 `11–12 batch/s`，未发生 OOM 或配置错误。本记录只证明训练已启动，不宣称 20 个 epoch 已完成。测试记录已追加。
- 2026-11-04：模仿学习调研报告结构、引用与关键数据核验 PASS。`IMITATION_LEARNING_RESEARCH_20261104.md` 为 28,188 bytes，包含 11 个规定章节、参考文献 `[1]–[18]` 且无 TODO/TBD/placeholder；报告中的原始训练集 `200 episodes / 136,348 frames / 192 success`、已转换首批恢复训练集 `20 / 580` 与 wave2 五份采集验证均与当前文件一致。wave2 train/val/test、train_a、train_b 的 `dataset_verification.json` 当前结果均为 PASS，Episodes/Frames/Success 分别为 `180/5,094/175`、`40/1,154/40`、`40/1,136/39`、`45/1,260/43`、`45/1,271/44`。另独立通过 Crossref、OpenAlex、arXiv 和 Europe PMC 核验报告引用元数据；本轮未重新训练或重新运行策略闭环。测试记录已追加。
- 2026-11-05：恢复增强模型 20-epoch 训练完成 PASS（仅离线验证，尚未闭环评估）。`outputs/diffusion_policy_oct_v4_recovery_weighted/training_history.json` 含 epoch 0–19，日志以“训练完成”结束；最佳 checkpoint 为 epoch 15，train loss `0.0850182`、val loss `0.0883270`、normalized action MAE `0.2792083`，对应 A/B/C MAE `0.0179058°/0.0101868°/0.00308390 mm`。相对旧 v3 best normalized MAE `0.6681988` 改善约 `58.2%`，C 轴 MAE 从 `0.0113111 mm` 降至 `0.00308390 mm`（约 `72.7%`）；但 A/B MAE 相对旧值 `0.0140630°/0.00734995°` 分别退化，因此不能据此宣称闭环问题已解决。epoch 19 未超过 epoch 15，best_model 正确保留 epoch 15；训练进程已退出。下一步必须使用独立 test split 做 MotorBackend/严格 RGB-D 闭环评估，重点检查过冲、最小误差和最终停稳。测试记录已追加。
- 2026-11-05：恢复增强模型独立 test 开环评估 PASS（仅动作预测，不代表闭环成功）。对 `data/lerobot_rcm_test_recovery_full_224` 按六个 OCT 距离分层各抽 128 个窗口、共 768 样本，使用 epoch 15 best checkpoint、CUDA 和固定 seed `20270305`：整体 normalized MAE `0.321798`，A/B/C MAE=`0.0190741°/0.0110150°/0.00412308 mm`；整体预测/目标负 `dC` 比例为 `10.16%/11.46%`。近目标 `<0.02 mm` 分层预测 C=`0.002307±0.007954 mm`、教师 `0.001078±0.002364 mm`，C MAE `0.00193262 mm`、符号正确率 `86.72%`；相比旧模型同一近目标分层 C MAE `0.035065 mm`，离线近目标制动误差下降约 `94.5%`。但预测 C 标准差仍大于教师值，需以闭环结果判定是否实际稳定。测试记录已追加。
- 2026-11-05：恢复增强模型闭环评估 FAIL（离线改善未转化为闭环成功）。使用 epoch 15 best checkpoint、严格 RGB-D、OCT navigation、MotorBackend、随机目标 seed `20270305`、1600 步上限评估 1 个 episode：`success=false`，最终误差 `24.7881 mm`，最低误差 `3.3433 mm`（第 644 步），最终 state A/B/C=`[-32.1750°,16.7025°,32.2550 mm]`，最终速度全零；`safety_limited_steps=352`，最后约 348 步因 C 轴超过编码器安全上限保持零动作。目标逆解为 A/B/C=`0.536456°/-11.889936°/19.395918 mm`。轨迹前约 650 步主要沿正向 C 接近，随后出现持续错误方向动作 A≈`-0.2287°`、B≈`+0.1493°`、C≈`+0.08694 mm`，逐步发散；约第 800 步后误差升至 `16 mm` 以上，随后在 A/B 大偏转下漂移并最终安全停住。该结果证明近目标 C 轴离线 MAE 的改善不足以保证闭环稳定；当前不能宣称新模型优于旧模型。输出见 `outputs/evaluation_oct_v4_recovery_weighted_test_20261105/evaluation_results.json`。测试记录已追加。
- 2026-11-05：闭环发散根因核查 PASS（确认扩散采样灾难性分支，非 OCT 失效/坐标错误）。在第 669–670 步对应状态、图像和目标上，OCT navigation 始终有效；第 670 步 OCT 向量为 `[-0.69146,-3.23189,-0.76663] mm`。训练集中该 state+OCT 条件最近的 20 帧动作均值为 A/B/C=`0.00162°/-0.000535°/0.042110 mm`，不属于显著分布外。对完全相同观测用 7 个独立采样 seed 推理，其中 6 个输出小幅合理动作，seed `42` 精确复现闭环中的饱和错误动作 `[-0.2287106°,0.1493193°,0.0869445 mm]`；该值正是 A/B/C 分别触及 `pred_x0_clip=4` 标准化边界的输出去归一化结果。诊断脚本首次因从根目录直接导入 `eval_policy.py` 缺少 `lerobot_training` 路径而 `ModuleNotFoundError`，修正 `sys.path` 后完成上述复测。结论：当前主要根因是扩散策略在部分随机采样路径上产生灾难性动作，随后被闭环状态反馈放大；不是 OCT 在该点失效，也不是 C 轴单位或目标坐标错误。TEST_LOG 已追加。
- 2026-11-05：闭环推理护栏实现与 smoke PASS。`eval_policy.py` 默认动作上限由 `(2,2,0.2)` 收紧为最终训练集逐轴最大值 `(0.289800°,0.144344°,0.0427121 mm)`，同观测默认并行采样 7 个动作并逐轴取中位数，另新增误差 `>30 mm` 或连续 20 步安全限位时的发散早停。首次断言误把裁剪前模型原始输出当作裁剪后动作，因而测试断言失败；修正测试后，seed `42` 和第 670 步原灾难性条件的新原始中位动作分别为 `[-0.002677,0.005726,0.015797]`、`[-0.013077,0.005579,0.024199]`，均不再触发极端分支，裁剪后也满足逐轴上限。`py_compile` PASS；3-step MotorBackend/严格 RGB-D CLI smoke 完成并正常写出 `outputs/evaluation_guard_smoke_20261105/evaluation_results.json`，其中记录 action limit、7-sample median 和早停契约。测试记录已追加。
- 2026-11-05：带隐藏状态的快速回归模型实现 smoke PASS（含两次测试夹具失败）。`train_oct_regression.py` 支持 `none/velocity/velocity_previous_action` 三种历史模式，从现有 state/action 按 episode 边界推导速度与上一步动作，并对训练期历史特征加入高斯噪声和 dropout；`eval_policy.py` 同步支持回归模型的速度、上一步实际执行动作及 episode reset。测试夹具前两次分别错误比较了位移数组的索引，以及错误假设每个 episode 的第二帧没有上一帧；修正预期边界计算后，速度/上一步动作逐元素匹配、episode 首帧在两路历史中均为 0；`py_compile` PASS；新 13 维回归模型在合成输入上输出有限 `[5,3]`，旧 7 维 regression checkpoint 仍可在 `history_mode=none` 下加载并推理。测试记录已追加。
- 2026-11-05：隐藏状态快速回归训练完成 PASS（离线动作指标，尚无闭环结果）。`velocity_previous_action` 模型在 epoch 326 达到 best normalized MAE `0.0751337`，A/B/C MAE=`0.00662345°/0.00311885°/0.000354562 mm`，连续 30 轮无改善后在 epoch 356 早停；`velocity-only` 模型在 epoch 396 达到 best normalized MAE `0.0746014`，A/B/C MAE=`0.00649860°/0.00298989°/0.000377260 mm`，在 epoch 426 早停。二者均显著优于旧回归基线 normalized MAE `0.104941`；但加入上一步动作没有优于仅速度，说明该字段暂未提供可测增益，也降低了“模型仅抄上一步动作”作为主要解释的可能性。仍需闭环验证，不能仅凭此宣称闭环问题修复。测试记录已追加。
- 2026-11-05：并行闭环评估低显存改造 smoke PASS。将 7-sample median 从一次 batch repeat 改为同一模型顺序采样 7 次再取中位数，降低每进程峰值显存；`py_compile` PASS，3-step MotorBackend/严格 RGB-D CLI smoke 完成并写出 `outputs/evaluation_guard_sequential_smoke_20261105/evaluation_results.json`。为避免单个 20-episode 进程耗时过长，已停止原串行评估并改为三路独立分片：`7/7/6` 个目标、seed `20271105/20271106/20271107`、输出目录 `evaluation_oct_v4_guard_parallel_20261105_g1/g2/g3`。并行评估已启动，尚未完成，不记录成功率。测试记录已追加。
- 2026-11-05：空间视觉、图像增强和 OCT 鲁棒性训练接口 smoke PASS（含两次实现错误）。为保留显微图像空间位置，`ResNetEncoder` 新增 `spatial_softmax` 模式，输出 7×7 特征图的逐通道期望坐标而非全局平均池化；训练图像新增轻度 `RandomResizedCrop` 与颜色抖动，验证图像保持确定性预处理。模型新增 OCT 高斯噪声、特征 dropout 和随机 invalid，并支持 `oct_dim=0` 的 RGB+state 消融。首次 spatial softmax 测试因二维坐标扁平化广播错误失败；改为 `meshgrid` 后通过。随后 OCT mask 赋值因布尔二维索引广播失败，改为 `masked_fill` 后通过。最终 smoke 验证：增强图像输出有限 `[3,224,224]`，spatial+OCT 训练 loss 有限、推理输出 `[2,4,3]`，spatial 无 OCT 输出 `[2,4,3]`；旧 global-vision checkpoint 仍可加载，新的 spatial/no-OCT checkpoint 可被 `PolicyEvaluator` 重建。测试记录已追加。
- 2026-11-05：空间图像消融训练启动与吞吐调整 PASS。首次 batch 48 + channels-last 尝试因 rank-5 `[B,T,C,H,W]` 张量直接使用 channels-last 抛出 `RuntimeError`；改为仅在 ResNet 内部 reshape 后的 rank-4 图像块上应用后模型 smoke PASS。但实测 batch 48+channels-last 仅约 `99 samples/s`，移除 channels-last 后 batch 24 约 `157 samples/s`，batch 32 约 `173 samples/s`。最终选择 batch 32、12 图像线程、无 channels-last，并顺序训练两个同预算消融：`diffusion_policy_v5_spatial_rgb_oct_fast5` 与 `diffusion_policy_v5_spatial_rgb_state_fast5`。共同配置为 spatial softmax、随机裁剪/颜色抖动、20 epochs、cosine 100-step diffusion、动作距离分层 8/4/2 权重；OCT 组另加三维噪声 `0.05 mm`、特征 dropout `0.10`、invalid 概率 `0.05`。当前 OCT 组已进入 epoch 0，约 `5.4 batch/s / 173 samples/s`，未 OOM；后续 RGB+state 消融将在其结束后自动启动。训练尚未完成，不能宣称图像消融结果。测试记录已追加。
- 2026-11-05：图像消融重新启动（已按定位精度修正增强并并行训练）。为优先执行该消融，已停止三个教师接管诊断进程及无 checkpoint 的旧消融任务。新增并 smoke 验证 `--freeze-vision-encoder`：ResNet18 主干 trainable 参数为 0，空间输出头仍可反向传播；同时确认 `RandomResizedCrop` 默认关闭，训练只保留 ColorJitter。两组使用相同数据、seed `20261105`、20 epochs、batch 48、12 线程、spatial softmax、冻结 ResNet、距离分层 8/4/2 和无几何增强；OCT 组额外使用噪声 `0.05 mm`、dropout `0.10`、invalid `0.05`。两进程已并行进入 epoch 0，各约 `2.9–3.2 batch/s`，合计约 `300 samples/s`，显存 `3076/6141 MiB`，GPU 约 `78%`，无 OOM。输出目录为 `outputs/diffusion_policy_v5_ablation_oct_run2` 和 `outputs/diffusion_policy_v5_ablation_nooct_run2`。训练尚未完成，不记录消融结论。测试记录已追加。
- 2026-11-05：图像消融按 4-epoch 上限重启 PASS。停止尚未产生 epoch checkpoint 的 20-epoch run2，两模型保持完全相同配置改为 4 epochs，新输出为 `diffusion_policy_v5_ablation_oct_run3` 与 `diffusion_policy_v5_ablation_nooct_run3`。两组仍并行使用冻结 ResNet18、spatial softmax、无随机几何裁剪、ColorJitter、batch 48、12 线程、距离分层 8/4/2 和同一 seed `20261105`；OCT 组额外保留 `0.05 mm` 噪声、`0.10` dropout、`0.05` invalid。两进程均已进入 epoch 0，各约 `3.6 batch/s`，显存 `3114/6141 MiB`，GPU 约 `55%`，无 OOM。训练尚未完成，测试记录已追加。
- 2026-11-05：空间图像 4-epoch 消融完成 PASS（离线动作指标）。两组均完整训练 epoch 0–3，训练进程已退出。`RGB+state+OCT` 最佳 normalized MAE `0.5012204`，A/B/C MAE=`0.0162452°/0.0107448°/0.0113735 mm`；`RGB+state` 最佳 normalized MAE `0.5006039`，A/B/C MAE=`0.0157247°/0.0103483°/0.0115908 mm`。无 OCT 模型的总体 normalized MAE 反而低 `0.123%`，A/B 略好，C 轴 MAE 高 `1.91%`；差异很小，当前 4-epoch 预算下无法认为 OCT 分支带来稳定优势。两组均显著差于此前的 global-ResNet v4 normalized MAE `0.279208`，说明冻结 ResNet + spatial softmax + 4-epoch 方案本身尚未收敛，不能据此否定图像或 OCT 的长期价值。checkpoint config 均记录 `vision_mode=spatial_softmax`、`geometric_augmentation=false`、`freeze_vision_encoder=true`。下一步应做闭环对比；若闭环均无改善，再增加 `state+OCT` 无图像第三臂，才能严格量化图像贡献。测试记录已追加。
- 2026-11-05：DAgger 第 0 步教师接管验证完成（部分 PASS：4/6 场景恢复到 100 µm 内）。新增并验证 `verify_teacher_takeover.py`，从三个失控 episode 的“持续发散起点”和“其后 100 步状态”分别重建场景、目标及 MotorBackend，再让严格 RGB-D 几何教师接管。g1：起点 `3.0710 mm`→`4.866 µm`（92 步），后段 `10.9838 mm`→`4.900 µm`（198 步），均停稳 PASS；g3：起点 `2.2175 mm`→`8.252 µm`（73 步），后段 `11.0364 mm`→`4.641 µm`（176 步），均 PASS。g2 起点 `4.2091 mm` 在第 19 步因 `INCONSISTENT_VISUAL_DEPTH` 失败，最终 `2.986 mm`；后段 `13.1214 mm` 因 `INSUFFICIENT_VISUAL_DEPTH` 失败，最终 `1.197 mm`。结论：教师能从大部分策略失败状态稳定恢复并产生高质量纠正标签，DAgger 可行；但不能无条件标注所有状态，必须在线过滤视觉深度不可靠、目标不可见或教师恢复失败的帧。按设计，后续只标注教师可见且恢复成功的状态，较大偏差默认限制在 `10 mm` 内。输出位于 `outputs/teacher_takeover_v2_20261105_g1/g2/g3`。测试记录已追加。
- 2026-11-05：DAgger 标签函数、采集器和转换器 smoke PASS（含开发期测试错误）。`VisualServoController.geometry_action_from_observation()` 在不移动场景时计算教师单步动作和深度有效性；先用 1 步上限导致没有生成命令字段，改为 2 步后测试继续，随后发现标签把 task-SI 增量乘 rad/deg，修正为除以 rad/deg 后与教师第一个真实 command 精确一致：A/B/C=`[-0.0250°,-0.0125°,-0.00375 mm]`。新增 `collect_dagger_dataset.py`：教师预跑到 3–15 mm、新随机目标、β 混合、误差反向/动作触限接管、40 帧保持、逐帧 policy/teacher/executed action 与教师深度 valid 记录；未知或无 checkpoint 状态仍不写入训练标签。转换器新增 `--action-source teacher`，按 `teacher_valid` 过滤无效标签。首轮真实 smoke 为 1 目标、1 个 15 mm 起点、50 帧：policy 28 帧、β-teacher 22 帧、教师标签有效 50/50，转换输出 50 帧且 action 数组逐元素等于 teacher label；另一合成测试确认 3 个源帧中 1 个教师 invalid 被过滤，转换后仅保留 2 个有效教师标签。初次 converter 测试断言误把第一条有效标签期望为 0，修正后 PASS。输出为 `runs/dagger_smoke_20261105_retry` 与 `data/dagger_smoke_teacher`。本轮尚未触发自动 teacher takeover/hold 分支，也未开始 60 目标首轮采集或训练。测试记录已追加。
- 2026-11-05：DAgger teacher-takeover 与到位保持 smoke PASS。用 1 个 3 mm 起点、`beta=1`、10 步初定接管、100 步接管上限和 40 帧保持运行完整采集：216 帧中 β-teacher 96 帧、自动 takeover 80 帧、零动作 hold 40 帧，策略自身 0 帧；教师标签有效 `216/216`，最终真值误差 `8.935 µm` 且正常停稳。检查发现自动接管在策略动作触限后发生，教师状态按实测 q/velocity 重置；末 20 帧教师标签均为零。`convert_to_lerobot.py --action-source teacher` 转换后仍为 216 帧，动作数组逐元素等于教师标签，OCT valid `216/216`；此前 3 帧 1-invalid 合成测试也确认无效教师标签被过滤。输出为 `runs/dagger_smoke_takeover_hold_20261105` 与 `data/dagger_smoke_takeover_teacher`。该 smoke 验证了采集、接管、保持和转换链路；尚未开始 60 目标正式采集，也未训练。测试记录已追加。
- 2026-11-05：DAgger 第 1 轮正式采集已启动（进行中）。三路并行，每路 20 个新目标、每个目标 2 个随机 3–15 mm 起点，共 60 个目标/约 120 个 rollout；seed 分别为 `20280101/20281101/20282101`，与现有 val/test seed 分离。共同配置：v4 epoch 15 behavior policy、β=0.5、教师接管 40 步（最多 400）、成功到位后 40 帧零动作保持、最大误差 15 mm、max-steps 1600、每步记录 policy/teacher/executed action 及教师深度有效性。三路均已加载模型并进入首目标，首起点误差分别为 `9.4471/9.0094/14.9775 mm`；当前显存 `5382/6141 MiB`，未 OOM。尚未完成采集或训练。测试记录已追加。
- 2026-11-05：v4 OCT 因果贡献测试完成（确认近目标未有效使用 OCT）。在 DAgger 采集保持运行时使用 CPU，对独立 val 数据按 6 个 OCT 距离分层各抽 64 个窗口，共 384 样本；固定种子生成噪声。比较原始 OCT、OCT 向量+valid 全零、跨不同 episode 且同距离分层置换 OCT：baseline normalized MAE `0.330994`，A/B/C MAE=`0.0232357°/0.0120579°/0.00323442 mm`；zero normalized MAE `0.299822`（较 baseline 改善 `9.4%`），A/B/C=`0.0219025°/0.0119831°/0.00240263 mm`；episode-shuffle normalized MAE `0.349201`（较 baseline 恶化 `5.5%`），A/B/C=`0.0243795°/0.0135751°/0.00316296 mm`。近目标 `<0.02 mm` 分层：baseline C MAE `0.00249304 mm`，zero `0.00124301 mm`（改善 `50.1%`），shuffle `0.00248752 mm`（几乎不变）。说明模型在较远 A/B 区域会利用 OCT，但近目标 C 制动几乎不依赖 OCT，且真实 OCT 条件反而令 MAE 变差；当前 OCT 接入没有形成有效的比例制动信号。首次测试因只替换当前帧而模型期望 `[历史帧,4]` 导致维度错误，改为整段历史置零/跨 episode 替换后完成。结论：不应继续增加同结构训练轮次，下游需要把 OCT 误差显式接入输出比例项或残差修正；DAgger 仍继续以教师标签和策略失败状态为主。测试记录已追加。
- 2026-11-05：逐步 OCT/A-B-C 诊断记录接口 smoke PASS。`eval_policy.py` 的轨迹新增逐控制步 `oct_navigation_mm`、`oct_navigation_norm_mm`、`oct_navigation_valid` 和 `simulation_time_s`，同时保留 A/B/C states、动作和独立误差。3-step MotorBackend/严格 RGB-D smoke 正常写出 3 个对齐记录，OCT valid 均为 true，向量范数约 `19.75 mm`，数组长度一致；`py_compile` PASS。随后已停止图像消融训练，开始并行复放三个代表性失控 episode：原 seed `20271105/20271106/20271107` 各自 episode 0。测试记录已追加。
- 2026-11-05：失控 episode 分流诊断完成 FAIL（根因不在 OCT 丢失）。逐步复放 3 个代表性失败 episode，结论均为“先到达近目标、OCT 仍有效时开始发散，之后才出现 OCT invalid”：g1 seed `20271105` 最低误差 `2.5377 mm`（第 660 步）、持续发散起点第 819 步、OCT 向量范数 `3.0794 mm` 且 valid=true，首次 invalid 为第 1087 步；g2 seed `20271106` 最低误差 `3.6643 mm`（第 659 步）、第 677 步开始发散、OCT 范数 `4.2075 mm` 且 valid=true，全过程 OCT 从未 invalid；g3 seed `20271107` 最低误差 `1.6644 mm`（第 667 步）、第 869 步持续发散、OCT 范数 `2.2237 mm` 且 valid=true，首次 invalid 为第 1151 步。三例在持续发散后均反复输出裁剪边界动作 A/B/C=`-0.22871°/+0.14434°/+0.042712 mm`，而 OCT 向量范数随后从约 `2–4 mm` 增至 `11–30 mm`。因此失败不是“OCT 先丢失导致盲走”，而是策略在有效近目标观测下做出错误/饱和动作，随后造成闭环状态偏差；主要修复方向是策略-数据覆盖与动作生成，而不是修 OCT 感知或重标定。输出位于 `outputs/oct_trace_diag_20261105_g1e0/g2e0/g3e0`。测试记录已追加。
- 2026-11-05：训练/评估图像预处理一致性与增强修正 PASS。对同一原始 `1920×1440` RGB 帧分别执行训练缓存路径（`1920×1440 -> 640×480 PNG -> 224×224`）和评估实时路径（`1920×1440 -> 640×480 -> 224×224`），两者均为 PIL bilinear、RGB、ToTensor+ImageNet Normalize，最终 uint8 图像逐像素完全一致（最大差 `0`、均值差 `0`），排除了训练/评估插值或 RGB/BGR 顺序不一致。首次核验使用了错误的原始图像路径，修正到 `runs/paper_stage1/train/dataset/images/episode_000000/frame_000000.png` 后完成。另确认两条路径均将 4:3 图像拉伸为 `224×224`，这是共同的几何变形而非训练/评估不一致。鉴于毫米级定位不能接受额外随机平移/缩放，已将训练几何增强与光度增强拆分：`RandomResizedCrop` 默认关闭，仅保留轻度 ColorJitter；显式传 `--geometric-augmentation` 才会启用随机裁剪。smoke 确认增强训练 transform 为 `ColorJitter/ToTensor/Normalize`、验证 transform 为 `ToTensor/Normalize`，`py_compile` PASS。测试记录已追加。
- 2026-11-05：空间图像消融正式训练启动 PASS。新增 `--freeze-vision-encoder`：冻结 ResNet18 主干参数并保持其特征提取为 eval 模式，仅训练 spatial-softmax 输出头、状态编码和扩散头；smoke 验证骨干可训练参数为 0、输出头仍有梯度、带/不带 OCT 两种模型前向及反向均有限。为提速，`RGB+state+OCT` 与 `RGB+state` 两组改为并行训练，均使用相同数据、seed `20261105`、20 epochs、batch 48、12 图像线程、spatial softmax、ColorJitter-only（无随机几何裁剪）、cosine 100-step 和距离分层采样。启动后两进程各约 `3.8–4.2 batch/s`（合计约 `380 samples/s`），GPU 利用率约 `48%`、显存 `2928/6141 MiB`，均无 OOM。输出目录为 `outputs/diffusion_policy_v5_ablation_oct_frozen_parallel` 与 `outputs/diffusion_policy_v5_ablation_nooct_frozen_parallel`。训练尚未完成，不记录消融结论。测试记录已追加。
- 2026-11-05：v4 推理护栏 20 目标闭环评估完成 FAIL。三路分片共 20 个独立随机目标全部失败（成功率 `0/20`）；最终误差均值/中位数 `21.591/30.021 mm`，最终误差范围 `1.284–30.095 mm`；每集最低误差均值/中位数 `2.794/2.577 mm`，全局最低单集误差 `0.276 mm`。所有 episode 的 `safety_limited_steps` 均为 0，说明逐轴教师动作上限成功消除了此前的安全限位和 `±4σ` 饱和动作；但 13 个 episode 因误差超过 `30 mm` 提前终止，另 7 个跑满 1600 步，最终误差范围 `1.284–9.607 mm`。7-sample median 明显压住了灾难性单步动作，但没有解除闭环方向漂移，因此不能把问题继续归因于单一扩散离群采样。结果位于 `outputs/evaluation_oct_v4_guard_parallel_20261105_g1/g2/g3`，汇总由三份 `evaluation_results.json` 计算。测试记录已追加。
- 2026-11-05：direct-OCT 比例前馈 + 有界残差轻量回归实现并完成 smoke/单元测试 PASS。新增 `lerobot_training/train_oct_guided_regression.py` 与 `verify_oct_guided_regression.py`：OCT camera-optical 误差按 `R_camera @ (e / [1,-1,-1])` 转世界系，用 `Reference.forward_kinematics` 的中心差分任务雅可比和阻尼伪逆计算 `K·sat(J⁺e)`；学习网络只输出随 OCT 距离收缩的残差，并输入单位方向、`log(d)`、有效位和基动作。首次训练发现残差监督上界与模型上界不一致，C 轴被错误硬截为约 `0.0148 mm`；修正为远距离使用训练动作逐轴上限、近距离最低收缩到 `10%` 后重训。测试覆盖相机逆变换、雅可比针尖预测（相对误差 `<0.2%`）、动作方向/饱和/无效 OCT、距离特征和残差边界；`py_compile` PASS。测试记录已追加。
- 2026-11-05：direct-OCT 轻量回归 120 epoch 训练与因果验证 PASS。最终 checkpoint epoch `112`，验证动作 A/B/C MAE=`0.00973681°/0.00506767°/0.00165412 mm`，normalized MAE `0.1470005`。在独立 val 上按 7 个距离层各 64 个窗口共 448 样本评估：真实 OCT、向量置零、跨不同 episode 且同距离层打乱三组的总体 A/B/C MAE 分别为 `0.005992/0.003341/0.004109`、`0.009482/0.004430/0.024790`、`0.032473/0.013676/0.005394`。近目标 `<0.02 mm` 的 C MAE：真实 `0.00054862 mm`、置零 `0.00086073 mm`、打乱 `0.00197700 mm`，真实 OCT 相对两组分别改善 `36.26%/72.25%`，超过预设 `10%` 门槛。结论：新 OCT 接入已通过“近目标真实 OCT 明显优于失效/错误 OCT”的因果验收，可用于下游策略结构；该无图像回归仍只作诊断，不作为交付策略。结果位于 `outputs/oct_guided_regression_v2_20261105`。测试记录已追加。
- 2026-11-05：guided-OCT 图像扩散集成 smoke PASS，4 epoch 微调已启动但尚无结果。`RCMDataset` 增加可选的 `guided_base_action.npy`/`guided_residual_limit.npy` sidecar；`SimpleDiffusionPolicy` 新增 guided 模式，图像策略输入改为 state、OCT 单位方向、`log(d)`、valid 和基动作，扩散只学习 clipped residual，最终采样严格限制为 `base + bounded_residual`。`train_oct_policy.py` 支持 `--guided-oct` 和 `--init-from`，从 v4 加载 134 个形状兼容权重，仅重新初始化改变的 state 输入层。新 `verify_guided_diffusion.py` 覆盖训练 loss、采样形状/残差边界、state 编码宽度和真实数据 sidecar 对齐，PASS；真实训练集单 batch（8×2 帧图像）前向/反向和采样均为有限值；旧 v4 checkpoint 经 `PolicyEvaluator` 仍按 7 维 state encoder 正常加载。已为 final train/val 224 数据生成 guided sidecar（test 未动），输出目录 `outputs/diffusion_policy_v6_guided_oct_finetune`；训练约 `3 it/s`，完成前不记录模型效果或开始新策略 DAgger。测试记录已追加。
- 2026-11-05：guided-OCT `PolicyEvaluator` 集成 smoke PASS。构造 guided diffusion checkpoint 后，`PolicyEvaluator` 正确识别 `guided_oct=true`、重建 11 维 state 输入，并在 CPU 上完成一次真实图像预处理、OCT 基动作计算、残差扩散采样，输出全部有限。旧 v4 checkpoint 同时复测可正常加载并按原 7 维 state encoder 构造，未受 guided 分支影响；`py_compile` PASS。输出位于 `runs/checks/guided_policy_eval_smoke_20261105`。测试记录已追加。
- 2026-11-05：DAgger 采集器可用性复核 PASS。重新执行 `collect_dagger_dataset.py --help`，CLI 正常列出模型、输出、目标数、起点数、seed、β、教师接管步数/上限、到位保持、动作采样数、显式起点误差及 reference 数据参数；`collect_dagger_dataset.py`、`convert_to_lerobot.py`、`eval_policy.py` 的 `py_compile` 退出码 0。代码复核确认支持教师 3–15 mm 预跑、策略/β/教师接管/零动作保持四种执行来源，逐帧保存 policy/teacher/executed action、teacher valid/depth valid，并在 manifest 记录 policy checkpoint、是否 guided OCT；转换器支持 `--action-source teacher` 且过滤 invalid 标签。旧 v4 未完成 pilot 已由 `runs/dagger_round1/STATUS_OLD_POLICY.md` 标为不可合并。采集器已就绪，正式 guided 第 1 轮仍等待 v6 训练最终 checkpoint。测试记录已追加。
- 2026-11-05：guided v6 图像策略 4 epoch 微调完成，最终 checkpoint 加载/推理验收 PASS。训练退出正常，4 个 epoch 的 normalized MAE 依次为 `0.325371/0.341216/0.301130/0.272570`；最佳为 epoch 3，A/B/C MAE=`0.0135156°/0.00663015°/0.00312630 mm`，略优于 v4 的 normalized MAE `0.279208`。`PolicyEvaluator` 从最终 `best_model.pth` 成功重建 11 维 guided state 输入，确认 `guided_oct=true`、checkpoint epoch=3；用真实 guided 路径完成两次连续推理，第一次历史缓冲返回零动作，第二次输出有限 `[0.0896531°,-0.0117031°,0.0258049 mm]`。最终 checkpoint 为 `outputs/diffusion_policy_v6_guided_oct_finetune/best_model.pth`。测试记录已追加。
- 2026-11-05：guided v6 DAgger 第 1 轮正式采集启动 smoke PASS。三路进程均成功加载 epoch 3 guided checkpoint，并分别进入首个 target/episode：`shard1` seed `20280101` 起点误差 `9.4471 mm`、`shard2` seed `20281101` 起点 `9.0094 mm`、`shard3` seed `20282101` 起点 `14.9775 mm`；无启动异常。三路配置均为 20 个新目标、每目标 2 个 3–15 mm 起点、β=`0.5`、教师接管 40 步（最多 400）、40 帧保持、单步动作 5 样本中位数、15 mm 失控终止。输出根目录为 `runs/dagger_round1_guided_v6_20261105`；旧 v4 pilot 不使用。当前仅记录启动成功，采集数量和成功率等完成后再报告。测试记录已追加。
- 2026-11-05：DAgger 采集后自动训练编排器实现与 dry-run PASS。新增 `auto_train_after_dagger.py`，等待三路 `manifest.json` 后依次执行：校验 shard 均来自 guided epoch 3、按 `--action-source teacher` 转换并核对有效教师帧数、与 `recovery_full` 训练集合并、生成 guided OCT sidecars、校验 sidecar 形状/有限值/非负上界，最后从 v6 epoch 3 checkpoint 自动微调 4 epoch 到 `outputs/diffusion_policy_v7_guided_dagger1`。任一 shard 无 manifest 退出或 12 小时超时即失败；不再覆盖已有输出，不混入旧 v4 pilot。`py_compile` 与 `--check-only` PASS，计划中的三个转换目录、合并目录、验证集和初始 checkpoint 均正确解析；三路采集进程 `12720/41524/10212` 当前仍存活且尚无 manifest。测试记录已追加。
- 2026-11-05：自动训练编排器进程存活检测 FAIL→PASS（systematic-debugging）。首次后台启动后，编排器在 12 秒内错误报告 shard1 PID 已退出；三联采集实际仍运行且在持续写图像，根因是 `tasklist /FI` 在当前沙箱返回 `Access denied`，旧实现把查询失败误判为进程死亡。已改为 Win32 `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` + `CloseHandle` 做存在性检查；复测三个采集 PID `12720/41524/10212` 全部为 alive，伪造 PID `999999` 为 dead，`py_compile` PASS。失败运行在写任何转换/合并/训练输出前停止，旧 `pipeline.lock` 已由 `finally` 清理，未启动训练。测试记录已追加。
- 2026-11-06：DAgger shard1 中断根因定位与可恢复修复 PASS（尚未完成剩余采集）。shard2/shard3 已完成，各 `40` episodes、`16,831/17,637` 帧，均使用 guided epoch 3，教师无效帧 `0/2`。shard1 完成 `episode_000000–000011` 后在 episode 12 的首帧前因预跑状态不在训练范围而触发 `DAgger episode produced no frames`，旧逻辑因此终止整个 shard；自动编排器随后检测到进程退出、未创建 manifest，并按要求停止且未进入转换/训练。已增强采集器：空 episode 目录重命名为 `failed_episode_*_no_frames` 后跳过并继续，而不是终止整路；新增 `--resume-start-episode`，可从既有输出继续并保留前缀 control reports。已将空的 `episode_000012` 安全归档为 `failed_episode_000012_no_frames`。同时修复编排器：只对仍缺 manifest 的 shard 检查 PID，已完成 shard 即使进程退出也不再误报。`py_compile`、resume CLI 和 Win32 PID 存活/死亡探针均 PASS。测试记录已追加。
- 2026-11-06：DAgger shard1 续采与自动训练等待 smoke PASS。shard1 以相同 seed `20280101` 从 `resume-start-episode=13` 启动，正确跳过已归档的空 episode 12，并从目标 6、起点 1、episode 13、起点误差 `9.4161 mm` 继续；进程 PID `41944`。自动训练编排器 PID `9308` 已重新启动，状态为 `waiting_for_collection`，`shards_ready=[shard2,shard3]`、`shards_missing=[shard1]`；转换、合并和 v7 训练尚未开始。测试记录已追加。
- 2026-11-06：剩余 DAgger 采集并行提速改造 smoke PASS。为缩短 shard1 剩余 27 集的长尾，新增 `--target-start/--target-end/--skip-episodes-before`；保持相同 seed 和 start-error 随机序列，将剩余目标拆为 `shard1_rem_a` 目标 6–10（跳过全局 episode 13 以前）、`shard1_rem_b` 目标 11–15、`shard1_rem_c` 目标 16–19，分别预计 9/10/8 集。已停止原串行续采并归档空的 `episode_000013`；由已有 12 个 control reports 通过新增 `finalize_dagger_manifest.py` 生成 shard1 前缀 manifest：12 episodes、5,289 帧、guided epoch 3。自动编排器扩展为六个来源，三轮续采均加载 epoch 3 并分别从 episode 13/22/32 正确启动；当前状态 `waiting_for_collection`，六来源中旧 shard1/shard2/shard3 已 ready，三路 rem 缺失，转换与训练尚未开始。`py_compile` 和 prefix manifest 校验 PASS。测试记录已追加。
- 2026-11-06：自动编排器陈旧锁恢复 FAIL→PASS。强制停止前一 watcher 后，`pipeline.lock` 残留 PID `33620`，新编排器首次启动被旧实现整体拒绝；确认该 PID 已 dead 后，将锁处理改为读取持有 PID，仅当 PID 仍存活或不可解析时拒绝，dead PID 自动回收并重建锁。重新启动后编排器正常进入 `waiting_for_collection`；三路并行采集与编排器 PID `39544/37980/31024/25800` 均存活，状态正确列出 `shard1/shard2/shard3` ready 与三个 rem 缺失。测试记录已追加。
- 2026-11-06：DAgger 合并 schema 不一致 FAIL→PASS（systematic-debugging）。采集全部完成后，转换数据统计正确，但首轮 merge 在 shard1 报告 `Dataset schemas differ`；逐字段对比确认数值/动作/OCT schema 完全一致，唯一差异是基准 recovery 集图像为 `[224,224,3]`、新 DAgger 转换为 `[480,640,3]`。原因是旧编排器直接把两类数据合并；失败 merge 只创建了空输出目录并已安全移除。修复为两阶段合并：先合并六份同构 DAgger 640 数据，再用 `prepare_image_cache.py` 生成 224 缓存，最后与 recovery 224 合并。同时修复缓存脚本处理 merge 相对路径 `../` 时可能越出输出目录的问题，改为统一输出 `images/frame_NNNNNNNNN.png` 并重写 `image_paths.json`。隔离测试使用 `../external.png` 输入，成功生成唯一 224 图像和 `images/frame_000000000.png` 安全路径；`prepare_image_cache.py` 与 `auto_train_after_dagger.py` 的 `py_compile` PASS。测试记录已追加。
- 2026-11-06：修复后的自动后处理重启 smoke PASS。使用 `--reuse-converted` 跳过已完成教师标签转换，六份来源正确进入 `merge_dagger_sources_640`，不再触发 recovery 224 与新 DAgger 640 schema 冲突；pipeline PID `40180`，状态 `running`。后续固定为 DAgger 六源合并 → 51,027 帧重采样到 224 → 与 recovery 合并 → guided sidecar → v7 训练。测试记录已追加。
- 2026-11-06：冻结视觉特征缓存加速训练 smoke PASS，首次后台全量缓存因 Windows DataLoader 多进程权限失败后修正。新增 `prepare_vision_feature_cache.py`：从 v6 checkpoint 加载冻结 ResNet，不增强、确定性预处理，将 195,578 训练帧和 29,000 验证帧预计算为 `[N,512]` float16 特征；元数据记录 checkpoint SHA/epoch、帧数和预处理。`RCMDataset` 支持 `use_vision_feature_cache`，可完全跳过图像读取；`SimpleDiffusionPolicy` 训练/验证支持直接使用 `observation.vision_feature`，实时推理仍走原图像编码器；`train_oct_policy.py` 新增 `--vision-feature-cache`。合成 2 帧测试生成 `[2,512]` float16 finite 特征，cached diffusion 前向/反向和采样 finite，cached dataset 不再返回 image，`py_compile` PASS。首次全量后台缓存因 Windows `multiprocessing.Queue CreateFile WinError 5` 失败，未生成 sidecar、未开始训练；根因是 DataLoader `workers=4` 受当前执行环境限制，编排器改为 `workers=0`，isolated feature-cache 样例仍 PASS。测试记录已追加。
- 2026-11-06：视觉特征预计算改为线程池并发 PASS。单进程顺序缓存实测约 `2.7–3.6 s/batch (256)`，完成训练集预计 30+ 分钟；改用已有 `ThreadBatchLoader` 的 12 个图像线程，在 GPU 上继续批量编码，避免 Windows 多进程 Queue 的 `WinError 5`。首次线程 smoke 因 `ThreadBatchLoader` 要求样本为字典而失败，已将图像样本封装为 `{"image": tensor}` 并复跑。16 帧、8 线程隔离测试生成 `[16,512]` float16 finite 特征，耗时 `8.55 s`（包含进程启动和 checkpoint 加载），缓存完整性 PASS；自动编排器已同步改为 `workers=12`。测试记录已追加。
- 2026-11-06：DAgger 后 v7 缓存特征训练完成 PASS（仅训练/离线指标，闭环待完成）。特征缓存 12 线程将 195,578 训练帧的 ResNet 预计算从预计 30+ 分钟降至约 3 分钟；训练阶段验证速度约 `50–75 it/s`。4 epoch normalized MAE 依次为 `0.363706/0.378044/0.314351/0.310172`，最佳 epoch 3，A/B/C MAE=`0.0117792°/0.00643861°/0.00479247 mm`。相对 v6 的 `0.272570` 总体指标变差，原因是 C 轴从 `0.00312630` 退到 `0.00479247 mm`，而 A/B 略有改善；不能据此判断闭环效果。最终 checkpoint 为 `outputs/diffusion_policy_v7_guided_dagger1_cached/best_model.pth`。已启动固定 20 目标的 7/7/6 三路并行闭环评估，seed=`20271105/20271106/20271107`，7 样本中位数、1600 步、30 mm 早停、训练集逐轴动作上限；输出 `outputs/evaluation_v7_guided_dagger1_cached_20261106`。测试记录已追加。
- 2026-11-06：三路评估汇总器实现与历史回归 smoke PASS。新增 `summarize_parallel_evaluations.py`，聚合 shard JSON 中的成功率、>30 mm 失控数、最终误差、每集最低误差及 safety-limited 统计。用 v4 的三份历史评估作 fixture，聚合结果精确复现已知 `20` 集、成功 `0`、失控 `13`、最终误差均值/中位数 `21.5909/30.0215 mm`、每集最低误差均值/中位数 `2.7944/2.5773 mm`、safety-limited 总数 `0`；`py_compile` PASS。v7 三路评估仍在运行，尚无结果。测试记录已追加。
- 2026-11-06：guided v7 DAgger 后固定 20 目标闭环评估完成 PASS（按 DAgger 验收门槛）。三路 `7/7/6` 共 20 个独立目标全部完成评估，无 >30 mm 失控、无 safety-limited 步；严格 20 µm 到达成功 `19/20`（95%），唯一未达标 episode 跑满 1600 步后稳定在 `47.926 µm`，没有发散。最终误差 mean/median/min/max=`11.249/9.920/4.624/47.926 µm`；每集最低误差 mean/median=`10.992/9.920 µm`，19 个成功集的最终误差最大 `14.106 µm`。相对 v4 基线：成功 `0/20→19/20`、>30 mm 失控 `13→0`、最终误差中位数 `30.021 mm→0.00992 mm`、每集最低误差中位数 `2.577 mm→0.00992 mm`。验收门槛“失控 <5/20、最低误差中位数 <1 mm”均通过；严格全 20 集成功仍未达到。结果见 `outputs/evaluation_v7_guided_dagger1_cached_20261106/aggregate_summary.json`。测试记录已追加。
- 2026-11-06：v7 闭环结果统计复核 PASS。成功 `19/20` 的 Wilson 95% 区间为 `76.39%–99.11%`，样本量仅 20，不能把 95% 点估计视为稳定总体成功率。最终误差 P50/P75/P90/P95/max=`9.920/11.396/12.353/15.797/47.926 µm`；19 个成功 episode 步数 mean/median/range=`776.4/724/707–1067`，全部 safety-limited=0。与同 seed、同 7-sample median 配置的 v4 逐 episode 配对，v7 的最终误差和每集最低误差均为 `20/20` 改善，最终误差改善量中位 `30.0109 mm`。唯一失败 episode 后 100 步误差恒为 `47.9261745 µm`（std≈`7e-18`）且无安全限位，属于稳定近目标平台，不是发散；下一步应针对 20–60 µm 的保持/微调、停止头和近目标残差精度，而不是重复同结构大范围 DAgger。测试记录已追加。
- 2026-11-06：20 mm 起始眼内导航条件复核 PASS。读取 v7 三路闭环每个 episode 的首步真实误差，确认本轮 20 个目标全部从约 20 mm 起始，范围 `19.5034–19.9978 mm`、均值 `19.7348 mm`、中位数 `19.7442 mm`，不是近目标重启或挑选容易样本。因此已有 `19/20` 成功、0 失控结果就是“从约 20 mm 进入眼内相机闭环导航”的实验结果。测试记录已追加。
- 2026-11-06：v7 约 20 mm 起始的独立重复眼内闭环评估启动 smoke PASS。为验证首轮 `19/20` 不是 seed 偶然，使用新独立 seed `20291105/20291106/20291107`（与 train/val/test/DAgger seed 分离），三路 `7/7/6` 共 20 目标；模型仍为 v7 epoch 3，眼内相机实时闭环、7-sample median、1600 步、30 mm 早停、训练集逐轴动作上限。三路均成功加载模型并进入 episode 评估，输出 `outputs/evaluation_v7_guided_20mm_repeat_20261106`；当前尚无最终结果。测试记录已追加。
- 2026-11-06：v7 约 20 mm 起始的独立重复闭环实验完成 PASS。新 seed `20291105/20291106/20291107` 共 20 个目标全部从约 20 mm 起始（首步误差范围 `19.5454–19.9436 mm`、均值 `19.7064 mm`、中位数 `19.6764 mm`）；三路均为 `100%`，总计严格 20 µm 成功 `20/20`、>30 mm 失控 `0/20`、safety-limited 步数 `0`。最终误差 mean/median/min/max=`9.840/10.160/7.509/11.298 µm`，每集最低误差 mean/median=`9.783/9.997 µm`，步数 mean/median/range=`736.95/717.5/704–988`。相对上一组独立目标（`19/20`、最终 max `47.926 µm`），重复实验变为 `20/20` 且最差最终误差 `11.298 µm`；两组合计 `39/40`（97.5%），仍限仿真眼内场景而非真实眼球/硬件。结果见 `outputs/evaluation_v7_guided_20mm_repeat_20261106/aggregate_summary.json`。测试记录已追加。
- 2026-11-06：v7 冻结包完整性与上传前校验 PASS。保存 checkpoint `outputs/diffusion_policy_v7_guided_dagger1_cached/best_model.pth`（`60,608,666` bytes，SHA-256 `59C8B186F83CD434B7F09C669FE9CC0F7AF9B8E0C0B8C477B8FDC9E439C2BDEB`）；两组评估目录各含 `7` 个文件，两个 `aggregate_summary.json` 均为 `PASS`，逐文件哈希已记录于 `FREEZE_MANIFEST_20261106.json`。Git 冻结分支基于远端 `main@50b18b6e18cb0d9864cd7fb04b45a5090e14d387`，目标分支为 `freeze/v7-guided-dagger1-20261106`；上传代理为 `127.0.0.1:26800`。24.0 GB 原始 `data/` 与无关历史训练输出不纳入分支。测试记录已追加。
- 2026-11-06：冻结分支远端逐字节校验 FAIL→修复（systematic-debugging）。独立从已推送提交下载后，checkpoint、TEST_LOG 与冻结清单 SHA-256 一致；两组 `aggregate_summary.json` 因 Git 自动文本规范化将本地 CRLF 转为 LF，远端哈希分别为 `D8DDD0E9…`、`CC52C308…`，与本地 `4FCDB919…`、`212C4ABC…` 不一致，字节数各少 `55`。已在 `.gitattributes` 对 `outputs/**` 禁用文本转换并重新登记全部冻结输出，后续将重新提交并复验。测试记录已追加。
- 2026-11-06：冻结分支远端逐字节复验 PASS（systematic-debugging 结案）。通过 `127.0.0.1:26800` 重新读取远端分支与 Raw 内容：ref、本地 HEAD 和远端广告 SHA 均为 `7cb14d3a4d95829b303cf13b70049816f47794a8`；checkpoint 远端/本地 SHA-256 均为 `59C8B186…`，两组汇总远端/本地 SHA-256 分别为 `4FCDB919…` 与 `212C4ABC…`，全部逐字节一致。`.gitattributes` 的 `outputs/** -text` 已消除 CRLF/LF 转换。测试记录已追加。

## 2026-11-06T16:31:02+08:00 | UNKNOWN
<!-- codex-test:b17b7aa01556dc2f6c14d0bc -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/b17b7aa01556dc2f6c14d0bc.log](test-logs/b17b7aa01556dc2f6c14d0bc.log)

### 命令

```text
$py='C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe'; & $py -m unittest -v tests.test_restricted_perception
```

### 输出末尾摘要

```text
test_coarse_bias_is_fixed_per_episode_and_varies_by_seed (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_bias_is_fixed_per_episode_and_varies_by_seed) ... ok
test_coarse_sigma_increases_with_distance (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_sigma_increases_with_distance) ... ok
test_config_round_trip (tests.test_restricted_perception.RestrictedPerceptionTests.test_config_round_trip) ... ok
test_fine_loss_coasts_two_frames_then_falls_back (tests.test_restricted_perception.RestrictedPerceptionTests.test_fine_loss_coasts_two_frames_then_falls_back) ... FAIL
test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction (tests.test_restricted_perception.RestrictedPerceptionTests.test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction) ... ok
test_precision_floor_requires_both_accuracy_and_closed_loop (tests.test_restricted_perception.RestrictedPerceptionTests.test_precision_floor_requires_both_accuracy_and_closed_loop) ... ok
test_switch_gate_rejects_insufficient_reduction (tests.test_restricted_perception.RestrictedPerceptionTests.test_switch_gate_rejects_insufficient_reduction) ... ok
test_truth_is_not_in_policy_features (tests.test_restricted_perception.RestrictedPerceptionTests.test_truth_is_not_in_policy_features) ... ok

======================================================================
FAIL: test_fine_loss_coasts_two_frames_then_falls_back (tests.test_restricted_perception.RestrictedPerceptionTests.test_fine_loss_coasts_two_frames_then_falls_back)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_restricted_perception.py", line 77, in test_fine_loss_coasts_two_frames_then_falls_back
    self.assertEqual(states[0], FallbackState.FINE_TRACKING)
AssertionError: <FallbackState.PRECISION_HOLD: 'PRECISION_HOLD'> != <FallbackState.FINE_TRACKING: 'FINE_TRACKING'>

----------------------------------------------------------------------
Ran 8 tests in 0.045s

FAILED (failures=1)
```

## 2026-11-06T16:36:35+08:00 | UNKNOWN
<!-- codex-test:999f6e56e0a6ed1adc5eede5 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/999f6e56e0a6ed1adc5eede5.log](test-logs/999f6e56e0a6ed1adc5eede5.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception
```

### 输出末尾摘要

```text
test_coarse_bias_is_fixed_per_episode_and_varies_by_seed (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_bias_is_fixed_per_episode_and_varies_by_seed) ... ok
test_coarse_sigma_increases_with_distance (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_sigma_increases_with_distance) ... ok
test_config_round_trip (tests.test_restricted_perception.RestrictedPerceptionTests.test_config_round_trip) ... ok
test_fine_loss_coasts_two_frames_then_falls_back (tests.test_restricted_perception.RestrictedPerceptionTests.test_fine_loss_coasts_two_frames_then_falls_back) ... ok
test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction (tests.test_restricted_perception.RestrictedPerceptionTests.test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction) ... ok
test_precision_floor_requires_both_accuracy_and_closed_loop (tests.test_restricted_perception.RestrictedPerceptionTests.test_precision_floor_requires_both_accuracy_and_closed_loop) ... ok
test_switch_gate_rejects_insufficient_reduction (tests.test_restricted_perception.RestrictedPerceptionTests.test_switch_gate_rejects_insufficient_reduction) ... ok
test_truth_is_not_in_policy_features (tests.test_restricted_perception.RestrictedPerceptionTests.test_truth_is_not_in_policy_features) ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.062s

OK
```

## 2026-11-06 Stage 1 restricted-perception gate PASS

- Normal restricted closed loop: 37/40 successes at the 20 um threshold, target-clustered 95% success interval available in `outputs/restricted_stage1_closed_loop_20261106/stage1_summary.json`.
- Zero restricted vector: 11/40 successes; success drop 65.0 percentage points, exact McNemar p=2.98e-8.
- Same-distance cross-episode mismatched restricted vector: 12/40 successes; success drop 62.5 percentage points, exact McNemar p=4.17e-7.
- Image causal gate PASS. Both zeroing and mismatching the restricted vector significantly degraded success and final-error P95.
- Primary comparison against v4 no-DAgger diffusion: candidate 37/40 (92.5%), baseline 13/40 (32.5%); paired success-risk difference 60.0 percentage points, target-clustered bootstrap 95% CI [47.83, 73.08] percentage points, exact McNemar p=8.05e-7. Final-error P95 improved 79.1%.
- No normal, zero, mismatch, or baseline episode exceeded 30 mm; all safety-limited counts were zero.
- Final summary SHA-256: `303118CC25CE567D487B2ACDBD2CA293209AD77CF4280FF8D6C541F5B5C0BDAD`.
- Fresh tests: 26 passed, 0 failures, 0 errors, exit code 0. Static compile passed. Training and validation sidecar verification both PASS.

## 2026-11-06T16:43:01+08:00 | UNKNOWN
<!-- codex-test:80774e410570fb5b211192e9 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/80774e410570fb5b211192e9.log](test-logs/80774e410570fb5b211192e9.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception
```

### 输出末尾摘要

```text
test_coarse_bias_is_fixed_per_episode_and_varies_by_seed (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_bias_is_fixed_per_episode_and_varies_by_seed) ... ok
test_coarse_sigma_increases_with_distance (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_sigma_increases_with_distance) ... ok
test_config_round_trip (tests.test_restricted_perception.RestrictedPerceptionTests.test_config_round_trip) ... ok
test_fine_loss_coasts_two_frames_then_falls_back (tests.test_restricted_perception.RestrictedPerceptionTests.test_fine_loss_coasts_two_frames_then_falls_back) ... ok
test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction (tests.test_restricted_perception.RestrictedPerceptionTests.test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction) ... ok
test_precision_floor_requires_both_accuracy_and_closed_loop (tests.test_restricted_perception.RestrictedPerceptionTests.test_precision_floor_requires_both_accuracy_and_closed_loop) ... ok
test_switch_gate_rejects_insufficient_reduction (tests.test_restricted_perception.RestrictedPerceptionTests.test_switch_gate_rejects_insufficient_reduction) ... ok
test_truth_is_not_in_policy_features (tests.test_restricted_perception.RestrictedPerceptionTests.test_truth_is_not_in_policy_features) ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.044s

OK
```

## 2026-11-06T16:54:54+08:00 | UNKNOWN
<!-- codex-test:a8d9ace81a3a9bf7fafa4f87 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/a8d9ace81a3a9bf7fafa4f87.log](test-logs/a8d9ace81a3a9bf7fafa4f87.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments
```

### 输出末尾摘要

```text
test_coarse_bias_is_fixed_per_episode_and_varies_by_seed (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_bias_is_fixed_per_episode_and_varies_by_seed) ... ok
test_coarse_sigma_increases_with_distance (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_sigma_increases_with_distance) ... ok
test_config_round_trip (tests.test_restricted_perception.RestrictedPerceptionTests.test_config_round_trip) ... ok
test_fine_loss_coasts_two_frames_then_falls_back (tests.test_restricted_perception.RestrictedPerceptionTests.test_fine_loss_coasts_two_frames_then_falls_back) ... ok
test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction (tests.test_restricted_perception.RestrictedPerceptionTests.test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction) ... ok
test_precision_floor_requires_both_accuracy_and_closed_loop (tests.test_restricted_perception.RestrictedPerceptionTests.test_precision_floor_requires_both_accuracy_and_closed_loop) ... ok
test_switch_gate_rejects_insufficient_reduction (tests.test_restricted_perception.RestrictedPerceptionTests.test_switch_gate_rejects_insufficient_reduction) ... ok
test_truth_is_not_in_policy_features (tests.test_restricted_perception.RestrictedPerceptionTests.test_truth_is_not_in_policy_features) ... ok
test_closed_loop_is_deterministic (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_is_deterministic) ... ok
test_precision_scan_exposes_all_levels (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok

----------------------------------------------------------------------
Ran 11 tests in 5.392s

OK
```

## 2026-11-06T16:56:10+08:00 | UNKNOWN
<!-- codex-test:c8125ee2ac3db589aaad68ba -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/c8125ee2ac3db589aaad68ba.log](test-logs/c8125ee2ac3db589aaad68ba.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments
```

### 输出末尾摘要

```text
test_coarse_bias_is_fixed_per_episode_and_varies_by_seed (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_bias_is_fixed_per_episode_and_varies_by_seed) ... ok
test_coarse_sigma_increases_with_distance (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_sigma_increases_with_distance) ... ok
test_config_round_trip (tests.test_restricted_perception.RestrictedPerceptionTests.test_config_round_trip) ... ok
test_delayed_fine_measurement_is_aligned_to_current_kinematic_tip (tests.test_restricted_perception.RestrictedPerceptionTests.test_delayed_fine_measurement_is_aligned_to_current_kinematic_tip) ... ok
test_fine_loss_coasts_two_frames_then_falls_back (tests.test_restricted_perception.RestrictedPerceptionTests.test_fine_loss_coasts_two_frames_then_falls_back) ... ok
test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction (tests.test_restricted_perception.RestrictedPerceptionTests.test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction) ... ok
test_precision_floor_requires_both_accuracy_and_closed_loop (tests.test_restricted_perception.RestrictedPerceptionTests.test_precision_floor_requires_both_accuracy_and_closed_loop) ... ok
test_switch_gate_rejects_insufficient_reduction (tests.test_restricted_perception.RestrictedPerceptionTests.test_switch_gate_rejects_insufficient_reduction) ... ok
test_truth_is_not_in_policy_features (tests.test_restricted_perception.RestrictedPerceptionTests.test_truth_is_not_in_policy_features) ... ok
test_closed_loop_is_deterministic (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_is_deterministic) ... ok
test_precision_scan_exposes_all_levels (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok

----------------------------------------------------------------------
Ran 12 tests in 5.440s

OK
```

## 2026-11-06T16:57:27+08:00 | UNKNOWN
<!-- codex-test:1311cd653fcf2c19fe32d673 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/1311cd653fcf2c19fe32d673.log](test-logs/1311cd653fcf2c19fe32d673.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments
```

### 输出末尾摘要

```text
ames_then_falls_back (tests.test_restricted_perception.RestrictedPerceptionTests.test_fine_loss_coasts_two_frames_then_falls_back) ... ok
test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction (tests.test_restricted_perception.RestrictedPerceptionTests.test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction) ... ok
test_precision_floor_requires_both_accuracy_and_closed_loop (tests.test_restricted_perception.RestrictedPerceptionTests.test_precision_floor_requires_both_accuracy_and_closed_loop) ... ok
test_switch_gate_rejects_insufficient_reduction (tests.test_restricted_perception.RestrictedPerceptionTests.test_switch_gate_rejects_insufficient_reduction) ... ok
test_truth_is_not_in_policy_features (tests.test_restricted_perception.RestrictedPerceptionTests.test_truth_is_not_in_policy_features) ... ok
test_closed_loop_does_not_stop_on_early_precision_hold (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_does_not_stop_on_early_precision_hold) ... FAIL
test_closed_loop_is_deterministic (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_is_deterministic) ... ok
test_precision_scan_exposes_all_levels (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok

======================================================================
FAIL: test_closed_loop_does_not_stop_on_early_precision_hold (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_does_not_stop_on_early_precision_hold)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_restricted_perception_experiments.py", line 61, in test_closed_loop_does_not_stop_on_early_precision_hold
    self.assertGreaterEqual(report["success_rate"], 0.8)
AssertionError: 0.65 not greater than or equal to 0.8

----------------------------------------------------------------------
Ran 13 tests in 6.854s

FAILED (failures=1)
```

## 2026-11-06T16:57:55+08:00 | UNKNOWN
<!-- codex-test:c11d52260e82cde76f839528 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/c11d52260e82cde76f839528.log](test-logs/c11d52260e82cde76f839528.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments
```

### 输出末尾摘要

```text
ames_then_falls_back (tests.test_restricted_perception.RestrictedPerceptionTests.test_fine_loss_coasts_two_frames_then_falls_back) ... ok
test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction (tests.test_restricted_perception.RestrictedPerceptionTests.test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction) ... ok
test_precision_floor_requires_both_accuracy_and_closed_loop (tests.test_restricted_perception.RestrictedPerceptionTests.test_precision_floor_requires_both_accuracy_and_closed_loop) ... ok
test_switch_gate_rejects_insufficient_reduction (tests.test_restricted_perception.RestrictedPerceptionTests.test_switch_gate_rejects_insufficient_reduction) ... ok
test_truth_is_not_in_policy_features (tests.test_restricted_perception.RestrictedPerceptionTests.test_truth_is_not_in_policy_features) ... ok
test_closed_loop_does_not_stop_on_early_precision_hold (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_does_not_stop_on_early_precision_hold) ... FAIL
test_closed_loop_is_deterministic (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_is_deterministic) ... ok
test_precision_scan_exposes_all_levels (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok

======================================================================
FAIL: test_closed_loop_does_not_stop_on_early_precision_hold (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_does_not_stop_on_early_precision_hold)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_restricted_perception_experiments.py", line 61, in test_closed_loop_does_not_stop_on_early_precision_hold
    self.assertGreaterEqual(report["success_rate"], 0.8)
AssertionError: 0.75 not greater than or equal to 0.8

----------------------------------------------------------------------
Ran 13 tests in 6.433s

FAILED (failures=1)
```

## 2026-11-06T16:59:19+08:00 | UNKNOWN
<!-- codex-test:149e4976ea79f2366b45e565 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/149e4976ea79f2366b45e565.log](test-logs/149e4976ea79f2366b45e565.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments
```

### 输出末尾摘要

```text
test_coarse_bias_is_fixed_per_episode_and_varies_by_seed (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_bias_is_fixed_per_episode_and_varies_by_seed) ... ok
test_coarse_sigma_increases_with_distance (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_sigma_increases_with_distance) ... ok
test_config_round_trip (tests.test_restricted_perception.RestrictedPerceptionTests.test_config_round_trip) ... ok
test_delayed_fine_measurement_is_aligned_to_current_kinematic_tip (tests.test_restricted_perception.RestrictedPerceptionTests.test_delayed_fine_measurement_is_aligned_to_current_kinematic_tip) ... ok
test_fine_loss_coasts_two_frames_then_falls_back (tests.test_restricted_perception.RestrictedPerceptionTests.test_fine_loss_coasts_two_frames_then_falls_back) ... ok
test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction (tests.test_restricted_perception.RestrictedPerceptionTests.test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction) ... ok
test_precision_floor_requires_both_accuracy_and_closed_loop (tests.test_restricted_perception.RestrictedPerceptionTests.test_precision_floor_requires_both_accuracy_and_closed_loop) ... ok
test_switch_gate_rejects_insufficient_reduction (tests.test_restricted_perception.RestrictedPerceptionTests.test_switch_gate_rejects_insufficient_reduction) ... ok
test_truth_is_not_in_policy_features (tests.test_restricted_perception.RestrictedPerceptionTests.test_truth_is_not_in_policy_features) ... ok
test_closed_loop_does_not_stop_on_early_precision_hold (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_does_not_stop_on_early_precision_hold) ... ok
test_closed_loop_is_deterministic (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_is_deterministic) ... ok
test_precision_scan_exposes_all_levels (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok

----------------------------------------------------------------------
Ran 13 tests in 6.401s

OK
```

## 2026-11-06T17:08:24+08:00 | UNKNOWN
<!-- codex-test:fd9afa0eab0fcc59b41a35e2 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/fd9afa0eab0fcc59b41a35e2.log](test-logs/fd9afa0eab0fcc59b41a35e2.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments tests.test_target_split
```

### 输出末尾摘要

```text
ok
test_checked_in_manifest_has_frozen_separation (tests.test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (tests.test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (tests.test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok

----------------------------------------------------------------------
Ran 16 tests in 32.141s

OK
```

## 2026-11-06T17:09:31+08:00 | UNKNOWN
<!-- codex-test:801ac33976fde096346309eb -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/801ac33976fde096346309eb.log](test-logs/801ac33976fde096346309eb.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_sidecar
```

### 输出末尾摘要

```text
.ZipFile(file, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\zipfile.py", line 1337, in __init__
    self.fp = io.open(file, filemode)
              ^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: 'C:\\Users\\Lenovo\\AppData\\Local\\Temp\\tmpd1cithp1\\dataset.npz'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\shutil.py", line 612, in _rmtree_unsafe
    with os.scandir(path) as scandir_it:
         ^^^^^^^^^^^^^^^^
PermissionError: [WinError 5] 拒绝访问。: 'C:\\Users\\Lenovo\\AppData\\Local\\Temp\\tmpd1cithp1'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_restricted_sidecar.py", line 20, in test_sidecar_schema_and_truth_omission
    with tempfile.TemporaryDirectory() as temporary:
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 943, in __exit__
    self.cleanup()
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 947, in cleanup
    self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 929, in _rmtree
    _shutil.rmtree(name, onerror=onerror)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\shutil.py", line 787, in rmtree
    return _rmtree_unsafe(path, onerror)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\shutil.py", line 615, in _rmtree_unsafe
    onerror(os.scandir, path, sys.exc_info())
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 890, in onerror
    _resetperms(path)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 287, in _resetperms
    _dont_follow_symlinks(_os.chmod, path, 0o700)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 278, in _dont_follow_symlinks
    func(path, *args)
PermissionError: [WinError 5] 拒绝访问。: 'C:\\Users\\Lenovo\\AppData\\Local\\Temp\\tmpd1cithp1'

----------------------------------------------------------------------
Ran 1 test in 0.061s

FAILED (errors=1)
```

## 2026-11-06T17:09:40+08:00 | UNKNOWN
<!-- codex-test:2b6a9a3675623fe958f8006a -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/2b6a9a3675623fe958f8006a.log](test-logs/2b6a9a3675623fe958f8006a.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_sidecar
```

### 输出末尾摘要

```text
   ^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: 'E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\\.test-tmp\\tmpyiezt_wy\\dataset.npz'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\shutil.py", line 612, in _rmtree_unsafe
    with os.scandir(path) as scandir_it:
         ^^^^^^^^^^^^^^^^
PermissionError: [WinError 5] 拒绝访问。: 'E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\\.test-tmp\\tmpyiezt_wy'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_restricted_sidecar.py", line 22, in test_sidecar_schema_and_truth_omission
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 943, in __exit__
    self.cleanup()
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 947, in cleanup
    self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 929, in _rmtree
    _shutil.rmtree(name, onerror=onerror)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\shutil.py", line 787, in rmtree
    return _rmtree_unsafe(path, onerror)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\shutil.py", line 615, in _rmtree_unsafe
    onerror(os.scandir, path, sys.exc_info())
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 890, in onerror
    _resetperms(path)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 287, in _resetperms
    _dont_follow_symlinks(_os.chmod, path, 0o700)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 278, in _dont_follow_symlinks
    func(path, *args)
PermissionError: [WinError 5] 拒绝访问。: 'E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\\.test-tmp\\tmpyiezt_wy'

----------------------------------------------------------------------
Ran 1 test in 0.023s

FAILED (errors=1)
```

## 2026-11-06T17:09:51+08:00 | UNKNOWN
<!-- codex-test:3e8162b137cbb478f657a2b9 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/3e8162b137cbb478f657a2b9.log](test-logs/3e8162b137cbb478f657a2b9.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_sidecar
```

### 输出末尾摘要

```text
r: [Errno 13] Permission denied: 'E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\\outputs\\test_artifacts\\tmpt_e8vbre\\dataset.npz'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\shutil.py", line 612, in _rmtree_unsafe
    with os.scandir(path) as scandir_it:
         ^^^^^^^^^^^^^^^^
PermissionError: [WinError 5] 拒绝访问。: 'E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\\outputs\\test_artifacts\\tmpt_e8vbre'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_restricted_sidecar.py", line 22, in test_sidecar_schema_and_truth_omission
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 943, in __exit__
    self.cleanup()
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 947, in cleanup
    self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 929, in _rmtree
    _shutil.rmtree(name, onerror=onerror)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\shutil.py", line 787, in rmtree
    return _rmtree_unsafe(path, onerror)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\shutil.py", line 615, in _rmtree_unsafe
    onerror(os.scandir, path, sys.exc_info())
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 890, in onerror
    _resetperms(path)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 287, in _resetperms
    _dont_follow_symlinks(_os.chmod, path, 0o700)
  File "C:\ProgramData\Miniconda3\envs\rcm_mujoco\Lib\tempfile.py", line 278, in _dont_follow_symlinks
    func(path, *args)
PermissionError: [WinError 5] 拒绝访问。: 'E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\\outputs\\test_artifacts\\tmpt_e8vbre'

----------------------------------------------------------------------
Ran 1 test in 0.030s

FAILED (errors=1)
```

## 2026-11-06T17:10:12+08:00 | UNKNOWN
<!-- codex-test:2ed75d350042d38ab3883c9c -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/2ed75d350042d38ab3883c9c.log](test-logs/2ed75d350042d38ab3883c9c.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_sidecar
```

### 输出末尾摘要

```text
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.558s

OK
```

## 2026-11-06T17:13:22+08:00 | UNKNOWN
<!-- codex-test:79f8c593a94f7aca20a9605b -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/79f8c593a94f7aca20a9605b.log](test-logs/79f8c593a94f7aca20a9605b.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_sidecar
```

### 输出末尾摘要

```text
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... FAIL
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok

======================================================================
FAIL: test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_restricted_sidecar.py", line 30, in test_batched_kinematics_matches_reference
    self.assertTrue(np.allclose(batched, expected, atol=1e-10))
AssertionError: False is not true

----------------------------------------------------------------------
Ran 2 tests in 0.667s

FAILED (failures=1)
```

## 2026-11-06T17:13:52+08:00 | UNKNOWN
<!-- codex-test:f6ae1d840f0f8211aae54ca2 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/f6ae1d840f0f8211aae54ca2.log](test-logs/f6ae1d840f0f8211aae54ca2.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_sidecar
```

### 输出末尾摘要

```text
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.636s

OK
```

## 2026-11-06T17:19:37+08:00 | UNKNOWN
<!-- codex-test:d090c3d66bac9b8351d0ca28 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/d090c3d66bac9b8351d0ca28.log](test-logs/d090c3d66bac9b8351d0ca28.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_sidecar
```

### 输出末尾摘要

```text
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok

----------------------------------------------------------------------
Ran 2 tests in 5.990s

OK
```

## 2026-11-06T17:32:25+08:00 | UNKNOWN
<!-- codex-test:c938bf4b00ae571ec5819855 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/c938bf4b00ae571ec5819855.log](test-logs/c938bf4b00ae571ec5819855.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_sidecar
```

### 输出末尾摘要

```text
�b�~���0 ��
����Nx:� �B�a�s�h�/�S�e�r�v�i�c�e�/�C�r�e�a�t�e�I�n�s�t�a�n�c�e�/�E�_�A�C�C�E�S�S�D�E�N�I�E�D��
�
```

## 2026-11-06T17:32:36+08:00 | UNKNOWN
<!-- codex-test:ca3daba458720aeac054ccaa -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/ca3daba458720aeac054ccaa.log](test-logs/ca3daba458720aeac054ccaa.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_sidecar
```

### 输出末尾摘要

```text
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok

----------------------------------------------------------------------
Ran 2 tests in 5.491s

OK
```

## 2026-11-06T17:34:41+08:00 | UNKNOWN
<!-- codex-test:cb50663c8913dd5198ca7597 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/cb50663c8913dd5198ca7597.log](test-logs/cb50663c8913dd5198ca7597.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_policy tests.test_restricted_sidecar
```

### 输出末尾摘要

```text
test_feature_matrix_shape (tests.test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (tests.test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok

----------------------------------------------------------------------
Ran 4 tests in 1.969s

OK
```

## 2026-11-06T17:35:55+08:00 | UNKNOWN
<!-- codex-test:b1b88e62b4017abb48006ebe -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/b1b88e62b4017abb48006ebe.log](test-logs/b1b88e62b4017abb48006ebe.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_causal_and_degradation tests.test_restricted_policy
```

### 输出末尾摘要

```text
test_counterfactual_contribution_and_gate (tests.test_causal_and_degradation.CausalGateTests.test_counterfactual_contribution_and_gate) ... ok
test_no_effect_is_leakage (tests.test_causal_and_degradation.CausalGateTests.test_no_effect_is_leakage) ... ok
test_one_factor_degradation (tests.test_causal_and_degradation.CausalGateTests.test_one_factor_degradation) ... ok
test_feature_matrix_shape (tests.test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (tests.test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok

----------------------------------------------------------------------
Ran 5 tests in 1.675s

OK
```

## 2026-11-06T17:36:43+08:00 | UNKNOWN
<!-- codex-test:90dd46c9b229f2224be6e328 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/90dd46c9b229f2224be6e328.log](test-logs/90dd46c9b229f2224be6e328.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_policy tests.test_causal_and_degradation tests.test_restricted_sidecar
```

### 输出末尾摘要

```text
test_feature_matrix_shape (tests.test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (tests.test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_source_filter_returns_boolean_mask (tests.test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_counterfactual_contribution_and_gate (tests.test_causal_and_degradation.CausalGateTests.test_counterfactual_contribution_and_gate) ... ok
test_no_effect_is_leakage (tests.test_causal_and_degradation.CausalGateTests.test_no_effect_is_leakage) ... ok
test_one_factor_degradation (tests.test_causal_and_degradation.CausalGateTests.test_one_factor_degradation) ... ok
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok

----------------------------------------------------------------------
Ran 8 tests in 2.279s

OK
```

## 2026-11-06T18:01:31+08:00 | UNKNOWN
<!-- codex-test:bbae22d2b1b2648120e14902 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/bbae22d2b1b2648120e14902.log](test-logs/bbae22d2b1b2648120e14902.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_statistics tests.test_target_split tests.test_causal_and_degradation
```

### 输出末尾摘要

```text
test_pairs_by_target_and_seed (tests.test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok
test_checked_in_manifest_has_frozen_separation (tests.test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (tests.test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (tests.test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok
test_counterfactual_contribution_and_gate (tests.test_causal_and_degradation.CausalGateTests.test_counterfactual_contribution_and_gate) ... ok
test_no_effect_is_leakage (tests.test_causal_and_degradation.CausalGateTests.test_no_effect_is_leakage) ... ok
test_one_factor_degradation (tests.test_causal_and_degradation.CausalGateTests.test_one_factor_degradation) ... ok

----------------------------------------------------------------------
Ran 7 tests in 0.203s

OK
```

## 2026-11-06T18:02:02+08:00 | UNKNOWN
<!-- codex-test:dcc82def592ca4478df0fb8f -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/dcc82def592ca4478df0fb8f.log](test-logs/dcc82def592ca4478df0fb8f.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments tests.test_restricted_sidecar tests.test_restricted_policy tests.test_target_split tests.test_causal_and_degradation tests.test_statistics
```

### 输出末尾摘要

```text
test_closed_loop_does_not_stop_on_early_precision_hold (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_does_not_stop_on_early_precision_hold) ... ok
test_closed_loop_is_deterministic (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_is_deterministic) ... ok
test_precision_scan_exposes_all_levels (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_feature_matrix_shape (tests.test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (tests.test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_source_filter_returns_boolean_mask (tests.test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_checked_in_manifest_has_frozen_separation (tests.test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (tests.test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (tests.test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok
test_counterfactual_contribution_and_gate (tests.test_causal_and_degradation.CausalGateTests.test_counterfactual_contribution_and_gate) ... ok
test_no_effect_is_leakage (tests.test_causal_and_degradation.CausalGateTests.test_no_effect_is_leakage) ... ok
test_one_factor_degradation (tests.test_causal_and_degradation.CausalGateTests.test_one_factor_degradation) ... ok
test_pairs_by_target_and_seed (tests.test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok

----------------------------------------------------------------------
Ran 26 tests in 7.726s

OK
```

## 2026-11-06 restricted perception implementation

- Stage 0 PASS. Report `outputs/restricted_perception_stage0_20261106/stage0_report.json`; Python exit code 0; config SHA-256 `74C966EBE685F0AEAFAA4A2A12932CAF9EC2B4EC7FDF0BE176CF226A200F9FE5`. Switch scan passed at 1/2/3 mm; selected 2 mm switch and 0.2 mm half-width. Fused P95 perception error was 4.00 um, closed-loop success 99.5%/89%/65% at 20/10/5 um, and blended P50 arrival did not exceed hard switching.
- Target split PASS. `configs/target_split_manifest.json` contains 254 training, 59 DAgger, 40 development, and 60 final targets. Final nearest-neighbor minima were 1.000167 mm to final, 1.001074 mm to training, 1.104997 mm to DAgger, and 1.018658 mm to development. The predeclared `z>=0.8` extended posterior region was required because `z>=0.9` admitted no additional 1 mm-separated targets.
- Restricted sidecars PASS. Training sidecar covers 195,578/195,578 frames and 608/608 episodes; validation covers 29,000/29,000 frames and 100/100 episodes. Verification reports confirm row alignment, schema, finite valid measurements, fallback codes, and absence of truth fields.
- Added and exercised one-factor image ladder: full 224, 112, 112+blur 1.5, and 112+blur 1.5+25% center occlusion. Offline near-target zero-vector action-error increase rose from 4.84% to 17.73%; mismatch-vector increase rose from 7.89% to 38.12%. These are offline diagnostics only.
- Closed-loop integration smoke completed on one development target with the real v9 renderer and MotorBackend; 5 steps completed through online restricted perception, frozen image encoding, policy inference, and safety hold. It is not a success-rate result.
- Latest combined test command: 26 tests, 0 failures, 0 errors, exit code 0. Static `compileall` over the new package, scripts, tests, and modified dataset loader passed.
- Stage 1 remains IN_PROGRESS because the 40-target paired closed-loop normal/zero/mismatch gate has not been completed. Stages 2-4 remain blocked and were not represented as passed.

## 2026-11-06T18:03:05+08:00 | UNKNOWN
<!-- codex-test:2ea8441c6b55632034dc469e -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/2ea8441c6b55632034dc469e.log](test-logs/2ea8441c6b55632034dc469e.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments tests.test_restricted_sidecar tests.test_restricted_policy tests.test_target_split tests.test_causal_and_degradation tests.test_statistics
```

### 输出末尾摘要

```text
test_closed_loop_does_not_stop_on_early_precision_hold (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_does_not_stop_on_early_precision_hold) ... ok
test_closed_loop_is_deterministic (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_is_deterministic) ... ok
test_precision_scan_exposes_all_levels (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_feature_matrix_shape (tests.test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (tests.test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_source_filter_returns_boolean_mask (tests.test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_checked_in_manifest_has_frozen_separation (tests.test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (tests.test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (tests.test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok
test_counterfactual_contribution_and_gate (tests.test_causal_and_degradation.CausalGateTests.test_counterfactual_contribution_and_gate) ... ok
test_no_effect_is_leakage (tests.test_causal_and_degradation.CausalGateTests.test_no_effect_is_leakage) ... ok
test_one_factor_degradation (tests.test_causal_and_degradation.CausalGateTests.test_one_factor_degradation) ... ok
test_pairs_by_target_and_seed (tests.test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok

----------------------------------------------------------------------
Ran 26 tests in 6.713s

OK
```

## 2026-11-06T18:33:24+08:00 | UNKNOWN
<!-- codex-test:b1a47d78affd2f3355ab4490 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/b1a47d78affd2f3355ab4490.log](test-logs/b1a47d78affd2f3355ab4490.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_causal_and_degradation tests.test_statistics
```

### 输出末尾摘要

```text
<3>WSL (11 - Relay) ERROR: CreateProcessCommon:735: execvpe(/bin/bash) failed: No such file or directory
```

## 2026-11-06T18:33:34+08:00 | UNKNOWN
<!-- codex-test:d36c92723c90ed8013181ee1 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/d36c92723c90ed8013181ee1.log](test-logs/d36c92723c90ed8013181ee1.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_causal_and_degradation tests.test_statistics
```

### 输出末尾摘要

```text
test_counterfactual_contribution_and_gate (tests.test_causal_and_degradation.CausalGateTests.test_counterfactual_contribution_and_gate) ... ok
test_no_effect_is_leakage (tests.test_causal_and_degradation.CausalGateTests.test_no_effect_is_leakage) ... ok
test_one_factor_degradation (tests.test_causal_and_degradation.CausalGateTests.test_one_factor_degradation) ... ok
test_pairs_by_target_and_seed (tests.test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok

----------------------------------------------------------------------
Ran 4 tests in 2.966s

OK
```

## 2026-11-06T19:42:49+08:00 | UNKNOWN
<!-- codex-test:5bd5302cd74971105a6ec781 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/5bd5302cd74971105a6ec781.log](test-logs/5bd5302cd74971105a6ec781.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments tests.test_restricted_sidecar tests.test_restricted_policy tests.test_target_split tests.test_causal_and_degradation tests.test_statistics
```

### 输出末尾摘要

```text
test_closed_loop_does_not_stop_on_early_precision_hold (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_does_not_stop_on_early_precision_hold) ... ok
test_closed_loop_is_deterministic (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_is_deterministic) ... ok
test_precision_scan_exposes_all_levels (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_feature_matrix_shape (tests.test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (tests.test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_source_filter_returns_boolean_mask (tests.test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_checked_in_manifest_has_frozen_separation (tests.test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (tests.test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (tests.test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok
test_counterfactual_contribution_and_gate (tests.test_causal_and_degradation.CausalGateTests.test_counterfactual_contribution_and_gate) ... ok
test_no_effect_is_leakage (tests.test_causal_and_degradation.CausalGateTests.test_no_effect_is_leakage) ... ok
test_one_factor_degradation (tests.test_causal_and_degradation.CausalGateTests.test_one_factor_degradation) ... ok
test_pairs_by_target_and_seed (tests.test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok

----------------------------------------------------------------------
Ran 26 tests in 8.155s

OK
```

## 2026-11-06 Stage 1 restricted-perception gate PASS

- Closed-loop paired evaluation completed on all 40 development targets with a 100 um initial error and a 200-frame budget.
- Normal restricted candidate: 37/40 successes (92.5%), final-error P95 22.97 um.
- Zero restricted vector: 11/40 successes (27.5%), final-error P95 228.77 um; success drop 65.0 percentage points, exact McNemar p=2.98e-8.
- Same-distance cross-episode mismatched vector: 12/40 successes (30.0%), final-error P95 294.51 um; success drop 62.5 percentage points, exact McNemar p=4.17e-7.
- Image causal gate PASS: both zeroing and mismatching the restricted vector produced significant paired success and final-error degradation.
- Primary v4 no-DAgger baseline: 13/40 successes (32.5%), final-error P95 110.00 um. Candidate-minus-baseline success difference was 60.0 percentage points, target-clustered bootstrap 95% CI [47.83, 73.08], exact McNemar p=8.05e-7; final-error P95 improved 79.1%.
- No normal, zero, mismatch, or baseline episode exceeded 30 mm; all safety-limited counts were zero.
- `stage1_summary.json` SHA-256: `303118CC25CE567D487B2ACDBD2CA293209AD77CF4280FF8D6C541F5B5C0BDAD`.
- Final unit suite: 26/26 PASS, exit code 0. Static compile PASS. Train and validation sidecar verification PASS.

## 2026-11-06 Stage 2 restricted DAgger result

- Collected 120 episodes (2,160 frames) over 60 fresh training targets with two starts per target in the 20-200 um range. One episode had insufficient teacher-valid frames; all recorded files were retained.
- DAgger-only: 38/40 successes, final-error P95 20.00 um.
- DAgger plus velocity: 40/40 successes, final-error P95 11.99 um.
- DAgger plus velocity plus distance-scaled residual cap: 39/40 successes, final-error P95 14.29 um.
- DAgger plus velocity plus residual cap plus stop head: 26/40 successes, final-error P95 731.01 um.
- All variants had zero >30 mm uncontrolled episodes and zero safety-limited steps.
- Stage 2 result: FAIL. Velocity input improved the policy, but the residual cap failed to retain the gain and the stop head caused severe false stopping. Stages 3-4 remain blocked.

## 2026-11-06 Stage 2 corrected 20 um residual-scale result

- Corrected the distance scale from 1 mm to 20 um and moved action clipping to deployment-time inference, avoiding zero gradients during training.
- Corrected residual-cap variant: 40/40 successes, final-error P50/P95 6.75/13.06 um, zero >30 mm and zero safety-limited steps.
- Corrected stop-head variant: 40/40 successes, final-error P50/P95 11.33/15.51 um, zero >30 mm and zero safety-limited steps.
- Velocity-only remains selected: 40/40, final-error P95 11.99 um.
- Paired against velocity-only, residual cap changes mean error by +1.07 um and stop head by +4.94 um; the corresponding 95% bootstrap intervals exclude improvement.
- Stage 2 result: PASS with `velocity` selected. Residual cap and stop head are recorded as rejected no-accuracy-gain features. Stage 3 is ready.
- Summary SHA-256: `546F21DF7D68A504C9DDF6D650A495515F45E21F7B1C43E1D1B0424C591E7A81`.

## 2026-11-06 Stage 2 final 10 um residual-scale retest

- Changed the residual-cap distance scale from 20 um to 10 um and retrained the residual-cap and stop-head variants.
- Residual-cap variant: 40/40 successes, final-error P50/P95 6.75/13.06 um.
- Residual-cap plus stop-head variant: 40/40 successes, final-error P50/P95 11.33/15.51 um.
- Both variants had zero >30 mm uncontrolled episodes and zero safety-limited steps.
- Results are unchanged from the 20 um retest; the stop head acts before the cap in the tested trajectories.
- Velocity-only remains selected at 40/40 with final-error P95 11.99 um. Residual cap and stop head remain rejected because paired mean errors increased by 1.07 um and 4.94 um respectively.
- Final 10 um summary SHA-256: `3554DDB2104B02E98A2D33FB362E4489D886242E9FD35F51EAD99458C2BAD240`.

## 2026-11-06T21:24:32+08:00 | UNKNOWN
<!-- codex-test:1dacd897b50e10345e877df9 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/1dacd897b50e10345e877df9.log](test-logs/1dacd897b50e10345e877df9.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_policy tests.test_causal_and_degradation tests.test_statistics
```

### 输出末尾摘要

```text
test_feature_matrix_shape (tests.test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (tests.test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_residual_cap_uses_twenty_micrometre_scale (tests.test_restricted_policy.RestrictedPolicyTests.test_residual_cap_uses_twenty_micrometre_scale) ... ok
test_source_filter_returns_boolean_mask (tests.test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_counterfactual_contribution_and_gate (tests.test_causal_and_degradation.CausalGateTests.test_counterfactual_contribution_and_gate) ... ok
test_no_effect_is_leakage (tests.test_causal_and_degradation.CausalGateTests.test_no_effect_is_leakage) ... ok
test_one_factor_degradation (tests.test_causal_and_degradation.CausalGateTests.test_one_factor_degradation) ... ok
test_pairs_by_target_and_seed (tests.test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok

----------------------------------------------------------------------
Ran 8 tests in 2.188s

OK
```

## 2026-11-06T21:36:35+08:00 | UNKNOWN
<!-- codex-test:2b1d413bf6ebad9570803509 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/2b1d413bf6ebad9570803509.log](test-logs/2b1d413bf6ebad9570803509.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments tests.test_restricted_sidecar tests.test_restricted_policy tests.test_target_split tests.test_causal_and_degradation tests.test_statistics
```

### 输出末尾摘要

```text
oes_not_stop_on_early_precision_hold) ... ok
test_closed_loop_is_deterministic (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_is_deterministic) ... ok
test_precision_scan_exposes_all_levels (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_feature_matrix_shape (tests.test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (tests.test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_residual_cap_uses_twenty_micrometre_scale (tests.test_restricted_policy.RestrictedPolicyTests.test_residual_cap_uses_twenty_micrometre_scale) ... ok
test_source_filter_returns_boolean_mask (tests.test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_checked_in_manifest_has_frozen_separation (tests.test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (tests.test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (tests.test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok
test_counterfactual_contribution_and_gate (tests.test_causal_and_degradation.CausalGateTests.test_counterfactual_contribution_and_gate) ... ok
test_no_effect_is_leakage (tests.test_causal_and_degradation.CausalGateTests.test_no_effect_is_leakage) ... ok
test_one_factor_degradation (tests.test_causal_and_degradation.CausalGateTests.test_one_factor_degradation) ... ok
test_pairs_by_target_and_seed (tests.test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok

----------------------------------------------------------------------
Ran 27 tests in 8.175s

OK
```

## 2026-11-06T21:39:38+08:00 | UNKNOWN
<!-- codex-test:80a5fd9a724b1ec105de29c8 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/80a5fd9a724b1ec105de29c8.log](test-logs/80a5fd9a724b1ec105de29c8.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_policy tests.test_causal_and_degradation tests.test_statistics
```

### 输出末尾摘要

```text
test_feature_matrix_shape (tests.test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (tests.test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_residual_cap_uses_ten_micrometre_scale (tests.test_restricted_policy.RestrictedPolicyTests.test_residual_cap_uses_ten_micrometre_scale) ... ok
test_source_filter_returns_boolean_mask (tests.test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_counterfactual_contribution_and_gate (tests.test_causal_and_degradation.CausalGateTests.test_counterfactual_contribution_and_gate) ... ok
test_no_effect_is_leakage (tests.test_causal_and_degradation.CausalGateTests.test_no_effect_is_leakage) ... ok
test_one_factor_degradation (tests.test_causal_and_degradation.CausalGateTests.test_one_factor_degradation) ... ok
test_pairs_by_target_and_seed (tests.test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok

----------------------------------------------------------------------
Ran 8 tests in 2.165s

OK
```

## 2026-11-06T21:51:24+08:00 | UNKNOWN
<!-- codex-test:cc154c56c84d65a83b195fb2 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/cc154c56c84d65a83b195fb2.log](test-logs/cc154c56c84d65a83b195fb2.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception tests.test_restricted_perception_experiments tests.test_restricted_sidecar tests.test_restricted_policy tests.test_target_split tests.test_causal_and_degradation tests.test_statistics
```

### 输出末尾摘要

```text
loop_does_not_stop_on_early_precision_hold) ... ok
test_closed_loop_is_deterministic (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_is_deterministic) ... ok
test_precision_scan_exposes_all_levels (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_feature_matrix_shape (tests.test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (tests.test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_residual_cap_uses_ten_micrometre_scale (tests.test_restricted_policy.RestrictedPolicyTests.test_residual_cap_uses_ten_micrometre_scale) ... ok
test_source_filter_returns_boolean_mask (tests.test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_checked_in_manifest_has_frozen_separation (tests.test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (tests.test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (tests.test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok
test_counterfactual_contribution_and_gate (tests.test_causal_and_degradation.CausalGateTests.test_counterfactual_contribution_and_gate) ... ok
test_no_effect_is_leakage (tests.test_causal_and_degradation.CausalGateTests.test_no_effect_is_leakage) ... ok
test_one_factor_degradation (tests.test_causal_and_degradation.CausalGateTests.test_one_factor_degradation) ... ok
test_pairs_by_target_and_seed (tests.test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok

----------------------------------------------------------------------
Ran 27 tests in 8.124s

OK
```

## 2026-11-06T22:14:34+08:00 | UNKNOWN
<!-- codex-test:1dbe18c0e9d7f1b4f4e615de -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/1dbe18c0e9d7f1b4f4e615de.log](test-logs/1dbe18c0e9d7f1b4f4e615de.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_restricted_perception
```

### 输出末尾摘要

```text
test_coarse_bias_is_fixed_per_episode_and_varies_by_seed (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_bias_is_fixed_per_episode_and_varies_by_seed) ... ok
test_coarse_sigma_increases_with_distance (tests.test_restricted_perception.RestrictedPerceptionTests.test_coarse_sigma_increases_with_distance) ... ok
test_config_round_trip (tests.test_restricted_perception.RestrictedPerceptionTests.test_config_round_trip) ... ok
test_delayed_fine_measurement_is_aligned_to_current_kinematic_tip (tests.test_restricted_perception.RestrictedPerceptionTests.test_delayed_fine_measurement_is_aligned_to_current_kinematic_tip) ... ok
test_extrinsic_matches_rcm_scene_v9_optical_camera_snapshot (tests.test_restricted_perception.RestrictedPerceptionTests.test_extrinsic_matches_rcm_scene_v9_optical_camera_snapshot) ... ok
test_fine_loss_coasts_two_frames_then_falls_back (tests.test_restricted_perception.RestrictedPerceptionTests.test_fine_loss_coasts_two_frames_then_falls_back) ... ok
test_fine_optical_vector_rotates_back_to_target_minus_tip (tests.test_restricted_perception.RestrictedPerceptionTests.test_fine_optical_vector_rotates_back_to_target_minus_tip) ... ok
test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction (tests.test_restricted_perception.RestrictedPerceptionTests.test_imprecise_coarse_fallback_holds_then_requests_bounded_retraction) ... ok
test_precision_floor_requires_both_accuracy_and_closed_loop (tests.test_restricted_perception.RestrictedPerceptionTests.test_precision_floor_requires_both_accuracy_and_closed_loop) ... ok
test_switch_gate_rejects_insufficient_reduction (tests.test_restricted_perception.RestrictedPerceptionTests.test_switch_gate_rejects_insufficient_reduction) ... ok
test_truth_is_not_in_policy_features (tests.test_restricted_perception.RestrictedPerceptionTests.test_truth_is_not_in_policy_features) ... ok

----------------------------------------------------------------------
Ran 11 tests in 3.694s

OK
```

## 2026-11-06T22:38:43+08:00 | UNKNOWN
<!-- codex-test:94143305470d113bca64e308 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/94143305470d113bca64e308.log](test-logs/94143305470d113bca64e308.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_stage34
```

### 输出末尾摘要

```text
<3>WSL (11 - Relay) ERROR: CreateProcessCommon:735: execvpe(/bin/bash) failed: No such file or directory
```

## 2026-11-06T22:38:51+08:00 | UNKNOWN
<!-- codex-test:040a48ae52f76610d3d8ea02 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/040a48ae52f76610d3d8ea02.log](test-logs/040a48ae52f76610d3d8ea02.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_stage34
```

### 输出末尾摘要

```text
test_geometric_action_moves_toward_target_minus_tip (tests.test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... FAIL
test_perturbations_change_one_sensor_factor (tests.test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (tests.test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok

======================================================================
FAIL: test_geometric_action_moves_toward_target_minus_tip (tests.test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_stage34.py", line 86, in test_geometric_action_moves_toward_target_minus_tip
    self.assertGreater(backward[2], 0.0)
AssertionError: np.float64(-0.014569245205752127) not greater than 0.0

----------------------------------------------------------------------
Ran 3 tests in 0.507s

FAILED (failures=1)
```

## 2026-11-06T22:39:03+08:00 | UNKNOWN
<!-- codex-test:27f67b93c790b04614f8ec8a -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/27f67b93c790b04614f8ec8a.log](test-logs/27f67b93c790b04614f8ec8a.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_stage34
```

### 输出末尾摘要

```text
test_geometric_action_moves_toward_target_minus_tip (tests.test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (tests.test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (tests.test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.328s

OK
```

## 2026-11-07T03:41:26+08:00 | UNKNOWN
<!-- codex-test:5aa4d3b5d27367405b5df226 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/5aa4d3b5d27367405b5df226.log](test-logs/5aa4d3b5d27367405b5df226.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_stage3_policy tests.test_stage34
```

### 输出末尾摘要

```text
test_residual_is_bounded (tests.test_stage3_policy.Stage3PolicyTests.test_residual_is_bounded) ... ok
test_zero_initialized_residual_preserves_base_action (tests.test_stage3_policy.Stage3PolicyTests.test_zero_initialized_residual_preserves_base_action) ... ok
test_geometric_action_moves_toward_target_minus_tip (tests.test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (tests.test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (tests.test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.242s

OK
```

## 2026-11-07T07:21:27+08:00 | UNKNOWN
<!-- codex-test:0c17e4d411d1afda6b2adf70 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/0c17e4d411d1afda6b2adf70.log](test-logs/0c17e4d411d1afda6b2adf70.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest discover -s tests -v
```

### 输出末尾摘要

```text
icrometre_scale) ... ok
test_source_filter_returns_boolean_mask (test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_batched_kinematics_matches_reference (test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_geometric_action_moves_toward_target_minus_tip (test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok
test_residual_is_bounded (test_stage3_policy.Stage3PolicyTests.test_residual_is_bounded) ... ok
test_zero_initialized_residual_preserves_base_action (test_stage3_policy.Stage3PolicyTests.test_zero_initialized_residual_preserves_base_action) ... ok
test_pairs_by_target_and_seed (test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok
test_checked_in_manifest_has_frozen_separation (test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok

======================================================================
FAIL: test_closed_loop_does_not_stop_on_early_precision_hold (test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_does_not_stop_on_early_precision_hold)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_restricted_perception_experiments.py", line 62, in test_closed_loop_does_not_stop_on_early_precision_hold
    self.assertLess(report["final_error_p95_um"], 100.0)
AssertionError: 114.75082182071675 not less than 100.0

----------------------------------------------------------------------
Ran 33 tests in 12.048s

FAILED (failures=1)
```

## 2026-11-07T07:22:20+08:00 | UNKNOWN
<!-- codex-test:984c719f9bc076071c5d7910 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/984c719f9bc076071c5d7910.log](test-logs/984c719f9bc076071c5d7910.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m py_compile restricted_perception\stage34.py restricted_perception\stage3_policy.py run_stage3_robustness.py run_stage3_legacy_baseline.py collect_stage3_distillation.py finalize_stage3_distillation.py train_stage3_distillation.py train_stage3_residual_distillation.py run_stage4_distilled.py run_stage4_median_control.py summarize_stage1_corrected.py summarize_stage2_corrected.py summarize_stage3.py summarize_stage4.py benchmark_stage3_latency.py train_restricted_policy.py train_stage2_variants.py evaluate_restricted_policy.py lerobot_training\eval_policy.py tests\test_stage3_policy.py tests\test_stage34.py
```

### 输出末尾摘要

```text
(无输出)
```

## 2026-11-07T07:22:39+08:00 | UNKNOWN
<!-- codex-test:6242442ecce3944812ef3f9d -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/6242442ecce3944812ef3f9d.log](test-logs/6242442ecce3944812ef3f9d.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest discover -s tests -v
```

### 输出末尾摘要

```text
ministic) ... ok
test_precision_scan_exposes_all_levels (test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok
test_feature_matrix_shape (test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_residual_cap_uses_ten_micrometre_scale (test_restricted_policy.RestrictedPolicyTests.test_residual_cap_uses_ten_micrometre_scale) ... ok
test_source_filter_returns_boolean_mask (test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_batched_kinematics_matches_reference (test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_geometric_action_moves_toward_target_minus_tip (test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok
test_residual_is_bounded (test_stage3_policy.Stage3PolicyTests.test_residual_is_bounded) ... ok
test_zero_initialized_residual_preserves_base_action (test_stage3_policy.Stage3PolicyTests.test_zero_initialized_residual_preserves_base_action) ... ok
test_pairs_by_target_and_seed (test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok
test_checked_in_manifest_has_frozen_separation (test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok

----------------------------------------------------------------------
Ran 33 tests in 11.910s

OK
```

## 2026-11-07T19:54:07+08:00 | UNKNOWN
<!-- codex-test:48d937c5737c2f395aef0a9e -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/48d937c5737c2f395aef0a9e.log](test-logs/48d937c5737c2f395aef0a9e.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_stage4_analysis tests.test_stage34 tests.test_stage3_policy
```

### 输出末尾摘要

```text
test_failure_classification_separates_hold_and_initialization (tests.test_stage4_analysis.Stage4AnalysisTests.test_failure_classification_separates_hold_and_initialization) ... ok
test_value_summary_uses_requested_population_statistics (tests.test_stage4_analysis.Stage4AnalysisTests.test_value_summary_uses_requested_population_statistics) ... ok
test_geometric_action_moves_toward_target_minus_tip (tests.test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (tests.test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (tests.test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok
test_residual_is_bounded (tests.test_stage3_policy.Stage3PolicyTests.test_residual_is_bounded) ... ok
test_zero_initialized_residual_preserves_base_action (tests.test_stage3_policy.Stage3PolicyTests.test_zero_initialized_residual_preserves_base_action) ... ok

----------------------------------------------------------------------
Ran 7 tests in 0.386s

OK
```

## 2026-11-07T19:54:49+08:00 | UNKNOWN
<!-- codex-test:2fa2531b7f26ecc4a73e22df -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/2fa2531b7f26ecc4a73e22df.log](test-logs/2fa2531b7f26ecc4a73e22df.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m py_compile restricted_perception\stage4_analysis.py analyze_stage4_failure_modes.py tests\test_stage4_analysis.py
```

### 输出末尾摘要

```text
(无输出)
```

## 2026-11-07T19:55:09+08:00 | UNKNOWN
<!-- codex-test:b0a5e1f3bd30dfcda1dcc67c -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/b0a5e1f3bd30dfcda1dcc67c.log](test-logs/b0a5e1f3bd30dfcda1dcc67c.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest discover -s tests -v
```

### 输出末尾摘要

```text
stances_and_gate) ... ok
test_feature_matrix_shape (test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_residual_cap_uses_ten_micrometre_scale (test_restricted_policy.RestrictedPolicyTests.test_residual_cap_uses_ten_micrometre_scale) ... ok
test_source_filter_returns_boolean_mask (test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_batched_kinematics_matches_reference (test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_geometric_action_moves_toward_target_minus_tip (test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok
test_residual_is_bounded (test_stage3_policy.Stage3PolicyTests.test_residual_is_bounded) ... ok
test_zero_initialized_residual_preserves_base_action (test_stage3_policy.Stage3PolicyTests.test_zero_initialized_residual_preserves_base_action) ... ok
test_failure_classification_separates_hold_and_initialization (test_stage4_analysis.Stage4AnalysisTests.test_failure_classification_separates_hold_and_initialization) ... ok
test_value_summary_uses_requested_population_statistics (test_stage4_analysis.Stage4AnalysisTests.test_value_summary_uses_requested_population_statistics) ... ok
test_pairs_by_target_and_seed (test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok
test_checked_in_manifest_has_frozen_separation (test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok

----------------------------------------------------------------------
Ran 35 tests in 14.069s

OK
```

## 2026-11-07T19:59:18+08:00 | UNKNOWN
<!-- codex-test:12018d365b1f2aae9f92191c -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/12018d365b1f2aae9f92191c.log](test-logs/12018d365b1f2aae9f92191c.log)

### 命令

```text
$before=(Get-Item -LiteralPath 'TEST_LOG.md').Length; Write-Output "BEFORE=$before"; $env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest -v tests.test_stage4_analysis.Stage4AnalysisTests.test_value_summary_uses_requested_population_statistics; $code=$LASTEXITCODE; Start-Sleep -Seconds 2; $after=(Get-Item -LiteralPath 'TEST_LOG.md').Length; Write-Output "EXIT=$code"; Write-Output "AFTER=$after"; Write-Output "DELTA=$($after-$before)"
```

### 输出末尾摘要

```text
BEFORE=289134
test_value_summary_uses_requested_population_statistics (tests.test_stage4_analysis.Stage4AnalysisTests.test_value_summary_uses_requested_population_statistics) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.012s

OK
EXIT=0
AFTER=289134
DELTA=0
```

## 2026-11-07T20:00:01+08:00 | SUCCESS
<!-- codex-test:91f56231f8ffe44a2885da44 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：0
- 输出记录：[test-logs/91f56231f8ffe44a2885da44.log](test-logs/91f56231f8ffe44a2885da44.log)

### 命令

```text
Write-Output "# codex-test manual hook probe"
```

### 输出末尾摘要

```text
manual probe completed
```

## 2026-11-07T20:00:03+08:00 | UNKNOWN
<!-- codex-test:028f7678543463a0730b8343 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/028f7678543463a0730b8343.log](test-logs/028f7678543463a0730b8343.log)

### 命令

```text
$before=(Get-Item -LiteralPath 'TEST_LOG.md').Length; $payload=@{session_id='manual-hook-probe';tool_use_id='manual-hook-probe-001';cwd=(Get-Location).Path;tool_input=@{command='Write-Output "# codex-test manual hook probe"';workdir=(Get-Location).Path};tool_response=@{exit_code=0;output='manual probe completed'}} | ConvertTo-Json -Depth 8 -Compress; $payload | py -3.11 -X utf8 'C:\Users\Lenovo\.codex\hooks\research_hook.py' test-log; $hookExit=$LASTEXITCODE; Start-Sleep -Milliseconds 500; $after=(Get-Item -LiteralPath 'TEST_LOG.md').Length; Write-Output "HOOK_EXIT=$hookExit BEFORE=$before AFTER=$after DELTA=$($after-$before)"; Get-Content -LiteralPath 'TEST_LOG.md' -Tail 25
```

### 输出末尾摘要

````text
HOOK_EXIT=0 BEFORE=290383 AFTER=290872 DELTA=489
OK
EXIT=0
AFTER=289134
DELTA=0
```

## 2026-11-07T20:00:01+08:00 | SUCCESS
<!-- codex-test:91f56231f8ffe44a2885da44 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：0
- 输出记录：[test-logs/91f56231f8ffe44a2885da44.log](test-logs/91f56231f8ffe44a2885da44.log)

### 命令

```text
Write-Output "# codex-test manual hook probe"
```

### 输出末尾摘要

```text
manual probe completed
```
````

## 2026-11-07T21:20:16+08:00 | SUCCESS
<!-- codex-test:pptmaster-20261107-212016 -->

- 来源：Manual verification record
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：0
- 输出记录：[rcm 11p final quality report](projects/rcm_11p_technical_review_ppt169_20261107/validation/final_quality_report.md)

### 命令

```text
py -3.11 svg_quality_checker.py projects/rcm_11p_technical_review_ppt169_20261107 --canonical-authoring --stage final --json
py -3.11 project_manager.py validate projects/rcm_11p_technical_review_ppt169_20261107
PowerPoint COM export: 11 slides -> 1600x900 PNG
python-pptx readback: slide count + per-slide notes
py -3.11 svg_to_pptx.py projects/rcm_11p_technical_review_ppt169_20261107 --no-animations
```

### 输出末尾摘要

```text
Final SVG quality gate: 11 files, blocking errors=0, advisory warning files=6
Project validation: complete
Charts: 3/3 horizontal bars coordinate-verified against declared plot areas
Speaker notes: 11/11 non-empty
PPTX postflight: passed-with-warnings, zip integrity=passed, slides=11, notes=11
Render review: 11/11 pages at 1600x900; P07 and P10 overlap issues repaired before final export
Earlier export dependency failure: ModuleNotFoundError pptx; minimal fix installed python-pptx; import and export then passed
```

## 2026-11-08T15:45:53+08:00 | UNKNOWN
<!-- codex-test:69cc96542d01c33535922c25 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/69cc96542d01c33535922c25.log](test-logs/69cc96542d01c33535922c25.log)

### 命令

```text
$env:PYTHONPATH='.'; & 'C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe' -m unittest discover -s tests -v
```

### 输出末尾摘要

```text
stances_and_gate) ... ok
test_feature_matrix_shape (test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_residual_cap_uses_ten_micrometre_scale (test_restricted_policy.RestrictedPolicyTests.test_residual_cap_uses_ten_micrometre_scale) ... ok
test_source_filter_returns_boolean_mask (test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_batched_kinematics_matches_reference (test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_geometric_action_moves_toward_target_minus_tip (test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok
test_residual_is_bounded (test_stage3_policy.Stage3PolicyTests.test_residual_is_bounded) ... ok
test_zero_initialized_residual_preserves_base_action (test_stage3_policy.Stage3PolicyTests.test_zero_initialized_residual_preserves_base_action) ... ok
test_failure_classification_separates_hold_and_initialization (test_stage4_analysis.Stage4AnalysisTests.test_failure_classification_separates_hold_and_initialization) ... ok
test_value_summary_uses_requested_population_statistics (test_stage4_analysis.Stage4AnalysisTests.test_value_summary_uses_requested_population_statistics) ... ok
test_pairs_by_target_and_seed (test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok
test_checked_in_manifest_has_frozen_separation (test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok

----------------------------------------------------------------------
Ran 35 tests in 23.462s

OK
```

## 2026-11-08 最近实验补记：受限感知、Stage 0–4 与失败分析

本节补齐 2026-11-06 至 2026-11-07 已产生正式产物但尚未形成完整审计结论的实验。内容依据 machine-readable JSON、阶段总结、失败分析报告及现有分片日志整理；除本节末尾的单元测试外，没有重新执行昂贵的闭环或训练实验，因此不将历史产物冒充本次新运行结果。

### 关键外参缺陷与重跑范围

- 发现并修正一项影响 Stage 0–2 的外参缺陷：早期受限感知配置使用了 legacy `Reference` 相机位姿，而实际 RGB-D/OCT 闭环观测使用 `RCMSceneV9.eye_internal`；两套位姿矩阵最大差异为 `1.091838353470798`。修正后重建训练/验证 sidecar、重训 Stage 1，并重新采集 120 条 Stage 2 DAgger episode。
- 修正后的 Stage 0–4 状态记录在 `outputs/restricted_experiment_status_20261107.json`：Stage 0/1/2/3 为 PASS，Stage 4 评估为 COMPLETE、性能为 FAIL，`deployment_claim_supported=false`。审计确认未使用冻结 final-test targets 参与训练。
- 当前状态文件 SHA-256 为 `EEA80478E1C1C27B03C2A68EDAF377DF1D6D18D1C354CC32D5170E2FA44A26EF`。

### Stage 0：外参修正后的感知与控制门限

- 正式报告：`outputs/restricted_perception_stage0_v2_20261106/stage0_report.json`，结果 PASS，报告 SHA-256 `D5030A529EE5B9A151187927328924212931FA8C5B28AE94E8857575D733FB06`。
- 配置 SHA-256 为 `02B2FD1E4193BDD0FDF9771318D9B356288D2366406772443B24F1A698611464`。选定粗/细切换距离 `2 mm`、过渡半宽 `0.2 mm`、精度门限 `20 µm`。
- 在 `1/2/3 mm` 切换距离下，continuous blending 相对 hard switching 的事件级 P95 步长降幅约为 `90.6%–91.7%`，达到 `>=80%` 要求。
- 近目标 fused P95 感知误差为 `3.7323280031177974 µm`；确定性闭环成功率在 `20/10/5 µm` 分别为 `96.5% / 85.0% / 64.5%`，因此只有 `20 µm` 同时通过感知精度和闭环成功率门限。
- Blended P50 到达/停止时间为 `7.475 s`，hard switching 为 `7.75 s`，增加量为 `-0.275 s`，通过 `+0.5 s` 非劣约束。

### Stage 1：修正后的配对闭环因果门

- 正式报告：`outputs/restricted_stage1_summary_v2_20261106.json`，结果 PASS，报告 SHA-256 `70EBAD2CE6EE67FB1A2EFDF0D92B9CC6C3C2F3386AC6C0E945F2B474722D509A`。旧的 offline causal ladder 因使用错误外参被标记为 superseded。
- 使用最终冻结开发集 40 个目标、`100 µm` 初始误差和 `200` 帧预算，三组正常/零向量/同距离错配向量均在相同目标与 episode 配对下运行。

| 变体 | 成功数 | 成功率 | 最终误差 P95 | 失败/安全说明 |
| --- | ---: | ---: | ---: | --- |
| normal restricted vector | 39/40 | 97.5% | 18.0678 µm | 0 个 >30 mm、0 安全限幅步 |
| zero vector | 10/40 | 25.0% | 100.1065 µm | 0 个 >30 mm、0 安全限幅步 |
| cross-episode mismatch vector | 9/40 | 22.5% | 82.7147 µm | 0 个 >30 mm、0 安全限幅步 |
| v4 no-DAgger diffusion baseline | 13/40 | 32.5% | 110.0044 µm | 0 个 >30 mm、0 安全限幅步 |

- 相对 zero vector，成功率提高 `72.5` 个百分点，target-clustered bootstrap 95% CI `[62.5, 83.33]` 个百分点，exact McNemar `p=3.73e-9`；相对 mismatch，提高 `75.0` 个百分点，CI `[65.22, 85.19]`，`p=1.86e-9`。图像因果门 PASS。
- 相对 v4 no-DAgger baseline，成功率提高 `65.0` 个百分点，CI `[53.85, 76.92]`，`p=2.98e-8`；最终误差 P95 相对改善 `83.58%`。

### Stage 2：修正外参后的 DAgger 变体排序

- 修正后重新采集 `120` 条受限 DAgger episode，共 `3,064` 帧。正式报告为 `outputs/restricted_stage2_acceptance_v5_20261106/summary.json`，结果 PASS，SHA-256 `12B33D100629E29128BA31BF12B706A652285AED5110116DFB82CEC06632A5CB`。
- 外参修正改变了单因素排序，最终选择 `dagger_only`，不再沿用修正前选择的 velocity-only 结论。

| 变体 | 成功数 | 成功率 | 最终误差 P50 | 最终误差 P95 | 安全限幅步 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DAgger only（selected） | 40/40 | 100% | 5.7537 µm | 16.6741 µm | 0 |
| + velocity | 34/40 | 85.0% | 11.1290 µm | 83.5997 µm | 0 |
| + velocity + residual cap | 38/40 | 95.0% | 5.7547 µm | 17.1161 µm | 0 |
| + velocity + residual cap + stop head | 38/40 | 95.0% | 10.6775 µm | 20.3407 µm | 0 |

- 相对 DAgger-only，velocity 的成功率下降 `15.0` 个百分点，残差上限和 stop-head 各下降 `5.0` 个百分点；所有变体均无 >30 mm uncontrolled episode。Stage 3 因此继承 DAgger-only 的 Stage 2 结论，而不是旧的 velocity-only 结论。

### Stage 3：基线、鲁棒性、蒸馏与延迟

- 协议：`configs/stage34_protocol.json`，seed `20270301`，40 个开发目标，`100 µm` 初始误差，`1600` 帧上限。正式报告为 `outputs/restricted_stage3_summary_20261106.json`，结果 PASS，SHA-256 `C43D344E1E6C683A4768DDB67F647B2E90BDE287B35B130F4B93CCA9C648BA9A`。

| Stage 3 变体 | 成功数 | 最终误差 P95 | 结论 |
| --- | ---: | ---: | --- |
| selected restricted student | 40/40 | 16.6741 µm | 选中基线 |
| guided no-DAgger action-chunk MLP | 39/40 | 17.0145 µm | 学习价值对照 |
| old diffusion structure + DAgger, 7-median | 39/40 | 10.4633 µm | 蒸馏教师 |
| guided OCT regression | 40/40 | 11.6934 µm | 通过但非主方案 |
| raw diffusion, 7-median | 0/40 | 6520.5475 µm | 失败；40 个 episode 触发安全限制 |

- 学习价值鲁棒性对比覆盖 default、2× noise、45% center occlusion、20% OCT dropout 和 3-frame delay。Student / geometric teacher 成功数分别为 `40/8`、`40/11`、`0/7`、`40/3`、`39/11`。四个非遮挡场景的 target-clustered bootstrap 成功差下界均大于 0；遮挡场景 student 为 `0/40`，是明确失败案例，不是通过项。
- 单步蒸馏尝试及处置如下；只有最后采用协议 seed 重跑的 residual single-step head 进入正式 Stage 3 汇总。

| 尝试 | 产物目录 | 成功数 | 处置 |
| --- | --- | ---: | --- |
| anchor v3 | `outputs/restricted_stage3_distilled_anchor_eval_v3_20261106/` | 32/40 | 淘汰 |
| anchor novel v4 | `outputs/restricted_stage3_distilled_anchor_eval_v4_20261106/` | 34/40 | 淘汰 |
| anchor strong v5 | `outputs/restricted_stage3_distilled_anchor_strong_eval_v5_20261106/` | 31/39 | 分片不完整，淘汰 |
| preserve v6 | `outputs/restricted_stage3_distilled_preserve_eval_v6_20261106/` | 19/25 | 分片不完整，淘汰 |
| residual distillation v7 | `outputs/restricted_stage3_residual_distilled_eval_v7_20261106/` | 40/40 | 候选 |
| residual distillation, protocol seed | `outputs/restricted_stage3_residual_distilled_protocol_seed_20261106/` | 40/40 | 正式汇总 |
| no-DAgger protocol seed | `outputs/restricted_stage3_nodagger_protocol_seed_20261106/` | 39/40 | 协议重跑对照 |

- 正式蒸馏使用 `2,270` 个 training-target 状态的 on-policy 标签，教师为 7-sample median diffusion；未使用 development 或 final-test 目标生成标签。Selected checkpoint 为 `outputs/restricted_stage3_residual_distilled_v7_20261106/best_model.pth`，`2,990,045` bytes，SHA-256 `202FFDB08AF82F67E92F4DFC1018CF657FF7BA25AA7D03F0F1500E420F867BC7`。
- 蒸馏结果 40/40，最终误差 P50/P95 为 `7.2300/14.3188 µm`；教师 39/40，P95 `10.4633 µm`。成功率差 bootstrap 下界为 `0.0`，超过预注册的非劣边界 `-5` 个百分点，故 `distilled_single_step_noninferior=true`。
- 延迟报告：`outputs/restricted_stage3_latency_20261106.json`，SHA-256 `03A719DB1B95C7AABDBACA598745118B374F6E1E627C1578496EA8ED2AD0085A`。CUDA batch-1 学生单步推理为 P50/P95 `0.38375/0.41338 ms`；7-sample diffusion median 为 `288.715/343.346 ms`。CPU batch-1 对应为 `0.37695/0.47028 ms` 与 `726.416/860.888 ms`。延迟只描述模型推理，不包含完整感知和渲染控制链。

### Stage 4：冻结 final test 完成但性能失败

- 正式报告：`outputs/restricted_stage4_summary_20261106.json`，`completion=COMPLETE`、`result=FAIL`，SHA-256 `5763CC8DBF45FC51DE912C1937E061137710BA1F159FA081A0417EBD144C5B19`。
- Primary distilled single-step policy 使用 60 个冻结 final-test targets × 3 seeds，共 180 episodes。All episodes present、same-target/same-seed pairing、无 residual resource exception；`uncontrolled_over_30mm=0`、`safety_limited_steps=0`。

| 策略 | Episode | 20 µm 成功 | 10 µm 成功 | 5 µm 成功 | 最终误差 P50/P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| distilled single-step primary | 180 | 20 (11.11%) | 14 (7.78%) | 7 (3.89%) | 103.9037 / 3734.7234 µm |
| 7-sample diffusion median control | 60 | 18 (30.0%) | 17 (28.33%) | 6 (10.0%) | 26.0911 / 50.1229 µm |

- 在 hash-selected paired 60 episodes 上，`20 µm` 成功差为 `-15.0` 个百分点，target-cluster 95% CI `[-30.77, 0.0]`；`10 µm` 差为 `-18.33` 个百分点，95% CI `[-30.56, -5.56]`。Distilled head 更快，但在 final targets 上没有达到 20 µm 非劣或 95% 成功率目标。

### Stage 4 失败归因

- 正式分析：`outputs/restricted_stage4_failure_analysis_20261107.json`（SHA-256 `473E5C8C29EA90941994440F05E832EB7AFDD23097168FB8C36EE390E84FD8D7`）及叙述版 `STAGE4_FAILURE_ANALYSIS.md`。
- 目标分布存在完全分离：development 的 normalized posterior 范围为 `0.9006366–0.9995540`、极角 `1.7114–25.7581°`；final 为 `0.8000630–0.8821612`、`28.0958–36.8639°`。两个坐标的 KS statistic 均为 `1.0`，因此该结果主要是分布外泛化测试。
- Fine fixed-bias 没有解释失败：final-test P95/max 为 `2.8610/3.3115 µm`，全部 180 个 episode 均低于 `20 µm`；development P95 为 `2.6540 µm`，两分布 KS 检验不显著。Stage 0 fused near-target P95 `3.7323 µm` 也表明 `20 µm` 阈值本身可达。
- 180 个 primary episodes 的失败模式为：20 个在 20 µm 成功；51 个至少一次进入 20 µm 但未完成三帧稳定 hold；8 个最小误差为 20–30 µm；41 个为 30–50 µm；45 个为 50–100 µm；6 个没有从 100 µm start 接近（2 个目标 × 3 seeds）；9 个在首帧前因 B 轴编码器超出标定安全范围而无效初始化（3 个目标 × 3 seeds）。
- 51 个 hold 失败 episode 的中位最小误差为 `5.54 µm`，但中位最终误差回升至 `96.29 µm`；fine channel 在控制帧中的中位可用率为 `99.52%`，coarse fallback 从未发生。主要已学习策略问题是 OOD 目标位置下的终点稳定和 hold 后继续动作，而不是 fine 通道缺失或 bias 超限。

### 本次测试与日志追加核验

- 2026-11-08 重新执行 `python -m unittest discover -s tests -v`，35 个测试全部通过（`Ran 35 tests in 23.462s`，`OK`，退出码 0）。
- 自动钩子已将本次执行追加到 `TEST_LOG.md`，记录 ID `69cc96542d01c33535922c25`，输出文件 `test-logs/69cc96542d01c33535922c25.log`；本节再次确认该测试记录已追加。
- 本轮只是补齐既有实验日志并复跑单元测试。所有精度、成功率和延迟结论均来自 MuJoCo 仿真及冻结产物，不支持真实 OCT、临床精度、组织接触或硬件安全声明。

## 2026-11-08T15:53:20+08:00 | UNKNOWN
<!-- codex-test:ad093c1c4a093938e4dd184c -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/ad093c1c4a093938e4dd184c.log](test-logs/ad093c1c4a093938e4dd184c.log)

### 命令

```text
Get-Content README.md -TotalCount 230; Write-Output '--- pytest availability ---'; py -3.11 -m pytest --version; Write-Output '--- CI/config ---'; rg --files -g 'pyproject.toml' -g 'pytest.ini' -g 'tox.ini' -g 'setup.cfg' -g '.github/**' -g 'requirements*.txt'
```

### 输出末尾摘要

````text
692 帧。三种仿真导航轨迹均 PASS；遮挡标签显示有部分针体遮挡，但没有完全遮挡帧。

| 方法 | 结果 | 帧数 | 运动时间 (s) | 最大到达误差 (μm) | 部分遮挡帧 | 完全遮挡帧 |
|---|---:|---:|---:|---:|---:|---:|
| 视觉伺服 | PASS | 1,580 | 78.739 | 5.30 | 604 | 0 |
| RRT（seed 7） | PASS | 2,056 | 102.750 | 14.60 | 979 | 0 |
| PRM（seed 7） | PASS | 2,056 | 102.750 | 14.60 | 979 | 0 |

任务模式有两种：`registered_coordinate_prior` 会将假设已配准的机器人坐标写入任务文本（当前示例坐标来自仿真，不是实测 OCT）；默认 `visual_current_target` 的策略输入文本不含目标坐标、目标 ID 或访问顺序。这个 pilot 中标靶逐个呈现，访问顺序由脚本依据测量选定并由各方法共享，不能证明策略自主识别并排序多个病灶。

专家与策略输入也要区分：专家使用仿真分割/渲染深度和标定信息；眼底视觉伺服还按已知目标姿态标定特征，并在回撤/转移阶段使用解析运动，RGB 在该阶段只记录、不参与纠偏，最后的 C 轴接近才使用视觉深度闭环。因此示范不是全程纯单目导航。已导出去图像输入包和对应动作标签，但项目尚未实现策略训练/推理或去图像消融；该 pilot 只有一个目标集合，不是完整 train/validation/test 数据集。

字段、轨迹、输入泄漏及遮挡标签验证结果见 `runs/历史记录/05_random_datasets/pilot_n5_visual_current_target_v1/episodes/train/episode_000000/dataset_verification.json`。复验命令（从本目录运行）：

```powershell
.\.venv\Scripts\python.exe verify_waypoint_comparison.py --run-dir runs\历史记录\05_random_datasets\pilot_n5_visual_current_target_v1\episodes\train\episode_000000
```

本版默认不启用 A/B/C 位置限位，默认针尖目标改为眼底路径点 P3 `[1.92,8.478,22.043]` mm；原深部示例 `[9.5,-1.5,16]` mm 仅作为历史回归目标保留。此前记录的编码器区间仅保留为观测数据；它们不能代表全行程，也不能用来判断眼底目标不可达。

```bash
python -m pip install -r requirements.txt
python run_navigation.py
```

程序执行：瞄准 → 进针 → 停留 → C 回零 → A/B 复位。P 暂停/继续，关闭窗口保存报告和 CSV。无窗口运行：

```bash
python run_navigation.py --no-viewer
python run_navigation.py --target-robot 9.5 -1.5 16 --solve-only
python run_navigation.py --targets examples/targets.json --no-viewer
```

执行方式是运动学回放 `mj_forward`，用于核对目标求解、针轴与 RCM 的关系及导航流程。反馈控制、驱动器响应和实机通信还没有接入。完整接口说明见 [NAVIGATION.md](NAVIGATION.md)。

## 零位、针和编码器

已确认的零位 ABC 编码器为 `[0,1466018,893518]`，E 为 0。启动和 R 复位都回到这个状态，相对任务坐标为 `[0,0,0]`。

| 轴 | 换算与相对坐标 |
|---|---|
| A | `A_deg = EA*0.0018`；相对零位角度写入 `A_azimuth`，单位 rad |
| B | `dB_mm = EB/K`；经连杆公式求 alpha；`qB = radians(alpha-alpha_zero)` |
| C | `insertion_mm = (EC-893518)/K`；`qC = insertion_mm/1000`，单位 m |

`K=63822.71488 count/mm`，当前 B 零位角 `alpha_zero≈54.236005250°`。默认正向计数，原点/反向参数仍在配置文件中。

针尖到 RCM 的距离：

```text
L_tip_mm = needle_tip_distance_at_reference_mm + insertion_mm
--- pytest availability ---
C:\Users\Lenovo\AppData\Local\Programs\Python\Python311\python.exe: No module named pytest
--- CI/config ---
requirements-mesh.txt
requirements.txt
mujoco_model\requirements.txt
lerobot_training\requirements.txt
````

## 2026-11-08T15:55:10+08:00 | UNKNOWN
<!-- codex-test:c9ef450edd66c103581a8a62 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/c9ef450edd66c103581a8a62.log](test-logs/c9ef450edd66c103581a8a62.log)

### 命令

```text
Get-Command python,py,pytest -ErrorAction SilentlyContinue | Select-Object Name,Source,Version; py -0p; Get-ChildItem -Force -Directory | Where-Object Name -Match 'venv|env' | Select-Object FullName; Get-Content requirements.txt; Write-Output '--- test imports ---'; Get-ChildItem tests -File | ForEach-Object { Write-Output "### $($_.Name)"; Get-Content $_.FullName -TotalCount 35 }
```

### 输出末尾摘要

```text
w s l :   �hKm0R  l o c a l h o s t   �NtM�n�FO*g\��P0R  W S L 0N A T   !j_N�v  W S L   N/ec  l o c a l h o s t   �Nt0 
 <3>WSL (13 - Relay) ERROR: CreateProcessCommon:735: execvpe(/bin/bash) failed: No such file or directory
```

## 2026-11-08T15:55:21+08:00 | UNKNOWN
<!-- codex-test:850f13bf7c5eca33c8475849 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/850f13bf7c5eca33c8475849.log](test-logs/850f13bf7c5eca33c8475849.log)

### 命令

```text
Get-Command python,py,pytest -ErrorAction SilentlyContinue | Select-Object Name,Source,Version; py -0p; Get-ChildItem -Force -Directory | Where-Object Name -Match 'venv|env' | Select-Object FullName; Write-Output '--- requirements ---'; Get-Content requirements.txt; Write-Output '--- test imports ---'; Get-ChildItem tests -File | ForEach-Object { Write-Output "### $($_.Name)"; Get-Content $_.FullName -TotalCount 35 }
```

### 输出末尾摘要

```text
T_DID_NOT_HOLD",
        )
        self.assertEqual(
            classify_failure(_episode(minimum_um=39.0, final_um=100.0)),
### test_statistics.py
from __future__ import annotations

import unittest

from restricted_perception.statistics import paired_comparison


def _episode(target, seed, success, error):
    return {
        "target_robot_mm": list(target),
        "seed": seed,
        "success": success,
        "final_error_mm": error,
    }


class PairedStatisticsTests(unittest.TestCase):
    def test_pairs_by_target_and_seed(self):
        baseline = [
            _episode([0, 0, 2], 1, False, 0.5),
            _episode([1, 0, 2], 1, True, 0.01),
        ]
        candidate = [
            _episode([1, 0, 2], 1, True, 0.005),
            _episode([0, 0, 2], 1, True, 0.01),
            _episode([2, 0, 2], 1, True, 0.01),
        ]
        report = paired_comparison(
            candidate,
            baseline,
            bootstrap_samples=100,
        )
        self.assertEqual(report["paired_episodes"], 2)
        self.assertAlmostEqual(report["success_rate_difference"], 0.5)
        self.assertLess(report["paired_error_difference_mean_um"], 0.0)
### test_target_split.py
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from restricted_perception.target_split import (
    canonical_point,
    nearest_neighbor_distances,
    target_hash,
)


ROOT = Path(__file__).resolve().parents[1]


class TargetSplitTests(unittest.TestCase):
    def test_hash_is_stable_and_seed_sensitive(self):
        point = [1.0, 2.0, 3.0]
        self.assertEqual(target_hash(point, 7), target_hash(point, 7))
        self.assertNotEqual(target_hash(point, 7), target_hash(point, 8))
        self.assertEqual(canonical_point([1.0 + 1e-9, 2.0, 3.0]), (1.0, 2.0, 3.0))

    def test_nearest_neighbor_and_self_exclusion(self):
        points = [[0, 0, 0], [0, 0, 1.2], [0, 0, 3.0]]
        self.assertEqual(
            nearest_neighbor_distances(points, points, exclude_self=True),
            [1.2, 1.2, 1.8],
        )

    def test_checked_in_manifest_has_frozen_separation(self):
        manifest_path = ROOT / "configs" / "target_split_manifest.json"
        self.assertTrue(manifest_path.exists())
        self.assertGreater(manifest_path.stat().st_size, 0)
```

## 2026-11-08T15:56:56+08:00 | UNKNOWN
<!-- codex-test:40cc15806cc60d703c0ac278 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/40cc15806cc60d703c0ac278.log](test-logs/40cc15806cc60d703c0ac278.log)

### 命令

```text
$before = (Get-Item -LiteralPath TEST_LOG.md).Length; Write-Output "TEST_LOG_BEFORE=$before"; py -3.11 -m unittest discover -s tests -v; $testExit = $LASTEXITCODE; Write-Output "UNITTEST_EXIT=$testExit"; $after = (Get-Item -LiteralPath TEST_LOG.md).Length; Write-Output "TEST_LOG_IMMEDIATE_AFTER=$after"; exit $testExit
```

### 输出末尾摘要

```text
UNITTEST_EXIT=1
TEST_LOG_IMMEDIATE_AFTER=317126
```

## 2026-11-08T15:58:59+08:00 | FAILED
<!-- codex-test:368146dc9c77af34fe0a8633 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未提供，由测试输出末尾判定为失败（推断）
- 输出记录：[test-logs/368146dc9c77af34fe0a8633.log](test-logs/368146dc9c77af34fe0a8633.log)

### 命令

```text
Write-Output 'PYTHON_ENV:'; python -c 'import sys, torch, numpy, PIL; print(sys.version); print(torch.__version__)'; python -m unittest discover -s tests -v; $testExit = $LASTEXITCODE; Write-Output "UNITTEST_EXIT=$testExit"; exit $testExit
```

### 输出末尾摘要

```text
atics_matches_reference (test_restricted_sidecar.RestrictedSidecarTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_restricted_sidecar.py", line 23, in test_batched_kinematics_matches_reference
    from rcm_reference import Reference
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\rcm_reference.py", line 15, in <module>
    import mujoco
ModuleNotFoundError: No module named 'mujoco'

======================================================================
ERROR: test_sidecar_schema_and_truth_omission (test_restricted_sidecar.RestrictedSidecarTests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_restricted_sidecar.py", line 74, in test_sidecar_schema_and_truth_omission
    metadata = build_restricted_sidecar(
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\restricted_perception\sidecar.py", line 134, in build_restricted_sidecar
    from rcm_reference import Reference
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\rcm_reference.py", line 15, in <module>
    import mujoco
ModuleNotFoundError: No module named 'mujoco'

======================================================================
ERROR: test_geometric_action_moves_toward_target_minus_tip (test_stage34.Stage34Tests)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\tests\test_stage34.py", line 70, in test_geometric_action_moves_toward_target_minus_tip
    from rcm_reference import Reference
  File "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225\rcm_reference.py", line 15, in <module>
    import mujoco
ModuleNotFoundError: No module named 'mujoco'

----------------------------------------------------------------------
Ran 35 tests in 25.495s

FAILED (errors=5)
UNITTEST_EXIT=1
```

## 2026-11-08T16:08:58+08:00 | SUCCESS
<!-- codex-test:952ad992efb78dcbf400d7cc -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未提供，由测试输出末尾判定为通过（推断）
- 输出记录：[test-logs/952ad992efb78dcbf400d7cc.log](test-logs/952ad992efb78dcbf400d7cc.log)

### 命令

```text
py -3.11 -c 'import sys, torch, mujoco, numpy, PIL; print(sys.version); print("torch", torch.__version__); print("mujoco", mujoco.__version__)'; py -3.11 -m unittest discover -s tests -v; $testExit = $LASTEXITCODE; Write-Output "UNITTEST_EXIT=$testExit"; exit $testExit
```

### 输出末尾摘要

```text
ok
test_feature_matrix_shape (test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_residual_cap_uses_ten_micrometre_scale (test_restricted_policy.RestrictedPolicyTests.test_residual_cap_uses_ten_micrometre_scale) ... ok
test_source_filter_returns_boolean_mask (test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_batched_kinematics_matches_reference (test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_geometric_action_moves_toward_target_minus_tip (test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok
test_residual_is_bounded (test_stage3_policy.Stage3PolicyTests.test_residual_is_bounded) ... ok
test_zero_initialized_residual_preserves_base_action (test_stage3_policy.Stage3PolicyTests.test_zero_initialized_residual_preserves_base_action) ... ok
test_failure_classification_separates_hold_and_initialization (test_stage4_analysis.Stage4AnalysisTests.test_failure_classification_separates_hold_and_initialization) ... ok
test_value_summary_uses_requested_population_statistics (test_stage4_analysis.Stage4AnalysisTests.test_value_summary_uses_requested_population_statistics) ... ok
test_pairs_by_target_and_seed (test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok
test_checked_in_manifest_has_frozen_separation (test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok

----------------------------------------------------------------------
Ran 35 tests in 80.858s

OK
UNITTEST_EXIT=0
```

## 2026-11-08 | Stage 2 四组单因素实验人工审计记录

来源报告：
`outputs/restricted_stage2_acceptance_v5_20261106/summary.json`

报告 SHA-256：
`12B33D100629E29128BA31BF12B706A652285AED5110116DFB82CEC06632A5CB`

该组结果使用相机外参修正后的受限 sidecar、重新采集的 120 个 DAgger
episode 和 3,064 帧数据。所有比较均使用 40 个开发目标、同目标同 seed
配对；各组安全限位步数均为 0，且无超过 30 mm 的失控 episode。

| 实验 | 单因素配置 | 成功数 | 最终误差 P50 | 最终误差 P95 |
| --- | --- | ---: | ---: | ---: |
| DAgger-only | 无速度、无残差上限、无停止头 | 40/40 | 5.75 µm | 16.67 µm |
| Velocity | 增加 3 维实际电机速度输入 | 34/40 | 11.13 µm | 83.60 µm |
| Residual cap | Velocity + 部署期距离缩放残差上限 | 38/40 | 5.75 µm | 17.12 µm |
| Stop head | Residual cap + 二分类停止头 | 38/40 | 10.68 µm | 20.34 µm |

相对上一组的增量结果：

- Velocity 相对 DAgger-only：成功率差 `-15.0` 个百分点，目标聚类
  bootstrap 95% CI `[-23.1, -7.1]`；exact McNemar `p=0.03125`；
  最终误差平均增加 `8.97 µm`，配对误差差 95% CI `[3.75, 13.36] µm`。
- Residual cap 相对 Velocity：成功率增加 `10.0` 个百分点，从 34/40
  恢复到 38/40；平均误差减少 `4.79 µm`，95% CI `[-7.75, -1.61] µm`；
  exact McNemar `p=0.125`。
- Stop head 相对 Residual cap：成功数不变，仍为 38/40；平均误差增加
  `2.73 µm`，95% CI `[1.40, 4.33] µm`；exact McNemar `p=1.0`。

残差上限公式为
`scale = clip(distance / 10 µm, 0.05, 1.0)`，训练阶段不截断，部署推理时
按距离缩放逐轴动作上限。停止头在 `sigmoid(stop_logit) >= 0.5` 时输出零动作。

最终选择：`DAgger-only`。速度输入造成显著退化和长尾误差；残差上限能够修复
速度版的大部分问题，但仍未超过无速度基础策略；停止头没有成功率收益并进一步
降低精度。因此后三个新增因素均标记为无精度增益，未进入最终策略。

- 2026-11-08 21:04 +08:00：PPT 项目工具环境核查。默认 `python` 为 3.8.5，
  `project_manager.py init --help` 因不支持 `list[...]` 注解而 FAIL（退出码 1）；
  改用 `py -3.11` 复跑同一命令 PASS（退出码 0），确认根因是解释器选择而非项目工具缺陷。

- 2026-11-08 21:07 +08:00：新建独立项目
  `projects/rcm_11p_arch_p01_p03_ppt169_20261108`，初始化与导入
  `RESTRICTED_PERCEPTION.md`、`MOTOR_DATASET.md`、`OCT_SIMULATION.md` 均 PASS。
  首次 `project_manager.py validate` FAIL：`icons.inventory` 不能为空；改为
  `inventory: none` 后复跑 PASS（仅保留 SVG 尚未生成的预期 warning）。设计锁定为
  3 页、1280×720、正文 26 px、注释/页脚 24 px、flat、无备注/动画/旁白。

- 2026-11-08 21:11 +08:00：`text_measure.py calibrate --outline` PASS，7 个角色
  均完成校准。首次 live preview 在 6060 FAIL：端口由既有预览占用，健康检查返回
  另一项目；保留旧进程，改用 `--port 6123` 启动 PASS，锁文件确认
  `pid=36440, port=6123`。

- 2026-11-08 21:14 +08:00：三页 SVG 首次最终质量门 FAIL（2 errors、7 advisories）：
  P02 标题区与导航区重叠，P03 标题/门控/白名单框有边界溢出，并存在可合并文本与继承样式告警。
  一次合并修复后运行 `compact_svg_styles.py --inplace`，复跑最终门 PASS：
  3/3 页、0 errors、0 warnings；`project_manager.py validate` 再次 PASS，项目结构 complete。

- 2026-11-08 21:22 +08:00：首版 PPTX 导出与 PowerPoint 1600×900 渲染完成。逐页复核发现
  P02 策略标签拥挤、P03 测量标签与门控数字过宽、状态轨和证据标签过紧；完成合并修复并复跑
  SVG 最终门 PASS（3/3、0 errors、0 warnings），重新导出后再次渲染三页，未再发现裁切或重叠。
  最终 PPTX 以 `--no-notes --no-animations` 导出，postflight PASS、0 warning categories；
  `pptx_delivery_check.py` PASS：3 slides、0 notes、0 media、0 transitions、0 object animations。
  独立 python-pptx 断言 PASS：16:9、3 slides、0 notes、必需文本齐全、0 路径/文件名泄漏。
  独立内容扫描首次因 PowerShell 传参去除 Python 字符串引号而 FAIL，改用单引号原始字符串后复跑 PASS。

- 2026-11-08 21:23 +08:00：完成前新鲜复验全部 PASS：项目结构 complete；SVG 最终门
  3/3 页、0 errors、0 warnings；`pptx_delivery_check.py` 通过且 0 advisories；
  python-pptx 独立断言确认 16:9、3 页、0 备注、必需文本齐全、0 路径泄漏，
  `exports/` 中仅保留最终文件 `rcm_11p_arch_p01_p03_20261108_212031.pptx`。
  两份中间 PPTX 及对应报告已移至 `backup/draft_exports/`，可恢复。

- 2026-11-08 22:51 +08:00：扩展完整 11 页项目。导入 Stage 3、Stage 4、失败分析、
  协议和汇总来源 6 份；Design Spec 与 lock 扩为 P01–P11，
  `project_manager.py validate` PASS（项目结构 complete）。P01–P03 的既有正文与视觉保持不变。

- 2026-11-08 23:11 +08:00：11 页规划重新执行 `text_measure.py calibrate --outline` PASS。
  完成 P04–P05 后运行第 5 页强制 early gate：首次 0 errors、3 advisories
  （页脚继承样式与 P04 同段文字拆分）；合并说明、运行 `compact_svg_styles.py --inplace`
  后复跑 early gate PASS：5/5 页、0 errors、0 warnings。

- 2026-11-08 23:14 +08:00：完成 P06–P11 与 11 页 SVG。最终质量门首轮 FAIL：
  P06 参数说明越界、P07 灰线箭头颜色不一致及右栏数值越界、P08/P10 右栏长文本越界，
  并有 3 处同段文本拆分告警。统一合并修复后最终门 PASS：11/11 页、0 errors、0 warnings。

- 2026-11-08 23:21 +08:00：完整 11 页 PPTX 导出并通过 postflight
  （11 slides、0 warning categories）。PowerPoint 将 11 页渲染为 1600×900 PNG；
  逐页复核发现 P05 权限说明、P07 延迟卡和双行标题过密，完成修正并复跑 SVG 最终门 PASS
  （11/11、0 errors、0 warnings），重新导出后再次渲染确认无裁切、重叠或连接断裂。
  `pptx_delivery_check.py` PASS：11 slides、0 notes、0 media、0 transitions、
  0 object animations、0 errors、0 advisories。python-pptx 独立断言 PASS：16:9、
  11 titles、11 footers 齐全、0 notes、0 路径/文件名泄漏。最终导出为
  `exports/rcm_11p_arch_p01_p03_20261108_231902.pptx`；旧 3 页交付和两份 11 页中间稿
  已移至 `backup/draft_exports/`，可恢复。

- 2026-11-08 23:21 +08:00：完成前新鲜复验全部 PASS：项目结构 complete；
  SVG 最终门 11/11 页、0 errors、0 warnings；`pptx_delivery_check.py`
  11 slides、0 notes、0 media、0 transitions、0 object animations、0 advisories；
  python-pptx 独立断言确认 16:9、11 个标题与页脚齐全、0 备注、0 路径/文件名泄漏；
  `exports/` 仅保留唯一完整文件 `rcm_11p_arch_p01_p03_20261108_231902.pptx`，
  `validation/final_render/` 恰有 11 张渲染图。

- 2026-11-10 19:00 +08:00：17 页汇报制作首次调用 `project_manager.py --help`
  FAIL（退出码 1）：默认 `python` 为 3.8.5，不支持 `list[Path]` 类型注解。
  系统调试确认根因与 2026-11-08 记录一致，为解释器选择问题；后续 PPT Master
  工具统一改用 `py -3.11`。
- 2026-11-10 19:00 +08:00：按 `systematic-debugging` 完成根因定位并通过
  `py -3.11` 检查解释器可用性 PASS；未修改 Skill 或项目工具源码。
- 2026-11-10 19:02 +08:00：使用 `py -3.11` 初始化
  `projects/rcm_imitation_learning_17p_ppt169_20261110` PASS；画布确认
  PPT 16:9（1280×720）。
- 2026-11-10 19:05 +08:00：`academic-research` Style 模板契约校验 PASS：
  1/1 文件、0 errors、0 warnings。
- 2026-11-10 19:05 +08:00：`report_core` Layout 模板契约校验 PASS（退出码 0）：
  13/13 文件、0 blocking errors；26 条均为模板自带的 compact-authoring advisories。
- 2026-11-10 19:07 +08:00：`apply_template.py` PASS，共安装 15 个文件。
  分段所有权为 identity:current-project、structure:report_core Layout、
  application_context:none、direction:academic-research Style；active roster 为
  `report_core`。
- 2026-11-10 19:09 +08:00：`project_manager.py import-sources` PASS；导入 3 份
  叙述来源、9 份协议/汇总/失败分析 JSON 和 5 张现有 PNG。JSON 按“仅归档”保留，
  5 张 PNG 已复制到运行图片池。
- 2026-11-10 19:09 +08:00：`analyze_images.py` PASS；5/5 图片可读并写入
  `analysis/image_analysis.csv`，无缺失或损坏文件。
- 2026-11-10 19:12 +08:00：首次 `icon_sync.py` 调用误用了不可用 shell 参数，
  未执行到脚本；切换 PowerShell 后 PASS，12 个 `tabler-outline` 图标成功同步。
- 2026-11-10 19:13 +08:00：`visualization_recall.py validate` PASS；
  `line_chart`、`column_chart`、`metric_table`、`scatter_chart`、`funnel_chart`
  五个引用全部解析到有效模板。
- 2026-11-10 19:18 +08:00：`project_manager.py validate` 首轮 FAIL：9 条
  `pptx_layouts` 映射将模板 SVG basename 直接当作源，并使用了改写后的 Master/Layout
  key；根因是结构化锁必须与原型根节点的 `data-pptx-master` / `data-pptx-layout`
  身份完全一致，源须写为 `template:<basename>`。
- 2026-11-10 19:20 +08:00：按原型身份修正 9 个 Master/Layout 映射后，
  `project_manager.py validate` PASS；仅保留 `svg_output/` 尚为空的预期 warning。
- 2026-11-10 20:03 +08:00：`text_measure.py calibrate --outline` 首跑 PASS；
  8 个锁定排版角色均完成宽度估算。随后补充 `kpi_family` 并复跑 PASS。
- 2026-11-10 20:07 +08:00：Live Preview 启动 PASS：
  `http://127.0.0.1:6060`，pid=`44740`。
- 2026-11-10 20:31 +08:00：P01–P05 首轮 early gate FAIL：
  4 页存在槽内文本溢出，P01/P03/P04/P05 的结构化页在 adaptive 模式改动固定
  atom 几何。修复：恢复原型 `rx`/日期槽宽度，正文改为语义分行，移除未投影的
  原生图表伴随文本。合并修复后 early gate PASS：5/5 页、0 errors、0 warnings。
- 2026-11-10 21:08 +08:00：17 页首轮 final gate FAIL：P12 槽内文本越界；
  P11 表格 fallback/JSON 不一致；P07 复用的 `process_timeline` 固定节点颜色漂移。
  另有两处图表 companion/axis 投影告警。
- 2026-11-10 21:15 +08:00：第二轮 final gate PASS（0 errors、2 advisories），随后
  清理 P11 表头对齐与 P12 冗余粗体声明；第三轮最终质量门 PASS：
  17/17 页、0 errors、0 warnings。
- 2026-11-10 21:22 +08:00：定量图校验完成并写入
  `validation/chart_verification.md`：P05 line 与 P09 column 用坐标计算器修正，
  P12 scatter 与计算值一致，P15 漏斗分解用分段宽度公式核验；修正后 final gate
  再次 PASS：17/17 页、0 errors、0 warnings。
- 2026-11-10 21:32 +08:00：`visual_review.py` 首次渲染 PASS：17/17 页、
  1280×720、0 failed、0 all-background。逐页 PNG 回查发现并修复 P03、P09、
  P12、P15 的 4 处硬碰撞/可读性问题；受影响页面重新渲染 PASS，逐页 JSON
  写入 `.review/`。
- 2026-11-10 21:37 +08:00：视觉修正后 final gate PASS：17/17 页、
  0 errors、0 warnings。
- 2026-11-10 21:44 +08:00：讲者备注完成覆盖检查 PASS：17 个页标题、
  3009 个非空白字符；按技术汇报约 210 字/分钟估算约 14.3 分钟，符合“约 14.5 分钟”。
- 2026-11-10 21:45 +08:00：`project_manager.py validate` PASS，项目结构 complete。
- 2026-11-10 21:48 +08:00：`total_md_split.py` PASS；17 个 SVG 与 17 个备注段
  一一对应，生成 17 份逐页 Markdown。备注目录共 18 个 Markdown 文件
  （含 `total.md` 总稿）。
- 2026-11-10 21:50 +08:00：首次 `finalize_svg.py` 调用因误用不可用 shell 参数
  未进入脚本；切换 PowerShell 后 PASS：11 个图标嵌入、4 张图片嵌入并生成
  `svg_final/` 17/17 份自包含预览。
- 2026-11-10 21:53 +08:00：首次 `svg_to_pptx.py --native-charts-and-tables`
  调用误用不可用 shell 参数；切换 PowerShell 后进入转换，但 P05 结构化
  `chart` 槽因 caption companion 生成 2 个原生子对象而 FAIL。
- 2026-11-10 22:06 +08:00：移除 P05/P09 槽内伴随文本后复跑，推进到 P12；
  P12 的 scatter 点标签 `notes` 生成 6 个原生子对象，仍不符合一槽一载体。
- 2026-11-10 22:12 +08:00：P12 改为槽内无 companion 的单一 scatter，点数值与
  颜色映射移入右侧独立说明槽；重新 stamp、final gate（17/17、0 errors、
  0 warnings）、finalize 后导出 PASS：
  `exports/rcm_imitation_learning_17p_20261110_212617_native_charts_tables.pptx`，
  postflight passed、17 slides、0 warning categories。
- 2026-11-10 22:17 +08:00：PowerPoint COM 将最终 PPTX 导出为 17 页 PDF 和
  17 张 1600×900 PNG；首次命令出现不可隐藏窗口的非致命警告，但导出文件完整。
  contact sheet 与 P05/P09/P11/P12/P15 原生渲染逐页回查通过；P12 原生 scatter
  四点颜色已确认保留。
- 2026-11-10 22:20 +08:00：`pptx_delivery_check.py` 对最终 PPTX PASS：
  17 slides、17 notes、2 masters、9 layouts、4 embedded media、0 external media、
  0 object animations、0 audio timings、0 errors、0 advisories。
- 2026-11-10 22:22 +08:00：独立 python-pptx/PIL/PDF 断言首次因缺少 `PyPDF2`
  FAIL；安装 `pypdf` 后复验 PASS：16:9、17 slides、3 native charts、
  1 native table、P12 四 series、17 条非空备注、17 张 PNG、PDF 17 页、
  无本地路径或文件后缀泄漏。
- 2026-11-10 22:24 +08:00：将较早的 `211418` 成功稿移至
  `backup/draft_exports/`（可恢复）；`exports/` 仅保留最终
  `212617_native_charts_tables.pptx`。
- 2026-11-10 22:28 +08:00：完成前新鲜复验 `project_manager.py validate`
  PASS：项目结构 complete。
- 2026-11-10 22:29 +08:00：完成前新鲜复验 SVG final gate PASS：
  17/17 页、0 errors、0 warnings。
- 2026-11-10 22:30 +08:00：完成前新鲜复验 `pptx_delivery_check.py`
  PASS：17 slides、17 notes、2 masters、9 layouts、4 embedded media、
  0 external media、0 object animations、0 errors、0 advisories。
- 2026-11-10 22:31 +08:00：完成前独立断言 PASS：`exports/` 仅 1 份 PPTX；
  16:9、17 slides、3 native charts、1 native table、P12 四个 series、
  17 条非空备注、17 张 1600×900 PNG、PDF 17 页、无本地路径泄漏；
  notes 18 份（17 页 + total）、最终质量报告与图表校验报告齐全，
  `TEST_LOG.md` 已确认包含本次复验记录。
- 2026-11-10 22:38 +08:00：最终数值词表复核发现 P09 页面只显示 39/10/9，
  未显式显示 10/40 等分母；补入“成功数：39/40 · 10/40 · 9/40”，
  P09 native chart 重新 stamp 后 final gate 回到 17/17、0 errors、0 warnings。
- 2026-11-10 22:54 +08:00：重新 finalize 并导出最终
  `rcm_imitation_learning_17p_20261110_215408_native_charts_tables.pptx`；
  PowerPoint 重新导出 17 页 PDF 与 17 张 1600×900 PNG；最终 P09 渲染确认
  三个分母可见。新稿 postflight passed、0 warning categories。
- 2026-11-10 22:56 +08:00：最终稿 `pptx_delivery_check.py` PASS；
  独立递归断言 PASS：16:9、17 slides、3 charts、1 table、P12 四 series、
  17 notes、PDF 17 页、PNG 17 张、24 个统计 token 齐全、无路径泄漏。
- 2026-11-10 22:57 +08:00：将较早的 `212617` 稿移至
  `backup/draft_exports/`（可恢复）；`exports/` 仅保留最终 `215408` 稿。
- 2026-11-10 22:59 +08:00：完成前最终 fresh check PASS：
  `exports/` 恰 1 份 PPTX、17 slides、17 张 1600×900 PNG、PDF 17 页、
  notes 18 份（17 页 + total）、最终质量报告存在、`TEST_LOG.md` 含本轮记录。
- 2026-11-10 21:39 +08:00：`figgenie-paper-diagram` 网络架构图首轮
  `spec_validate.py` 因当前 Python 3.8 的 `jsonschema==3.2.0` 缺少
  `Draft202012Validator` 而 FAIL。升级到 `jsonschema==4.23.0` 后，spec 校验
  PASS：0 errors、0 warnings。注意：该环境升级与已安装的
  `label-studio-sdk==1.0.5` 的 `jsonschema==3.2.0` 固定依赖冲突，后续使用该 SDK
  时应改用独立虚拟环境。
- 2026-11-10 21:32 +08:00：`figgenie-paper-diagram/check.py` 对工作 SVG 首轮
  FAIL：`n:visual-feature` 文本越界，另有 10 条对齐、边连接、字体和尺寸 warning。
  修正节点盒、连线端点、字号层级与描边体系后复跑 PASS：0 errors、0 warnings；
  仅保留预期 image crop info。
- 2026-11-10 21:32 +08:00：`embed_fonts.py --font arial --cjk source-han-sans
  --strict` PASS；嵌入 Arimo + Source Han Sans 子集，1 张 PNG 内联到 SVG，
  115/115 个字符覆盖。为使 Windows 上的 `NotoSansSC-VF.ttf` 可被 Skill 字体解析器
  找到，补充了本地 `fonts.json` 系统字体匹配模式。
- 2026-11-10 21:33 +08:00：`check.py fig.final.svg --spec spec.json --scale 3`
  PASS：120 elements、0 errors、0 warnings；PNG 为 1512×708，视觉回查无重叠、
  越界、错误箭头或遮挡。
- 2026-11-10 21:35 +08:00：`export_pdf.py fig.final.svg` PASS；生成
  504×236 pt、1 页 PDF，字体与 1 张位图嵌入。提示 CJK 字体在 PDF 中为 Type3，
  可用于 LaTeX/Pages，但部分出版社字体检查可能拒绝。
- 2026-11-10 21:37 +08:00：独立 PDF/PNG 核验首跑因未安装 `pypdf` FAIL；
  改用已安装的 PyMuPDF 后 PASS：PDF 1 页、504×235.92 pt、436 个文本字符、
  1 张图像、56 个矢量绘图；PNG 1512×708；SVG 含 6 个字体 face 和 1 个内联图像。
- 2026-11-10 21:41 +08:00：为避免保留全局依赖冲突，修正
  `figgenie-paper-diagram/scripts/spec_validate.py`：旧版 `jsonschema` 缺少
  `Draft202012Validator` 时自动回退到内置 subset checker；随后将环境恢复为项目
  原有 `jsonschema==3.2.0`。再次执行 `check.py fig.final.svg --spec spec.json
  --scale 3` PASS：120 elements、0 errors、0 warnings。
- 2026-11-10 21:55 +08:00：按“每块内容均可见”要求制作网络图 v2，并保留 v1
  到 `versions/v1/`。v2 增加 ResNet 内部阶段、18 维受限向量分组、两组 MLP
  完整层序、逐轴残差上限、动作单位，以及完整二维 Diffusion 分支。
- 2026-11-10 21:55 +08:00：v2 spec 校验 PASS（0 errors、0 warnings）；
  `check.py fig.v2.svg --spec spec.json --scale 3` 首轮发现 legend 与
  `bounded Δ` 标签重叠，移除冗余边标签并重新对齐后 PASS：504×300 pt、
  0 errors、0 warnings；仅保留 12% 图像有意裁切 info。
- 2026-11-10 22:00 +08:00：v2 `embed_fonts.py --strict` PASS：122/122 字符被
  Arimo + Source Han Sans 子集覆盖，1 张 PNG 内联；`check.py fig.final.svg`
  PASS：147 elements、0 errors、0 warnings。
- 2026-11-10 22:01 +08:00：v2 `export_pdf.py` PASS：单页 504×300 pt；
  PyMuPDF 独立核验 PASS：709 个文本字符、1 张图像、65 个矢量绘图；
  PNG 为 1512×900 RGB；SVG 含 6 个字体 face 和 1 个内联图像。
- 2026-11-11 10:04 +08:00：新增 `render_v9_motion_gif.py` 后
  `py_compile` PASS。
- 2026-11-11 10:04 +08:00：24 帧 GIF 冒烟测试首轮 FAIL：
  `KeyError: tip_rcm_distance_mm`。按 `systematic-debugging` 检查
  `RCMSceneV9.metrics()`，确认该字段位于 `needle_eye_geometry` 子字典。
- 2026-11-11 10:04 +08:00：修正字段路径后复跑同一 24 帧 GIF 冒烟测试 PASS：
  1280×760、24 帧、每帧 80 ms、无限循环，文件和元数据均成功写出。
- 2026-11-11 10:05 +08:00：最终 72 帧、12 fps 的
  `outputs/current_v9_motion.gif` 渲染 PASS；峰值预览显示
  A≈+5.0°、B≈−2.0°、C≈+2.0 mm，整机与眼内相机同步。
- 2026-11-11 10:06 +08:00：完成前新鲜复验 PASS：`py_compile` 与
  `git diff --check` 均通过；独立检查 GIF 为 72 帧、1280×760、每帧 80 ms、
  无限循环，文件 5,691,031 bytes；首帧到峰值帧、峰值帧到末帧 RGB RMS
  分别为 10.937 和 10.929，确认画面确有运动；预览与元数据均存在。
- 2026-11-11 10:23 +08:00：提速与可读性调整后 24 帧、20 fps GIF 冒烟
  测试 PASS：overview 相机临时从 40° 拉近到 27° FOV，机械臂显示区域扩大；
  新周期速度相对旧版提高，模型姿态接口与 A/B/C 演示幅度未改变。
- 2026-11-11 10:24 +08:00：最终提速版 GIF 渲染 PASS：48 帧、20 fps、
  总周期 2.4 s，较旧版 72 帧、12 fps、6.0 s 的往返速度提高 2.5 倍；
  预览确认整机放大后无裁切，眼内视角与关节读数同步。
- 2026-11-11 10:25 +08:00：完成前新鲜复验 PASS：`py_compile` 与
  `git diff --check` 均通过；最终 GIF 为 48 帧、1280×760、20 fps、
  每帧 50 ms、周期 2.4 s、无限循环，文件 4,615,763 bytes；首帧到峰值帧、
  峰值帧到末帧 RGB RMS 分别为 15.252 和 15.259；元数据中的动画文件名、
  FOV 与时长均正确。
- 2026-11-11 10:38 +08:00：16:9 半尺寸版式 24 帧冒烟测试 PASS：
  输出 960×540，整机视图、眼内视图、关节读数和底部进度条均成功渲染。
- 2026-11-11 10:39 +08:00：半尺寸视觉回查发现底部说明与进度线重叠，
  将进度线移至画布最底部后复跑同一 24 帧冒烟测试 PASS，文字与进度条分离。
- 2026-11-11 10:40 +08:00：最终 PPT 16:9 半尺寸 GIF 渲染 PASS：
  960×540、48 帧、20 fps、周期 2.4 s；预览确认文字、进度条和两个相机视图
  均无重叠或裁切。
- 2026-11-11 10:41 +08:00：完成前新鲜复验 PASS：`py_compile` 与
  `git diff --check` 均通过；GIF 与 PNG 均为严格 960×540、16:9，GIF
  48 帧、20 fps、每帧 50 ms、周期 2.4 s、无限循环，文件 2,832,669 bytes；
  首帧到峰值帧、峰值帧到末帧 RGB RMS 分别为 15.089 和 14.991；元数据确认
  画布为 1920×1080 PPT 画布的一半。
- 2026-11-11 10:52 +08:00：核对仓库现有 16:9 PPT 基准为 1280×720
  （SVG viewBox 与 round-trip 清单一致），因此半尺寸应为 640×360；
  更新自适应字号与读数布局后 24 帧冒烟测试 PASS。
- 2026-11-11 10:53 +08:00：最终 640×360 GIF 渲染命令首次误用不可用 bash，
  未进入脚本；按 `systematic-debugging` 确认属于 shell 调用错误，改用
  PowerShell 后渲染 PASS：48 帧、20 fps、2.4 s。
- 2026-11-11 10:54 +08:00：完成前新鲜复验 PASS：`py_compile` 与
  `git diff --check` 均通过；GIF 与 PNG 均为 640×360、严格 16:9，基准明确为
  1280×720 PPT 画布的一半；GIF 48 帧、20 fps、每帧 50 ms、2.4 s、
  无限循环，文件 1,581,567 bytes；关键帧 RGB RMS 为 15.869 / 15.730。
- 2026-11-11 13:22 +08:00：PowerPoint COM 无保存试插入 PASS：在
  960×540 pt 的 16:9 页面中，当前 GIF 默认插入尺寸为 480×270 pt，
  宽高均恰为页面 50%，证明 GIF 文件本身尺寸正确。
- 2026-11-11 13:23 +08:00：GIF DPI 元数据探针显示 Pillow 不会在 GIF
  `info` 中保留 `dpi=(96,96)`；PowerPoint 实测仍按 96 PPI 将
  640×360 px 映射为 6.667×3.75 in，与半页物理尺寸一致。
- 2026-11-11 14:31 +08:00：经 `127.0.0.1:7897` 代理访问 GitHub 与
  `git ls-remote` 均 PASS；目标驱动仓库
  `Kiborgik/aic8800dc-linux-patched` 的 `main` 提交为
  `ea3043a1b7dd3ff69fdd6954430b3e419279ac75`。
- 2026-11-11 14:31 +08:00：一次 README 拉取 FAIL，原因为将 `shell` 错设为
  `bash` 后经由不可用的 WSL 启动；按 `systematic-debugging` 定位为调用层
  错误，改用 PowerShell 后同一 GitHub 代理连接 PASS。未进入安装或驱动测试。
- 2026-11-11 14:33 +08:00：AIC8800FC Ubuntu 离线包静态测试 PASS。
  `install-aic8800fc.sh` 与 `diagnose-aic8800fc.sh` 均通过 Git Bash
  `bash -n`；源码包 SHA-256 为
  `2e887d820fadd6b12ba6e117e42666b321da76c9c5b1cf82132d46754d440af5`，
  `sha256sum -c` PASS。
- 2026-11-11 14:33 +08:00：离线压缩包解压 PASS，根目录正确，确认源码包含
  `USB_PRODUCT_ID_AIC8800FC_U2 0x88de`，并包含 165,835-byte
  `lmacfw_rf_8800dc.bin` 固件。上游固定提交 `ea3043a` 的 GitHub Actions
  6.12 LTS、6.18、7.1、latest、mainline-rc 与 scripts 检查均为 success。
  尚未在目标 Ubuntu 内核上编译或进行真实联网测试。
- 2026-11-11 14:34 +08:00：将安装包复制到 `F:\aic8800fc-ubuntu` FAIL，
  Windows 返回“媒体受写保护”。按 `systematic-debugging` 核查：
  `Get-Disk -Number 2` 显示 `AIC flash`、USB、`IsReadOnly=True`；
  `StorageDevicePolicies` 不存在，`USBSTOR Start=3`，因此不是常见注册表
  写保护策略，而是 AIC 驱动盘模式报告的只读介质。原始
  `aic8800fc_windows_wifi_driver.exe` 哈希仍为
  `450C4AEC...A716F0058`。安装包保留在工作区
  `deliverables/aic8800fc-ubuntu/`，未改写或格式化该设备。
- 2026-11-11 14:35 +08:00：完成前新鲜复验 PASS：离线包 5 个文件均存在，
  源码包 443,844 bytes、157 个归档条目；实际 SHA-256 与
  `SHA256SUMS.txt` 及安装脚本内固定值一致。安装/诊断脚本
  `bash -n` 退出码均为 0；归档内确认 `0x88de` 和 RF 固件存在。
  因目标 Ubuntu 环境尚未提供，实际 DKMS 编译、网卡模式切换和联网仍待
  Ubuntu 实机验证；当前未宣称无线功能已最终可用。
- 2026-11-11 14:48 +08:00：`Set-Disk -Number 2 -IsReadOnly $false` 返回
  Access Denied。按 `systematic-debugging` 检查发现磁盘枚举已变化：当前
  磁盘 2 是 64 GB `aigo U310` Ubuntu 24.04 启动盘，`IsReadOnly=False`，
  不再需要清除只读标志；该失败仅为不必要的提权调用，未改变磁盘。
- 2026-11-11 14:48 +08:00：AIC8800FC 离线包已写入
  `F:\aic8800fc-ubuntu`。复验 PASS：5 个文件与工作区源文件 SHA-256
  逐一相同；F 盘脚本 `bash -n` 退出码均为 0；`sha256sum -c` 退出码 0；
  驱动归档 157 个条目，源码包哈希
  `2e887d820fadd6b12ba6e117e42666b321da76c9c5b1cf82132d46754d440af5`，
  并确认 `0x88de` 与 RF 固件存在。F 盘仍为 FAT32、
  `IsReadOnly=False`，剩余 56,209,997,824 bytes。
- 2026-11-11 14:55 +08:00：新增 `安装命令.txt`，写入安装、USB 标识检查、
  Wi-Fi 扫描/连接、诊断日志与卸载命令。首次复制命令错误指定 Bash shell，
  经不可用 WSL 启动而 FAIL；按 `systematic-debugging` 确认是调用层错误，
  改用 PowerShell 后复制 PASS。一次 PowerShell 内容检查因字符串转义方式
  错误误报 `False`，改用 `Select-String -SimpleMatch` 后确认
  `sudo bash "$INSTALLER"` 存在，命令行检查 PASS。
- 2026-11-11 14:55 +08:00：F 盘安装目录最终复验 PASS：共 6 个文件、
  454,594 bytes；`安装命令.txt` 与工作区 SHA-256 一致
  `1375BBD9EBC4F44CFEB628169ECBC0C3B685B8AADE1BF58BE0C550A23DA01434`；
  README 已同步并包含命令文件入口，驱动归档哈希仍为
  `2e887d820fadd6b12ba6e117e42666b321da76c9c5b1cf82132d46754d440af5`。
- 2026-11-11 15:03 +08:00：读取 `F:\aic8800fc-ubuntu` 的三项检查均 FAIL，
  错误为“驱动器 F 不存在”。按 `systematic-debugging` 判断这属于预期环境
  变化：F 是当前正在启动的 Ubuntu Live USB，Windows 失去盘符并不表示文件
  丢失。为处理用户报告的 `cd: 参数太多`，已在工作区
  `安装命令.txt` 增加“含空格路径必须加引号”和
  `cd /cdrom/aic8800fc-ubuntu` 两种正确命令；因 F 正被 Ubuntu 占用，本轮
  无法从 Windows 同步回 F，待切回 Windows 后可再次同步。
- 2026-11-11 15:04 +08:00：一个只读内容检查再次误将 PowerShell 命令交给
  不可用 WSL Bash，未执行检查；改用 PowerShell 后 PASS。工作区
  `安装命令.txt` 同时包含带引号路径
  `cd "/media/$USER/UBUNTU 24_0/aic8800fc-ubuntu"` 与
  `cd /cdrom/aic8800fc-ubuntu`，两者均已通过精确字符串检查。
- 2026-11-11 15:18 +08:00：查明“未找到 install.sh”根因：压缩包中的
  `install.sh` 实际存在，但 Windows 生成归档时权限为 `-rw-rw-rw-`；
  安装器错误使用 `[[ -x ... ]]` 判断，导致误报。已改为 `[[ -f ... ]]`
  并使用 `bash ./install.sh`，不再依赖可执行位。
- 2026-11-11 15:18 +08:00：直接在 F 盘写入根目录 `install.sh` 一键入口，
  并将完整驱动源码解压至 `F:\aic8800fc-ubuntu\aic8800dc-linux-patched`。
  复验 PASS：根目录与源码目录的两个 `install.sh` 均存在，源码目录共
  147 个文件、2,378,818 bytes；三个 Bash 入口脚本 `bash -n` 退出码均为 0。
  一次文件统计命令误交不可用 WSL Bash 而未执行，改用 PowerShell 后 PASS。
- 2026-11-11 15:18 +08:00：F 盘更新文件复验 PASS：`install.sh`、
  `install-aic8800fc.sh`、`安装命令.txt`、README、校验清单和驱动归档
  均与工作区 SHA-256 一致；安装器确认只使用 `-f` 检查并调用
  `bash ./install.sh`。F 盘为 `aigo U310`、FAT32、
  `IsReadOnly=False`，剩余 56,203,345,920 bytes。
- 2026-11-11 15:51 +08:00：Ubuntu 安装测试 FAIL：
  `./install.sh: 行 2: $'\r': 未找到命令`，随后出现 `do\r` 语法错误。
  按 `systematic-debugging` 检查 F 盘驱动源码，139 个文本文件中有 133 个
  为 CRLF。根因是 Windows 检出和打包时写入了 CRLF，Bash 与 Make 不能直接
  使用。已将工作区与 F 盘全部构建文本机械转换为 UTF-8/LF。
- 2026-11-11 15:51 +08:00：重新生成 Linux 归档并同步 F 盘 PASS：
  新归档 441,860 bytes、157 个条目，`install.sh` 权限为 `-rwxr-xr-x`；
  F 盘 139 个文本文件剩余 CRLF 数量为 0；7 个 shell 脚本全部通过
  `bash -n`。F 盘驱动源码与工作区备份均为 147 个文件且 SHA-256
  逐文件一致。新归档 SHA-256 为
  `d22212e25da5ea52ff4a81a0e018a0d3cf0416ff5c9754093feb44fdb07e806d`，
  `sha256sum -c` 与安装器内固定哈希均 PASS。
- 2026-11-11 15:51 +08:00：两次只读验证命令误将 PowerShell 代码交给不可用
  WSL Bash，均在执行前 FAIL；改用 PowerShell 后同一验证 PASS。未修改
  磁盘内容。Secure Boot 仍为开启状态，实际模块加载需要关闭 Secure Boot
  或完成 MOK 签名。
- 2026-11-11 16:12 +08:00：Ubuntu 6.17.0-14-generic 实机首轮驱动加载
  FAIL。DKMS 显示 `aic8800dc/6.4.3.0-patched.20` 已安装，USB 已进入
  Wi-Fi 模式 `a69c:88dc AICSemi AIC8800DC`，但 `ip` 只有 `lo` 和
  `enp3s0`，没有无线接口。按 `systematic-debugging` 确认 `88dc` 是驱动
  源码正式支持的 AIC8800DC ID，原安装器仅检查 `88de`，因此验收判断有误。
- 2026-11-11 16:12 +08:00：已修改并同步 F 盘安装器：支持完整的
  AIC8800DC/FC USB ID 列表；`modprobe` 不再吞掉错误；无无线接口时输出
  相关 dmesg 并提示检查 Secure Boot。更新后的脚本 `bash -n` 退出码为 0。
  因输出中未见 `aic_load_fw` / `aic8800_fdrv` 模块，仍需在 Ubuntu 中确认
  `mokutil --sb-state` 与 modprobe 报错，尚未确认无线功能可用。
- 2026-11-11 16:20 +08:00：重新核对驱动源码 PASS：
  `aic8800_fdrv` 的 USB 表包含 `a69c:88dc`，`CONFIG_USB_BT=y`，
  `aicwf_usb_chipmatch()` 明确把该 ID 映射为 `PRODUCT_ID_AIC8800DC`。
  因此 `88dc` 不是不支持或错误设备；udev 将存储模式切到 `88dc` 也不能
  证明 WiFi 模块已绑定。当前证据可确定安装器的 `88de` 专属验收规则有误，
  但“没有 wlan 接口”的根因仍取决于 Ubuntu 中的 modprobe 返回值和 dmesg，
  不能单独归因于 Secure Boot。
- 2026-11-11 16:28 +08:00：实机诊断确认真实根因 PASS：Ubuntu 输出
  `SecureBoot enabled`；`modprobe -v aic8800_fdrv` 返回
  `Key was rejected by service`；dmesg 记录
  `Loading of module with unavailable key is rejected`。因此 USB 虽已切换到
  `a69c:88dc AIC8800DC`，内核仍无法加载驱动，也就不会创建无线接口或显示
  Wi-Fi 图标。DKMS 构建与设备 ID 均无问题。
- 2026-11-11 16:28 +08:00：F 盘安装器已同步增强：当主模块
  `aic8800_fdrv` 加载失败且 Secure Boot 开启时，立即报告密钥被拒绝并退出，
  不再等待 30 秒或将问题误报为 USB ID 不匹配。更新脚本 `bash -n` PASS。
- 2026-11-11 16:44 +08:00：VS Code Remote SSH 连接 `yanke_4090` FAIL。
  `ssh -G` 确认目标为 `him1@192.168.1.107:22`；Windows 本机源地址为
  `192.168.1.102`，`ping 192.168.1.107` 两次成功（约 4 ms），但
  `Test-NetConnection -Port 22` 的 `TcpTestSucceeded=False`。结论为远端
  主机在线但 22 端口无 SSH 服务监听；不是 VS Code 代理、DNS 或物理链路故障。
- 2026-11-11 16:44 +08:00：同一诊断批次中一条 `ssh -G` 命令误交不可用
  WSL Bash，另一条 PowerShell 命令误交 cmd，均未执行目标检查；改用
  PowerShell 后 `ssh -G` 验证 PASS，并通过独立端口测试完成根因定位。
- 2026-11-11 19:16 +08:00：CC-Switch AppImage 本地文件检查 PASS：
  `D:\浏览器下载\CC-Switch-v4.0.24-Linux-x86_64.AppImage` 存在，
  大小 97,454,584 bytes，SHA-256 为
  `ef915ece18c12598485662516c5cb76fdd88dac728fedaf35a041e9a26abe440`。
  该哈希可用于 `scp` 传输后的完整性核对。
- 2026-11-11 20:02 +08:00：Ubuntu 主机硬件检查 PASS：主机名
  `him1-Z790-D-DDR4`，可见显卡仅 Intel UHD Graphics 770（i915）；
  `lspci -nn | grep -i nvidia` 与 `/dev/nvidia*` 均无结果，
  `ubuntu-drivers devices` 也为空，因此当前主机不存在可供系统使用的
  NVIDIA GPU，不能运行 `nvidia-smi`，也不应安装 NVIDIA 驱动。
- 2026-11-11 20:02 +08:00：同一检查显示 AIC USB Wi-Fi 已工作：
  接口 `wlx688fc92e9ce5` 为 UP 状态并取得
  `192.168.1.107/24` IPv4 地址；此前 Secure Boot 拒绝驱动的问题已解除。
- 2026-11-11 20:14 +08:00：GPU 与 SSH 目标交叉检查 PASS：Windows 本机
  `nvidia-smi` 实际报告 RTX 4050 Laptop、6,141 MiB 显存；Windows SSH
  配置中 `yanke_4090` 映射为 `him1@192.168.1.107:22`。用户口中的目标机
  该远端只枚举到 Intel UHD 770，没有 NVIDIA PCI 设备。若 `yanke_4090`
  确是目标机，则 4090 未对该 Linux 暴露，可能为未安装、供电/PCIe 未工作
  或虚拟机缺少 GPU 直通，而不是 NVIDIA 驱动缺失。
- 2026-11-11 20:14 +08:00：同一检查批次中两条 PowerShell 命令误交不可用
  WSL Bash，未执行；其余 `nvidia-smi`、SSH 配置和局域网邻居检查成功。
- 2026-11-11 20:18 +08:00：`lspci -tv` / `lshw -C display` 实机检查 PASS：
  Gigabyte Z790 D DDR4 物理拓扑中只有一个显示控制器
  Intel UHD 770（`00:02.0`，i915）；不存在任何 `10de:` NVIDIA PCI
  设备。00:06 根端口下为 Sandisk NVMe，x16 GPU 端点未枚举。结论：
  4090 没有暴露给操作系统，问题发生在驱动之前的硬件/PCIe/固件层；
  当前 Linux 内无法通过安装驱动修复。
- 2026-11-11 20:24 +08:00：主机确认 `systemd-detect-virt=none`，为物理机。
  `dmidecode -t slot` 将 x16 插槽 `J6B2 / 00:01.0` 报告为
  `Current Usage: In Use`，但 `lspci -nn` 与 dmesg 中完全没有
  `00:01.0` 根端口或 `10de:` 设备。该矛盾说明 4090 很可能物理在位，
  但 PCIe link training、供电或 BIOS 根端口启用失败，问题在驱动加载之前。
  下一步应检查 12VHPWR 供电、重插显卡、BIOS 显示/PCIe 设置和 BIOS 版本。
- 2026-11-11 20:55 +08:00：RCM 数据集到终测交互式教程静态验证 PASS。
  使用 `C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe` 解析
  `outputs/rcm_dataset_to_result_guide_20261111.html`：60,172 bytes、
  15 个 HTML id、47 个 href、9 张表、1 个内联脚本；47 个链接全部可定位，
  内部锚点全部存在，无外部 `http`/`https`/脚本资源依赖。
  同一批次用 NumPy 核对目标 manifest、基础与聚合 dataset、视觉缓存、
  sidecar metadata、Stage 2 贯穿 episode 及 Stage 4 汇总，各项断言 PASS。
- 2026-11-11 20:59 +08:00：在应用内浏览器检查 RCM 教程 FAIL（工具环境）。
  `node_repl` 连最小 `nodeRepl.write(nodeRepl.cwd)` 也返回
  `Mcp error: -32602: missing field sandboxPolicy`；重置内核后同一最小
  用例仍稳定复现，证明失败发生在页面加载之前，不是 HTML/JS 错误。
  页面未执行，后续改用本机 Chrome/Edge 无头模式做离线渲染回退验证。
- 2026-11-11 21:03 +08:00：Chrome 156 无头交互验证 PASS。
  通过独立 CDP 端口在离线网络模式下打开本地 `file://` 页面：
  12 个章节、12 个进度复选框、47 个链接、9 张表、14 个问答均渲染；
  页面高度 13,005 px，视口 1440×1,200，无外部脚本或媒体资源。
  勾选首节后进度由 `0 / 12` 变为 `1 / 12`，localStorage 保存为 `1`；
  重载后仍为 `1 / 12`；目录跳到 `#results` 后结果段可见，问答可展开，
  Runtime/Log 未记录脚本错误。首屏与结果页截图均完成目检。
  同批次一次测试脚本曾因普通 Node 中不存在仅 MCP 提供的 `nodeRepl`
  输出对象而终止；浏览器动作和截图已完成，改用 `console.log` 后全量重跑
 PASS，产品文件未因该测试脚本问题修改。
- 2026-11-11 21:05 +08:00：完成前最终验证 PASS。
  最终静态/数据审计使用 `C:\ProgramData\Miniconda3\envs\rcm_mujoco\python.exe`：
  教程 60,166 bytes，SHA-256
  `41B366B6F52705DC34D21064F7E7FFCA8C1CA10753F909ECA7534B9F960302E0`；
  15 个 id、47 个链接、9 张表；12 项内容清单全部满足，47 个链接与
  内部锚点全部有效，无外部资源。目标计数、608/195,578 聚合数据、
  `(195578,512) float16` 视觉缓存、sidecar 白名单与真值排除、
  Stage 2 贯穿 episode 行区间、Stage 4 `20/180` 与 `18/60` 均断言 PASS。
  最终浏览器复测再次在离线 Chrome 156 中 PASS：12 节、`0/12 → 1/12`
  勾选、重载持久化、`#results` 跳转与问答展开正常，脚本错误数为 0；
  首屏和结果页截图均已人工目检。最终一次断言命令曾因 memmap 不支持
  上下文管理器而在测试脚本中终止，改为直接读取后完整重跑 PASS；
  该问题未影响产品或数据。
  已知限制：应用内 `node_repl` 浏览器工具受会话 `sandboxPolicy` 元数据
  错误阻塞，浏览器结论由本机 Chrome 无头 CDP 回退取得；大型 `runs/`、
  `data/`、`outputs/` 被 Git 忽略，换机器后本地证据可能不可用。
- 2026-11-11 20:31 +08:00：重试硬件检查 PASS：RTX 4090 已正常枚举为
  `01:00.0 [10de:2684] NVIDIA AD102`，HDMI Audio 为
  `01:00.1 [10de:22ba]`，PCIe 根端口 `00:01.0` 正常出现。dmesg 显示
  `nouveau 0000:01:00.0: NVIDIA AD102 (192000a1)`。此前未见 GPU 属于
  一次未被操作系统枚举的瞬时/重启前状态；当前硬件识别正常，下一步应安装
  Ubuntu 推荐的 NVIDIA 专有驱动。
- 2026-11-11 20:38 +08:00：新内核 DKMS 预构建 FAIL。用户复制命令时
  `linux-hwe-7.0-tools` 在连字符后换行，Shell 将 `linux-hwe-7.0-` 和
  `tools` 当成两条命令；同时 `linux-headers-7.0.0-42-generic` 实际未安装，
  所以 `dkms autoinstall -k 7.0.0-42-generic` 因找不到
  `/lib/modules/7.0.0-42-generic/build` 而失败。当前 `aic8800dc`
  仅对旧内核 `6.17.0-14-generic` 安装成功。修复为先单独安装精确头文件包，
 再对新内核重新执行 DKMS 构建。
- 2026-11-11 20:43 +08:00：新内核 DKMS 验证 PASS：
  `dkms install -m aic8800dc -v 6.4.3.0-patched.20 -k 7.0.0-42-generic`
  返回 `already installed on kernel 7.0.0-42-generic`，说明 AIC WiFi
  模块已针对 NVIDIA 驱动随附的新内核成功构建安装。无需 `--force`，
  下一步应重启并在新内核上验证 NVIDIA、WiFi 与 DKMS 状态。
- 2026-11-11 20:47 +08:00：用户确认 Ubuntu 重启后验证全部成功。最终状态为
  Linux `7.0.0-42-generic`，AIC8800FC WiFi DKMS 驱动在该内核工作，
  RTX 4090 由 Ubuntu `nvidia-driver-595-open` 驱动。此前 Secure Boot 拒绝
  DKMS 模块、压缩包 CRLF、安装器设备 ID 判定及新内核头文件缺失问题均已解决。
- 2026-11-12 09:21 +08:00：Linux 容器主入口冒烟测试 PASS。
  使用本机 Docker `linux/amd64` 与 `python:3.11-slim`，安装
  `requirements.txt` 及 EGL/OpenGL 运行库后，环境为
  Linux 6.6（WSL2 内核）、Python 3.11.15、MuJoCo 3.3.7、NumPy 2.4.6。
  `v9_eye_scene.xml` 原生加载 PASS（`nq=7`、`nbody=13`、`ngeom=104`）；
  `MUJOCO_GL=egl python run_reference.py --no-viewer` PASS，
  返回 `model_backend=RCMSceneV9`，RCM 位于眼表且针尖位于眼内。
  目标 Ubuntu 主机 `192.168.1.108:22` 端口可达，但当前会话没有该主机的
  公钥或密码认证凭据，远程实机执行未进行。
- 2026-11-12 09:24 +08:00：Linux 容器单元测试首轮 FAIL（29 项中 28 项
  PASS、1 项 ERROR）。错误出现在
  `tests.test_restricted_sidecar.test_sidecar_schema_and_truth_omission`，
  `np.savez` 返回 `OSError: [Errno 30] Read-only file system`，
  目标为 `/workspace/outputs/test_restricted_sidecar/dataset.npz`。
  按 `systematic-debugging` 检查堆栈和测试源码后确认：该测试固定写入
  `ROOT / "outputs" / "test_restricted_sidecar"`，而本轮为保护工作区
  把 `/workspace` 以只读方式挂载。当前证据指向测试运行环境，尚不能据此
  判定 Linux 产品代码失败；修正方式应是为测试提供可写的隔离输出层。
- 2026-11-12 09:26 +08:00：按 `systematic-debugging` 完成最小假设验证
  PASS。以只读绑定保留项目源码，仅在 `/workspace/outputs` 覆盖 512 MiB
  tmpfs 后，测试探针确认输出目录可写且源码仍只读；随后单独复跑
  `tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission`
  PASS（1/1）。根因确认是此前测试容器的只读挂载，而非 Linux 代码路径。
- 2026-11-12 09:27 +08:00：Linux 容器非 PyTorch 单元测试全量复跑 PASS。
  同一隔离可写输出配置下执行 8 个测试模块，共 29/29 PASS，耗时 13.009 s；
  覆盖 restricted perception、sidecar、目标划分、统计、Stage 3/4 分析、
  相机外参和批处理运动学。无失败或跳过。
- 2026-11-12 09:30 +08:00：Linux 容器完整单元测试 PASS。因
  `requirements.txt` 未声明 PyTorch，额外安装 CPU 版
  `torch 2.15.0+cpu` 后执行 `python -m unittest discover -v -s tests`，
  共 35/35 PASS，耗时 15.275 s，包括完整的 restricted policy 与 Stage 3
  残差策略测试。无失败或跳过。该结果证明代码在 Linux 上可运行，同时确认
  Linux 基础安装清单不含 restricted-learning 功能所需的 PyTorch。
- 2026-11-12 09:33 +08:00：Linux 无窗口 EGL 渲染 PASS。
  `MUJOCO_GL=egl python run_reference.py --render /tmp/rcm-v9-overview.png
  --camera overview` 成功生成 131,706 bytes 的 RGB PNG；Pillow 完整性检查
  与重新解码均 PASS，尺寸为 1600×1080。该结果确认 Linux 容器中的
  MuJoCo OpenGL/EGL 渲染路径可用，而不仅是模型解析可用。

## 2026-11-12T09:21:44+08:00 | FAILED
<!-- codex-test:c4a0f89491459d82e0f3fa02 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未提供，由测试输出末尾判定为失败（推断）
- 输出记录：[test-logs/c4a0f89491459d82e0f3fa02.log](test-logs/c4a0f89491459d82e0f3fa02.log)

### 命令

```text
docker exec -e PYTHONPATH=. -e MUJOCO_GL=egl -e MPLBACKEND=Agg rcm-linux-verify-20261112 bash -lc "python -m unittest -v tests.test_causal_and_degradation tests.test_restricted_perception tests.test_restricted_perception_experiments tests.test_restricted_sidecar tests.test_target_split tests.test_statistics tests.test_stage34 tests.test_stage4_analysis"
```

### 输出末尾摘要

```text
nsitive) ... ok
test_nearest_neighbor_and_self_exclusion (tests.test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok
test_pairs_by_target_and_seed (tests.test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok
test_geometric_action_moves_toward_target_minus_tip (tests.test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (tests.test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (tests.test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok
test_failure_classification_separates_hold_and_initialization (tests.test_stage4_analysis.Stage4AnalysisTests.test_failure_classification_separates_hold_and_initialization) ... ok
test_value_summary_uses_requested_population_statistics (tests.test_stage4_analysis.Stage4AnalysisTests.test_value_summary_uses_requested_population_statistics) ... ok

======================================================================
ERROR: test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/workspace/tests/test_restricted_sidecar.py", line 64, in test_sidecar_schema_and_truth_omission
    np.savez(
  File "/usr/local/lib/python3.11/site-packages/numpy/lib/_npyio_impl.py", line 673, in savez
    _savez(file, args, kwds, False, allow_pickle=allow_pickle)
  File "/usr/local/lib/python3.11/site-packages/numpy/lib/_npyio_impl.py", line 779, in _savez
    zipf = zipfile_factory(file, mode="w", compression=compression)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numpy/lib/_npyio_impl.py", line 112, in zipfile_factory
    return zipfile.ZipFile(file, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/zipfile.py", line 1311, in __init__
    self.fp = io.open(file, filemode)
              ^^^^^^^^^^^^^^^^^^^^^^^
OSError: [Errno 30] Read-only file system: '/workspace/outputs/test_restricted_sidecar/dataset.npz'

----------------------------------------------------------------------
Ran 29 tests in 12.756s

FAILED (errors=1)
```

## 2026-11-12T09:22:39+08:00 | UNKNOWN
<!-- codex-test:331dd5b0560f057d7f7a859c -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未知，不能据此断言测试通过
- 输出记录：[test-logs/331dd5b0560f057d7f7a859c.log](test-logs/331dd5b0560f057d7f7a859c.log)

### 命令

```text
docker exec -e PYTHONPATH=. -e MUJOCO_GL=egl -e MPLBACKEND=Agg rcm-linux-verify-20261112 python -m unittest -v tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission
```

### 输出末尾摘要

```text
w s l :   �hKm0R  l o c a l h o s t   �NtM�n�FO*g\��P0R  W S L 0N A T   !j_N�v  W S L   N/ec  l o c a l h o s t   �Nt0 
 <3>WSL (2317 - Relay) ERROR: CreateProcessCommon:735: execvpe(/bin/bash) failed: No such file or directory
```

## 2026-11-12T09:22:46+08:00 | SUCCESS
<!-- codex-test:e8ea4edcf0c4d4a09967c338 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未提供，由测试输出末尾判定为通过（推断）
- 输出记录：[test-logs/e8ea4edcf0c4d4a09967c338.log](test-logs/e8ea4edcf0c4d4a09967c338.log)

### 命令

```text
docker exec -e PYTHONPATH=. -e MUJOCO_GL=egl -e MPLBACKEND=Agg rcm-linux-verify-20261112 python -m unittest -v tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission
```

### 输出末尾摘要

```text
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok

----------------------------------------------------------------------
Ran 1 test in 1.040s

OK
```

## 2026-11-12T09:23:12+08:00 | SUCCESS
<!-- codex-test:32f69bcfa38b981936b324fe -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未提供，由测试输出末尾判定为通过（推断）
- 输出记录：[test-logs/32f69bcfa38b981936b324fe.log](test-logs/32f69bcfa38b981936b324fe.log)

### 命令

```text
docker exec -e PYTHONPATH=. -e MUJOCO_GL=egl -e MPLBACKEND=Agg rcm-linux-verify-20261112 bash -lc "python -m unittest -v tests.test_causal_and_degradation tests.test_restricted_perception tests.test_restricted_perception_experiments tests.test_restricted_sidecar tests.test_target_split tests.test_statistics tests.test_stage34 tests.test_stage4_analysis"
```

### 输出末尾摘要

```text
is_not_in_policy_features) ... ok
test_closed_loop_does_not_stop_on_early_precision_hold (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_does_not_stop_on_early_precision_hold) ... ok
test_closed_loop_is_deterministic (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_closed_loop_is_deterministic) ... ok
test_precision_scan_exposes_all_levels (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_precision_scan_exposes_all_levels) ... ok
test_switch_scan_has_required_distances_and_gate (tests.test_restricted_perception_experiments.RestrictedPerceptionExperimentTests.test_switch_scan_has_required_distances_and_gate) ... ok
test_batched_kinematics_matches_reference (tests.test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (tests.test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_checked_in_manifest_has_frozen_separation (tests.test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (tests.test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (tests.test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok
test_pairs_by_target_and_seed (tests.test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok
test_geometric_action_moves_toward_target_minus_tip (tests.test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (tests.test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (tests.test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok
test_failure_classification_separates_hold_and_initialization (tests.test_stage4_analysis.Stage4AnalysisTests.test_failure_classification_separates_hold_and_initialization) ... ok
test_value_summary_uses_requested_population_statistics (tests.test_stage4_analysis.Stage4AnalysisTests.test_value_summary_uses_requested_population_statistics) ... ok

----------------------------------------------------------------------
Ran 29 tests in 13.009s

OK
```

## 2026-11-12T09:24:41+08:00 | SUCCESS
<!-- codex-test:90c60f287593519ce896c3a6 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未提供，由测试输出末尾判定为通过（推断）
- 输出记录：[test-logs/90c60f287593519ce896c3a6.log](test-logs/90c60f287593519ce896c3a6.log)

### 命令

```text
docker exec -e PYTHONPATH=. -e MUJOCO_GL=egl -e MPLBACKEND=Agg rcm-linux-verify-20261112 python -m unittest discover -v -s tests
```

### 输出末尾摘要

```text
ch_scan_has_required_distances_and_gate) ... ok
test_feature_matrix_shape (test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_residual_cap_uses_ten_micrometre_scale (test_restricted_policy.RestrictedPolicyTests.test_residual_cap_uses_ten_micrometre_scale) ... ok
test_source_filter_returns_boolean_mask (test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_batched_kinematics_matches_reference (test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_geometric_action_moves_toward_target_minus_tip (test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok
test_residual_is_bounded (test_stage3_policy.Stage3PolicyTests.test_residual_is_bounded) ... ok
test_zero_initialized_residual_preserves_base_action (test_stage3_policy.Stage3PolicyTests.test_zero_initialized_residual_preserves_base_action) ... ok
test_failure_classification_separates_hold_and_initialization (test_stage4_analysis.Stage4AnalysisTests.test_failure_classification_separates_hold_and_initialization) ... ok
test_value_summary_uses_requested_population_statistics (test_stage4_analysis.Stage4AnalysisTests.test_value_summary_uses_requested_population_statistics) ... ok
test_pairs_by_target_and_seed (test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok
test_checked_in_manifest_has_frozen_separation (test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok

----------------------------------------------------------------------
Ran 35 tests in 15.275s

OK
```

- 2026-11-12 09:33 +08:00：Linux 无窗口 EGL 渲染 PASS。
  `MUJOCO_GL=egl python run_reference.py --render /tmp/rcm-v9-overview.png
  --camera overview` 生成 131,706 bytes 的 PNG；Pillow 完整性检查和重新解码
  均 PASS，尺寸 1600×1080、RGB。确认 MuJoCo 的 Linux EGL 渲染路径可用。
- 2026-11-12 09:34 +08:00：Linux 单步视觉伺服运行时冒烟检查 PASS。
  执行
  `MUJOCO_GL=egl MPLBACKEND=Agg python run_visual_servo.py --no-viewer
  --max-iterations 1 --output-dir /tmp/rcm-visual-smoke-20261112`。
  进程退出码为 1，但报告明确为 `result=FAIL`、`reason=MAX_ITERATIONS`，
  与人为设置的 1 步上限一致，不是异常崩溃。实际完成了 strict RGB-D 感知、
  1920×1440 深度帧、目标/针尖几何测量、`RCMSceneV9` 相机渲染和
  `MotorBackend` 调用；最终视觉误差约 19.75 mm，未宣称导航成功。

## 2026-11-12T09:27:39+08:00 | SUCCESS
<!-- codex-test:4f731ce10a087107d1ce7886 -->

- 来源：PostToolUse
- 工作目录："E:\\30_RESOURCES\\32_课程教程\\医学\\中科院\\rcm-mujoco-reference-ffd7b15d41c51c2d2f8a15596e9575106f81c225"
- 退出码：未提供，由测试输出末尾判定为通过（推断）
- 输出记录：[test-logs/4f731ce10a087107d1ce7886.log](test-logs/4f731ce10a087107d1ce7886.log)

### 命令

```text
docker exec -e PYTHONPATH=. -e MUJOCO_GL=egl -e MPLBACKEND=Agg rcm-linux-verify-20261112 python -m unittest discover -v -s tests
```

### 输出末尾摘要

```text
ch_scan_has_required_distances_and_gate) ... ok
test_feature_matrix_shape (test_restricted_policy.RestrictedPolicyTests.test_feature_matrix_shape) ... ok
test_policy_output_shape_and_action_clip (test_restricted_policy.RestrictedPolicyTests.test_policy_output_shape_and_action_clip) ... ok
test_residual_cap_uses_ten_micrometre_scale (test_restricted_policy.RestrictedPolicyTests.test_residual_cap_uses_ten_micrometre_scale) ... ok
test_source_filter_returns_boolean_mask (test_restricted_policy.RestrictedPolicyTests.test_source_filter_returns_boolean_mask) ... ok
test_batched_kinematics_matches_reference (test_restricted_sidecar.RestrictedSidecarTests.test_batched_kinematics_matches_reference) ... ok
test_sidecar_schema_and_truth_omission (test_restricted_sidecar.RestrictedSidecarTests.test_sidecar_schema_and_truth_omission) ... ok
test_geometric_action_moves_toward_target_minus_tip (test_stage34.Stage34Tests.test_geometric_action_moves_toward_target_minus_tip) ... ok
test_perturbations_change_one_sensor_factor (test_stage34.Stage34Tests.test_perturbations_change_one_sensor_factor) ... ok
test_run_specs_are_frozen_and_pairing_safe (test_stage34.Stage34Tests.test_run_specs_are_frozen_and_pairing_safe) ... ok
test_residual_is_bounded (test_stage3_policy.Stage3PolicyTests.test_residual_is_bounded) ... ok
test_zero_initialized_residual_preserves_base_action (test_stage3_policy.Stage3PolicyTests.test_zero_initialized_residual_preserves_base_action) ... ok
test_failure_classification_separates_hold_and_initialization (test_stage4_analysis.Stage4AnalysisTests.test_failure_classification_separates_hold_and_initialization) ... ok
test_value_summary_uses_requested_population_statistics (test_stage4_analysis.Stage4AnalysisTests.test_value_summary_uses_requested_population_statistics) ... ok
test_pairs_by_target_and_seed (test_statistics.PairedStatisticsTests.test_pairs_by_target_and_seed) ... ok
test_checked_in_manifest_has_frozen_separation (test_target_split.TargetSplitTests.test_checked_in_manifest_has_frozen_separation) ... ok
test_hash_is_stable_and_seed_sensitive (test_target_split.TargetSplitTests.test_hash_is_stable_and_seed_sensitive) ... ok
test_nearest_neighbor_and_self_exclusion (test_target_split.TargetSplitTests.test_nearest_neighbor_and_self_exclusion) ... ok

----------------------------------------------------------------------
Ran 35 tests in 15.184s

OK
```

- 2026-11-12 09:29 +08:00：Linux 完成前新鲜复验 PASS。
  重新执行全部 35 项单元测试，结果 `35/35 OK`（15.184 s）。
  重新执行 `MUJOCO_GL=egl python run_reference.py --verify`，
  返回 `status=PASS`、`rcm_on_eye_surface=true`、`tip_inside_eye=true`；
  重新解码最终渲染图 PASS（131,706 bytes、1600×1080、RGB）。
  一次组合校验包装命令在最后的 PowerShell 内联 Python 摘要处因引号剥离
  而 FAIL，但被包装的 verify 与 render 已先成功；拆分为独立命令后两项均
  PASS，确认失败发生在命令包装层，不影响产品执行。
  目标 Ubuntu `192.168.1.108:22` 再次复核仍返回
  `Permission denied (publickey,password)`，因此远程实机尚未执行。
- 2026-11-12 09:31 +08:00：Linux 受限策略入口检查 PASS（补装依赖后）。
  `requirements.txt` 只声明 MuJoCo、NumPy、Pillow、Matplotlib、trimesh 和
  SciPy；完整 restricted-learning 路径还导入 PyTorch，策略评估入口还导入
  torchvision。额外安装 CPU 版 `torch 2.15.0+cpu` 与
  `torchvision 0.29.2+cpu` 后，
  `python train_restricted_policy.py --help` 和
  `python evaluate_restricted_policy.py --help` 均在 Linux 容器退出码 0。
  结论：代码路径可用，但当前基础安装清单不足以直接启动该功能，需要补装
  PyTorch/torchvision 或增加单独的可选依赖清单。
- 2026-11-12 09:36 +08:00：Linux 完整视觉伺服闭环 PASS。
  在 `MUJOCO_GL=egl`、CPU 容器中执行默认参数的
  `python run_visual_servo.py --no-viewer --output-dir
  /tmp/rcm-visual-full-20261112`，进程退出码 0。结果为
  `result=PASS`、`reason=POSITION_ERROR_AND_SETTLED`，
  共 683 次迭代、682 个控制步、34.1 s 仿真时间。最终真值针尖误差
  6.46 µm，几何测量误差 7.25 µm，连续停稳 3 帧；
  `independent_accuracy_acceptance_pass=true`，低于 20 µm 独立验收阈值。
  过程实际使用 1920×1440 strict RGB-D、`RCMSceneV9`、可见几何感知与
  `MotorBackend`，完成 683 条 OCT 观测记录。该结果是 Linux 仿真可用性证据，
  仍不代表真实硬件、真实 OCT、组织接触或临床精度。
- 2026-11-12 10:08 +08:00：4090 迁移目录 Linux 清单校验 PASS。
  迁移目录为 `E:\30_RESOURCES\32_课程教程\医学\中科院\rcm_4090_deploy`。
  在只读 Linux 容器中运行 `python validate_migration.py`，16 个指定数据集和
  16 个 `image_paths.json` 全部存在；651,683 条图像路径均为 POSIX 相对路径
  且所指文件存在；171 个数据 JSON 均可解析，残留 `E:\`/`E:/` 绝对路径为 0。
  16 个清单共机械替换反斜杠 1,447,919 处，152 个含源机绝对路径的元数据 JSON
  已改写为相对路径。冻结 v7 检查点仍不存在，校验报告将其明确标记为
  `frozen_v7_checkpoint_present=false`。测试记录已追加。
- 2026-11-12 10:17 +08:00：迁移副本代码运行检查 PASS（修复缺项后）。
  首次运行迁移副本的 `run_reference.py` FAIL：迁移清单漏掉该核心入口；
  按 `systematic-debugging` 核对 README 入口与目录后，已补入根目录全部
  109 个 Python 文件。随后 `MUJOCO_GL=egl python run_reference.py --verify`
  返回 `status=PASS`，EGL 渲染 PASS（1600×1080 RGB）。
  181 个 Python 文件全部通过 Linux Python 3.11 语法编译检查。
  `train_stage3_residual_distillation.py --help`、
  `evaluate_restricted_policy.py --help` 均通过。
- 2026-11-12 10:17 +08:00：迁移副本训练入口检查 PASS（补装依赖后）。
  `lerobot_training/train_diffusion_policy.py --help` 首次因缺少 `tqdm`
  FAIL；安装 `tqdm 4.70.1` 后退出码 0，且 `wandb` 被正确识别为可选依赖。
  已安装但不属于 `requirements.txt` 的运行依赖为
  `torch 2.15.0+cpu`、`torchvision 0.29.2+cpu`、`tqdm 4.70.1`；
  在 4090 主机上应安装对应 CUDA 版本。
- 2026-11-12 10:17 +08:00：迁移目录最终清单校验 PASS。
  第一次更新后校验 FAIL，是因为带可写 tmpfs 的验证容器遮住了
  `outputs/` 下模型目录；改用不覆盖 `outputs/` 的只读容器后，
  `validate_migration.py` 返回 `status=PASS`：16 个数据集、
  651,683 条图像路径、171 个数据 JSON、0 个绝对 Windows 路径，
  6 组必需模型/结果目录全部存在。冻结 v7 检查点仍缺失。
  迁移目录实际为 254,240 个文件、6,625,036,008 bytes（6.17 GiB）；
  18 个数据目录的文件数均与源目录一致，非 JSON 资产文件大小全部一致，
  长度差异仅出现在规范化过的 JSON。目标机仍因 SSH 认证被拒而未传输。
- 2026-11-12 10:20 +08:00：迁移目录完成前新鲜复验 PASS。
  在未覆盖 `outputs/` 的只读 Linux 容器中再次运行
  `python validate_migration.py`，仍为 `status=PASS`、0 errors，
  651,683 条图像路径全部存在。抽查的 9 个关键代码、配置和场景文件
  （含 `run_reference.py`、`run_visual_servo.py`、
  `train_diffusion_policy.py`、MuJoCo 场景与模型加载器）源/目标
  SHA-256 全部一致，`hash_mismatches=0`。两份 `TEST_LOG.md` 均已追加记录。
- 2026-11-12 10:34 +08:00：v7 冻结产物恢复与校验 PASS。
  经 `127.0.0.1:7897` 代理获取 GitHub 分支
  `freeze/v7-guided-dagger1-20261106`。远端 HEAD `3f59660` 是
  `Trim generated artifacts from branch archive`，所需文件已从历史提交
  `7cb14d3` 恢复。checkpoint 为 60,608,666 bytes，SHA-256
  `59C8B186F83CD434B7F09C669FE9CC0F7AF9B8E0C0B8C477B8FDC9E439C2BDEB`，
  与 `FREEZE_MANIFEST_20261106.json` 完全一致；两个 evaluation 各 7 个文件，
  14 个文件哈希全部匹配。产物已回填到源项目和
  `rcm_4090_deploy/outputs/`，两处校验均为
  `checkpoint_match=true`、`evaluation_hash_mismatches=0`。
  PyTorch `torch.load` 成功，`PolicyEvaluator` 以 CPU 实例化模型 PASS
  （`type=oct_diffusion_policy`、`epoch=3`）。
- 2026-11-12 10:37 +08:00：v7 回填后迁移最终校验 PASS。
  首次复验发现 `dagger_guided_v6_round1_all_224/data/image_paths.json`
  在 10:26 被重新覆盖为反斜杠；按 `systematic-debugging` 重新规范化该文件
  的 51,025 条路径后，只读 Linux 容器运行 `validate_migration.py` 返回
  `status=PASS`、0 errors、`frozen_v7_checkpoint_present=true`，
  651,683 条图像路径全部存在，绝对 Windows 路径残留为 0。
- 2026-11-12 16:03 +08:00：项目本地安装与运行验收 PASS。
  在 `rcm_4090_deploy/.venv` 创建 Python 3.12.3 虚拟环境并安装
  `requirements.txt` 全部核心依赖；`pip check` 返回无依赖冲突。
  `MUJOCO_GL=egl MPLBACKEND=Agg python run_reference.py --no-viewer`
  返回 `status=PASS`，`python verify_current_model.py` 全部检查 PASS；
  EGL 渲染生成 1600×1080 `runs/install_smoke_20261112/current_v9.png`。
  标准 `source .venv/bin/activate` 激活成功，GLFW GUI 启动后由 6 秒定时
  关闭，确认交互窗口能够打开。
- 2026-11-12 16:07 +08:00：Codex 改动审计 hook 验收 PASS。
  `PostToolUse` hook 已由项目配置加载并标记为 trusted；隔离自测覆盖新增、
  修改、删除和 unified diff，项目基线共跟踪 642 个文件。本次真实变更用于
  验证日志会记录工具、会话、文件路径及差异内容。
- 2026-11-12 16:12 +08:00：改动审计日志容量管理 PASS。
  `events.jsonl` 达到 20 MiB 后轮转为 gzip，最多保留 3 个、30 天和 30 MiB；
  每日摘要最多 30 个、30 天和 10 MiB；整个日志目录硬上限 80 MiB，
  `errors.log` 上限 512 KiB。快照对象仅保留当前 manifest 引用内容且上限
  40 MiB，
  自动清理及 `--prune` 均通过隔离测试，日志不会无界增长。
- 2026-11-12 16:12 +08:00：RTX 4090 CUDA / EGL 优先运行验收 PASS。
  系统识别 `NVIDIA GeForce RTX 4090`、49140 MiB 显存、驱动 595.99.02；
  项目虚拟环境已固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和
  `CUDA_VISIBLE_DEVICES=0`。安装 `torch 2.14.1+cu126`、
  `torchvision 0.29.1+cu126` 和 `tqdm 4.70.1` 后，
  `torch.cuda.is_available()` 为 true，设备计算能力 8.9，cuDNN 91002；
  FP16 矩阵乘法以及 FP16/BF16 AMP 均完成且结果有限。
  MuJoCo EGL 上下文报告 renderer 为 RTX 4090，并生成
  `runs/gpu_smoke_20261112/current_v9_rtx4090.png`；
  `train_diffusion_policy.py`、`train_oct_policy.py` 和 `eval_policy.py`
  的入口检查均 PASS。
- 2026-11-12 16:23 +08:00：部署目录结构化整理 PASS。
  根目录 Python 文件由 109 个降至 24 个，只保留核心 `rcm_*` 模块、v9
  模型资产和 4 个主入口；87 个辅助脚本迁入 `scripts/datasets`、
  `scripts/applications`、`scripts/tools`、`scripts/verification` 和
  `scripts/research`，专题文档迁入 `docs/`。新增 `docs/README.md`、
  `scripts/README.md` 和根 `README.md` 目录导航，并统一为
  `python -m scripts...` 模块调用。
  全部 87 个脚本导入和 `py_compile` PASS；
  `run_reference.py --no-viewer`、模块化 `verify_current_model`、
  `verify_dataset_collection` 6/6、`verify_v9_alignment`、
  `verify_occlusion_strategy` 7/7、遮挡 dry-run 90 cases 和
  `validate_migration --code-only` 均 PASS。数据、冻结产物、`outputs/`
  和 vendored driver 未移动。
  默认完整迁移检查仍报告既有数据清单中的 Windows 反斜杠图像路径，
  本次未改写数据清单；该问题与目录迁移代码检查相互独立。
- 2026-11-12 16:28 +08:00：核心包化与根目录精简 PASS。
  根目录现有 12 个文件，其中只有 `run_reference.py` 一个 Python 文件。
  20 个核心模块迁入 `rcm/` 包，导航、视觉伺服和多目标三个应用入口迁入
  `scripts/applications/`；所有 Python 导入和文档命令已同步更新。
  `py_compile` PASS，20 个 `rcm` 模块及 90 个脚本模块全部导入 PASS。
  `run_reference.py --no-viewer`、模块化 `verify_current_model`、
  三个应用入口 `--help`、`verify_dataset_collection` 6/6、
  `verify_v9_alignment`、`verify_occlusion_strategy` 7/7、
  训练与评估入口 `--help`、`validate_migration --code-only` 均 PASS。
  `mujoco_model/run_model.py --verify` 运行内部四项验证全部 PASS，
  报告写入 `runs/organization_20261112/mujoco_model_verify.json`。
- 2026-11-12 16:49 +08:00：包化后全功能分层验收，核心运行 PASS，
  自动验收套件仍有已知项未全绿。完整汇总见
  `runs/verification_20261112/SUMMARY.md`。
  依赖一致性、RTX 4090 CUDA、110 个模块编译/导入、当前 v9 模型与 GUI、
  参考运动学、标定、导航、电机执行、视觉伺服、OCT 接口与渲染、
  多目标协议、视觉导航、遮挡恢复、受限侧车、9 个模型检查点加载、
  真实数据 GPU 前向/反向和策略评估 smoke 均 PASS。
  端到端采集完成 1 集 682 帧，导航 PASS，最终误差 10.1004 µm，
  677 个电机滞后转换，离线回放与策略输入隔离 PASS；教师接管两项均 PASS。
  验收期间补装完整 `requirements-mesh.txt`，恢复生成
  `examples/three_waypoints.json`，并修复包化遗留的
  `verify_v9_viewer` patch 路径、`collect_v9_dataset` 最终导入、
  legacy 验证导入及 `rcm.vision_navigation` 动态导入。
  未全绿项包括：严格 RGB-D/视觉几何仍执行记录中已确认过时的 5 µm
  目标阈值，当前为 9.38–10.14 µm；`verify_smooth_shells` 与
  `verify_needle_geometry` 断言仍对应旧资产/字段；guided-diffusion
  与 teacher-step 检查引用未保留的历史夹具；`verify_v9_viewer`
  可复现地出现 18 个 RGB 通道值相差 1；两个文档示例目标文件仍缺失。
- 2026-11-12 17:06 +08:00：验收失败项修复与全量复跑 PASS。
  严格 RGB-D 与视觉几何检查改用配置中的 15 µm 几何容差，分别通过
  10/10 和 9 个姿态；smooth-shell 检查识别当前
  `distal_guide_front_display.obj`，针尖检查改用
  `tip_marker_radius_mm`，两项均 PASS。
  guided-diffusion 改用真实保留特征缓存并通过 5 项检查；
  teacher-step 改为可选评估 episode/step 并 PASS；
  v9 viewer 使用有界 1 级 RGB 舍入容差后 4/4 PASS，
  实测仅 8 个 RGB 通道值相差 1。
  新增并验证 `examples/targets.json`、
  `examples/rejected_targets.json` 和 `three_waypoints.json`；
  有效目标可解，越界目标按预期返回 `OUTSIDE_EYE`。
  `validate_migration --code-only`、110 个模块编译/导入、
  `verify_motor_execution` 7/7、主模型验证及 682 帧数据回放均 PASS，
  数据回放最终误差 10.1004 µm。
  加速度硬限幅和前馈逆模型均经端到端试验后回滚，因为会降低闭环成功率；
  一阶电机模型的实际速度/加速度超出名义命令限值仍作为未辨识硬件动力学
  限制由数据集验证器单独报告。
- 2026-11-12 17:19 +08:00：v8 模仿学习全量解冻训练 PASS（仅离线动作指标，
  尚未闭环评估）。两路均从 v7 checkpoint 初始化，关闭视觉特征缓存和编码器
  冻结，保持 guided OCT、距离加权采样及原动作结构。`batch=48, lr=5e-5`
  的 epoch 3 normalized MAE 为 `0.2865439`；并行 `batch=96, lr=1e-4` 的
  epoch 3 为 `0.2738137`，相对 v7 的 `0.3101718` 改善约 `11.7%`，A/B/C MAE
  为 `0.0106388°/0.0059851°/0.0040336 mm`。模型、训练日志和历史分别规范保存
  在 `outputs/diffusion_policy_v8_full_unfreeze_20261112` 与
  `outputs/diffusion_policy_v8_full_unfreeze_b96_lr1e4_20261112`。下一验收
  必须使用严格 RGB-D、MotorBackend 和独立目标闭环评估。
- 2026-11-12 17:59 +08:00：v8 batch 96 全量解冻模型闭环评估完成，未优于
  v7 的成功率。沿用 v7 的三路随机目标 seed `20271105/20271106/20271107`
  和 `7/7/6` 分片，使用严格 RGB-D、MotorBackend、OCT、7-sample median、
  1600 步上限、30 mm 早停及同动作限幅。v8 为 `19/20` 成功（95%），
  `0/20` 失控、safety-limited 步数为 0；最终误差
  mean/median/min/max=`11.848/11.117/9.712/26.791 um`。同 seed 逐集与 v7
  配对时，v8 仅 `6/20` 最终误差改善，平均增加 `0.599 um`，但最差最终误差
  从 `47.926 um` 降至 `26.791 um`。结论：离线 MAE 改善未转化为闭环整体
  增益，不能替换 v7 作为当前主模型；下一轮应聚焦近目标样本/目标函数。
  结果目录 `outputs/evaluation_v8_full_unfreeze_b96_lr1e4_20261112`，
  checkpoint SHA-256 `e93337c065ce1692a7e500530d8b34f341b7345ea4878893cd7070b63508b3ff`。
- 2026-11-12 22:32 +08:00：模仿学习组会汇报 PPT PASS。
  使用 `ppt-master` Default Generate 生成 12 页 16:9 中文演示文稿，其中正文
  10 页、备用 2 页；内容覆盖现状与问题、模仿学习定义、数据构造、
  Diffusion Policy、OCT 引导残差、DAgger、闭环评价判据、v7/v8 对比及
  证据边界。视觉采用白底、克莱因蓝 `#002FA7` 主色、深紫与深青同深度
  冷色辅助。12 页原生 DrawingML 对象和 12 页演讲备注导出成功；
  `svg_quality_checker --stage final` 为 0 error，
  `[POSTFLIGHT] status=passed-with-warnings`、12 slides、1 个告警类别，
  PPTX ZIP 完整性检查 PASS。导出后递归检索幻灯片与备注，未发现文件路径、
  Markdown 文件名或蒸馏内容。交付目录：
  `outputs/group_meeting_imitation_20261112/imitation_learning_talk_ppt169_20261112/`。
- 2026-11-13：Terminal DAgger round 2 采集与数据准备 PASS。
  使用 v7 checkpoint、guided OCT、`beta=0.5`、教师接管、40 帧 hold 和
  7-sample policy 动作，从 20 个恢复 profile 直接初始化近目标状态。
  原始批次共 200 集、18,551 帧；20 个 profile 各 10 集；196 集正常停稳，
  4 集未通过三帧 settled hold（其中 3 集最终误差仍小于 20 µm，1 集中途
  超出训练状态范围）。7 个无效教师标签按既有规则过滤，OCT 有效率为 100%。
  目标与 development/final-test 的最小距离为 `1.0032 mm`，未改动冻结划分。
  转换后教师数据为 `data/dagger_terminal_round2_all_224`：200 集、18,544 帧。
  与 recovery baseline 合并并生成 guided sidecars 后，
  `data/lerobot_rcm_train_guided_terminal2_full_224` 共 690 集、163,097 帧；
  163,097 条图像路径均存在，OCT 有效 `163,095/163,097`，
  guided base/limit 形状均为 `[163097, 3]`、全部有限且 limit 非负。
  汇总见 `runs/dagger_terminal_round2_20261112/collection_summary.json`。
- 2026-11-13：v9 terminal-DAgger Diffusion Policy 训练 PASS（仅离线动作指标，
  尚未闭环评估）。从 v7 checkpoint 初始化，视觉编码器保持冻结并使用 v7
  对应的预计算特征缓存，开启 guided OCT；训练数据为
  `data/lerobot_rcm_train_guided_terminal2_full_224`（690 集、163,097 帧），
  验证仍使用 `data/lerobot_rcm_val_recovery_full_224`。配置为 4 epochs、
  batch 48、lr `5e-5`、2-frame observation、8-step action prediction 和
  diffusion 动作采样验证。最佳 epoch 为 1，normalized action MAE `0.231271`，
  优于 v7 的 `0.310172`；A/B/C MAE 为
  `0.0116817°/0.0061930°/0.0031425 mm`。最终 epoch 的 validation loss
  继续下降，但 normalized MAE 未改善，因此按动作 MAE 保留 epoch 1。
  checkpoint 为 `outputs/diffusion_policy_v9_terminal_dagger_20261113/best_model.pth`，
  SHA-256 `fc6d35240f9543171929c2ad4ad5ab1ded0155e6f05f515954b89e91a493f0dd`。
  离线改善不能作为闭环提升证据，下一验收必须使用严格 RGB-D、MotorBackend、
  OCT 和未参与训练的独立目标。
- 2026-11-13：v9 terminal-DAgger 固定 20 目标闭环评估 PASS（按既有验收门，
  未替换 v7 baseline）。沿用 v7/v8 的 seed
  `20271105/20271106/20271107`、`7/7/6` 分片、严格 RGB-D、MotorBackend、
  OCT、7-sample median、1600 步、20 µm 和三帧停稳判据。三路成功数为
  `6/7、7/7、6/6`，合计 `19/20`（95%），`0/20` 超过 30 mm，
  safety-limited steps 为 0。最终误差
  mean/median/min/max=`11.159/10.865/7.437/23.146 µm`。
  与 v7 配对时仅 `5/20` 改善、`15/20` 恶化，平均变化 `-0.091 µm`、
  中位变化 `+0.945 µm`，最差误差从 `47.926 µm` 降至 `23.146 µm`。
  唯一失败集跑满 1600 步，末端 OCT invalid、动作和速度为 0，停在
  `23.146 µm`，没有失控。结论：terminal DAgger 改善了失败尾部，但没有
  提升成功数量或中位闭环精度，v7 仍为工程 baseline。结果目录
  `outputs/evaluation_v9_terminal_dagger_20261113`，汇总 SHA-256
  `5908edf2762702b24f9da4c9a7e2a39efb007113c4e119bd4cdbc144c202d9b8`。
- 2026-11-13：v10 temporal-conditioned policy 训练 PASS（仅离线动作指标，
  闭环待评估）。模型从 v9 epoch 1 初始化，新增四帧观测历史、状态差分速度
  和上一步执行动作；视觉编码器保持冻结，继续使用 guided OCT 和视觉特征
  缓存。条件融合由 `1152→512→256` 扩展为
  `(512+64)×4+64=2368→512→256`，动作仍预测 8 步。训练数据为 690 集、
  163,097 帧，有效训练窗口 156,205；配置为 4 epochs、batch 48、lr `5e-5`。
  最佳 checkpoint 为 epoch 0，normalized action MAE `0.219186`，
  A/B/C MAE=`0.0106900°/0.0058183°/0.0030266 mm`，优于 v9 的
  `0.231271`。checkpoint 位于
  `outputs/diffusion_policy_v10_temporal_20261113/best_model.pth`，
  SHA-256 `f451aaadfbbb7a95efac6214a9a16c69243f9f732960c550d192e4eaf9c7b5c4`。
  闭环入口同时增加 1/2/4 步 action-chunk 执行和 7-sample batch 并行采样；
  下一步必须与 v9 在相同目标和协议下比较。
- 2026-11-13：v10 并行采样与 action-chunk 执行筛选 PASS（部分目标范围，
  不作为最终成功率）。`PolicyEvaluator` 将 7 个 Diffusion 样本由 CPU 驱动
  的顺序调用改为单次 CUDA batch 推理，并保留 OOM 顺序回退；新增
  `--n-action-execute-steps`，可在一次预测后连续执行 1/2/4 步动作，任何
  安全缩放都会清空剩余 chunk。并行采样 smoke 中一次 7-sample 调用为
  `0.229 s`，动作队列 smoke 通过。使用 seed `20271105` 的同一组 7 个目标
  筛选 v10 执行 1/2/4 步：三者均为 `6/7`；最终误差中位数分别为
  `10.201/11.411/12.873 µm`，最大误差分别为
  `45.648/41.116/42.750 µm`。执行 4 步明确变差；执行 2 步平均误差基本
  持平但中位更差。结果分别位于
  `outputs/evaluation_v10_temporal_screen_20261113/exec1`、
  `exec2`、`exec4`。严格 RGB-D 与 MuJoCo 逐帧观测仍是闭环墙钟时间主项，
  因而 batching 未显著缩短单 episode 总时间。
- 2026-11-13：v10 temporal-conditioned policy 完整 20 目标闭环评估 PASS
  （按既有验收门，未替换 baseline）。使用与 v7/v8/v9 相同的三组 seed 和
  `7/7/6` 目标分片，分别评估执行 1 步与 2 步。两者均为 `19/20`、无
  `>30 mm` 失控、无 safety-limited steps。v10 exec1 最终误差
  mean/median/min/max=`12.464/10.418/8.917/45.648 µm`；
  v10 exec2 为 `12.559/11.181/7.569/41.116 µm`。
  与 v9 配对，v10 exec1 改善 `10/20`、恶化 `9/20`、持平 `1/20`，平均最终
  误差增加 `1.305 µm`；执行 2 步相对执行 1 步改善 `7/20`、恶化 `11/20`、
  持平 `2/20`，平均增加 `0.096 µm`，中位增加 `0.504 µm`。唯一失败仍为
  同一目标，成功数为 0 前已进入 OCT invalid、零动作和零速度停驻。
  结论：四帧/速度/上一步动作改善了离线 MAE，但没有改善成功数量或失败尾部；
  执行 2 步未显示精度收益，4 步屏幕测试更差，当前继续执行第 1 步并保留
  已实现的并行采样与可配置 chunk 入口。结果目录
  `outputs/evaluation_v10_temporal_20261113` 与
  `outputs/evaluation_v10_temporal_exec2_20261113`，汇总 SHA-256 分别为
  `ea96528c28facccbe0bf6b4daaaf62a2aae2a1c390aa278f4b7b1217f123e164`、
  `e41f1883e091fba66ab666f6b30ad7bfc31525374bcdc6d46604ae3d369c2491`。
- 2026-11-13：action-horizon 训练消融 PASS（闭环待评估）。
  固定 v10 的四帧历史、状态速度、上一步动作、视觉冻结、guided OCT、
  数据、初始化、4 epochs、batch 48 和 lr `5e-5`，只改变预测步数
  `H=1/2/4`；既有 v10 为 `H=8` 对照。最佳 normalized action MAE：
  `H1=0.223821`、`H2=0.219375`、`H4=0.246795`、`H8=0.219186`。
  对应 A/B/C MAE：H1 `0.0096437/0.0063323/0.0035771 mm`，
  H2 `0.0099122/0.0060389/0.0034026 mm`，
  H4 `0.0108076/0.0068738/0.0038053 mm`。H2 几乎追平 H8，H4 明显变差；
  说明更长未来窗口没有带来单调的离线动作收益。checkpoint 分别位于
  `outputs/diffusion_policy_v11_h1_20261113`、
  `outputs/diffusion_policy_v11_h2_20261113`、
  `outputs/diffusion_policy_v11_h4_20261113`；SHA-256 分别为
  `6322223b0149f3c0ca0d6b33d84bcae90618694c30d1db0e08d8e95889d2ccc9`、
  `cd993e0636394045a0be7536ddd434e4485d332409a05099b7a842b3e7163189`、
  `4223375c5f20546b700022f378fc59061990c81601ebb880a21c0ea4c1dab4b7`。
- 2026-11-13：horizon 完整闭环补跑 INTERRUPTED（partial，不作为结果）。
  H1/H2/H4 的 g2、g3 六路并行运行约 10 分钟后被系统以退出码 `143`
  终止，子进程均未保存 `evaluation_results.json`，仅留下 console 进度；
  因此没有将该部分运行计入成功率。H1/H2/H4 的 g1 7 目标结果完整保留。
  后续改为每波最多 3 个评估进程，并分别重新运行 g2 和 g3。
- 2026-11-13：action-horizon 7 目标闭环筛选 PASS（screen scope only）。
  固定 seed `20271105`、同一组 7 个目标、执行第 1 步和相同闭环协议，
  `H=1/2/4/8` 均为 `6/7`；最终误差 mean/median/max 分别为：
  H1 `16.407/11.282/49.670 µm`、H2 `17.211/12.079/49.068 µm`、
  H4 `12.609/10.094/27.691 µm`、H8 `15.136/10.201/45.648 µm`。
  H4 在该屏幕上取得最低 mean、median 和最大误差；H1/H2 没有显示更好收益。
  四种模型都在同一目标进入 OCT invalid 后的零动作停驻，H4 停驻残差最小。
  结果目录 `outputs/evaluation_horizon_screen_20261113` 与既有的
  `outputs/evaluation_v10_temporal_screen_20261113/exec1`。该结果只用于选择
  完整验证候选，不作为 20 目标最终结论；启动 H4 的完整验证仍是必要步骤。
- 2026-11-13：horizon 补跑 g2 PARTIAL PASS（14/20 scope，不构成最终结论）。
  H1/H2/H4 使用 seed `20271106` 的 g2 七目标均完成且全部 `7/7`；最终误差
  mean/median 分别为 H1 `10.124/10.074 µm`、H2 `11.965/10.891 µm`、
  H4 `10.314/10.891 µm`。三组均无 safety-limited step。结合此前 g1
  `6/7`，当前各自完成 14 个目标，均为 `13/14`；g3 六目标尚未运行。
  该 partial 结果只能说明 H1/H2/H4 在 g2 上没有失败，不能据此给出完整
  20 目标排名。结果位于
  `outputs/evaluation_v11_h1_20261113`、
  `outputs/evaluation_v11_h2_20261113`、
  `outputs/evaluation_v11_h4_20261113`。
- 2026-11-13：H4 action-horizon 完整 20 目标闭环 PASS（按既有验收门，
  未替换 v9/v7）。H4 在 g1/g2/g3 上分别 `6/7、7/7、6/6`，合计 `19/20`；
  最终误差 mean/median/min/max=`11.459/10.807/6.264/27.691 µm`，
  `0/20` 超过 30 mm，safety-limited steps 为 0。与 v9 配对时改善 `10/20`、
  恶化 `10/20`，平均增加 `0.300 µm`、中位增加 `0.067 µm`、最大增加
  `4.545 µm`；与 H8 配对时改善 `9/20`、恶化 `11/20`，平均改善
  `1.005 µm`、中位增加 `0.417 µm`、最大改善 `17.957 µm`。唯一失败仍是
  同一目标，最终停在 `27.691 µm`，末端 OCT invalid、动作和速度为 0，
  没有失控。结论：H4 比 H8 的失败尾部更稳定，但未超过 v9；完整 H1/H2
  在 g3 尚未运行，当前只有筛选及部分 g2 证据，不纳入完整排名。
  结果目录 `outputs/evaluation_v11_h4_20261113`，checkpoint SHA-256
  `4223375c5f20546b700022f378fc59061990c81601ebb880a21c0ea4c1dab4b7`。
  完成后的 aggregate SHA-256
  `2a86bbb42342b491e5da1e2471d5108c7b2e405a434aca7a868f5e0c72da2f92`。
- 2026-11-13：H4 final-test 四分片启动 INTERRUPTED（no result）。
  H4 在冻结 60 final-test targets 上以 4 分片、每片 15 集启动；运行约 21 秒
  后尚未产生任何 `evaluation_results.json`。为按用户要求提高并行度并避免
  直接覆盖，主动终止该次尝试，原始 console 保留在
  `outputs/evaluation_v11_h4_final_test_20261113/g1..g4`，不计入评价指标。
  随后改用 6 分片、每片 10 集的新目录重新运行。
- 2026-11-13：H4 final-test 六分片启动 INTERRUPTED（no result）。
  六分片运行约 100 秒后仍未保存 `evaluation_results.json`；实测六路共占
  约 19/31 GB 主机内存和 12.8/49.1 GB GPU 显存。由于主机内存是瓶颈，
  为按用户要求继续提高并行度，主动终止该次尝试并保留 console 为未完成
  记录，不纳入指标。随后切换为 8 分片、每片 7–8 个目标的新目录运行。
- 2026-11-13：H4 final-test 8 分片运行 PARTIAL（22/60 completed，
  不作为最终成功率）。8 路运行约 39 分钟时，日志中已有 22 个 episode
  完整结束，其中成功 `12/22`、失败 `10/22`；已完成最终误差
  mean/median/max=`72.18/17.5/392.0 µm`。失败包括多个 1600 步平台
  `14–48 µm`，以及 `223/279/332/392 µm` 大偏差样本；一个 episode 在
  715 步触发连续 safety-limit 早停。所有 shard 尚未写出
  `evaluation_results.json`，因此这些日志计数只用于观察进度，不进入
  最终 60 目标指标。运行目录：
  `outputs/evaluation_v11_h4_final_test_parallel8_20261113`。
- 2026-11-13：H4 final-test 8 分片 PARTIAL UPDATE（27/60 completed，
  仍非最终结果）。运行约 48 分钟后，日志中已有 27 个 episode 完整结束：
  成功 `16/27`、失败 `11/27`，已完成最终误差
  mean/median/max=`61.19/16.0/392.0 µm`。其中 g1 `0/2`、g2 `1/3`、
  g3 `2/3`、g4 `2/4`、g5 `4/5`、g6 `5/5`、g7 `2/3`、g8 `0/2`。
  shard 尚无最终 JSON；失败包含大偏差和多次 1600 步未停稳。该更新仍不进入
  最终 60 目标统计。
- 2026-11-13：模仿学习汇报 PPT 技术逻辑调整与结构校验 PASS。
  源文件为仓库根目录的 `模仿学习v1_第13页BC-DAgger重绘.pptx`，重排为
  “问题定义 → 专家标签 → 时序样本 → 策略输入与训练 → 开发集表现 →
  时序不可观测 → BC 分布偏移与 DAgger → 最终测试 → 结论”，并将实验规模与
  建模演进页移至附录 G/H。第 13 页改为“训练期专家重标注”，明确部署时无
  专家在环，同时修正图例中的“专家示范流形”和“在线专家纠正”表述。
  使用 `runs/20261113/ppt_review/logic_adjustment/apply_logic_adjustment.py`
  生成调整版。校验结果：21 页完整，128 个 ZIP entry，121 个 XML/RELS 部件
  全部可解析，无缺失 slide relationship；第 13 页旧措辞计数均为 0，标题、
  副标题和底部口径宽度分别约为文本框的 63%/52%/76%。当前环境缺少
  LibreOffice/PowerPoint 渲染器，未做最终像素级跨软件渲染检查。输出为
  `outputs/imitation_learning_logic_adjustment_20261113/`
  `模仿学习v1_技术逻辑调整版_20261113.pptx`，SHA-256
  `47d9b23c137b989e5c95153632396a0612f72bf8d1bb7515b3fdc1f0cfb608c9`。
- 2026-11-13：上一条 PPT 技术逻辑调整记录更正（最终复核 PASS）。
  复核发现附录 B 与附录 F 仍引用重排前的主汇报页码；已将附录 B 的
  “主汇报 P11”改为“主汇报 P09 / 12”，将附录 F 的“对应 P14”改为
  “对应 P12 / 12”。重新生成后再次通过 ZIP、121 个 XML/RELS 部件、
  21 页顺序与 slide relationship 校验。最终文件仍位于
  `outputs/imitation_learning_logic_adjustment_20261113/`
  `模仿学习v1_技术逻辑调整版_20261113.pptx`，最终 SHA-256 为
  `5f2436e90ff36fb8d51dba167ca2bac4c87bf056630c08718e3c2f29a27d6c86`；
  上一条中的 `47d9...` 已被本版本替代。
- 2026-11-13：模仿学习汇报 PPT 最新成果同步与去冗余 PASS。
  基于技术逻辑调整版生成 20 页精简版，不新增页面并删除重复的“五阶段实验设计”
  附录。P05 直接改为 v10 四帧观测历史与八步动作块；P08 改为 Guided Diffusion
  部署网络；P09 改为 v7/v9/v10 固定 20 目标结果；P10/P11/P13 直接替换旧的
  “速度未显式输入”和“下一步加入速度”结论；附录 A/B/D/F 更新模型维度、
  Baseline、训练数据版本与 Restricted 结果边界。未新增小字差异备注。
  校验结果：20 页、128 个 ZIP entry、全部 XML/RELS 可解析、无缺失 slide
  relationship；旧 `13/40`、`39/40`、`40/40`、`34/40`、单步 MLP 最终部署、
  `A–H` 等文案在主页面中计数为 0。生成脚本为
  `runs/20261113/ppt_review/latest_results/apply_latest_results_trim.py`。
  输出为 `outputs/imitation_learning_latest_results_20261113/`
  `模仿学习v1_最新成果精简版_20261113.pptx`，SHA-256
  `bd4a4b0fdbdc3215f31b2c67fa86c2ec7e2c4124545ac5d2de551a69208631f0`。
- 2026-11-13：模仿学习汇报 PPT 的 H4 四步执行修正 PASS。
  复核 v11 H4 checkpoint 配置，`n_obs_steps=4`、`n_action_steps=4`；
  汇报第 5 页原先仍保留“八步动作块、只执行第一步”，与当前 H4 执行口径
  不一致。已直接改为“四帧观测 + 四步动作块 + 连续执行四步”，并同步
  P08 Guided Diffusion 部署说明、附录 A 与相关演讲备注。未新增页面或小字
  差异备注。当前 `eval_policy.py` CLI 的 `--n-action-execute-steps` 默认值
  仍为 `1`，因此实际 H4 运行必须显式传入 `4`；PPT 依据当前 H4 执行设置。
  校验结果：ZIP 完整，全部 XML/RELS 可解析；P05 保留四个动作框
  `a(t)..a(t+3)`，全部标记“执行”，旧的八步与首步文案计数为 0。输出为
  `outputs/imitation_learning_h4_20261113/`
  `模仿学习v1_H4四步执行版_20261113.pptx`，SHA-256
  `518b7baf0f568af8d1afd8fee88dd3b88b3f75cbc9d26ec6e046392b5ec3bdc8`。
- 2026-11-13：v11 H4 四步执行完整 20 目标评估 PARTIAL（不得作为最终结果）。
  当前结果目录 `outputs/evaluation_v11_h4_20261113` 中，g1 已完成 `6/7`，
  g2 完成 `7/7`，g3 仅完成 `1/6`；累计完成 15 个目标，其中 14 个成功、
  1 个失败。唯一已完成的失败目标最终误差为 `27.691468580371914 µm`，
  达到 1600 步上限。g3 仍有 5 个目标未运行，因此暂不计算完整 20 目标
  成功率和总体误差统计。当前 `eval_policy.py` 的
  `--n-action-execute-steps` 默认值仍为 1；本轮 H4 口径为四帧观测、
  四步动作块并连续执行四步。
- 2026-11-13：模仿学习汇报 PPT 的 H4 后续工作与进度同步 PASS。
  P13 已直接改为 H4 四帧观测、四步动作、连续执行；累计进度写成
  “15 个目标完成、14 成功、1 失败”，下一步第一项改为完成 g3 剩余 5 个
  目标并汇总完整 20 目标结果。附录 B 的 H4 行同步为 `14/15` 和
  “g3 已完成 1/6；余 5 目标”。校验：ZIP 完整，128 个条目和全部
  XML/RELS 可解析。输出为
  `outputs/imitation_learning_h4_20261113/`
  `模仿学习v1_H4后续工作版_20261113.pptx`，SHA-256
  `303798a5f45d4cfb1acc698572c9ad0a539a4b4c0655e582fa7928f62d905c72`。
- 2026-11-13：模仿学习 H4 组会逐页演讲稿生成 PASS。
  按 20 页 H4 后续工作版生成中文技术型逐页讲稿，正文 13 页约 18 分钟，
  附录 A–G 按问答展开。内容覆盖任务判据、专家标签、四帧观测与四步连续
  执行、Restricted 输入消融、v7/v9/v10 固定 20 目标结果、terminal DAgger、
  Restricted 冻结最终集和当前 H4 部分结果。校验包含 20 个页面标题，未保留
  旧的“速度尚未输入”和“下一步加入速度”表述；H4 当前进度按
  `15 个目标完成、14 成功、1 失败、g3 余 5 目标`记录。输出为
  `outputs/imitation_learning_h4_20261113/`
  `模仿学习v1_H4_组会逐页演讲稿_20261113.md`。
- 2026-11-13：H4 完整 60 目标 final-test 评估完成，基础设施 PASS、
  性能 FAIL。使用冻结 final-test 60 targets、每目标 1 个 seed、严格 RGB-D、
  MotorBackend、OCT、四帧历史、预测 4 步、执行 1 步、7-sample batch median、
  1600 步和 20 µm 三帧停稳判据。20 µm 成功 `28/60`（46.7%），10 µm
  `9/60`，5 µm `0/60`；最终误差
  mean/median/P95/max=`622.638/21.295/332.104/19156.335 µm`。
  60 个 episode 中 35 个曾进入 20 µm，其中 7 个仍未完成停稳成功；
  25 个从未进入 20 µm。32 个失败中 29 个跑满 1600 步，3 个因连续
  safety-limit 早停，safety-limited 总步数 60。目标簇差异极大：
  g1、g8 为 `0/8、0/7`，g6 为 `7/7`；2 个 episode 最终误差分别达到
  `19.156 mm` 和 `15.734 mm`。聚合器的 `result=PASS` 仅表示未触发其宽松
  基础设施门和 30 mm 失控门，不表示 20 µm 性能通过。固定 development
  20 目标的 `19/20` 不能外推到 final-test。结果目录
  `outputs/evaluation_v11_h4_final_test_parallel8_20261113`，aggregate
  SHA-256 `7816bfdc5d1837b7a38982d43ea2038b4dc4f1f3151ea35c7e2204163558baa6`。
- 2026-11-14：v12 零误差数据集 pilot 门控 PASS（仅 `episode_000000`）。
  源数据为 `data/lerobot_rcm_train_guided_dagger1_full_224` 的首个 episode；
  使用严格 RGB-D、MotorBackend 和源终点整数编码器锁定。原始 pilot 的
  `685/685` 帧通过图像哈希、实际编码器状态、时间、协议动作、policy-input
  隔离及离线电机重放检查；终点针尖与目标世界坐标逐分量位相等，float64
  欧氏误差严格为 `0.0 µm`，电机编码器停稳。pilot 224 转换为 `685` 帧，
  `state/action` 为 `float32`、图像为 `224×224×3`，加载 smoke PASS。
  证据目录 `runs/20261114/zero_error_v12/pilot_224/`，门控报告
  `runs/20261114/zero_error_v12/pilot_gate.json`。该 PASS 只证明仿真目标
  真值和软件数据管线，不代表真实设备或临床精度；本轮未训练策略。
- 2026-11-14：v12 严格零误差聚焦回归 PASS。目标坐标逐分量复制、整数编码器
  终点复现、受限关节路径、源索引追溯、原子 manifest、
  视觉 PASS 后精确终点锁定，以及视觉失败后 MotorBackend fallback 再接精确
  锁定均通过；相关 Python 文件 `py_compile` PASS。另以正式 pilot 对
  `--expected-episodes 1` 的 raw prefix 和临时 224 prefix 做复验，均 PASS。
- 2026-11-14：v12 批量采集并发调优 PARTIAL（非最终结果）。正式目录先以
  单路串行续采到 21 集；为满足资源与总时长约束，先后尝试 5 路和 10 路
  正式目录直写，均在已完成若干离散 episode 后主动中断并调整并发数。已完成
  episode 结果保留，未覆盖；未完成目录及 `dataset/` 分片统一移入
  `aborted_episode_*` 证据目录。最终 14 路运行中，按连续完成前缀原子更新
  manifest。相关记录见 `runs/20261114/zero_error_v12/parallel_collection.json`
  和 `parallel_logs/`；该条不代表 608 集验收完成。
- 2026-11-14：v12 零误差训练数据采集与发布 PASS。源数据为
  `data/lerobot_rcm_train_guided_dagger1_full_224`，使用 pilot
  `episode_000000` 门控、严格 RGB-D、MotorBackend、源终点量化整数锁定和
  14 路正式目录直写；每波 worker 单集后退出以限制内存，协调器按连续完成
  前缀原子更新 manifest。最终 raw 为 `608/608` 集、`187,440` 帧；
  `608/608` 终点针尖与目标坐标逐分量位相等，float64 欧氏误差严格为
  `0.0 µm`，图像哈希、时间、实际编码器状态、协议动作、policy-input 隔离
  和离线电机重放全部 PASS。视觉 PASS `598` 集，受限关节 fallback `10`
  集；执行来源帧数为 strict RGB-D `181,916`、exact terminal lock `4,785`、
  privileged fallback `131`、terminal hold `608`。目标相对旧名义目标的
  平均位移范数 `4.547 mm`，最大 `8.607 mm`。最终 224 数据集
  `data/lerobot_v12_zero_error_train_224/` 为 `608` 集、`187,440` 帧，
  `state/action` 为 `float32`、`224×224×3` RGB，加载与目标真值隔离 smoke
  PASS。原始证据和验收报告位于 `runs/20261114/zero_error_v12/`，
  总采集耗时约 `8,785 s`。本轮未训练策略；以上仅验证 MuJoCo 仿真数据
  终点和软件管线，不代表真实设备、OCT 或临床精度。
- 2026-11-14：v12 OCT-aware 训练派生集构建与加载检查 PASS。为了实际
  OCT 输入实验，从 `runs/20261114/zero_error_v12/episode_*/oct_observations.jsonl`
  读取逐帧 `navigation_depth.delta_xyz_m`（`target_minus_tip`），转为 mm
  后按 `frame_audit.control_frame_index -> OCT iteration` 对齐。输出
  `data/lerobot_v12_zero_error_train_oct_224/`：608 集、187,440 帧，
  OCT 有效 182,514 帧（97.372%）、无效 4,926 帧、无未匹配视觉帧。
  fallback、exact-terminal 与 hold 帧按设计保持 `valid=false`。该值来源为
  注册 RGB-D 可见深度仿真，不是物理 OCT；metadata 已明确记录此边界。
  使用 `RCMDataset` 检查得到 182,576 个有效训练窗口，样本形状为 RGB
  `[2,3,224,224]`、state `[2,3]`、OCT `[2,4]`、action `[8,3]`。
- 2026-11-14：v12 DAgger OCT 子集构建 PASS。从 608 集 OCT-aware 训练集
  筛选 `source_dataset` 含 `dagger` 的 118 集，生成
  `data/lerobot_v12_zero_error_dagger_oct_224/`：38,969 帧，OCT 有效
  38,016 帧（97.554%），38,025 个有效训练窗口；图像以硬链接复用，
  数据与训练加载 smoke PASS。该子集保留 `v12_full_train_episode_index`
  和原 `source_episode_index` 映射。其语义是既有 DAgger rollout 状态分布
  由 v12 严格教师重采集/重标注，不是新策略产生的新 DAgger round。
- 2026-11-14：v12 OCT Diffusion Policy 首次训练 FAIL（基础设施失败，
  不作为模型结果）。命令使用 `data/lerobot_v12_zero_error_train_oct_224`
  和 `data/lerobot_rcm_val_224`，batch `192`、8 process workers、FP16 AMP、
  4 epochs。训练集 `182,576` 个窗口和 951 batches/epoch 均正常完成，
  epoch 0 train loss 约 `0.189`；验证阶段因冻结 val 的
  `image_paths.json` 含 Windows 反斜杠，在 Linux 上解析成错误文件名而退出。
  未生成 checkpoint。失败目录保留于
  `outputs/diffusion_policy_v12_oct_20261114/`。已修正训练 DataLoader
  的跨平台路径规范化，并对 train/val 单样本加载重新 smoke PASS。
- 2026-11-14：v12 OCT Diffusion Policy 训练 PASS（离线训练 scope）。
  使用 `data/lerobot_v12_zero_error_train_oct_224`（608 集、187,440 帧、
  182,514 个有效 OCT 帧）和 `data/lerobot_rcm_val_224`（26,970 个有效
  windows），物理 GPU 0 RTX 4090、FP16 AMP、batch `192`、8 process
  workers、2 帧观测历史、8 步动作块、OCT 输入启用、seed `20261114`，
  训练 4 epochs。train/val loss 从 `0.374025/0.190276` 降至
  `0.150758/0.144923`，四个 epoch 验证 loss 均改善，最佳 checkpoint 为
  epoch 3。输出 `outputs/diffusion_policy_v12_oct_20261114_retry1/`，
  `best_model.pth` SHA-256
  `146555bd69b91c50447c55b67426564e7d8a11537b5ff03f4aedbba976406bb9`。
  该结果仅为离线扩散去噪训练结果，尚未执行 v12 OCT-aware 闭环评估，不能
  据此宣称控制成功率或临床精度。首次因 Windows 路径失败的目录保留于
  `outputs/diffusion_policy_v12_oct_20261114/`。
- 2026-11-15：v12 OCT Diffusion Policy development target 0 闭环 smoke
  FAIL（基础设施 PASS，策略性能 FAIL）。使用最佳 checkpoint epoch 3、
  严格 RGB-D、MotorBackend、7-sample median、执行动作块第 1 步、
  `1600` 步上限和 `20 µm` 三帧停稳门。策略在 1032 步后因连续 20 次
  B 轴安全限位早停，最终误差 `22.7588 mm`；起始误差 `19.5481 mm`，
  最优误差 `18.8506 mm`。动作持续在 A/B/C 限幅附近振荡，未收敛。
  OCT 输入有效率为 `100%`，模型确实加载 `use_oct=true`，因此该失败不是
  OCT 缺失或输入单位错误。输出位于
  `outputs/evaluation_v12_oct_smoke_20261115/target_0/`。下一步按同一协议
  跑固定 development 20 目标，量化失败范围。
- 2026-11-15：v12 OCT Diffusion Policy 固定 development 20 目标闭环
  验证完成，基础设施 PASS、策略性能 FAIL。四个并行分片均完成 5/5，
  协议为 strict RGB-D、MotorBackend、OCT 输入、7-sample median、执行
  action chunk 第 1 步、1600 步上限和 20 µm 三帧停稳判据。成功率
  `0/20`；最终误差 mean/median/min/max =
  `22.495/22.815/18.528/23.697 mm`；每集轨迹最小误差
  mean/median/min/max = `19.123/19.196/17.521/19.762 mm`。
  `19/20` 集出现安全限位，共 `4016` 个 safety-limited steps；没有
  episode 超过 30 mm 失控门。最终目标相对向量主要留下约 `13.6 mm`
  的 X 向和 `17.9 mm` 的 Z 向偏差，说明策略发生了错误方向运动并在
  A/B 边界停留，而非没有输出动作。原始结果位于
  `outputs/evaluation_v12_oct_dev20_20261115/g1..g4/`，汇总位于
  `outputs/evaluation_v12_oct_dev20_20261115/aggregate_summary.json`。
  当前模型仅完成离线 behavior-cloning 训练，不是新的在线 DAgger round。
- 2026-11-15：v12 OCT 在线 DAgger pilot PASS。使用 epoch 3 checkpoint、
  strict RGB-D 教师、MotorBackend、OCT 输入和 β=`0.3`；单个 target/start
  episode 从 `9.0227 mm` 开始，教师预跑交接后策略在线运行，共 351 帧，
  最终误差 `0.011267 mm`（11.267 µm）并停稳。教师无效帧 `0`，OCT 有效
  帧 `351/351`。执行来源为 teacher `328`、policy `2`、
  beta-teacher `1`、hold `20`；策略动作触发限幅 `318` 帧，证明策略偏离
  状态和教师接管样本已进入数据。输出
  `runs/20261115/online_dagger_v12/pilot_target_0/`。
- 2026-11-15：按“专家终点最多 0.1 µm”收紧 DAgger 的验收检查。
  先前以 20 µm 门限完成的 pilot 为 `11.267 µm`，批量中断前完成的 16 集
  终点为 `9.941–13.698 µm`，全部隔离到
  `runs/20261115/online_dagger_v12/round1_aborted_20um/`，不得用于训练。
  修改后在线 DAgger 在视觉接近后追加 privileged integer MotorBackend
  终点锁定，并由实际终点重建目标。新 pilot 共 370 帧、教师无效帧 0，
  终点针尖与目标逐分量位相等，`final_error_um=0.0`，满足 0.1 µm 上限；
  目标相对原名义目标移动范数 `1.588 µm`。证据位于
  `runs/20261115/online_dagger_v12/pilot_exact_0p1um/`。
- 2026-11-15：固定可达目标 DAgger pilot PASS。名义 target 先映射到整数
  MotorBackend 可达终点，并直接把该可达针尖坐标定义为固定 target；
  随后目标在 rollout 中不再移动。pilot 共 366 帧，教师无效帧 0，
  终点 error `5.56e-11 µm`，`strict_zero_error=true`，满足 0.1 µm
  上限；名义 target 到固定可达 target 的构造偏移为 `2.628 µm`，单独记录。
  输出 `runs/20261115/online_dagger_v12/pilot_fixed_reachable_target/`。
  此前的 reconstructed-target pilot 与 20 µm 批量数据均不作为训练输入。
- 2026-11-15：固定可达目标在线 DAgger round 1 完成。8 路分片合计
  `40/40` 集、14,201 帧；`38/40` 集 PASS 且 `strict_zero_error=true`，
  最终误差满足 `≤0.1 µm`，成功集误差中位数
  `1.73e-12 µm`。2 集 FAIL：episode 6 为
  `LOW_QUALITY_VISUAL_GEOMETRY`、误差 `94.440 µm`；episode 14 同样是
  `LOW_QUALITY_VISUAL_GEOMETRY`、误差 `67.226 µm`。失败集不得作为
  成功专家示范。帧级执行来源为 teacher `13,031`、hold `798`、
  exact-terminal `304`、policy `50`、beta-teacher `18`；策略动作触碰
  限幅 `12,654` 帧，再次确认当前学生策略几乎全程需要教师纠正。
  OCT 有效 `13,859/14,201` 帧。分片结果位于
  `runs/20261115/online_dagger_v12/fixed_target_round1_shards/g1..g8/`；
  尚未转换、合并或用于训练。
- 2026-11-15：v12 OCT online-DAgger round 1 转换、合并与增量训练 PASS。
  转换器新增 224 输出和严格成功过滤；8 个 shard 保留
  `38` 个 `strict_zero_error=true` episode、13,503 帧，排除终点失败
  episode 6/14。与基础 608 集合并为
  `data/lerobot_v12_oct_dagger1_fixed_224/`：646 集、200,943 帧、
  195,675 个有效 OCT 帧，形成 195,775 个训练窗口。从 epoch 3
  checkpoint 增量训练 4 epochs，最佳 epoch 7，train/val loss
  `0.137604/0.129817`，较基础 `0.150758/0.144923` 改善。输出
  `outputs/diffusion_policy_v12_oct_dagger1_fixed_20261115/`，
  `best_model.pth` SHA-256
  `966fa042d80b692f7117b3061b9e8305683f117706ebbb41afb2a3598fdd5175`。
  下一步用同一 development 20 目标做配对闭环验证。
- 2026-11-15：v12 OCT DAgger 增量模型 development 20 目标闭环 FAIL
  （离线训练改善，控制性能仍不合格）。新模型成功率仍为 `0/20`，但
  最终误差 mean/median/min/max 从旧模型
  `22.495/22.815/18.528/23.697 mm` 降至
  `16.097/18.034/4.855/23.974 mm`；每集轨迹最小误差中位数从
  `19.196 mm` 降至 `14.715 mm`。20 集配对中 `16` 集改善、`4` 集恶化，
  最终误差平均改善 `6.398 mm`、中位改善 `5.292 mm`，最大改善
  `17.464 mm`。安全限位从 `19/20` 集、4016 步降至 `13/20` 集、
  2550 步。四个目标已达到约 `4.85–5.16 mm`，但没有任何 episode 进入
  20 µm 三帧停稳门。原始分片位于
  `outputs/evaluation_v12_oct_dagger1_fixed_dev20_20261115/g1..g8/`，
  汇总为 `aggregate_summary.json`。结论：在线 DAgger 有效改善失败尾部，
  但当前普通 Diffusion BC 架构仍不可部署，下一步应引入 OCT guided base
  action 和有界 residual。
- 2026-11-15：Flow Temporal U-Net 训练 PASS（离线训练 scope）。模型实现
  ResNet-18 GroupNorm/Spatial Softmax 视觉编码、state+4D OCT+velocity+
  previous-action 门控融合、`T_p=16` 的 1D temporal U-Net、逐残差块
  FiLM 条件调制和 4-step Euler Flow Matching；训练数据仅为
  `data/lerobot_v12_oct_dagger1_fixed_224` 的严格零误差集。训练 4 epochs，
  参数量 `34.24M`，最佳 epoch 3，train/val flow loss
  `0.158621/0.183125`。checkpoint
  `outputs/flow_temporal_unet_v12_zeroerror_20261115_retry3/best_model.pth`，
  SHA-256
  `85c38dfbffaa2c15b3e677f4c419940f014a66ec92ae1a250ae0837eeeb7a87b`。
  `PolicyEvaluator` 已增加该 checkpoint 的完整推理契约：直接 224 图像、
  完整 `[dx,dy,dz,valid]` OCT、由最近两帧计算的 velocity、上一实际执行
  动作和 4-step Flow 采样，并记录成功率阈值、Flow 积分步数和动作聚合方式。
- 2026-11-15：Flow Temporal U-Net 单目标闭环 smoke FAIL（`0.1 µm` 验收）。
  使用 development target 0、strict RGB-D/OCT、MotorBackend、单样本 Flow、
  执行预测动作块首步、1600 步上限，最终误差 `490.049 µm`，轨迹最小误差
  `154.746 µm`；OCT valid `1600/1600`，无安全限位或发散早停。该结果说明
  Flow 架构把同目标从上一 DAgger 模型的 `19.6237 mm` 降至亚毫米，但距
  `0.1 µm` 门仍差约 `1547×`。结果位于
  `outputs/evaluation_flow_temporal_unet_v12_dev1_smoke_20261115/`。
- 2026-11-15：Flow Temporal U-Net development 20 目标 `0.1 µm` 闭环 FAIL。
  因用户将最终门槛收紧为 `error <= 0.0001 mm`，先前按 `20 µm` 门启动的
  8 路批次在尚无完整 episode 时中止，保留于
  `outputs/evaluation_flow_temporal_unet_v12_0p1um_dev20_20261115/`，不得
  作为验收结果。最终批次按 `0.1 µm`、连续 3 帧达标且电机停稳、单样本
  Flow、执行动作块首步、1600 步上限运行 10 路，每目标 2 集。结果为
  `0/20`，没有任何 episode 的最终误差或轨迹最小误差达到 `0.1 µm`。
  最终误差 mean/median/min/max/P95 =
  `1731.480/1010.959/479.490/15219.883/2352.584 µm`；每集最小误差
  mean/median/min/P95 =
  `1470.546/654.012/154.746/2326.599 µm`。20 集共 `32,000` 帧，
  OCT valid `31,995/32,000`，安全限位 `0` 步。相对同目标上一 DAgger
  模型，`19/20` 改善、`1/20` 恶化，最终误差平均/中位改善
  `14.365/17.188 mm`；恶化的是 target 14，从 `6.855 mm` 变为
  `15.220 mm`。原始结果位于
  `outputs/evaluation_flow_temporal_unet_v12_0p1um_dev20_fast10_20261115/g01..g10/`，
  汇总为同目录 `strict_summary.json`。结论：Flow Temporal U-Net 显著改善
  大范围接近行为并消除了安全限位，但亚毫米平台距 `0.1 µm` 仍有
  `2–3` 个数量级差距；下一步必须增加显式 OCT 比例微调、近目标制动/保持，
  以及特权整数编码器终点锁定，不能把当前策略直接作为 `0.1 µm` 闭环结果。
  以上均为 MuJoCo/RGB-D 模拟结果，不代表真实 OCT 或临床精度。
- 2026-11-15：Flow Temporal U-Net 700 步平台诊断完成。误差曲线显示典型
  目标在约 `600–900` 步进入近目标平台；最佳误差步数中位数为 `882`，
  IQR `719–1229`。策略并未停止输出或触发安全限位：动作范数均值从
  `500–600` 步的 `0.04502` 降至 `600–700` 步的 `0.03904`、
  `700–800` 步的 `0.01398`，之后稳定在约 `0.009`；`700–1600` 步中
  只有 `47.94%` 的步骤实际减小误差，平均每个控制步仅改善约
  `0.174 µm`，但电机仍在缓慢漂移，B 轴平均净漂移 `-1.82°`。700 步时
  OCT 全部有效且其范数与真值误差一致，后 700 步 C 轴动作平均绝对值
  `0.00553 mm`，远高于编码器单计数 `1.57e-5 mm`，因此平台不是 OCT
  丢失、动作全零、限幅或编码器量化造成，而是策略在近目标从粗接近
  错误切换到低速但方向效率很低的微动作。诊断图位于
  `outputs/evaluation_flow_temporal_unet_v12_0p1um_dev20_fast10_20261115/plots/`
  和 `plots_phase_diagnostics/`。
- 2026-11-15：Flow covariate-shift DAgger round 2 启动 PARTIAL。教师接管
  hook 新增 `near_target_stagnation` 条件：在三维误差不超过 `2.5 mm`
  时，若连续 `120` 步没有至少 `5 µm` 的有效改善，则学生让出控制权；
  原有连续恶化、策略动作触限、教师无效、精确整数终点锁定和 `0.1 µm`
  最终门保持不变。每个 episode 的 `control_report.json` 现额外记录
  `executor_counts` 和 `takeover_trigger_reasons`，用于审计 policy、
  teacher、hold、exact-terminal 各阶段。pilot 使用新目标 seed
  `20263115`、起点 `0.5/1.0/2.0 mm`、学生单样本 Flow、`beta=0`，
  3/3 集 PASS；总 `412` 帧，教师无效帧 `0`，终点误差最大
  `1.7347e-12 µm`。pilot 224 转换输出
  `data/online_dagger_covshift2_pilot_224`，合并回放集为
  `data/lerobot_v12_oct_dagger2_covshift_pilot_224`（649 集、
  201,355 帧）。转换数据 OCT valid `385/412`，无效帧均来自
  exact-terminal/hold 阶段。`train_flow_temporal_unet.py` 新增
  `--init-from`，可从未污染的 epoch 3 Flow checkpoint 做增量 DAgger
  训练。正式 round 2 使用 20 个新目标、每个目标 3 个近目标起点，
  共 60 集，按 8 个分片写入
  `runs/20261115/online_dagger_v12/covariate_shift_round2_shards/g01..g08/`。
  g05 的 target 12/start 1 在首帧前因预跑状态不满足范围被安全跳过，
  其余采集仍在进行；完成前不将本轮视为最终数据或模型结果。
- 2026-11-15：Codex PostToolUse change-log hook 超时修复 PASS。故障表现为
  `Hook failed / hook timed out after 30s`；根因是 DAgger 运行目录持续写入
  数千张 1920×1440 帧，hook 每次工具调用都遍历整个工作区，状态清单膨胀到
  `53.9 MB`、日志目录达到 `80 MB`。修复后 hook 排除生成的
  `data/`、`outputs/`、`runs/`、`rcm_4090_deploy/{data,outputs,runs}/`
  和全部 `images/` 目录，仅跟踪源码/配置/文档；超时由 `30s` 提升到 `60s`，
  README 补充恢复和验证命令，自测增加生成帧目录排除断言。
  `--init` 后 tracked files `313`、日志目录约 `0.08 MiB`；初始化和状态
  查询实测约 `0.10/0.07 s`，远低于 30 秒限制。hook 自测 PASS，未删除
  既有审计事件。
- 2026-11-15：Flow covariate-shift DAgger round 2 采集、训练和配对闭环验证
  完成，采集/训练 PASS、控制验收 FAIL。20 个新目标、每个
  `0.5/1.0/2.0 mm` 起点，共请求 60 集；实际严格成功 `55` 集、`8,102`
  帧，`2` 集因 `LOW_QUALITY_VISUAL_GEOMETRY` 失败（约
  `70.829/79.664 µm`），`3` 集因预跑状态超范围在首帧前跳过。成功集
  teacher invalid `0`，最大终点误差 `5.5538e-11 µm`。执行来源为 policy
  `3,439`、teacher `3,054`、hold `1,155`、exact-terminal `454` 帧；
  takeover 触发统计为动作触限 `63`、恶化 `23`、近目标停滞 `6`。严格过滤
  后转换 8 个分片并与原 646 集合并，得到
  `data/lerobot_v12_oct_dagger2_covshift_verified_224`：701 集、
  209,045 帧，严格元数据审计 `701/701` PASS，最大专家终点误差
  `5.56e-11 µm`。从原 Flow epoch 3 增量训练 4 epochs，新增 7,222 个
  DAgger 窗口按 10 倍 replay 权重采样（有效权重占比 `27.48%`）；
  train/val flow loss 从 `0.158621/0.183125` 变为
  `0.136582/0.170068`。checkpoint
  `outputs/flow_temporal_unet_v12_oct_dagger2_covshift_20261115/best_model.pth`，
  SHA-256
  `675a9360d9f971d9d962ceae647c09d708cf74f408dcf3d1e6d5fb7d9dce5fea`。
  同一 development 20 目标、严格 RGB-D/OCT、MotorBackend、单样本 Flow、
  `0.1 µm` 三帧停稳门结果仍为 `0/20`。最终误差
  mean/median/min/max/P95 =
  `986.731/448.589/162.972/5065.026/3295.588 µm`；每集最小误差
  mean/median/min =
  `456.846/210.973/40.487 µm`。相对上一 Flow checkpoint，
  `15/20` 改善、`5/20` 恶化，最终误差平均/中位改善
  `744.749/321.881 µm`；安全限位 `0`，OCT valid `31,997/32,000`。
  原 E14 从 `15.219883 mm` 降至 `0.252049 mm`，最小误差
  `40.487 µm`；但 E07、E10 分别恶化至 `5.065/3.202 mm`。典型残余方向
  从上一模型的 `+X/+Z` 平台变为 `-X/+Z`，说明 DAgger 已改变近目标动作，
  但微动作仍未收敛。结果位于
  `outputs/evaluation_flow_temporal_unet_v12_oct_dagger2_covshift_0p1um_dev20_20261115/`，
  汇总与轨迹图位于同目录 `strict_summary.json`、`plots/` 和
  `plots_phase_diagnostics/`。结论：协变量偏移 teaching 明显缩小中位
  误差并把最坏异常拉回亚毫米，但纯 Flow BC 仍不能完成最后
  `162–448 µm` 平台；继续增加同构 DAgger 的收益有限，下一步需要显式
  OCT 比例微调或有界残差控制器。以上均为 MuJoCo 仿真结果。
- 2026-11-16：OCT proportional + bounded residual 分阶段实现完成；中间
  阶段未执行闭环评估。第一阶段从 covariate-shift DAgger 中只保留
  `0.1–2500 µm`、OCT valid、teacher valid、教师动作非零且执行来源为
  policy/teacher 的连续片段，排除 hold 和 exact-terminal，得到
  `data/lerobot_v12_oct_covshift2_finetune_v3_224`：68 段、6,471 帧，
  误差 `8.636–1,997.898 µm`、中位 `187.667 µm`，离线加载
  `6,471/6,471` PASS。为避免目标泄漏，按 shard 隔离为 train
  `g01–g05`（4,819 帧）和 val `g06–g08`（1,652 帧）。第二阶段生成
  `gain=0.5` 的 OCT 阻尼 Jacobian 比例基动作和近目标收缩 residual 侧车，
  训练 `GuidedOCTResidualPolicy`；最佳 epoch 64，验证动作 MAE
  A/B/C=`0.002408/0.005559/0.007545 deg/deg/mm`，normalized MAE
  `0.473844`。验证集近目标真实 OCT 的 C MAE=`0.001442 mm`，OCT 置零
  `0.003248 mm`，跨 episode 置换 `0.002306 mm`，相对 zero/shuffle
  分别改善 `55.6%/37.4%`，因果验收 PASS。checkpoint
  `outputs/oct_residual_covshift2_20261115/best_model.pth`。
- 2026-11-16：粗 Flow + OCT residual + 整数终点混合控制器 smoke PASS。
  `PolicyEvaluator` 新增 `--fine-model`、`--fine-switch-error-mm` 和
  `--terminal-refine-error-mm`；OCT 距离大于 `1 mm` 使用 DAgger Flow，
  `<=1 mm` 使用 OCT proportional+bounded residual，`<=0.2 mm` 使用当前
  state、OCT 和标定运动学反解整数编码器终点。无场景 rollout 的三模式
  推理 smoke 为 coarse/fine/terminal=`1/1/1`，动作均有限；合成可达终点
  搜索 125 个邻域候选后预测量化误差 `0.0 µm`，控制未使用目标真值。
- 2026-11-16：粗 Flow + OCT residual + 整数终点 development 20 目标
  `0.1 µm` 闭环 FAIL，且出现安全回归。最终成功 `0/20`；最终误差
  mean/median/min/max/P95 =
  `4800.836/1606.998/8.182/14571.062/14124.144 µm`，轨迹最小误差
  mean/median/min=`624.999/300.269/2.204 µm`。其中 11 集最终进入
  `8.18–11.54 µm` 的 OCT 测量/量化地板，但没有一集达到 `0.1 µm`；
  另外 9 集因终点反复反解和错误方向运动恶化到毫米至厘米级。控制模式
  总计 coarse/fine/terminal=`15,076/5,614/9,031` 帧，OCT valid
  `28,991/29,721`，安全限位 `5` 集共 `100` 步。相对上一纯 Flow+DAgger
  模型，`10` 集改善、`8` 集恶化、`2` 集持平；最终误差中位改善
  `76.618 µm`，但均值恶化 `3814.105 µm`。结果位于
  `outputs/evaluation_flow_oct_residual_hybrid_0p1um_dev20_20261115/`，
  汇总 `strict_summary.json`，轨迹图 `plots_v2/` 和
  `plots_phase_diagnostics/`。结论：混合控制器证明可以稳定到达约
  `10 µm`，但当前模拟 OCT 的可见深度系统误差约为数微米，纯感知闭环
  无法可靠证明 `0.1 µm`；重复整数终点求解还会放大目标估计误差。若要
  最终满足 `0.1 µm`，下一步只能引入时间稳定目标估计、终点命令去抖/单次
  锁定和独立的末端计量闭环，不能继续每帧重解终点。以上均为 MuJoCo
  仿真结果。
- 2026-11-16：冻结最近整数编码器可达目标 PASS。使用独立
  `RCMSceneV9` 正/逆运动学，对 development 40 个名义目标在每轴
  `±2 counts` 邻域枚举整数终点，按世界坐标欧氏距离选择最近可达点，
  并冻结 nominal target、reachable target、A/B/C counts、目标偏移向量和
  quantization error。输出
  `configs/v12_reachable_development_targets_0p35um.json` 及 SHA-256；
  40/40 PASS，目标偏移最小/中位/最大 =
  `0.006660/0.151372/0.351480 µm`，全部满足用户允许的 `0.35 µm` 量级。
  扩大搜索到每轴 `±8 counts` 后最小误差不变，确认不是搜索不足。
- 2026-11-16：OCT 终点目标锁存与单次 counts 锁定 smoke PASS。终端控制器
  连续 8 帧估计并取中位目标，之后只锁定一套整数编码器终点；每次动作由
  同一锁定 counts 减当前 A/B/C 得到，不再逐帧重新搜索。合成可达目标
  验证 `stabilize=7` 后进入 terminal，锁存后预测终点误差
  `6.54e-9 µm`。`eval_policy.py` 支持读取冻结可达目标 manifest，并分别
  记录 nominal/reachable target、目标偏移和冻结 counts。
- 2026-11-16：冻结可达目标 + 单次终点锁存 development 20 目标闭环 FAIL。
  验收门为相对冻结可达目标 `<=0.35 µm`、连续 3 帧停稳。结果
  `0/20`，没有任何 episode 最终或瞬时达到 `0.35 µm`。最终误差
  mean/median/min/max/P95 =
  `4890.596/2609.548/8.152/14399.986/14116.395 µm`；每集最小误差
  mean/median/min=`706.695/712.420/3.761 µm`。结果呈两极分化：
  `9/20` 集最终约为 `8.15–11.39 µm`，其余 `11/20` 在进入稳定终点前
  失败，误差为 `1.864–14.400 mm`。控制模式
  coarse/fine/stabilize/terminal=`16,046/5,632/63/8,022` 帧，安全限位
  `5` 集共 `100` 步。相对上一版未冻结目标混合控制器，`13/20` 改善、
  `7/20` 恶化，最终误差中位改善 `0.035 µm`、均值恶化 `89.761 µm`，
  说明停止逐帧重解减少了部分漂移，但没有解决 OCT 目标估计偏置和部分
  接近失败。原始结果位于
  `outputs/evaluation_flow_oct_residual_hybrid_reachable_0p35um_dev20_20261116/`，
  汇总 `strict_summary.json`，轨迹图 `plots_v2/` 和
  `plots_phase_diagnostics/`。结论：目标可达性已不再是主要阻塞；当前
  稳定平台约 `8–11 µm`，高于允许的 `0.35 µm`，主要来自模拟可见
  RGB-D/OCT 的目标估计误差，而大误差尾部来自终点锁定前接近失败。
  下一步必须直接降低目标估计误差并限制何时允许 terminal lock。
- 2026-11-16：主策略架构切换为 DiT + Rectified Flow，训练与推理 benchmark
  PASS（闭环评估尚未执行）。新增
  `lerobot_training/dit_rectified_flow.py`：ResNet-18 spatial token 编码、
  state+4D OCT+velocity+previous-action 几何融合、`T_p=16` action
  tokens、6 层 DiT、adaLN-Zero 条件调制、visual cross-attention 和
  4-step Euler rectified-flow 采样；训练目标直接回归
  `v = x_1 - x_0`，不使用 DDPM 噪声调度或 U-Net 去噪骨干。模型训练
  4 epochs，参数 `34.455M`，最佳 epoch 3，train/val rectified-flow loss
  `0.164041/0.189048`，输出
  `outputs/dit_rectified_flow_v12_oct_dagger2_20261116/best_model.pth`，
  SHA-256
  `4533a7f428f93011a5eb485d1a38f498641c1e3211f17cd21be7ff140650ff81`。
  同条件 RTX 4090 action-chunk 推理 benchmark：DiT/RF 4-step batch1
  `6.506 ms`、batch32 `15.501 ms`、`2064.4 sample/s`、
  `10.432 GFLOP/sample`、峰值 allocated `156/572 MiB`；优化后 1/2-step
  batch1 为 `2.320/3.745 ms`。与同等 observation-cache 优化后的
  1D U-Net Flow 4-step 相比，DiT batch1 `6.506 vs 5.223 ms`、FLOPs
  `10.432 vs 8.205 G`，DiT 约慢 `24.6%`、计算量高 `27.1%`；batch32
  吞吐 `2064 vs 2592 sample/s`。与历史 U-Net DDPM batch1 `9.274 ms`、
  `7.340 GFLOP/sample` 相比，DiT/RF 延迟低约 `29.8%`，但参数/模型文件
  约 `34.455M/394.5 MiB`，高于 DDPM 的 `12.581M/144.1 MiB`。训练吞吐
  约 `16.5 iter/s`（batch64），GPU 利用率可达 `99%`、训练显存约
  `5.74/49.14 GB`；显存充足但计算已饱和。benchmark 原始结果位于
  `runs/20261116/inference_benchmark/`。DiT/RF 单规划延迟远低于
  `50 ms` 控制周期，闭环主要延迟仍是 1920×1440 RGB-D 渲染与分割，
  不是策略网络。
- 2026-11-16：纯 DiT + Rectified Flow development 20 名义目标闭环 FAIL。
  评估明确禁用旧 U-Net、DDPM、冻结目标、OCT residual 和特权终点
  counts；使用原始 `target_split_manifest.json` 前 20 个 development
  目标、4-step RF、单样本、执行动作块首步、`0.35 µm` 三帧停稳门。
  成功 `0/20`，最终或瞬时达到 `0.35 µm` 均为 `0/20`。最终误差
  mean/median/min/max/P95 =
  `4080.069/2230.331/302.240/16439.474/15916.858 µm`；每集最小误差
  mean/median/min=`2399.433/130.471/41.532 µm`。全部 1600 步，
  safety-limited `0`。相对上一版 U-Net Flow + DAgger，仅 `2/20` 改善、
  `18/20` 恶化，最终误差平均/中位变化
  `+3093.338/+1296.779 µm`（正值表示退化）。最终误差方向以 `+Y/+Z`
  为主：排除 E14 后 Y 轴平均绝对误差 `2597.509 µm`，Y/Z 正向分别为
  `16/19` 和 `17/19`。说明 DiT/RF 能接近到约 `40–200 µm` 的瞬时最小
  误差，但当前直接在完整数据集上从零训练 4 epochs 的闭环稳定性和
  近目标动作质量明显不足。结果位于
  `outputs/evaluation_dit_rectified_flow_nominal_0p35um_dev20_20261116/`，
  汇总 `strict_summary.json`，轨迹图 `plots_v2/` 和
  `plots_phase_diagnostics/`。
- 2026-11-16：DiT + Rectified Flow 在线 DAgger round 1 采集完成，结果为
  PARTIAL PASS。以纯 DiT checkpoint 为 student、严格 RGB-D/OCT teacher
  为监督，使用 20 个非 development 目标、每目标 `0.5/1.0/2.0 mm`
  三种起始偏移，8 个并行 shard 共请求 60 集。实际完成 60 集；
  `59/60` 严格零误差，1 集终点失败，最终误差 `79.447254 µm`；失败集
  原样保留在 `runs/20261116/online_dagger_dit_rf_round1_shards/g05/`。
  采集总计 9,435 帧，teacher-invalid 3 帧。按 `strict_zero_error=true`
  且最终误差 `<=0.1 µm` 筛选后转换 59 集、9,388 帧到 8 个
  `data/online_dagger_dit_rf_round1_g01_224` 至
  `g08_224` 数据集，教师动作作为监督。原始结果位于
  `runs/20261116/online_dagger_dit_rf_round1_shards/`；以上均为
  MuJoCo 仿真数据，不代表物理 OCT 或临床精度。
- 2026-11-16：DiT + Rectified Flow DAgger round 1 增量训练与纯感知闭环
  FAIL。将 59 个严格零误差 DAgger 集与 701 集 covariate-shift 基线合并
  为 `data/lerobot_v12_oct_dit_rf_dagger1_round1_verified_224`：760 集、
  218,433 帧，`760/760` 显式严格零误差，最大末端误差
  `5.56195e-11 µm`；数据数组和加载 smoke PASS，dataset SHA-256
  `d619c510ee8348af6b84e9cd0462320bd6a6d2a68dcdd94758b69b020001c8f9`。
  从纯 DiT checkpoint 增量训练 4 epochs，batch 64，lr `5e-5`，新 DAgger
  样本权重 10；最佳 epoch 2，train/val loss
  `0.138848/0.158255`，验证损失较增量前的 `0.189048` 下降约
  `16.3%`。最佳 checkpoint 为
  `outputs/dit_rectified_flow_v12_oct_dit_rf_dagger1_20261116/best_model.pth`，
  SHA-256
  `3ad95651e66c4cb62adbe65ce765fd4408f59748a1619919d3c664319dd85d07`。
  使用原始 development 20 名义目标，无冻结、无 residual、无特权终点，
  4-step RF、单样本、执行动作块首步、`0.35 µm` 三帧停稳门，严格成功
  `0/20`，最终和瞬时达到门限也均为 `0/20`。最终误差
  mean/median/min/max/P95 =
  `2334.450/1546.263/173.753/4406.408/4376.837 µm`；每集最小误差
  mean/median/min/max=`217.851/229.868/56.182/340.374 µm`。全部
  1600 步，safety-limited `0`。相对上一版纯 DiT，最终误差
  `10/20` 改善、`10/20` 恶化，均值改善 `1745.619 µm`，中位退化
  `309.192 µm`；15–16 mm 大误差尾部消失。相对旧 U-Net Flow + DAgger，
  仅 `5/20` 改善，均值/中位分别退化
  `1347.719/941.997 µm`。最终残差以 `+Y/+Z` 为主，排除 E14 后平均
  绝对分量 X/Y/Z=`1203.552/545.251/1712.266 µm`，Y/Z 正向分别为
  `18/19` 和 `16/19`。结论：DAgger 显著改善纯 DiT 的大误差尾部和
  平均误差，但没有解决近目标协变量偏移与动作收敛，离线 loss 改善未
  转化为 `0.35 µm` 闭环成功。结果位于
  `outputs/evaluation_dit_rectified_flow_dagger1_nominal_0p35um_dev20_20261116/`，
  汇总 `strict_summary.json`、`strict_summary_vs_flow_unet.json`，
  轨迹图 `plots_v2/` 和 `plots_phase_diagnostics/`。
- 2026-11-16：DiT + Rectified Flow 近目标在线 DAgger round 2 采集完成，
  PARTIAL PASS。使用 DiT DAgger1 最佳 checkpoint 作为 student，围绕同一组
  20 个非 development 目标，以 `0.35/0.5/1.0/2.0 mm` 四种近目标起点
  共请求 80 集；实际 `79/80` 严格零误差，1 集因
  `LOW_QUALITY_VISUAL_GEOMETRY` 失败，最终误差 `90.637375 µm`。成功集
  总计 15,883 帧，失败集保留在
  `runs/20261116/online_dagger_dit_rf_round2_near_target_shards/g06/`。
  按严格零误差过滤转换为 8 个 224 数据集，并与 DiT DAgger1 数据合并为
  `data/lerobot_v12_oct_dit_rf_dagger2_near_target_verified_224`：
  839 集、234,316 帧、`839/839` 显式严格零误差，最大专家终点误差
  `7.85046e-11 µm`；dataset SHA-256
  `4d3d5cb9871a92b1a12dec2f0459968f574bd158a211f2fa56f7aecaa8c6114e`。
  以上均为 MuJoCo 仿真结果。
- 2026-11-16：DiT + Rectified Flow 近目标 DAgger round 2 增量训练与纯感知
  闭环 FAIL，但闭环指标继续改善。从 DAgger1 checkpoint 增量训练 4 epochs，
  batch 64，lr `5e-5`，第 760 集后的新 DAgger 样本权重 10；最佳 epoch 2，
  train/val loss `0.130223/0.158247`，相对 DAgger1
  `0.138848/0.158255` 的离线验证 loss 基本持平。checkpoint
  `outputs/dit_rectified_flow_v12_oct_dit_rf_dagger2_near_target_20261116/best_model.pth`，
  SHA-256
  `686b78ae7744884ce34369e545f760253251ac0ccfa3c30691528cadf3f15424`。
  使用原始 development 20 名义目标、无冻结、无 residual、无特权终点，
  4-step RF、单样本、执行动作块首步、`0.35 µm` 三帧停稳门，严格成功
  `0/20`，最终和瞬时达到门限也均为 `0/20`。最终误差
  mean/median/min/max/P95 =
  `1029.075/668.484/70.093/2280.636/2265.680 µm`；每集最小误差
  mean/median/min/max=`68.675/67.443/29.381/159.527 µm`。全部 1600 步，
  safety-limited `0`。相对 DiT DAgger1，最终误差 `15/20` 改善、`5/20`
  恶化，均值/中位分别改善 `1305.375/1063.676 µm`，最大误差从
  `4406.408 µm` 降到 `2280.636 µm`。与旧 U-Net Flow + DAgger2 相比，
  `10/20` 改善、`10/20` 恶化，均值/中位分别退化
  `42.344/14.407 µm`，两者最终误差已基本同一量级；但 DiT DAgger2 的
  每集最小误差均值 `68.675 µm` 明显低于 U-Net 的 `456.846 µm`，说明它
  能更稳定地接近目标，仍不能保持到终点。最终残差以 `-X/+Y/+Z` 为主，
  平均绝对分量 `591.759/173.865/795.878 µm`，X/Z 负向/正向分别占
  `19/20` 和 `18/20`。结果位于
  `outputs/evaluation_dit_rectified_flow_dagger2_near_target_nominal_0p35um_dev20_20261116/`，
  汇总 `strict_summary.json`、`strict_summary_vs_flow_unet.json`，
  轨迹图 `plots_v2/` 和 `plots_phase_diagnostics/`。结论：近目标 round 2
  显著改善 DiT 的接近能力和最终误差平台，但最后数十至数百微米仍无法
  收敛，纯 Flow 动作块仍缺少终点保持能力。
- 2026-11-16 22:23:46 CST：DAgger 动作语义、缺帧连续性、编码器真停稳门控
  修复与回归测试 PASS。数据契约 v2 新增 `executed_action`、
  `source_frame_index`、`timestamp_s`：`action` 保留专家监督标签，历史输入
  和速度只使用实际接受命令与原始时间戳；过滤缺帧会切断观测历史和 8 步动作
  窗口。历史模式遇到旧教师数据缺少执行动作字段时严格失败。闭环评估的
  `success` 改为误差达标且 `MotorExecution.settled()` 连续 3 帧，低速阈值
  仅作为诊断；旧口径汇总结果会被标记为 legacy。`python -m unittest
  discover -v -s tests` 15/15 PASS，涉及转换器、数据集、训练历史、缺帧、
  成功门控和 legacy 汇总；相关模块 `py_compile` PASS。本轮未训练、未运行
  新闭环评估。
- 2026-11-16 22:23:46 CST：DAgger2 动作历史数据迁移 PASS。从
  `runs/20261116/online_dagger_dit_rf_round2_near_target_shards` 重新转换
  8 个 shard，输出
  `data/online_dagger_dit_rf_round2_acthist_20261116_g01_224` 至
  `g08_224`，并合并为
  `data/lerobot_v12_oct_dit_rf_dagger2_acthist_20261116_224`：839 集、
  234,316 帧，动作源按 episode 标记为 executed/teacher，历史动作统一来自
  `executed_action`。第二轮原始 79 条合格轨迹共 15,886 帧，过滤 3 个
  `LOW_QUALITY_VISUAL_GEOMETRY` 后保留 15,883 帧；其中 5,617 帧由策略
  执行，5,614 个有效策略帧的教师标签与实际执行命令全部不同，未发现完全
  相同帧。A/B/C 教师-执行差值 RMS=`0.00923622/0.00740038/0.00953227`
  deg/deg/mm，P99 绝对差=`0.0321656/0.0210013/0.03`。三个缺帧位于
  global episode 776、801、832，训练窗口审计确认无观测历史或动作块跨缺口。
  完整迁移报告位于
  `runs/20261116/verification/dagger2_acthist_migration/verification_report.json`，
  含逐行对照原始 JSONL 的标签、执行命令、原始帧号和时间戳；有效训练窗口
  227,580，单样本图像/历史/动作块 smoke PASS。旧数据与旧评估结果均未覆盖；
  合并元数据曾错误继承基础集 `action_source=executed`，已修正为
  `per_episode` 并重建、复验，过程留档于
  `runs/20261116/migration/dagger2_acthist/`。结果均为 MuJoCo 仿真数据与
  软件语义验证，不代表硬件或临床精度。
- 2026-11-16 22:23:46 CST：电机真停稳专项复核 PASS。
  `python -m scripts.verification.verify_motor_execution -v` 7/7 PASS，
  覆盖协议目标转换、非法命令不瞬移、50 ms/2 ms 多子步滞后、反馈解码、
  sub-handle 累积及 `settled()`；新闭环成功口径依赖的电机定义未改变。
  新数据 `RCMDataset(include_previous_action=True)` 加载成功；旧
  DAgger2 数据在缺少执行动作契约时按预期立即报错，无静默回退。
- 2026-11-16 22:24:48 CST：最终交付物复核 PASS。全量单测再次
  15/15 PASS，全部修改模块 `py_compile` PASS。迁移数据集
  `data/lerobot_v12_oct_dit_rf_dagger2_acthist_20261116_224` 的
  `dataset.npz` SHA-256 为
  `f55feb52aa131bcd5a77f8432b29a3fa3894cf2a243711f2ccefa74cfc5f911c`，
  `metadata.json` SHA-256 为
  `8f05e8f88cc44a3261a372bd22e32c5e0804af626f392743b72fa2d4d47d8acc`；
  迁移报告 SHA-256 为
  `6b858f5c70b41687be0770d38b84f5aa49862962c575c415c54ac27333c0a8aa`。
  本轮没有触发或静默回退到 CPU 训练/推理；迁移和验证均为 CPU I/O 与数据
  正确性工作，未执行新的模型训练或闭环评估。
- 2026-11-16 22:25:11 CST：新数据集跨来源图像/历史 smoke PASS。分别从
  round1 起始 episode 0、round2 首集 760、三个缺口所在 episode
  776/801/832 和末集 838 读取一个有效训练窗口；图像均为
  `[2,3,224,224]`，上一动作和速度有限。确认合并后的相对图像路径同时覆盖
  保留的旧数据和新转换 shard。
- 2026-11-17 13:51:52 CST：acthist DiT DAgger2 重训前启动与验证集迁移
  FAIL。首次启动因输出目录被预先创建而立即拒绝启动（0 训练）；
  第二次启动已加载训练集 220,844 窗口，但旧
  `data/lerobot_rcm_val_224` 缺少 action-history 契约，严格失败；
  验证集升级第一次因元数据无 `action_source` 失败，增加显式
  `--legacy-action-source executed` 后，又因历史 Windows 反斜杠图像路径失败。
  四次失败均未改动旧数据、未完成 epoch、未产生模型 checkpoint，日志与临时
  输出保留在 `runs/20261117/training/`。修复模块导入、显式 legacy 语义和
  路径规范化后，验证集升级为
  `data/lerobot_rcm_val_acthist_20261117_224`：40 集、27,290 帧、
  26,650 有效窗口；`dataset.npz` SHA-256
  `2b0760827c6d2a53dcf9c02ae589ad043b8de3156782ee0befed8ea475c32ec7`。
- 2026-11-17 13:51:52 CST：acthist DiT DAgger2 重训完成 PASS（离线训练，
  尚未闭环评估）。使用 RTX 4090、CUDA 12.6、FP16 AMP，从 DiT DAgger1
  epoch 2 checkpoint 初始化；数据为
  `data/lerobot_v12_oct_dit_rf_dagger2_acthist_20261116_224`，训练/验证窗口
  220,844/26,650，batch 64，lr `5e-5`，2 帧观测、16 步动作块、4-step
  rectified flow，最后 79 集权重 10，共 4 epochs、961.68 s。各 epoch
  train/val loss：`0.149197/0.167302`、`0.142475/0.169941`、
  `0.138629/0.162381`、`0.135912/0.161541`；best/last 均为 epoch 3，
  val loss `0.161541`。相比旧 DAgger2 最佳 val loss `0.158247` 退化
  `0.003294`（约 `2.08%`），因此本次修复数据语义后的离线验证指标暂未改善。
  输出
  `outputs/dit_rectified_flow_v12_oct_dit_rf_dagger2_acthist_20261117/`；
  `best_model.pth` SHA-256
  `c7156b4d6e2542d808b2397dc7135001175af25cb7bbe8098dc69e4a5b44ce0e`，
  模型加载成功、全部参数有限，`PolicyEvaluator` 可在 CUDA 上加载该
  checkpoint。`training.log` SHA-256
  `ccc87d4ad6c770db80ecd1b814e107b34d164ffd9cb7fa5a8c4160c035e6839b`。
  全量单测随后 15/15 PASS。该结果仅为 MuJoCo 仿真离线训练，不能代表闭环
  成功率或硬件精度；下一步需用新的编码器真停稳口径评估该 checkpoint。
- 2026-11-17 15:11:17 CST：acthist DiT DAgger2 development 20 目标闭环
  FAIL。使用
  `outputs/dit_rectified_flow_v12_oct_dit_rf_dagger2_acthist_20261117/best_model.pth`
  在相同 20 个 development 目标上并行 10 分片复评，每集 1600 步，
  `0.35 µm` 误差阈值、单动作样本、执行动作块首步、4-step rectified flow。
  新评估契约 20/20 正确记录 `version=2` 与
  `terminal_condition=MotorExecution.settled`。结果为编码器真停稳成功
  `0/20`，最终或瞬时进入 `0.35 µm` 均为 `0/20`；末步低速仅 `2/20`，
  末步编码器真停稳 `0/20`，20/20 均跑满 1600 步，safety-limited 总数为 0。
  最终误差 mean/median/min/max/P95 =
  `1737.487/2012.021/238.893/2342.127/2315.729 µm`；每集最小误差
  mean/median/min/max=`143.065/129.572/37.391/279.056 µm`。
  与旧 DiT DAgger2 同组结果配对比较：最终误差 `5/20` 改善、`15/20`
  恶化，均值/中位分别退化 `708.412/635.352 µm`；旧版最终误差
  mean/median=`1029.075/668.484 µm`。因此 action-history 修复后的本轮
  重训模型闭环性能明显下降，不能归因于成功口径变化（两组均 0 成功）。
  结果保存于
  `outputs/evaluation_dit_rectified_flow_dagger2_acthist_20261117_dev20_0p35um/`，
  汇总 `strict_summary_vs_old.json` SHA-256
  `990e58e59377e6d4560c40e8acf2f81ac3c17994a729b1d3cbddc813edfba50a`。
  全流程中 10 个评估子进程均 PASS 退出；从启动到分片完成约 52 分钟。结果
  仍为 MuJoCo 仿真性能，不代表硬件或临床精度。
- 2026-11-17 15:18:23 CST：误差图生成与 GitHub 上传 PASS。新增
  `plot_closed_loop_errors.py`，使用正确 `0.35 µm` 门限生成 20 条误差曲线、
  单目标小图、best/final 误差图和旧模型配对误差图；输出目录
  `outputs/evaluation_dit_rectified_flow_dagger2_acthist_20261117_dev20_0p35um/plots_acthist_20261117/`。
  采用 GitHub SSH 部署密钥推送独立分支
  `experiment/dit-rf-dagger2-acthist-20261117`，提交
  `4f02aff`（数据契约与真停稳修复）和 `02af806`（训练/评估报告）。
  上传报告位于
  `rcm_4090_deploy/github_reports/model_status_20261117/`，包含四个 PNG、
  20 目标汇总、训练历史、迁移证据、训练/验证元数据及 4.7 MB 完整评估压缩包；
  未上传 790 MB checkpoint 或原始 RGB-D 数据。远端 `MODEL_STATUS.md`、
  `error_curves_all.png` 和评估压缩包均已验证 HTTP 200。
- 2026-11-17 15:27:55 CST：state-conditioned temporal attention 增量包接入
  与 GPU smoke PASS（仅接入/训练路径验证，未进行完整微调或闭环评估）。
  新增
  `lerobot_training/dit_state_attention.py`，SHA-256
  `415b8a303c8b01aa5ab4d1f33d7ace8406f14a8760cdf3185c9adcdb9b0b0766`；
  相对交付包仅增加 `TORCH_DISABLE_NATIVE_JIT=1` 兼容开关，避免当前
  PyTorch 2.14/Triton 在注意力偏置反传时因缺少 `Python.h` 而即时编译失败。
  使用 RTX 4090、CUDA 12.6、checkpoint
  `outputs/dit_rectified_flow_v12_oct_dit_rf_dagger2_acthist_20261117/best_model.pth`
  （epoch 3，val loss `0.1615405332`）验证：原权重严格热加载 PASS，新增
  注意力参数 83,136 个，增强前后初始 GPU 输出最大绝对差 `0`；真实 checkpoint
  FP32 rectified-flow 反传得到有限梯度，新增门控梯度非零；`eval_policy`
  能构造增强策略并加载全部新增权重；backbone/attention 学习率分组为
  `1e-5/1e-4`。连续 8 个真实训练样本的 FP16 AMP 训练步 PASS，loss
  `0.418949`，自适应缩放后成功更新 6/8 步，门控/主干最大权重变化
  `5.139e-4/5.585e-5`，最终 scale `16384`，全部参数有限。语法检查和
  `python -m unittest discover -v -s tests` 15/15 PASS。完整证据及失败诊断
  位于 `runs/20261117/verification/state_attention_integration_retry3/`、
  `runs/20261117/verification/state_attention_amp_real_samples8_retry1/` 和
  `runs/20261117/verification/state_attention_unittest/`；过程未回退到 CPU。
- 2026-11-17 15:27:55 CST：state-attention 测试迭代中的失败与诊断记录。
  `runs/20261117/verification/state_attention_integration/` 首次 GPU 反传
  FAIL：checkpoint 加载与 GPU 前向均已 PASS，但 PyTorch/Triton 即时编译因
  系统缺少 `Python.h` 失败；随后 `integration_retry1` 因测试脚本属性名错误
  FAIL（未进入模型计算），`integration_retry2` 在设置兼容开关后 PASS。
  `state_attention_amp_step/`、`state_attention_amp_step_retry1/`、
  `state_attention_amp_step_retry2/` 和
  `state_attention_amp_real_sample/` 的单批次 AMP 更新断言 FAIL；经
  `state_attention_amp_grad_diagnostic/`、
  `state_attention_amp_grad_baseline_compare/`、
  `state_attention_amp_real_sample_ab/` 及
  `state_attention_amp_optimizer_diagnostic_retry1/` 复核，原 DiT 和增强
  DiT 在同一随机 flow 状态下都会被初始 `scale=65536` 的梯度缩放器判定
  溢出并跳过更新，因此不是新增注意力回归。最终
  `state_attention_amp_real_samples8/` 已证明参数可更新，但其“恰好一步”
  断言不符合 AMP 自适应跳过语义而 FAIL；放宽为“两组步数一致且至少一次”
  后在 `state_attention_amp_real_samples8_retry1/` PASS。所有失败日志均
  保留，未覆盖旧结果。
- 2026-11-17 15:51:29 CST：state-conditioned temporal attention v1
  acthist DiT 微调完成 PASS（离线训练，尚未闭环评估）。使用
  `lerobot_training/dit_state_attention.py train`，训练/验证数据分别为
  `data/lerobot_v12_oct_dit_rf_dagger2_acthist_20261116_224` 和
  `data/lerobot_rcm_val_acthist_20261117_224`，训练/验证窗口
  220,844/26,650；从
  `outputs/dit_rectified_flow_v12_oct_dit_rf_dagger2_acthist_20261117/best_model.pth`
  （epoch 3，val loss `0.1615405332`）热启动。配置为 RTX 4090、CUDA 12.6、
  FP16 AMP、4 epochs、batch 64、2 帧观测、16 步动作块、4-step
  rectified flow、DAgger episode start 760、repeat 10、seed 20261116；
  新注意力 lr `1e-4`，主干 lr `1e-5`。各 epoch train/val loss：
  `0.120944/0.151613`、`0.117699/0.150786`、
  `0.115918/0.150879`、`0.114257/0.149865`；best/last 均为 epoch 3，
  val loss `0.1498646501`，较热启动基线降低 `0.0116758831`
  （`7.227835%`）。总训练历史 elapsed `979.608 s`。输出目录
  `outputs/dit_rectified_flow_state_attention_v1_acthist_20261117/`；
  `best_model.pth` SHA-256
  `d1e2527fe087ad1f619450bdd83eac24e63d1e3a0d96186dc2cdd849fdfbc424`，
  `last_model.pth` SHA-256
  `f7627f65ffc2eadc13d96e3828d271440c1b5099c7d9bdc8bc2f5b466e551b78`，
  `training.log` SHA-256
  `c01d28387fcdf1e1e92481eb0e66ff44d9b3d7633529405cd6371e7c3d1ce798`。
  新 checkpoint 通过 `eval_policy.PolicyEvaluator` 在 CUDA 加载，全部参数
  有限，架构 metadata 为 `state_conditioned_temporal_attention_v1`，新增
  注意力参数 83,136。验证日志位于
  `runs/20261117/verification/state_attention_trained_checkpoint_retry1/`。
  首次 checkpoint 校验脚本因硬编码 val loss 多写一位不一致而 FAIL，修正为
  从 `training_history.json` 读取真值后 PASS；失败日志保留在
  `runs/20261117/verification/state_attention_trained_checkpoint/`。结果仅为
  MuJoCo 仿真离线训练指标，不能代表闭环成功率、硬件或临床精度。
- 2026-11-17 16:48:23 CST：state-attention v1 development 20 目标闭环评估
  FAIL。使用 `best_model.pth`（epoch 3，val loss `0.1498646501`）和冻结的
  `configs/v12_reachable_development_targets_0p35um.json`，按 acthistory
  基线相同协议评估：10 个并行分片、每片 2 个目标、1600 步上限、`0.35 µm`
  误差门限、连续 3 帧 `MotorExecution.settled`、单动作样本、执行动作块首步、
  4-step rectified flow、50 ms 控制周期。评估
  `2026-11-17T15:52:38+08:00` 启动，`16:44:17+08:00` 完成，历时约
  3099 s；10/10 shard 均 exit 0。严格成功 `0/20`，最终或瞬时进入
  `0.35 µm` 均为 `0/20`，safety-limited steps 为 0。最终误差
  mean/median/min/max/P95 =
  `434.397/258.933/134.901/2128.198/1131.427 µm`；每目标最小误差
  mean/median/min/max/P95 =
  `93.680/89.067/45.103/201.134/145.855 µm`。与 acthistory 基线逐目标
  配对比较：18 个改善、2 个恶化，最终误差均值/中位数改善
  `1303.090/1668.789 µm`；基线最终误差 mean/median=
  `1737.487/2012.021 µm`。因此闭环误差显著改善，但仍未达到严格成功门限。
  结果目录
  `outputs/evaluation_dit_rectified_flow_state_attention_v1_acthist_20261117_dev20_0p35um/`；
  `strict_summary_vs_acthist.json` 与 `summary.log` SHA-256 均为
  `004a3844dd64728a2f31f9c144a2f7ce7e7c73a6a19e3f6bb3c926ad5d683ea0`，
  `error_curves_all.png` SHA-256
  `c5a6313d826405e49aec987a26f08c42ab0da85067baf83646ecb0dbbe093c4d`。
  首次绘图尝试因 stdout 重定向到尚未创建的 plot 目录而未执行；改为先由
  绘图脚本创建目录，并把生成日志写到既有结果根目录后 PASS。结果仅为
  MuJoCo 仿真闭环性能，不代表硬件或临床精度。
- 2026-11-17 16:57:36 CST：统一模型流水线接入与入口 smoke PASS。新增
  `lerobot_training/run_model_pipeline.py`（SHA-256
  `1a3fb5e1220f187d4c63da03665d7f63207c4252fa845572ed70cf0c565176ac`）、
  `configs/state_attention_pipeline.json`（SHA-256
  `a1e56663f58107f34e71ab39759755b954567b5b9c7ba1072e0c9cf6a5c6e777`）
  和 `docs/MODEL_PIPELINE.md`（SHA-256
  `5c255edd432f8844f0dc0c268a6d8c4d406f5321adcc157c0cfe83ff7015f9ee`）。
  `py_compile` 与 `--help` 入口检查 PASS；配置解析为
  `state_attention_v1`、入口 `lerobot_training/dit_state_attention.py`、
  development 目标 0–20、10 分片。此检查未启动训练或闭环评估，也未生成
  模型产物。
- 2026-11-17 17:02:54 CST：统一模型流水线分段结果输出 smoke PASS。训练与
  验证完成后立即生成 `training/result.json` 和 `training/REPORT.md`，
  记录最佳 epoch、train/val loss、训练耗时、checkpoint 路径与 SHA-256，
  并把 `status.json` 标记为 `training_completed`。每个闭环评测分片完成后
  立即生成 `evaluation/gXX/result.json`，刷新
  `evaluation/partial_summary.json` 与 `partial_episodes.csv`，并保存
  `evaluation/progress/after_NN_shards/` 不可变累计快照；部分结果明确标记
  `status=partial`。临时 fixture 验证 1/2 与 2/2 shard 两个阶段、目标排序、
  成功数、CSV 和快照均 PASS，临时目录已自动清理。修改后
  `run_model_pipeline.py` SHA-256
  `7e2f3385e84b52655e6724e0a433a9d499b64ed6b5d4cac47f6abad1a96996d4`，
  `docs/MODEL_PIPELINE.md` SHA-256
  `aeb097e91ebbc7cd1138dc9885402cecf9f3cd5f8defcb264473b1d8954d246b`；
  验证日志位于
  `runs/20261117/verification/model_pipeline_progress/`。未启动真实训练或
  闭环评估。
- 2026-11-17 18:38:44 CST：state-attention v1 冻结为首个正式对照基线 PASS。
  基线目录 `configs/baselines/state_attention_v1/` 保存完整流水线配置副本、
  训练历史、20 目标严格评估汇总、机器可读 `BASELINE.json` 和
  `BASELINE.md`；基线 ID 为 `state_attention_v1_20261117`，状态
  `frozen`。基线指标为：canonical settled success `0/20`，最终误差
  mean/median `434.397058/258.933493 µm`，每目标最小误差均值
  `93.680006 µm`，safety-limited steps `0`。`BASELINE.json` SHA-256
  `3679d03cc2465bcb58c2eae247bd290736cd2ee7cd7da3434a6f48fdc533bb55`；
  模型源码、v1 checkpoint、冻结配置和目标清单的 SHA-256 均已记录并通过
  完整性复算。工作配置
  `configs/state_attention_pipeline.json` 默认通过 `baseline_manifest`
  指向该基线。流水线新增最终自动对照：强制校验目标清单 SHA-256、episode
  索引和 target ID；匹配时生成 `comparison.json` 与 `COMPARISON.md`，
  不匹配时拒绝输出不可比指标。隔离 smoke 验证 1 个改善、1 个恶化、成功
  差值、两类报告及错误目标拒绝均 PASS；验证日志位于
  `runs/20261117/verification/model_pipeline_baseline_comparison/`。
  修改后流水线 SHA-256
  `16057dfc3ed70ee47a0a20a19827c2f07c2f61b088f22f01e9c9020ed1ee7734`，
  工作配置 SHA-256
  `77c1a23c6060aff8f4cf4aa51aa475f08e80020d73b9b926e20a19026adb38c0`，
  文档 SHA-256
  `2f6b3715ad040b6bf20c57af4f50f6e425d6c0644af182413d216a5ead8ec368`。
  本次未启动真实训练或闭环评估。
- 2026-11-17 20:48:59 CST：E1 state-attention v1 同预算续训 FAIL。
  使用新流水线配置
  `configs/state_attention_v1_continue_pipeline.json`（SHA-256
  `fd930f3ac43762074665eef1b9277c07b2c51ba0f935189d1ce611a921a71fce`），
  从 v1 checkpoint 初始化，新偏置 lr `1e-5`、主干 lr `1e-6`，
  保持 4 epochs、batch 64、seed 20261116、DAgger start 760 / repeat 10。
  best epoch 1：train/val loss `0.1107269140/0.1467070205`；最后 epoch 3：
  `0.1089468125/0.1477184232`；训练耗时 973.384 s。best checkpoint SHA-256
  `4bc962b328ac57cc9f00e137f7866afe28ce5aa3592c3d166c525619d5f0610b`。
  使用冻结 20 目标严格协议完成闭环评估，10/10 shard 正常结束。正式成功
  `0/20`；最终误差 mean/median/min/max =
  `978.815/637.927/143.672/2548.990 µm`；每目标最小误差均值
  `72.531 µm`；safety-limited steps 0。与冻结 v1 baseline
  `state_attention_v1_20261117` 自动对照：成功数不变；最终误差均值恶化
  `544.418 µm`、中位数恶化 `378.994 µm`；最小误差均值改善 `21.149 µm`；
  逐目标 7 个改善、13 个变差。因此本轮不能保留为 v1 的替代版本，KV8
  初筛应以 E1 为同预算对照，而不能只看验证 loss。输出目录
  `outputs/state_attention_v1_continue_20261117_1/`；训练阶段独立结果
  `training/result.json` SHA-256
  `b00a6b96f40eaa27f89c953a600c8d5d33584323b27d76c2b962010ff5250856`，
  最终 `summary.json` SHA-256
  `cc43b8f02558707fbc5d13656df9cffc86d34c5b63d95a800d31339d6296463e`，
  `comparison.json` SHA-256
  `62f94397a34a27041eb493cba2cc99f1092244a4224dab9445a61a8ed0d7bb3c`。
  执行中发现旧流水线把仍只在写第一目标的 shard 误计为已完成；修复为只统计
  已正常退出 shard，隔离测试 PASS。E1 启动时使用旧代码快照，故其
  `progress_pre_fix/` 与 `partial_summary_pre_fix.json` 保留为问题证据；
  最终结果已重新生成正确的 `progress/after_01_shards/` 至
  `after_10_shards/`，新生成到 `after_10_shards` 时为正确的 10 shard /
  20 episode / success 0。修复后流水线 SHA-256
  `a54dbf166aec3612af359aa5150f0f5530d88648332839e800c2c8cc634c00c4`。
  结果仅为 MuJoCo 仿真，不代表硬件或临床精度。
- 2026-11-19 08:52:20 CST：经验回放边界、共享索引和比例实验完成。
  `valid_window_indices` 修复了审计器多检查一帧的问题；20 帧、2 帧历史、
  16 步动作测试得到 4 个窗口。训练与审计共享 220,816 个有效索引，SHA-256
  `3f852d9c7b296cdcc4c4abbca6a12e0e099050bc5bbfc4254aae979830fb927d`；
  自然纠正比例 `6.5987%`。剔除 episode 边界 `13,408`、源帧间断 `48`、
  时间异常 `0`、无效输入 `28`。未知标签来源改为直接报错。契约报告
  SHA-256
  `aa08478b826b9caf732041b637f8899f635f6fe06f632c319c168493e12a1ae7`。
- 2026-11-19 08:52:20 CST：比例实验零更新复现 PASS。220 个权重张量完全
  一致，固定噪声单步动作完全一致；3 个固定开发目标的 states/errors/
  velocities/settle flags 逐帧一致，原始动作最大差 `2.015e-6`。报告
  `runs/20261118/verification/zero_update_reproduction/zero_update_verification.json`
  SHA-256
  `26f13c8bd6fcf4f8d5cded916fb8b9da5e1e08a78ed85dd411811a17125d0d91`，
  轨迹对照
  `runs/20261118/verification/zero_update_trajectory_parity.json` SHA-256
  `1555f47806836e34c01c4c0aa15c36f91f31da68544fe715dbd6d0edab454bf4`。
- 2026-11-19 08:52:20 CST：纠正比例 R0/R10/R20 完整流水线全部完成，
  均未达到严格成功门限，均无编码器停稳，safety-limited steps 均为 0。
  R0 自然 `6.5987%` 最终误差 mean/median/P90/max=
  `247.936/170.736/566.363/658.363 µm`，与 v1 配对 19 改善、1 恶化；
  R10 为 `980.699/549.848/2435.211/2521.840 µm`，9 改善、11 恶化；
  R20 为 `589.481/227.022/1430.254/2202.496 µm`，11 改善、9 恶化。结论为
  自然比例最佳，提高纠正比例会造成退化。汇总
  `runs/20261119/analysis/experience_ratio/ratio_comparison.json` SHA-256
  `a72959710b0f356cfdb8050d33052c7fca40fc715c996c78bebb259e3531f599`。
  结果目录
  `outputs/state_attention_experience_ratio_20261118_1/`。
- 2026-11-19 08:52:20 CST：规划 2 采集试点完成并开始剩余训练数据采集。
  初次编排目录 `data/experience_replay_plan2_20261118/` 只生成 seeds，
  因创建顺序错误未采轨迹；`_retry1` 因 DAgger 未分派状态注意力架构而未采
  轨迹，两者均保留。修复后 `_retry2` 完成 D1/D2/D4 共 15 条，另用
  `data/experience_replay_plan2_d3_pilot_beta1_20261119/` 完成专家主导
  D3 5 条；验收试点共 20 条、8,538 帧。D3 `beta=0.5` 试点中 2/5 恶化，
  改为 `beta=1.0` 后已采集轨迹从 `20–77 µm` 改善到 `0.016–4.081 µm`，
  原 D3 试点作为 superseded 保留。剩余 180 条训练轨迹已启动，目录
  `data/experience_replay_plan2_20261119_remaining1/`；验证和测试各 40 条
  待训练集完成后采集。紧凑汇总
  `runs/20261119/analysis/plan2_collection/plan2_collection.json` SHA-256
  `a492e128bb05457ae2017697d9ab4fdb945243498e3b9d9e758bfe9061f7969d`。
  采集数据仅为仿真数据，不代表硬件或临床精度。
- 2026-11-19 15:40:00 CST：规划 2 专家纠正数据集采集完成。按用户确认将
  测试集上限从 40 调整为 35 条，最终接受训练/验证/测试
  `200/40/35`，共 `275` 条、`110,771` 帧。分类结果为：训练
  D1/D2/D3/D4=`70/50/50/30`，验证=`14/10/10/6`，测试=`14/10/10/1`。
  训练集 D4 有一条重复 target seed，已从接受计数中排除（重复帧 1,600）；
  测试 D4 分片在达到 35 条上限后停止，未完成目录保留为 interrupted
  evidence，不计入最终数据。并行续采期间，所有中断中的 episode 均归档为
  `partial_episode_*`，已完成 control report 保留。最终原始数据目录为
  `data/experience_replay_plan2_20261118_retry2/`、
  `data/experience_replay_plan2_d3_pilot_beta1_20261119/`、
  `data/experience_replay_plan2_20261119_remaining1/`、
  `data/experience_replay_plan2_20261119_validation1/`、
  `data/experience_replay_plan2_20261119_test1/`；最终清单
  `github_reports/model_status_20261119_plan2_collection/plan2_final_summary.json`。
  所有采集进程已结束，结果仅为 MuJoCo 仿真数据。
- 2026-11-17 20:52:25 CST：E1 轨迹误差图生成 PASS。绘图脚本新增可配置
  candidate/baseline 图例与配对文件名；使用 E1 20 目标完整轨迹和冻结 v1
  基线生成 4 张图：全部轨迹、20 宫格、best/final 对比、E1 对 v1 最终误差
  配对柱状图。图中成功门限为 `0.35 µm`，图例明确标为
  `State-attention E1` 与 `State-attention v1 baseline`。输出目录
  `outputs/state_attention_v1_continue_20261117_1/plots_state_attention_e1_20261117/`；
  `error_curves_all.png` SHA-256
  `ee4ad3b9effae39dca7e3a794b1b2841331d0b33355c2d2f82a80ccfde338c90`，
  `error_curves_small_multiples.png` SHA-256
  `dc452fe8cb04f3f1967a629b00f97e53a2654f6cc7d395ba79e5665f1a6b8f07`，
  `best_and_final_error.png` SHA-256
  `8eaaf50188bac17e9e0c4c45195575bfa766b79bb218820e03dab2ed5bfbb147`，
  `final_error_vs_v1.png` SHA-256
  `0c79ef36ee9829f1592d21860629fb2a83b6b6995f16fc0e193f4f00b4abd08d`。
  绘图脚本 SHA-256
  `f6478b50296dacacf9d44b4bbed6cf452e89b467e60fc47e90fe7e4d9bb70a67`。
  图像已同步到 E1 GitHub 报告目录，尚未另行推送远端。
- 2026-11-18 08:46:12 CST：KV8/E2 完整流水线 FAIL。Q/输出保持 16 token，
  K/V/偏置键使用固定相邻平均压缩到 8，v1 checkpoint 严格加载且参数量仍为
  34,538,115；GPU 前向有限。模型源码
  `lerobot_training/dit_state_attention_kv8.py` SHA-256
  `1b57216b9bea596c73d64fee9bcee01848f0e7b546b18dc6c8d1071823fddad8`，
  配置 SHA-256
  `c9b5a2602dacf0f10976371b01b1956366e0fe11260d35ef86b44870b74abce6`。
  4×1000 步训练 best epoch 3，best val loss `0.174219`；best checkpoint
  SHA-256
  `06d36598f35b83ec87e8f71498201f16fea733d98b74508f364d4a9db91ccea2`。
  20 目标闭环 canonical success `0/20`，最终误差
  mean/median/min/max=`1513.620/1713.822/166.260/2983.236 µm`，每目标最小
  误差均值 `80.416 µm`，safety-limited steps 0。与 v1 配对：4 改善、16
  恶化，最终误差均值恶化 `1079.223 µm`。结果目录
  `outputs/state_attention_kv8_20261117_1/`；`summary.json` SHA-256
  `c3fcedb5502a05311d7d6fbfc1a237f253446fe1a4073c41170a63d998ef4e55`，
  `comparison.json` SHA-256
  `0b0277e46b88bf8d54dd4d93e5a90458fb701ab89cee55a6de9de80623095039`。
  KV8 不能替代 v1，也不值得进入 KV4。
- 2026-11-18 08:46:12 CST：Replay actor-critic v1 完整流水线 FAIL（严格
  成功门限），但为当前最佳最终误差版本。Replay 6/6 数据与梯度单元测试 PASS；
  真实数据 GPU smoke PASS；经验池包含 225,614 个连续、OCT 有效转移。
  入口源码 SHA-256
  `10dbd2a0cc51d01ba13a77d2a8a6c8c90f02f212837d28fd42604ff0274961c0`，
  回放模块 SHA-256
  `50104076ef5aa617c294f70f8187d28ec4ec4ef1ab5d838fde5c6879d4aeff22`，
  配置 SHA-256
  `338d7cb54248291fa70d13e11072fcad633ec03edd70c9698493cf4b7487e46a`。
  4×1000 步训练 best epoch 3，best val loss `0.148059`；best checkpoint
  SHA-256
  `015bb3d2330693f9d8a46967c4f40698e2e54b708f2da2a8e1e87398fe740c8c`。
  20 目标 canonical success `0/20`，最终误差
  mean/median/min/max=`171.028/161.439/92.671/269.203 µm`，20/20 最终误差
  ≤300 µm，每目标最小误差均值 `77.841 µm`，全组最好瞬间误差 `28.463 µm`，
  从未进入 `0.35 µm`；safety-limited steps 0。与 v1 配对：19 改善、
  1 恶化，均值/中位数改善 `263.369/74.930 µm`；与 E1 配对：17 改善、
  3 恶化，均值改善 `807.787 µm`。Policy Q 均值从 `-2.256` 漂移至
  `-10.455`，critic 尺度稳定性仍需修复。结果目录
  `outputs/state_attention_replay_v1_20261117_1/`；`summary.json` SHA-256
  `9eca5e1699cda6cb76e3d842a480ca1e57ba876de2938987a0d16e607d87e443`，
  `comparison.json` SHA-256
  `4d1448281a0d6d0ee55d35b5039050952428af2089b59a927a2c651912bc16cd`。
  结果仅为 MuJoCo 仿真，不代表硬件或临床精度。
- 2026-11-18 13:44:11 CST：ILOSAC 来源与公式审计接入 PASS。新增
  `docs/ILOSAC_SOURCE_FORMULA_AUDIT.md`、原始论文
  `docs/references/ILOSAC_cit2.70076_original.pdf`（SHA-256
  `14fedf6a282501f1b356868c3f775d4dd22343d818c62251f3a7aa042faa5816`）
  和机器可读来源状态
  `docs/references/ILOSAC_source_status_20261118.json`；PDF 哈希与状态 JSON
  完全一致。文档索引已加入 ILOSAC 审计和 Replay actor-critic 条目，
  `REPLAY_ACTOR_CRITIC.md` 已明确当前实现不是已验证的 ILOSAC/THMM 复现，
  官方源码状态仍为 `not_verified`。已核对原 PDF 第 7 页式(17)软 Q 项确实
  带平方，式(20)使用示范策略 KL 约束；当前 Flow matching 不能等价替代该
  KL。本次仅接入审计资料，没有实现或启动新的 ILOSAC 复现实验。
- 2026-11-18 15:15:05 CST：ILOSAC-inspired Flow v2 完整流水线 FAIL
  （严格成功门限），但为当前最佳最终误差版本。模块标记为
  `flow_ilosac_inspired_offline_v1`，不是 ILOSAC 复现。实现加入冻结示范
  策略动作锚定（KL 近似）和 TD target `[-5,5]` 裁剪；明确未实现 THMM、
  持续时间模型、在线约束与 entropy temperature。6/6 replay 单元测试和 GPU
  smoke PASS。4×1000 步训练 best epoch 1，best/last val loss
  `0.1482578143/0.1484347544`；checkpoint SHA-256
  `1ab324fca7f388b5747cc976d42ccc62e24781c47a78e86e9131fcde021d0015`。
  20 目标 canonical success `0/20`，最终误差
  mean/median/min/max=`162.215/148.591/88.933/342.324 µm`，19/20 最终误差
  ≤300 µm，每目标最小误差均值 `63.193 µm`，全组最好瞬间误差
  `31.557 µm`，从未进入 `0.35 µm`，safety-limited steps 0。与 v1 配对：
  18 改善、2 恶化，均值/中位数改善 `272.182/81.240 µm`。Policy Q 均值从
  `-2.031` 到 `-3.712`，明显优于 Replay v1 的 `-10.455` 漂移。结果目录
  `outputs/state_attention_ilosac_flow_v2_20261118_1/`；`summary.json`
  SHA-256
  `62831313b3eaf5ae4d38adb830ef2358d7ec656b22fa2affde133c538a35a6a7`，
  `comparison.json` SHA-256
  `5dc358495ea3a484213a5cb3f450292f11594a36e420a83ad084d37cdadbdbd0`。
  模型入口 SHA-256
  `3777214a3cb87cece928771cd33d1cd342b8b594cc40ca4b39560b1a01b730a5`，
  配置 SHA-256
  `81481e8f6ba1d6e82aa995374852dfa53ed00899bf8a09bd2e3dec2144da59fe`。
  结果仅为 MuJoCo 仿真，不代表硬件或临床精度。
- 2026-11-18 16:05:48 CST：Frontiers 2022 Algorithm 1 DDPG + DiT-Flow/RCM
  完整流水线 FAIL（严格成功门限）。从
  `Frontiers2022_DDPG_linux_20261118.zip` 接入独立实现，并在共享 Replay
  入口增加可替换 policy/critic/update/reward 组件；保留当前 ILOSAC-inspired
  Flow v2 的 twin-Q、TD clip、demonstration anchor 路径。5/5 Frontiers 单元
  测试和 6/6 Replay 回归测试 PASS。启动命令为
  `bash scripts/run_frontiers2022_ddpg.sh --output-dir outputs/state_attention_frontiers2022_ddpg_20261118_1`，
  配置 `configs/frontiers2022_ddpg_pipeline.json`；训练集
  `lerobot_v12_oct_dit_rf_dagger2_acthist_20261116_224`、验证集
  `lerobot_rcm_val_acthist_20261117_224`、初始 checkpoint
  `dit_rectified_flow_state_attention_v1_acthist_20261117/best_model.pth`、
  seed `20261117`、固定 Flow 噪声 seed `20261118`。回放缓存保留 225,614 个
  连续 OCT-valid transitions；4×1000 步训练完成 3,750 次 actor 更新，best
  epoch 0 val loss `1.103461`，last epoch val loss `16.239620`。best
  checkpoint SHA-256
  `5559f5b3f1b986aceeda6aab76d64694880d931a58f9c8b4e65124ffdc51be4d`。
  20 目标闭环 canonical success `0/20`，最终误差
  mean/median/min/max=`15909.594/13546.894/11030.918/23944.568 µm`，
  每目标最小误差均值 `12334.938 µm`，退出时编码器停稳 `5/20`，
  safety-limited steps `108`。与 `state_attention_replay_v1_20261117_1`
  逐目标配对：0 改善、20 恶化，最终误差均值恶化 `15738.567 µm`。结果目录
  `outputs/state_attention_frontiers2022_ddpg_20261118_1/`；`summary.json`
  SHA-256
  `a49cd89b248d62602165aa3d3fd8f0f2a5588e28191fddf5b9dd7c170cec61b0`，
  `comparison.json` SHA-256
  `63d44dcef7a64fa4b7827e30a8246c0f4cf35984c1e105f21026db8ca11de3ce`。
  结果仅为 MuJoCo 仿真，不代表硬件或临床精度。
- 2026-11-18 18:21:33 CST：专家纠正历史经验回放 A/B/C 接入，A 完整
  FAIL、B 评测中、C 待运行。新增纯 Flow 训练入口
  `lerobot_training/dit_experience_replay.py`、固定 8+8 平衡采样器、
  `run_experience_replay_abc.sh`、契约审计和 A/B/C 汇总器。单元测试合计
  11/11 PASS，覆盖平衡批次、source 拆分、可复现性、reservoir 和旧 Replay
  回归。首次契约审计因窗口掩码初始化为全 False，错误报告
  `valid_windows=0`，判定 FAIL 并保留于
  `runs/20261118/verification/experience_replay_contract/`；修正后用
  `runs/20261118/verification/experience_replay_contract_retry1/` 重跑
  PASS：234,316 帧、839 episodes、219,967 个连续有效动作窗口，其中教师
  纠正 14,489 个；教师近目标 `<30/30–100/100–300 µm` 窗口数为
  `2048/2863/5268`。真实 GPU 1-step balanced smoke 和 2-step 评测入口
  smoke PASS，目录为 `runs/20261118/smoke/experience_replay_balanced/`
  与 `runs/20261118/smoke/experience_replay_eval/`。
- 2026-11-18 18:21:33 CST：Experience Replay A 原始继续训练完整流水线
  FAIL（严格成功门限）。命令
  `bash scripts/run_experience_replay_abc.sh outputs/state_attention_experience_replay_abc_20261118_1`，
  配置 `configs/state_attention_experience_replay_a_original_20261118.json`；
  从 state-attention v1 epoch 3 开始，4×1000 updates、batch 16、backbone
  lr `1e-6`、attention lr `1e-5`。best epoch 3 val loss `0.144550`。
  20 目标 canonical success `0/20`，最终误差
  mean/median/min/max=`1087.778/883.749/112.620/2466.592 µm`，仅 6/20
  最终误差 ≤300 µm，编码器停稳 0/20，safety-limited steps 0。与 v1 配对：
  7 改善、13 恶化，最终误差均值恶化 `653.381 µm`；每目标最小误差均值从
  `93.680` 改善至 `63.988 µm`，说明接近能力局部改善但无法保持。结果目录
  `outputs/state_attention_experience_replay_abc_20261118_1/a_original/`；
  `summary.json` SHA-256
  `9de7f96b0498a5d589261b1ea5c9ddc5697f189316303fa7f60875d521f3548b`，
  `comparison.json` SHA-256
  `dfd62c594dcf4f9e29dc8c5d2d737a45089644dabd2fd6e567c6607804989c4e`。
- 2026-11-18 18:21:33 CST：Experience Replay B/C 为部分状态，不作最终
  结论。B static 训练完成，best epoch 3 val loss `0.145008`，checkpoint
  SHA-256 `edc72afcd36b71fc9e969d0c58ba2aa92c97b317f95b5bda3f957f3af58afa56`；
  上传时 B 处于 10 分片闭环评测中，C balanced 因顺序执行尚未启动。状态
  快照位于 `outputs/state_attention_experience_replay_abc_20261118_1/b_static/status.json`。
  结果仅为 MuJoCo 仿真，不代表硬件或临床精度。
- 2026-11-18 19:24:05 CST：Experience Replay B/C 完成，A/B/C 全量对照
  结论为“新增专家纠正数据有效，但固定 50/50 回放组织无效”。B static
  最终误差 mean/median/P90/max=`207.290/141.257/510.406/677.360 µm`，
  17/20 最终误差 ≤300 µm，与 v1 配对 18 改善、2 恶化，均值改善
  `227.107 µm`；best epoch 3 val loss `0.145008`。C balanced 最终误差
  mean/median/P90/max=`1486.514/1637.206/2661.735/2915.647 µm`，仅
  4/20 最终误差 ≤300 µm，与 v1 配对 4 改善、16 恶化，均值恶化
  `1052.117 µm`；相对 B 仅 3 改善、17 恶化。三组 canonical success 均为
  `0/20`，编码器停稳均为 `0/20`，safety-limited steps 均为 0。B 在
  19/20 目标曾进入 100 µm，但 19/20 均在首次进入后离开，说明仍需补足
  保持和停稳行为。结果目录
  `outputs/state_attention_experience_replay_abc_20261118_1/`；B
  `summary/comparison` SHA-256
  `26f1ad0a070779a4cf1310cc37ec20637797d1d9beeb1155588de19e6231ba9f`/
  `11769abe6340597abf7f491684bf2d17de6b219c88764ea8ac7408e72d3d8e57`；
  C `summary/comparison` SHA-256
  `0d4cc724a077f4bb95b9d18ad68f41863b3d519858c97abcb8d5e1d86bc43a54`/
  `2fcd8de80578da155f4ecce0d6bb5136ea45a7226644e007aba81fa62dab348b`。
  汇总目录 `runs/20261118/analysis/experience_replay_abc/`；`abc_comparison.json`
  SHA-256
  `81ae86f7557801e33a5e0cf535ba85c0b8f5c9404b09c6848e968e92fe56513d`。
  结果仅为 MuJoCo 仿真，不代表硬件或临床精度。

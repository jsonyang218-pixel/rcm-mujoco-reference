# MuJoCo建模与RCM运动学模型（独立7页）

沿用用户原模板；原模板、原π0PPT与main均未修改。约6分20秒，含可编辑的RCM运动学图、正运动学、RCM约束、名义编码器映射与逐页讲稿。

- [下载PPTX](mujoco_rcm_kinematic_model_20261023.pptx?raw=true)
- [查看PDF](mujoco_rcm_kinematic_model_20261023.pdf)
- [下载完整资料包](mujoco_modeling_slides_20261023.zip?raw=true)：包括PPT、PDF、讲稿、独立模型图、图片与源码证据、逐页审查、字体。
- [运动学图PNG](rcm_kinematic_model.png) / [运动学图PDF](rcm_kinematic_model.pdf)

截图是固定零位的说明用MuJoCo静态重渲染。三姿态核对仅验证解析FK与MuJoCo site的理想运动学一致性，不是新增导航实验或实机精度结果。当前4mm针尖偏置与安装外参未实测，外观杆件为示意，无actuator或动力学积分。

资料包中的README包含完整使用说明与scientific-slides、Supervisor-Skills规范来源。结构/数值证据、逻辑与视觉检查见本目录evidence。若PPT字体替换，请安装资料包fonts内Noto Sans SC，或直接使用PDF。

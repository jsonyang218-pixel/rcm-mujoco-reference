# MuJoCo建模与RCM运动学模型：逐页讲稿
独立7页建模PPT；计划约6分20秒，可按组会需要选页。

## 1. MuJoCo建模与RCM运动学模型（15秒）
这一份单独说明MuJoCo中怎样建立RCM机器人模型，重点是关节树、运动学模型、针尖坐标变换与RCM约束。它是独立文件，不修改之前的π0与导航汇报。

来源：rcm_reference.py；rcm_recovered_model.py；config.json；外部模型图片为说明用静态重渲染，不是新导航实验。

## 2. MuJoCo场景与建模层级（45秒）
场景由眼球椭球、目标、相机、针具和固定RCM点构成。图片是当前模型的实际外部渲染，不是绘制出来的机构效果图。运动层有三个独立任务关节，并带四个从属显示关节，所以MuJoCo的nq与nv等于七，但控制任务自由度仍是三。当前通过设置qpos并调用mj_forward计算运动学，没有actuator，nu等于零。外观等效杆件并未按实机逐项标定，不能以截图推断真实机构尺寸或动力学。

来源：rcm_reference.py:446–760；config.json:illustrative_linkage，measured=false；visual_scale=0.67；model_assets/中的渲染说明；无新导航运行。

## 3. 主运动链：两个转动关节与一个移动关节（55秒）
真实主链从固定安装体依次进入A_stage、B_RCM_task_frame、C_carriage_and_needle和针尖site。A是绕自身局部负Y轴的hinge，B是绕局部正X轴的hinge，C沿局部负Z轴平移。因此不能直接套用常见的绕世界Z和Y旋转的公式。A、B是放在RCM处的等效任务旋转轴，展开框图只表示父子关系，不表示两轴在空间分离，也不是声称真实电机轴都安装在RCM。四个可视B关节由任务B正负赋值，不是额外控制量。

来源：rcm_reference.py:21、239–242、647–692；scene.xml相应body/joint及equality条目。

## 4. RCM运动学模型：两轴定向，沿针轴插入（75秒）
这张图是运动学模型的核心。安装系M的原点设在固定RCM。蓝色A轴沿负Y_M，橙色B轴随A旋转，沿转后的正X_A；绿色C方向是针轴局部负Z_B。A和B改变方向，C只沿当前针轴推进，所以无论姿态如何变化，针轴都应穿过固定RCM。虚线表示参考姿态，实线表示示意姿态。RCM到针尖的长度是参考偏置L零加插入增量c。图由PowerPoint原生线、箭头、圆点和文字构成，可继续编辑；只表达模型关系，不按实机尺寸绘制。

来源：rcm_reference.py:647–692；关节局部轴0 −1 0 / 1 0 0 / 0 0 −1；图示投影参数记录在evidence/kinematic_diagram.json。

## 5. 正运动学：从关节坐标到针尖位置（75秒）
正运动学严格按MuJoCo主链写。先平移到世界系中的RCM，再施加当前配置的45度X安装旋转，随后绕负Y进行方位转动、绕X进行俯仰，最后沿局部负Z平移到针尖。内部任务坐标是弧度、弧度、米，界面显示是度、度、毫米，两者需转换。方位角phi含零位phi零，俯仰beta含beta零，长度L含L零。当前phi零为0度，beta零约负13.764度，L零为4毫米，因此ABC显示为零并不代表针尖在RCM。这个4毫米是暂定模型偏置，不能作为已经实测的几何标定。

来源：rcm_reference.py:67–104、216–222、282–286、647–692；config.json:installation、encoder_reference；源机器人系换基R_mount diag(1,−1,−1)与此安装系写法等价，不是额外物理安装。

## 6. RCM约束与模型数值一致性检验（60秒）
RCM约束可以写为固定点到当前针轴直线的距离。d是单位针轴方向，I减dd转置把向量投影到针轴的垂直平面，所以约束为零意味着针轴经过固定RCM。我们做了最小的只读数值检验：检查三个任务变量和四个显示从属关节，再在三个确定姿态比较解析正运动学和MuJoCo针尖site，并计算轴到RCM的距离。差异在浮点精度量级，说明公式与编译模型一致。这是程序和理想模型的一致性检查，不能解释为实机达到这种数值精度。

来源：evidence/numeric_checks.json；rcm_reference.py:state_from_task、set_task、metrics；rcm_recovered_model.py:needle_axis、forward

## 7. 编码器映射与当前模型边界（55秒）
最后区分任务运动学、名义编码器映射和实机标定。编码器通过零位、符号和比例换算为执行器量，B执行器还有非线性位移到角度的映射。这里给出的公式来自原程序恢复的名义参数，并非实机闭链已经测量完成；B直线位移s_B也不是任务角增量b。α是原程序俯仰角，β=α−68°；因为映射跨极小值后并非一一对应，逆解需约束分支并参考前一姿态。当前明确的是等效RCM运动学、场景和几何观测接口。外观杆件、针尖绝对偏置、相机外参和真实机构映射仍需实测。模型没有actuator或动力学积分，惯量为占位、重力和接触关闭，因此不能用于宣称真实受力、制动或组织穿刺效果。

来源：rcm_recovered_model.py:b_displacement、EncoderMap；config.json:kinematics、encoders、encoder_reference、illustrative_linkage；rcm_reference.py:set_task及build_xml；当前动力学与接触范围说明见model_evidence.md。

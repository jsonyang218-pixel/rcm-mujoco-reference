<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- primary_language: zh-CN
- audience: 课题组技术评审教师与同学
- objective: 先报告开发集成效与最终测试失败，再解释闭环和学习机制并推动下一阶段协议与终端控制决策。
- core_message: 20 µm 闭环控制已在开发集达成，但外围目标外推尚未成立；先修测试协议与终端保持，再增加模型复杂度。
- consumption_mode: presentation

## mode
- mode: custom
- mode_behavior: Conclusion-first engineering review: open with the global verdict, explain each mechanism as a claim-evidence pair, and close by converting the failure evidence into ordered protocol and controller actions.

## visual_style
- visual_style: custom
- visual_style_behavior: Sharp scientific-review geometry on a white field with thin grids, pale blue-gray measurement surfaces, deep-blue structural bands, restrained 16 px corners, generous whitespace, and one teal focal accent per page; no shadows, decoration gradients, 3D, or dense card walls.

## colors
- background: #FFFFFF
- secondary_bg: #EAF2F7
- primary: #0B4F8A
- accent: #14988F
- secondary_accent: #78A83D
- body_text: #172633
- secondary_text: #526571
- divider: #CBD9E2
- warning: #D97706
- negative: #B42318

## typography
- font_family: Microsoft YaHei, Arial
- title_family: Microsoft YaHei, Arial
- body_family: Microsoft YaHei, Arial
- data_family: Microsoft YaHei, Consolas
- formula_family: Microsoft YaHei, Cambria Math
- body: 28
- title: 44
- subtitle: 32
- annotation: 22
- footer: 20
- hero_number: 96
- section_label: 24

## icons
- library: tabler-outline
- stroke_width: 2
- inventory: icons/tabler-outline/eye.svg, icons/tabler-outline/target.svg, icons/tabler-outline/route.svg, icons/tabler-outline/cpu.svg, icons/tabler-outline/database.svg, icons/tabler-outline/git-branch.svg, icons/tabler-outline/chart-bar.svg, icons/tabler-outline/chart-line.svg, icons/tabler-outline/alert-triangle.svg, icons/tabler-outline/circle-check.svg, icons/tabler-outline/clock.svg, icons/tabler-outline/git-compare.svg

## page_rhythm
- P01: anchor
- P02: dense
- P03: dense
- P04: dense
- P05: anchor
- P06: dense
- P07: dense
- P08: breathing
- P09: breathing
- P10: dense
- P11: anchor

## page_visualizations
- P03: table/comparison_matrix
- P08: chart/horizontal_bar_chart
- P09: chart/horizontal_bar_chart
- P10: chart/horizontal_bar_chart

## pptx_structure
- mode: flat

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters
- 不使用阴影、装饰渐变或 3D (user)
- 不添加未提供的校徽、机构标识或虚构徽标 (user)
- 页面不出现源码文件名、目录名、实验输出路径和脚注式小段解释 (user)
- 不声称学生优于教师 (user)
- 不把 0.384 ms 冒充完整 RGB-to-action 延迟 (user)

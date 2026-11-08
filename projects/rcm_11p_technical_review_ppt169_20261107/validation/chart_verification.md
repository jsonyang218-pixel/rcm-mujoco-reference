# Chart Verification

## P08 development-success-bars

- Type: horizontal bar
- Scale: 0–40 successes from visible axis ticks
- Plot area: 280,210–880,460
- PowerPoint category slots: `gap_width=50`
- Calculated values:
  - 几何教师: `x=280`, `y=218.3`, `width=120`, `height=33.3`
  - DAgger-only: `x=280`, `y=268.3`, `width=600`, `height=33.3`
  - Velocity: `x=280`, `y=318.3`, `width=510`, `height=33.3`
  - Residual cap: `x=280`, `y=368.3`, `width=570`, `height=33.3`
  - Stop head: `x=280`, `y=418.3`, `width=570`, `height=33.3`
- Result: SVG marks updated to the calculated slot geometry.

## P09 final-success-bars

- Type: horizontal bar
- Scale: 0–35 percent from visible axis ticks
- Plot area: 220,232–560,430
- PowerPoint category slots: `gap_width=50`
- Calculated values:
  - 蒸馏 20/180: `x=220`, `y=248.5`, `width=107.8`, `height=66`
  - 七样本 18/60: `x=220`, `y=347.5`, `width=291.4`, `height=66`
- Result: SVG marks updated; displayed percentages are 11.1% and 30.0%.

## P10 failure-mode-bars

- Type: horizontal bar
- Scale: 0–60 episodes from visible axis ticks
- Plot area: 250,222–690,552
- PowerPoint category slots: `gap_width=45`
- Calculated values:
  - 成功: `x=250`, `y=229.3`, `width=146.7`, `height=32.5`
  - 到达未保持: `x=250`, `y=276.5`, `width=374`, `height=32.5`
  - 20–30 µm: `x=250`, `y=323.6`, `width=58.7`, `height=32.5`
  - 30–50 µm: `x=250`, `y=370.7`, `width=300.7`, `height=32.5`
  - 50–100 µm: `x=250`, `y=417.9`, `width=330`, `height=32.5`
  - 未接近: `x=250`, `y=465`, `width=44`, `height=32.5`
  - 初始化不安全: `x=250`, `y=512.2`, `width=66`, `height=32.5`
- Result: SVG marks updated to the calculated slot geometry; episode counts sum to 180.

## Final State

All three charts carry object-scoped plot-area markers and fresh SVG-first fallback hashes. The final SVG quality report records 0 blocking errors.

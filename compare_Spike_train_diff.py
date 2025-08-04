import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 读取 CSV 数据
baseline_csv = 'EVDet200K_baseline_last_layer_binary_combinations.csv'
tde_csv = 'EVDet200K_TDE_last_layer_binary_combinations.csv'

baseline_df = pd.read_csv(baseline_csv)
tde_df = pd.read_csv(tde_csv)

# 提取数据并补足4位
combinations = baseline_df['Binary Combination'].astype(str).str.zfill(4)
freq_baseline = baseline_df['Frequency'].values
freq_tde = tde_df['Frequency'].values

# 排除 '0000' 和 '1111'
exclude_patterns = ['0000', '1111']
mask = ~combinations.isin(exclude_patterns)
combinations_filtered = combinations[mask]
freq_baseline_filtered = freq_baseline[mask]
freq_tde_filtered = freq_tde[mask]

# 计算百分比变化 (TDE相对Baseline)
percent_change_raw = np.array([
    ((tde - base) / base) * 100 if base > 0 else 0.0
    for base, tde in zip(freq_baseline_filtered, freq_tde_filtered)
])
percent_change_abs = np.abs(percent_change_raw)

# y 轴位置
y_pos = np.arange(len(combinations_filtered))

plt.figure(figsize=(15, 14))  # 增加宽度

# 固定颜色
increase_color = '#F8C8C0'  # 橙红色 
decrease_color = '#9AC1EF'  # 蓝青色
colors = [increase_color if raw > 0 else decrease_color for raw in percent_change_raw]

# 绘制柱状图
bars = plt.barh(y_pos, percent_change_abs, color=colors, alpha=0.9, height=0.6,
                edgecolor='black', linewidth=1.5)

# 自动设置 x 轴范围，留出标签空间
plt.xlim(0, max(percent_change_abs) * 1.2)

# 数据标签：+xx.x% 或 -xx.x%
for i, (pct_raw, pct_abs) in enumerate(zip(percent_change_raw, percent_change_abs)):
    plt.text(pct_abs + max(percent_change_abs) * 0.02, i, f'{pct_raw:+.1f}%',
             va='center', ha='left',
             fontsize=25, weight='bold', color=colors[i])

# y 轴标签
plt.yticks(y_pos, combinations_filtered, fontsize=20, weight='bold')

# 坐标轴标签和标题
plt.xlabel('Change Percentage (%)', fontsize=20, weight='bold', labelpad=10)
plt.ylabel('Spike Stream', fontsize=20, weight='bold', labelpad=10)
plt.title('Last LIF Layer Fire Pattern Change Percentage',
          fontsize=20, weight='bold', pad=20)

# 图例
legend_elements = [
    Patch(facecolor=increase_color, edgecolor='black', 
          label='Increase: (w TDE - w/o TDE) / w/o TDE > 0'),
    Patch(facecolor=decrease_color, edgecolor='black', 
          label='Decrease: (w TDE - w/o TDE) / w/o TDE < 0')
]
plt.legend(handles=legend_elements, loc='upper right',
           bbox_to_anchor=(1, 0.55), fontsize=20,
           frameon=True, edgecolor='black')

# 美化
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig('Last_LIF_Layer.png', dpi=300)
plt.show()

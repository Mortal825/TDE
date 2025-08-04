import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 数据
baseline_csv = 'EVDet200K_baseline_first_layer_binary_combinations.csv'
tde_csv = 'EVDet200K_TDE_first_layer_binary_combinations.csv'

baseline_df = pd.read_csv(baseline_csv)
tde_df = pd.read_csv(tde_csv)

# 提取数据并转换为百分比
combinations = baseline_df['Binary Combination'].astype(str).str.zfill(4)
freq_baseline = baseline_df['Frequency'].values
freq_tde = tde_df['Frequency'].values

freq_baseline = (freq_baseline / freq_baseline.sum()) * 100
freq_tde = (freq_tde / freq_tde.sum()) * 100

y_pos = np.arange(len(combinations))
bar_height = 0.6

plt.figure(figsize=(11, 8))  # 增加图的宽度

# 绘制两组柱状图
plt.barh(y_pos, freq_baseline, height=bar_height,
         color = '#9AC1EF', alpha=0.6, label='w/o TDE',
          linewidth=1.5)

plt.barh(y_pos, freq_tde, height=bar_height * 0.8,
         color='#F8C8C0', alpha=0.8, label='w TDE',
         linewidth=1.5)

# 获取最大值并扩展 x 轴范围
max_val = max(freq_baseline.max(), freq_tde.max())
plt.xlim(0, max_val * 1.1)  # 留出15%的空间显示标签

# 显示百分比值（自动调整防止重叠）
for i, (fb, ft) in enumerate(zip(freq_baseline, freq_tde)):
    # 初始偏移
    offset_fb = max_val * 0.01
    offset_ft = max_val * 0.01

    if fb > 0 and ft > 0:
        diff = abs(fb - ft)
        # 根据差值大小动态计算偏移，差值小 -> 偏移大
        # 当差值为0时偏移达到 max_val * 0.08，当差值大于 max_val * 0.05 时不额外增加
        if diff < max_val * 0.08:
            dynamic_offset = max_val * (0.10 - (diff / (max_val * 0.05)) * 0.07)
            if fb >= ft:
                offset_fb = dynamic_offset
            else:
                offset_ft = dynamic_offset

    # 绘制文本
    if fb > 0:
        plt.text(fb + offset_fb, i, f'{fb:.1f}%', va='center',
                 fontsize=16, color="#4998F3", weight='bold')
    if ft > 0:
        plt.text(ft + offset_ft, i, f'{ft:.1f}%', va='center',
                 fontsize=16, color="#FC674C", weight='bold')


plt.yticks(y_pos, combinations, fontsize=16, weight='bold')
plt.xlabel('Percentage (%)', fontsize=12, weight='bold')
plt.ylabel('Spike Stream', fontsize=16, weight='bold', labelpad=15)
plt.title('First LIF Layer Fire Pattern Comparison: w/o TDE vs TDE',
          fontsize=16, weight='bold')

plt.legend(loc='center right', fontsize=16, frameon=True, edgecolor='black')
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig('First_LIF_layer.png', dpi=300)
plt.show()

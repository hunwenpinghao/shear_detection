"""
按卷分析：可视化每一卷钢卷的磨损状态
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11

# 读取数据
df = pd.read_csv('results/features/wear_features.csv')

print("="*80)
print("按卷分析：将2249帧分割为9个钢卷（第4-12卷）")
print("="*80)

# 方法1：均匀分割（假设每卷帧数大致相等）
total_frames = len(df)
n_coils = 9  # 第4-12卷，共9个卷

# 简单均分
coil_size = total_frames // n_coils
print(f"总帧数: {total_frames}, 每卷约 {coil_size} 帧")

# 分配卷
df['coil_id'] = df.index // coil_size + 4  # 从第4卷开始
df.loc[df['coil_id'] > 12, 'coil_id'] = 12  # 最后的归到第12卷

print("\n每卷帧数分布:")
coil_counts = df['coil_id'].value_counts().sort_index()
for coil_id, count in coil_counts.items():
    print(f"  第{int(coil_id)}卷: {count}帧")

# 核心特征
key_features = {
    'avg_rms_roughness': 'RMS粗糙度',
    'max_notch_depth': '最大缺口深度',
    'right_peak_density': '右侧峰密度（剪切面）',
    'avg_gradient_energy': '梯度能量（锐度）',
    'tear_shear_area_ratio': '撕裂/剪切面积比'
}

# ==================== 可视化1：所有卷的箱线图对比 ====================
print("\n生成可视化1: 所有卷的箱线图对比...")

fig, axes = plt.subplots(3, 2, figsize=(20, 16))
axes = axes.flatten()

for idx, (feature, label) in enumerate(list(key_features.items())[:5]):
    ax = axes[idx]
    
    # 准备每个卷的数据
    coil_data = []
    coil_labels = []
    
    for coil_id in sorted(df['coil_id'].unique()):
        coil_df = df[df['coil_id'] == coil_id]
        coil_data.append(coil_df[feature].values)
        coil_labels.append(f'卷{int(coil_id)}')
    
    # 绘制箱线图
    bp = ax.boxplot(coil_data, labels=coil_labels, patch_artist=True,
                     widths=0.6,
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     medianprops=dict(linewidth=2, color='red'))
    
    # 用渐变色显示卷的顺序（浅→深代表磨损递增）
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(coil_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 计算每卷的均值并连线
    means = [np.mean(data) for data in coil_data]
    ax.plot(range(1, len(means)+1), means, 'bo-', linewidth=3, 
           markersize=8, label='均值趋势', zorder=10)
    
    # 拟合趋势线
    x = np.arange(len(means))
    z = np.polyfit(x, means, 1)
    trend = np.poly1d(z)
    ax.plot(range(1, len(means)+1), trend(x), 'g--', linewidth=3, 
           label=f'线性趋势(斜率={z[0]:.4f})', alpha=0.8)
    
    # 计算首尾变化
    first_mean = means[0]
    last_mean = means[-1]
    change_pct = ((last_mean - first_mean) / first_mean) * 100
    
    # 显示趋势判断
    if change_pct > 5:
        trend_text = f'✓ 显著递增 +{change_pct:.1f}%'
        box_color = 'lightgreen'
    elif change_pct > 0:
        trend_text = f'轻微递增 +{change_pct:.1f}%'
        box_color = 'lightyellow'
    elif change_pct > -5:
        trend_text = f'基本平稳 {change_pct:.1f}%'
        box_color = 'lightgray'
    else:
        trend_text = f'递减 {change_pct:.1f}%'
        box_color = 'lightcoral'
    
    ax.text(0.5, 0.98, trend_text,
           transform=ax.transAxes, fontsize=14, fontweight='bold',
           ha='center', va='top',
           bbox=dict(boxstyle='round,pad=1', facecolor=box_color, 
                    alpha=0.8, edgecolor='black', linewidth=2))
    
    ax.set_xlabel('钢卷编号', fontweight='bold', fontsize=13)
    ax.set_ylabel(label, fontweight='bold', fontsize=13)
    ax.set_title(f'{label}\n按卷演变趋势', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

# 移除多余的子图
axes[-1].axis('off')

plt.suptitle('剪刀磨损按卷分析 - 第4卷至第12卷演变', 
            fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/visualizations/coil_by_coil_boxplot.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/coil_by_coil_boxplot.png")

# ==================== 可视化2：每卷的统计指标柱状图 ====================
print("\n生成可视化2: 每卷统计指标柱状图...")

fig, axes = plt.subplots(3, 2, figsize=(20, 16))
axes = axes.flatten()

for idx, (feature, label) in enumerate(list(key_features.items())[:5]):
    ax = axes[idx]
    
    coil_ids = []
    coil_means = []
    coil_maxes = []
    coil_stds = []
    
    for coil_id in sorted(df['coil_id'].unique()):
        coil_df = df[df['coil_id'] == coil_id]
        coil_ids.append(int(coil_id))
        coil_means.append(coil_df[feature].mean())
        coil_maxes.append(coil_df[feature].max())
        coil_stds.append(coil_df[feature].std())
    
    x = np.arange(len(coil_ids))
    width = 0.35
    
    # 绘制均值和最大值
    bars1 = ax.bar(x - width/2, coil_means, width, label='均值',
                   color='steelblue', edgecolor='navy', linewidth=2, alpha=0.8)
    bars2 = ax.bar(x + width/2, coil_maxes, width, label='最大值',
                   color='coral', edgecolor='darkred', linewidth=2, alpha=0.8)
    
    # 添加数值标签
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2, height1,
               f'{height1:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(bar2.get_x() + bar2.get_width()/2, height2,
               f'{height2:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 添加趋势线
    ax.plot(x, coil_means, 'b--', linewidth=2, alpha=0.6)
    ax.plot(x, coil_maxes, 'r--', linewidth=2, alpha=0.6)
    
    # 计算趋势
    z_mean = np.polyfit(x, coil_means, 1)
    change_pct = ((coil_means[-1] - coil_means[0]) / coil_means[0]) * 100
    
    trend_text = f'均值变化: {change_pct:+.1f}%\n斜率: {z_mean[0]:.4f}'
    box_color = 'lightgreen' if change_pct > 0 else 'lightcoral'
    
    ax.text(0.02, 0.98, trend_text,
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           va='top', bbox=dict(boxstyle='round,pad=0.8', 
                              facecolor=box_color, alpha=0.7))
    
    ax.set_xlabel('钢卷编号', fontweight='bold', fontsize=13)
    ax.set_ylabel(label, fontweight='bold', fontsize=13)
    ax.set_title(f'{label}\n各卷统计对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'卷{cid}' for cid in coil_ids])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

axes[-1].axis('off')

plt.suptitle('剪刀磨损按卷统计分析 - 均值与最大值对比', 
            fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/visualizations/coil_by_coil_bars.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/coil_by_coil_bars.png")

# ==================== 可视化3：热力图展示 ====================
print("\n生成可视化3: 特征×卷 热力图...")

fig, ax = plt.subplots(figsize=(14, 8))

# 构建矩阵：特征 × 卷
feature_names = list(key_features.values())
matrix_data = []

for feature in key_features.keys():
    row = []
    for coil_id in sorted(df['coil_id'].unique()):
        coil_df = df[df['coil_id'] == coil_id]
        row.append(coil_df[feature].mean())
    matrix_data.append(row)

matrix = np.array(matrix_data)

# 按行归一化（每个特征独立归一化）
matrix_norm = np.zeros_like(matrix)
for i in range(matrix.shape[0]):
    row = matrix[i, :]
    matrix_norm[i, :] = (row - row.min()) / (row.max() - row.min() + 1e-8)

# 绘制热力图
im = ax.imshow(matrix_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# 设置刻度
coil_ids = sorted(df['coil_id'].unique())
ax.set_xticks(np.arange(len(coil_ids)))
ax.set_yticks(np.arange(len(feature_names)))
ax.set_xticklabels([f'第{int(cid)}卷' for cid in coil_ids], fontsize=12)
ax.set_yticklabels(feature_names, fontsize=12)

# 添加数值
for i in range(len(feature_names)):
    for j in range(len(coil_ids)):
        text = ax.text(j, i, f'{matrix_norm[i, j]:.2f}',
                      ha="center", va="center", color="black", 
                      fontsize=10, fontweight='bold')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('归一化特征值 (0=最小, 1=最大)', fontsize=12, fontweight='bold')

ax.set_title('各卷磨损特征热力图\n（颜色越红=该特征在该卷的值越大）', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('钢卷编号', fontsize=13, fontweight='bold')
ax.set_ylabel('磨损特征', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('results/visualizations/coil_heatmap.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/coil_heatmap.png")

# ==================== 可视化4：雷达图对比（首、中、尾卷）====================
print("\n生成可视化4: 雷达图对比（第4、8、12卷）...")

from math import pi

# 选择三个代表性的卷
representative_coils = [4, 8, 12]
coil_labels = ['第4卷(开始)', '第8卷(中期)', '第12卷(结束)']
colors = ['blue', 'orange', 'red']

fig, axes = plt.subplots(1, 3, figsize=(20, 7), subplot_kw=dict(projection='polar'))

for plot_idx, (coil_id, coil_label, color) in enumerate(zip(representative_coils, coil_labels, colors)):
    ax = axes[plot_idx]
    
    coil_df = df[df['coil_id'] == coil_id]
    
    if len(coil_df) == 0:
        ax.text(0.5, 0.5, f'{coil_label}\n无数据', 
               transform=ax.transAxes, ha='center', va='center')
        continue
    
    # 计算特征值
    categories = list(key_features.values())
    values = []
    
    for feature in key_features.keys():
        values.append(coil_df[feature].mean())
    
    # 归一化（使用全局最大最小值）
    global_max = []
    global_min = []
    for feature in key_features.keys():
        global_max.append(df[feature].max())
        global_min.append(df[feature].min())
    
    values_norm = [(v - vmin) / (vmax - vmin + 1e-8) 
                   for v, vmin, vmax in zip(values, global_min, global_max)]
    
    # 雷达图需要闭合
    values_norm += values_norm[:1]
    
    # 角度
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    # 绘制
    ax.plot(angles, values_norm, 'o-', linewidth=3, color=color, 
           label=coil_label, markersize=8)
    ax.fill(angles, values_norm, alpha=0.25, color=color)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax.set_title(coil_label, fontsize=15, fontweight='bold', pad=20)

plt.suptitle('雷达图对比：开始、中期、结束卷的磨损特征', 
            fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/visualizations/coil_radar_comparison.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/coil_radar_comparison.png")

# ==================== 可视化5：逐卷递进趋势图 ====================
print("\n生成可视化5: 逐卷递进趋势图（重点推荐！）...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 1, hspace=0.3)

# 重点关注的三个指标
focus_features = {
    'right_peak_density': '右侧峰密度（剪切面微缺口）',
    'avg_gradient_energy': '梯度能量（刀口锐度）',
    'max_notch_depth': '最大缺口深度'
}

for idx, (feature, label) in enumerate(focus_features.items()):
    ax = fig.add_subplot(gs[idx])
    
    coil_ids = []
    coil_means = []
    coil_maxes = []
    coil_mins = []
    coil_q25 = []
    coil_q75 = []
    
    for coil_id in sorted(df['coil_id'].unique()):
        coil_df = df[df['coil_id'] == coil_id]
        values = coil_df[feature].values
        
        coil_ids.append(int(coil_id))
        coil_means.append(np.mean(values))
        coil_maxes.append(np.max(values))
        coil_mins.append(np.min(values))
        coil_q25.append(np.percentile(values, 25))
        coil_q75.append(np.percentile(values, 75))
    
    coil_ids = np.array(coil_ids)
    coil_means = np.array(coil_means)
    coil_maxes = np.array(coil_maxes)
    coil_mins = np.array(coil_mins)
    coil_q25 = np.array(coil_q25)
    coil_q75 = np.array(coil_q75)
    
    # 绘制范围区域
    ax.fill_between(coil_ids, coil_mins, coil_maxes, 
                    alpha=0.2, color='gray', label='最小-最大范围')
    ax.fill_between(coil_ids, coil_q25, coil_q75, 
                    alpha=0.3, color='lightblue', label='25%-75%分位数')
    
    # 绘制均值线（加粗）
    ax.plot(coil_ids, coil_means, 'o-', linewidth=4, markersize=12,
           color='darkblue', label='均值', markeredgewidth=2, 
           markeredgecolor='white', zorder=10)
    
    # 绘制最大值线
    ax.plot(coil_ids, coil_maxes, 's-', linewidth=3, markersize=10,
           color='darkred', label='最大值', alpha=0.7, zorder=9)
    
    # 拟合均值趋势
    z = np.polyfit(coil_ids, coil_means, 1)
    trend = np.poly1d(z)
    ax.plot(coil_ids, trend(coil_ids), '--', linewidth=3, 
           color='green', label=f'均值趋势线', alpha=0.8)
    
    # 计算变化
    change_pct = ((coil_means[-1] - coil_means[0]) / coil_means[0]) * 100
    
    # 判断趋势（针对不同特征的物理意义）
    if feature == 'avg_gradient_energy':
        # 梯度能量下降代表磨损
        is_wear_increasing = (change_pct < 0)
        trend_desc = f'锐度下降{abs(change_pct):.1f}% → 磨损加重' if change_pct < 0 else f'锐度上升{change_pct:.1f}%'
    else:
        # 其他特征上升代表磨损
        is_wear_increasing = (change_pct > 0)
        trend_desc = f'递增{change_pct:.1f}% → 磨损加重' if change_pct > 0 else f'递减{abs(change_pct):.1f}%'
    
    # 显示结论
    if is_wear_increasing:
        conclusion_text = f'✓ {trend_desc}'
        box_color = 'lightgreen'
    else:
        conclusion_text = f'{trend_desc}'
        box_color = 'lightyellow'
    
    ax.text(0.98, 0.98, conclusion_text,
           transform=ax.transAxes, fontsize=14, fontweight='bold',
           ha='right', va='top',
           bbox=dict(boxstyle='round,pad=1', facecolor=box_color, 
                    alpha=0.8, edgecolor='black', linewidth=2))
    
    # 添加第一卷和最后一卷的标注
    ax.annotate(f'起始\n{coil_means[0]:.2f}', 
               xy=(coil_ids[0], coil_means[0]),
               xytext=(coil_ids[0]-0.5, coil_means[0]*1.1),
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.annotate(f'结束\n{coil_means[-1]:.2f}', 
               xy=(coil_ids[-1], coil_means[-1]),
               xytext=(coil_ids[-1]+0.5, coil_means[-1]*1.1),
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.set_xlabel('钢卷编号', fontweight='bold', fontsize=13)
    ax.set_ylabel(label, fontweight='bold', fontsize=13)
    ax.set_title(f'{label} - 逐卷演变', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(coil_ids)

plt.suptitle('剪刀磨损逐卷演变分析 - 关键指标', 
            fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/visualizations/coil_progression_detailed.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/coil_progression_detailed.png")

# ==================== 生成结论报告 ====================
print("\n" + "="*80)
print("按卷分析结论")
print("="*80)

for feature, label in focus_features.items():
    coil_means = []
    for coil_id in sorted(df['coil_id'].unique()):
        coil_df = df[df['coil_id'] == coil_id]
        coil_means.append(coil_df[feature].mean())
    
    change_pct = ((coil_means[-1] - coil_means[0]) / coil_means[0]) * 100
    
    print(f"\n【{label}】")
    print(f"  第4卷均值: {coil_means[0]:.4f}")
    print(f"  第12卷均值: {coil_means[-1]:.4f}")
    print(f"  变化率: {change_pct:+.1f}%")
    
    # 统计递增的卷数
    increases = sum(1 for i in range(len(coil_means)-1) 
                   if coil_means[i+1] > coil_means[i])
    total = len(coil_means) - 1
    
    print(f"  逐卷递增次数: {increases}/{total} = {increases/total*100:.0f}%")
    
    if feature == 'avg_gradient_energy':
        if change_pct < 0:
            print(f"  ✓ 锐度下降 → 刀口磨钝，符合磨损预期")
    else:
        if change_pct > 0:
            print(f"  ✓ 数值递增 → 磨损加重，符合预期")

print("\n" + "="*80)
print("分析完成！")
print("="*80)
print("\n推荐查看:")
print("  - coil_progression_detailed.png (最推荐！)")
print("  - coil_by_coil_boxplot.png")
print("  - coil_heatmap.png")


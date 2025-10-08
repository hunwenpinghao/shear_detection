"""
增强可视化：清晰展示剪刀磨损递增趋势
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks, argrelextrema
from scipy.ndimage import uniform_filter1d

# 设置中文字体和更大的字号
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

# 读取数据
df = pd.read_csv('results/features/wear_features.csv')

# 核心磨损指标
wear_features = {
    'avg_rms_roughness': '平均RMS粗糙度',
    'max_notch_depth': '最大缺口深度',
    'left_peak_density': '左侧峰密度'
}

# ==================== 方法1：峰值连线图 ====================
print("生成方法1：峰值连线图（只显示局部最大值的趋势）")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (feature, label) in enumerate(wear_features.items()):
    ax = axes[idx]
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 找到所有局部最大值（窗口大小=50）
    peaks, _ = find_peaks(values, distance=50, prominence=0.5)
    
    # 绘制原始数据（浅色背景）
    ax.plot(frames, values, '-', alpha=0.15, color='gray', linewidth=0.5)
    
    # 绘制峰值点和连线
    peak_frames = frames[peaks]
    peak_values = values[peaks]
    
    ax.plot(peak_frames, peak_values, 'ro-', linewidth=3, markersize=8, 
           label='局部峰值', alpha=0.8)
    
    # 拟合峰值趋势线
    if len(peak_values) > 2:
        z = np.polyfit(peak_frames, peak_values, 1)
        trend = np.poly1d(z)
        ax.plot(peak_frames, trend(peak_frames), 'b--', linewidth=3, 
               label=f'峰值趋势线(斜率={z[0]:.6f})')
        
        # 判断并高亮显示
        if z[0] > 0:
            ax.set_facecolor('#e8f5e9')  # 浅绿色背景
            trend_text = '✓ 峰值递增趋势'
            box_color = 'lightgreen'
        else:
            ax.set_facecolor('#ffebee')  # 浅红色背景
            trend_text = '✗ 峰值递减趋势'
            box_color = 'lightcoral'
        
        ax.text(0.5, 0.95, trend_text, transform=ax.transAxes,
               fontsize=16, fontweight='bold', ha='center', va='top',
               bbox=dict(boxstyle='round,pad=1', facecolor=box_color, alpha=0.8))
    
    ax.set_xlabel('帧编号', fontsize=14, fontweight='bold')
    ax.set_ylabel(label, fontsize=14, fontweight='bold')
    ax.set_title(f'{label}\n峰值点趋势分析', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

plt.suptitle('方法1：局部峰值连线 - 磨损趋势清晰可见', 
            fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/visualizations/peaks_trend.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/peaks_trend.png\n")

# ==================== 方法2：周期起终点对比图 ====================
print("生成方法2：周期起终点对比图（每个周期的开始vs结束）")

def detect_cycles(values, threshold=1.5):
    """检测周期的起点和终点"""
    # 找到所有下降点（可能的周期分界）
    diff = np.diff(values)
    drops = np.where(diff < -threshold)[0]
    
    cycles = []
    start = 0
    for drop in drops:
        if drop - start > 100:  # 周期至少100帧
            cycles.append((start, drop))
            start = drop + 1
    
    if len(values) - start > 100:
        cycles.append((start, len(values)-1))
    
    return cycles

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (feature, label) in enumerate(wear_features.items()):
    ax = axes[idx]
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 检测周期
    cycles = detect_cycles(values)
    
    # 提取每个周期的起点和终点值
    cycle_starts = []
    cycle_ends = []
    cycle_ids = []
    
    for cycle_id, (start, end) in enumerate(cycles[:12]):  # 最多显示12个周期
        cycle_starts.append(values[start])
        cycle_ends.append(values[end])
        cycle_ids.append(cycle_id + 1)
    
    # 绘制起点和终点的对比
    x = np.arange(len(cycle_ids))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cycle_starts, width, label='周期开始', 
                   color='lightblue', edgecolor='blue', linewidth=2)
    bars2 = ax.bar(x + width/2, cycle_ends, width, label='周期结束', 
                   color='lightcoral', edgecolor='red', linewidth=2)
    
    # 添加连接线显示变化
    for i in range(len(x)):
        change = cycle_ends[i] - cycle_starts[i]
        color = 'green' if change > 0 else 'red'
        ax.plot([x[i]-width/2, x[i]+width/2], [cycle_starts[i], cycle_ends[i]], 
               color=color, linewidth=2, alpha=0.5)
    
    # 拟合终点值的趋势
    if len(cycle_ends) > 2:
        z = np.polyfit(x, cycle_ends, 1)
        trend = np.poly1d(z)
        ax.plot(x, trend(x), 'b--', linewidth=3, label=f'终点趋势(斜率={z[0]:.4f})')
        
        # 判断趋势
        increasing_cycles = sum(1 for i in range(len(cycle_ends)) 
                               if cycle_ends[i] > cycle_starts[i])
        ratio = increasing_cycles / len(cycle_ends)
        
        ax.text(0.5, 0.95, 
               f'递增周期: {increasing_cycles}/{len(cycle_ends)} = {ratio:.0%}\n'
               f'终点趋势斜率: {z[0]:.6f}',
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               ha='center', va='top',
               bbox=dict(boxstyle='round,pad=1', 
                        facecolor='lightgreen' if ratio > 0.5 else 'lightcoral', 
                        alpha=0.8))
    
    ax.set_xlabel('周期编号', fontsize=14, fontweight='bold')
    ax.set_ylabel(label, fontsize=14, fontweight='bold')
    ax.set_title(f'{label}\n周期内磨损情况', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'#{i}' for i in cycle_ids])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('方法2：周期起终点对比 - 显示每个周期的净磨损', 
            fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/visualizations/cycle_comparison.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/cycle_comparison.png\n")

# ==================== 方法3：累积磨损指数 ====================
print("生成方法3：累积磨损指数（只累积上升，忽略下降）")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (feature, label) in enumerate(wear_features.items()):
    ax = axes[idx]
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 计算累积磨损（只累积正增长）
    cumulative_wear = np.zeros_like(values)
    cumulative_wear[0] = 0
    
    for i in range(1, len(values)):
        increase = max(0, values[i] - values[i-1])
        cumulative_wear[i] = cumulative_wear[i-1] + increase
    
    # 归一化
    if cumulative_wear[-1] > 0:
        cumulative_wear_norm = cumulative_wear / cumulative_wear[-1]
    else:
        cumulative_wear_norm = cumulative_wear
    
    # 绘制累积磨损
    ax.plot(frames, cumulative_wear_norm, '-', linewidth=3, color='darkred', 
           label='累积磨损指数')
    ax.fill_between(frames, 0, cumulative_wear_norm, alpha=0.3, color='red')
    
    # 拟合趋势
    z = np.polyfit(frames, cumulative_wear_norm, 1)
    trend = np.poly1d(z)
    ax.plot(frames, trend(frames), '--', linewidth=2, color='blue', 
           label=f'线性趋势(斜率={z[0]:.6f})')
    
    # 添加里程碑标记
    milestones = [len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
    for milestone in milestones:
        if milestone < len(cumulative_wear_norm):
            ax.plot(frames[milestone], cumulative_wear_norm[milestone], 
                   'go', markersize=12, markeredgecolor='darkgreen', markeredgewidth=2)
            ax.text(frames[milestone], cumulative_wear_norm[milestone] + 0.05, 
                   f'{cumulative_wear_norm[milestone]:.2f}',
                   ha='center', fontsize=10, fontweight='bold')
    
    ax.text(0.5, 0.95, f'✓ 累积净磨损: {cumulative_wear_norm[-1]:.2f}',
           transform=ax.transAxes, fontsize=16, fontweight='bold',
           ha='center', va='top',
           bbox=dict(boxstyle='round,pad=1', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('帧编号', fontsize=14, fontweight='bold')
    ax.set_ylabel('归一化累积磨损', fontsize=14, fontweight='bold')
    ax.set_title(f'{label}\n累积磨损指数', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

plt.suptitle('方法3：累积磨损指数 - 忽略周期性下降，只看净增长', 
            fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/visualizations/cumulative_wear.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/cumulative_wear.png\n")

# ==================== 方法4：对比图（第一卷 vs 最后一卷）====================
print("生成方法4：首尾对比图（第一卷 vs 最后一卷）")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 定义第一卷和最后一卷的范围
first_coil = df.iloc[:250]  # 前250帧
last_coil = df.iloc[-250:]   # 后250帧

for idx, (feature, label) in enumerate(wear_features.items()):
    ax = axes[idx]
    
    first_values = first_coil[feature].values
    last_values = last_coil[feature].values
    
    # 计算统计量
    first_stats = {
        'mean': np.mean(first_values),
        'max': np.max(first_values),
        'std': np.std(first_values)
    }
    
    last_stats = {
        'mean': np.mean(last_values),
        'max': np.max(last_values),
        'std': np.std(last_values)
    }
    
    # 绘制箱线图对比
    bp = ax.boxplot([first_values, last_values], 
                     labels=['第一卷\n(前250帧)', '最后一卷\n(后250帧)'],
                     widths=0.6, patch_artist=True,
                     boxprops=dict(linewidth=2),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2),
                     medianprops=dict(linewidth=3, color='red'))
    
    # 设置颜色
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    # 添加均值点
    ax.plot([1, 2], [first_stats['mean'], last_stats['mean']], 
           'go-', linewidth=3, markersize=12, label='均值', markeredgewidth=2)
    
    # 添加最大值点
    ax.plot([1, 2], [first_stats['max'], last_stats['max']], 
           'rs-', linewidth=3, markersize=12, label='最大值', markeredgewidth=2)
    
    # 计算变化
    mean_change = ((last_stats['mean'] - first_stats['mean']) / first_stats['mean']) * 100
    max_change = ((last_stats['max'] - first_stats['max']) / first_stats['max']) * 100
    
    # 显示变化
    change_text = (f'均值变化: {mean_change:+.1f}%\n'
                  f'最大值变化: {max_change:+.1f}%')
    
    box_color = 'lightgreen' if mean_change > 0 else 'lightcoral'
    
    ax.text(0.5, 0.95, change_text,
           transform=ax.transAxes, fontsize=14, fontweight='bold',
           ha='center', va='top',
           bbox=dict(boxstyle='round,pad=1', facecolor=box_color, alpha=0.8))
    
    # 添加数值标签
    ax.text(1, first_stats['mean'], f'{first_stats["mean"]:.2f}',
           ha='right', va='bottom', fontsize=10, fontweight='bold')
    ax.text(2, last_stats['mean'], f'{last_stats["mean"]:.2f}',
           ha='left', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel(label, fontsize=14, fontweight='bold')
    ax.set_title(f'{label}\n首尾卷对比', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('方法4：首尾卷对比 - 直观显示整体磨损程度', 
            fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/visualizations/first_last_comparison.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/first_last_comparison.png\n")

print("="*60)
print("所有增强可视化已完成！")
print("="*60)
print("\n生成的图表：")
print("1. peaks_trend.png - 峰值连线图")
print("2. cycle_comparison.png - 周期起终点对比")
print("3. cumulative_wear.png - 累积磨损指数")
print("4. first_last_comparison.png - 首尾卷对比")
print("\n这些图表更直观地展示了剪刀的磨损递增趋势！")


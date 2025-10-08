"""
智能平滑长期趋势：过滤换卷波动，展示清晰的磨损曲线
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage import maximum_filter1d, uniform_filter1d

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 13

# 读取数据
df = pd.read_csv('results/features/wear_features.csv')

# 核心磨损特征
wear_features = {
    'avg_rms_roughness': '平均RMS粗糙度',
    'max_notch_depth': '最大缺口深度',
    'left_peak_density': '左侧峰密度'
}

# ==================== 方法1：移动最大值包络线 ====================
print("方法1：移动最大值包络线（提取波峰趋势）")

fig, axes = plt.subplots(3, 1, figsize=(20, 15))

for idx, (feature, label) in enumerate(wear_features.items()):
    ax = axes[idx]
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 原始数据（浅色）
    ax.plot(frames, values, '-', alpha=0.2, color='gray', linewidth=0.5, label='原始数据')
    
    # 移动最大值（窗口=200）
    window = 200
    max_envelope = maximum_filter1d(values, size=window, mode='nearest')
    ax.plot(frames, max_envelope, '-', color='orange', linewidth=2, 
           alpha=0.7, label=f'移动最大值包络(窗口={window})')
    
    # 对包络线进行平滑（Savitzky-Golay滤波）
    if len(max_envelope) > 51:
        smooth_envelope = savgol_filter(max_envelope, window_length=51, polyorder=3)
        ax.plot(frames, smooth_envelope, '-', color='red', linewidth=3, 
               label='平滑包络线', zorder=10)
        
        # 拟合趋势
        z = np.polyfit(frames, smooth_envelope, 1)
        trend = np.poly1d(z)
        ax.plot(frames, trend(frames), '--', color='blue', linewidth=3, 
               label=f'线性趋势(斜率={z[0]:.6f})', alpha=0.8)
    
    ax.set_xlabel('帧编号', fontweight='bold')
    ax.set_ylabel(label, fontweight='bold')
    ax.set_title(f'{label} - 移动最大值包络线法', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.suptitle('方法1：移动最大值包络线 - 平滑长期趋势', 
            fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/visualizations/smooth_method1_envelope.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/smooth_method1_envelope.png\n")

# ==================== 方法2：周期代表点拟合 ====================
print("方法2：周期代表点拟合（每个周期取最大值）")

def detect_cycles_advanced(values, min_drop=1.5, min_cycle_length=100):
    """高级周期检测"""
    # 计算帧间差分
    diff = np.diff(values)
    
    # 找到大幅下降点（可能的换卷点）
    drops = []
    for i in range(len(diff)):
        if diff[i] < -min_drop:
            drops.append(i)
    
    # 合并接近的下降点
    if len(drops) == 0:
        return [(0, len(values)-1)]
    
    cycles = []
    start = 0
    
    for drop in drops:
        if drop - start > min_cycle_length:
            cycles.append((start, drop))
            start = drop + 1
    
    # 最后一段
    if len(values) - start > min_cycle_length:
        cycles.append((start, len(values)-1))
    
    return cycles

fig, axes = plt.subplots(3, 1, figsize=(20, 15))

for idx, (feature, label) in enumerate(wear_features.items()):
    ax = axes[idx]
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 原始数据
    ax.plot(frames, values, '-', alpha=0.2, color='gray', linewidth=0.5, label='原始数据')
    
    # 检测周期
    cycles = detect_cycles_advanced(values)
    print(f"{label}: 检测到 {len(cycles)} 个周期")
    
    # 提取每个周期的代表点（最大值）
    cycle_frames = []
    cycle_maxes = []
    cycle_ends = []
    
    for start, end in cycles:
        # 周期最大值
        max_idx = start + np.argmax(values[start:end+1])
        cycle_frames.append(frames[max_idx])
        cycle_maxes.append(values[max_idx])
        
        # 周期结束值
        cycle_ends.append(values[end])
    
    cycle_frames = np.array(cycle_frames)
    cycle_maxes = np.array(cycle_maxes)
    
    # 绘制周期代表点
    ax.plot(cycle_frames, cycle_maxes, 'ro', markersize=10, 
           label='周期最大值点', zorder=5)
    
    # 使用样条插值连接代表点（平滑曲线）
    if len(cycle_frames) > 3:
        # 三次样条插值
        spline = UnivariateSpline(cycle_frames, cycle_maxes, k=min(3, len(cycle_frames)-1), s=5)
        smooth_frames = np.linspace(frames[0], frames[-1], 500)
        smooth_values = spline(smooth_frames)
        
        ax.plot(smooth_frames, smooth_values, '-', color='red', linewidth=4, 
               label='样条插值平滑曲线', zorder=10, alpha=0.8)
        
        # 线性趋势
        z = np.polyfit(cycle_frames, cycle_maxes, 1)
        trend = np.poly1d(z)
        ax.plot(cycle_frames, trend(cycle_frames), '--', color='blue', linewidth=3, 
               label=f'线性趋势(斜率={z[0]:.6f})', alpha=0.8)
        
        # 判断趋势
        if z[0] > 1e-6:
            trend_text = f'✓ 长期递增趋势\n斜率: {z[0]:.6f}'
            box_color = 'lightgreen'
        else:
            trend_text = f'长期趋势\n斜率: {z[0]:.6f}'
            box_color = 'lightyellow'
        
        ax.text(0.02, 0.98, trend_text, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top',
               bbox=dict(boxstyle='round,pad=1', facecolor=box_color, alpha=0.8))
    
    ax.set_xlabel('帧编号', fontweight='bold')
    ax.set_ylabel(label, fontweight='bold')
    ax.set_title(f'{label} - 周期峰值拟合法', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('方法2：周期峰值拟合 - 基于代表点的平滑趋势', 
            fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/visualizations/smooth_method2_peaks.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/smooth_method2_peaks.png\n")

# ==================== 方法3：分段线性+全局平滑 ====================
print("方法3：分段线性拟合 + 全局二次曲线")

fig, axes = plt.subplots(3, 1, figsize=(20, 15))

for idx, (feature, label) in enumerate(wear_features.items()):
    ax = axes[idx]
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 原始数据
    ax.plot(frames, values, '-', alpha=0.15, color='gray', linewidth=0.5, label='原始数据')
    
    # 检测周期
    cycles = detect_cycles_advanced(values)
    
    # 提取每个周期的趋势线端点
    segment_starts = []
    segment_ends = []
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(cycles)))
    
    for cycle_idx, (start, end) in enumerate(cycles):
        seg_frames = frames[start:end+1]
        seg_values = values[start:end+1]
        
        if len(seg_values) > 10:
            # 对每段做线性拟合
            z = np.polyfit(seg_frames, seg_values, 1)
            trend = np.poly1d(z)
            
            # 绘制分段趋势
            ax.plot(seg_frames, trend(seg_frames), '-', 
                   color=colors[cycle_idx], linewidth=2, alpha=0.5)
            
            # 记录端点
            segment_starts.append(trend(seg_frames[0]))
            segment_ends.append(trend(seg_frames[-1]))
    
    # 提取周期的终点作为关键点
    cycle_key_frames = []
    cycle_key_values = []
    
    for start, end in cycles:
        if end - start > 10:
            # 取周期内的移动平均最大值
            window = min(50, (end - start) // 2)
            seg_smooth = uniform_filter1d(values[start:end+1], size=window)
            max_idx = start + np.argmax(seg_smooth)
            
            cycle_key_frames.append(frames[max_idx])
            cycle_key_values.append(values[max_idx])
    
    cycle_key_frames = np.array(cycle_key_frames)
    cycle_key_values = np.array(cycle_key_values)
    
    # 绘制关键点
    ax.plot(cycle_key_frames, cycle_key_values, 'go', markersize=10, 
           markeredgewidth=2, markeredgecolor='darkgreen',
           label='周期代表点', zorder=5)
    
    # 全局二次拟合（更平滑的曲线）
    if len(cycle_key_frames) > 3:
        z = np.polyfit(cycle_key_frames, cycle_key_values, 2)  # 二次拟合
        poly = np.poly1d(z)
        smooth_frames = np.linspace(frames[0], frames[-1], 500)
        smooth_curve = poly(smooth_frames)
        
        ax.plot(smooth_frames, smooth_curve, '-', color='darkred', linewidth=4, 
               label='全局二次平滑曲线', zorder=10, alpha=0.9)
        
        # 计算等效斜率（首尾差值）
        slope_equiv = (smooth_curve[-1] - smooth_curve[0]) / (smooth_frames[-1] - smooth_frames[0])
        
        # 判断
        first_val = smooth_curve[0]
        last_val = smooth_curve[-1]
        change_pct = ((last_val - first_val) / first_val) * 100
        
        trend_text = (f'起点: {first_val:.3f}\n'
                     f'终点: {last_val:.3f}\n'
                     f'变化: {change_pct:+.1f}%')
        
        box_color = 'lightgreen' if change_pct > 0 else 'lightcoral'
        
        ax.text(0.98, 0.98, trend_text, transform=ax.transAxes,
               fontsize=13, fontweight='bold', va='top', ha='right',
               bbox=dict(boxstyle='round,pad=1', facecolor=box_color, alpha=0.8))
    
    ax.set_xlabel('帧编号', fontweight='bold')
    ax.set_ylabel(label, fontweight='bold')
    ax.set_title(f'{label} - 全局二次平滑曲线', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.suptitle('方法3：全局二次平滑 - 最平滑的长期趋势曲线', 
            fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/visualizations/smooth_method3_global.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/smooth_method3_global.png\n")

# ==================== 综合对比图 ====================
print("生成综合对比图")

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

for idx, (feature, label) in enumerate(wear_features.items()):
    # 左侧：原始数据 + 三种平滑方法
    ax_left = fig.add_subplot(gs[idx, 0])
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 原始数据
    ax_left.plot(frames, values, '-', alpha=0.15, color='gray', 
                linewidth=0.5, label='原始数据')
    
    # 方法1：移动最大值包络
    window = 200
    max_envelope = maximum_filter1d(values, size=window, mode='nearest')
    if len(max_envelope) > 51:
        smooth_env = savgol_filter(max_envelope, window_length=51, polyorder=3)
        ax_left.plot(frames, smooth_env, '-', color='orange', 
                    linewidth=2.5, label='方法1:包络线', alpha=0.7)
    
    # 方法2：周期峰值样条插值
    cycles = detect_cycles_advanced(values)
    cycle_frames = []
    cycle_maxes = []
    
    for start, end in cycles:
        max_idx = start + np.argmax(values[start:end+1])
        cycle_frames.append(frames[max_idx])
        cycle_maxes.append(values[max_idx])
    
    if len(cycle_frames) > 3:
        spline = UnivariateSpline(cycle_frames, cycle_maxes, 
                                  k=min(3, len(cycle_frames)-1), s=5)
        smooth_frames = np.linspace(frames[0], frames[-1], 500)
        ax_left.plot(smooth_frames, spline(smooth_frames), '-', 
                    color='green', linewidth=2.5, label='方法2:样条', alpha=0.7)
    
    # 方法3：全局二次拟合
    cycle_key_frames = []
    cycle_key_values = []
    
    for start, end in cycles:
        if end - start > 10:
            window_size = min(50, (end - start) // 2)
            seg_smooth = uniform_filter1d(values[start:end+1], size=window_size)
            max_idx = start + np.argmax(seg_smooth)
            cycle_key_frames.append(frames[max_idx])
            cycle_key_values.append(values[max_idx])
    
    if len(cycle_key_frames) > 3:
        z = np.polyfit(cycle_key_frames, cycle_key_values, 2)
        poly = np.poly1d(z)
        smooth_frames = np.linspace(frames[0], frames[-1], 500)
        ax_left.plot(smooth_frames, poly(smooth_frames), '-', 
                    color='red', linewidth=3, label='方法3:二次曲线', alpha=0.8)
    
    ax_left.set_xlabel('帧编号', fontweight='bold')
    ax_left.set_ylabel(label, fontweight='bold')
    ax_left.set_title(f'{label}\n三种平滑方法对比', fontsize=13, fontweight='bold')
    ax_left.legend(fontsize=10)
    ax_left.grid(True, alpha=0.3)
    
    # 右侧：只显示最平滑的方法3（推荐）
    ax_right = fig.add_subplot(gs[idx, 1])
    
    # 浅色原始数据背景
    ax_right.fill_between(frames, 0, values, alpha=0.1, color='gray')
    
    if len(cycle_key_frames) > 3:
        # 绘制关键点
        ax_right.plot(cycle_key_frames, cycle_key_values, 'go', 
                     markersize=12, markeredgewidth=2.5, 
                     markeredgecolor='darkgreen', label='周期代表点', zorder=5)
        
        # 平滑曲线
        smooth_curve = poly(smooth_frames)
        ax_right.plot(smooth_frames, smooth_curve, '-', 
                     color='darkred', linewidth=5, 
                     label='推荐平滑曲线', zorder=10)
        
        # 添加置信区间（简化版）
        std_dev = np.std(cycle_key_values - poly(cycle_key_frames))
        ax_right.fill_between(smooth_frames, 
                             smooth_curve - std_dev, 
                             smooth_curve + std_dev,
                             alpha=0.2, color='red', label='±1标准差')
        
        # 计算趋势
        first_val = smooth_curve[0]
        last_val = smooth_curve[-1]
        change = last_val - first_val
        change_pct = (change / first_val) * 100
        
        # 大字显示结论
        if change_pct > 0:
            conclusion = f'✓ 磨损递增 +{change_pct:.1f}%'
            color = 'green'
        else:
            conclusion = f'磨损变化 {change_pct:.1f}%'
            color = 'orange'
        
        ax_right.text(0.5, 0.95, conclusion,
                     transform=ax_right.transAxes, fontsize=16,
                     fontweight='bold', ha='center', va='top',
                     bbox=dict(boxstyle='round,pad=1', 
                              facecolor=('lightgreen' if change_pct > 0 else 'lightyellow'), 
                              alpha=0.9, edgecolor=color, linewidth=3))
    
    ax_right.set_xlabel('帧编号', fontweight='bold')
    ax_right.set_ylabel(label, fontweight='bold')
    ax_right.set_title(f'{label}\n推荐方法（二次平滑）', fontsize=13, fontweight='bold')
    ax_right.legend(fontsize=10, loc='lower right')
    ax_right.grid(True, alpha=0.3)

plt.suptitle('剪刀磨损长期趋势 - 方法对比与推荐', 
            fontsize=20, fontweight='bold', y=0.995)
plt.savefig('results/visualizations/smooth_comparison_final.png', dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/smooth_comparison_final.png\n")

print("="*60)
print("所有平滑趋势分析完成！")
print("="*60)
print("\n推荐使用：")
print("- smooth_method3_global.png - 最平滑的二次曲线")
print("- smooth_comparison_final.png - 完整对比图")


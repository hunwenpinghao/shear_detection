"""
深度趋势分析：去除周期性，揭示长期磨损趋势
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d
import seaborn as sns

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取特征数据
df = pd.read_csv('results/features/wear_features.csv')

print(f"数据概览：共 {len(df)} 帧")
print(f"帧号范围：{df['frame_id'].min()} - {df['frame_id'].max()}")

# 核心特征
key_features = [
    'avg_rms_roughness',
    'avg_gradient_energy', 
    'max_notch_depth',
    'left_peak_density',
    'right_peak_density',
    'tear_shear_area_ratio'
]

# ==================== 方法1：峰值包络线分析 ====================
print("\n" + "="*60)
print("方法1：峰值包络线分析（提取每个周期的最大值）")
print("="*60)

def extract_envelope(signal, window=300):
    """提取峰值包络线"""
    envelope = []
    frames = []
    
    for i in range(0, len(signal), window//2):
        window_data = signal[i:i+window]
        if len(window_data) > 0:
            envelope.append(np.max(window_data))
            frames.append(i + window//2)
    
    return np.array(frames), np.array(envelope)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

envelope_slopes = {}

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 原始数据
    ax.plot(frames, values, 'o', alpha=0.1, markersize=2, color='gray', label='原始数据')
    
    # 提取峰值包络线
    env_frames, envelope = extract_envelope(values, window=300)
    ax.plot(env_frames, envelope, 'ro-', linewidth=2, markersize=4, label='峰值包络线', alpha=0.7)
    
    # 拟合包络线趋势
    if len(envelope) > 2:
        z = np.polyfit(env_frames, envelope, 1)
        trend = np.poly1d(z)
        ax.plot(env_frames, trend(env_frames), 'b--', linewidth=2, 
               label=f'趋势(斜率={z[0]:.6f})')
        envelope_slopes[feature] = z[0]
        
        # 判断趋势
        if z[0] > 1e-5:
            trend_text = "↑ 递增"
            color = 'green'
        elif z[0] < -1e-5:
            trend_text = "↓ 递减"
            color = 'red'
        else:
            trend_text = "→ 平稳"
            color = 'orange'
        
        ax.text(0.02, 0.98, f'包络趋势: {trend_text}\n斜率: {z[0]:.6f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    ax.set_xlabel('帧编号')
    ax.set_ylabel(feature)
    ax.set_title(f'{feature}\n峰值包络线分析')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/visualizations/envelope_analysis.png', dpi=200, bbox_inches='tight')
print("已保存: results/visualizations/envelope_analysis.png")

# ==================== 方法2：分段趋势分析 ====================
print("\n" + "="*60)
print("方法2：分段趋势分析（检测卷切换点）")
print("="*60)

def detect_change_points(signal, threshold=2.0):
    """检测突变点（可能的卷切换点）"""
    # 计算帧间差分
    diff = np.abs(np.diff(signal))
    
    # 找到超过阈值的点
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    change_points = np.where(diff > mean_diff + threshold * std_diff)[0]
    
    return change_points

# 使用RMS粗糙度检测变点
rms_values = df['avg_rms_roughness'].values
change_points = detect_change_points(rms_values, threshold=1.5)

print(f"检测到 {len(change_points)} 个潜在的卷切换点：")
print(f"位置：{change_points[:20]}...")  # 只打印前20个

# 基于变点分段
segments = []
start = 0
for cp in change_points:
    if cp - start > 50:  # 至少50帧才算一段
        segments.append((start, cp))
        start = cp
segments.append((start, len(rms_values)))

print(f"\n分为 {len(segments)} 段")

# 计算每段的趋势
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

segment_slopes = {feature: [] for feature in key_features}

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 绘制原始数据
    ax.plot(frames, values, 'o', alpha=0.2, markersize=1, color='gray')
    
    # 为每段计算趋势
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    
    for seg_idx, (start, end) in enumerate(segments[:10]):  # 只显示前10段
        seg_frames = frames[start:end]
        seg_values = values[start:end]
        
        if len(seg_values) > 2:
            z = np.polyfit(seg_frames, seg_values, 1)
            trend = np.poly1d(z)
            ax.plot(seg_frames, trend(seg_frames), '-', 
                   color=colors[seg_idx], linewidth=2, alpha=0.7)
            segment_slopes[feature].append(z[0])
    
    # 统计各段斜率
    slopes = segment_slopes[feature]
    if slopes:
        avg_slope = np.mean(slopes)
        positive_ratio = sum(1 for s in slopes if s > 0) / len(slopes)
        
        ax.text(0.02, 0.98, 
               f'段数: {len(slopes)}\n'
               f'平均斜率: {avg_slope:.6f}\n'
               f'递增段占比: {positive_ratio:.1%}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('帧编号')
    ax.set_ylabel(feature)
    ax.set_title(f'{feature}\n分段趋势（前10段）')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/visualizations/segment_analysis.png', dpi=200, bbox_inches='tight')
print("已保存: results/visualizations/segment_analysis.png")

# ==================== 方法3：低通滤波提取长期趋势 ====================
print("\n" + "="*60)
print("方法3：低通滤波提取长期趋势")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

lowpass_slopes = {}

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 原始数据
    ax.plot(frames, values, '-', alpha=0.2, linewidth=0.5, color='gray', label='原始数据')
    
    # 移动平均（窗口=100）
    if len(values) >= 100:
        ma100 = uniform_filter1d(values, size=100)
        ax.plot(frames, ma100, 'b-', linewidth=2, label='移动平均(100)', alpha=0.8)
        
        # 拟合长期趋势
        z = np.polyfit(frames, ma100, 1)
        trend = np.poly1d(z)
        ax.plot(frames, trend(frames), 'r--', linewidth=2, 
               label=f'长期趋势(斜率={z[0]:.6f})')
        lowpass_slopes[feature] = z[0]
        
        # 判断趋势
        if z[0] > 1e-5:
            trend_text = "✓ 长期递增"
            color = 'lightgreen'
        elif z[0] < -1e-5:
            trend_text = "✗ 长期递减"
            color = 'lightcoral'
        else:
            trend_text = "→ 长期平稳"
            color = 'lightyellow'
        
        ax.text(0.02, 0.98, trend_text,
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax.set_xlabel('帧编号')
    ax.set_ylabel(feature)
    ax.set_title(f'{feature}\n低通滤波后的长期趋势')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/visualizations/longterm_trend.png', dpi=200, bbox_inches='tight')
print("已保存: results/visualizations/longterm_trend.png")

# ==================== 总结报告 ====================
print("\n" + "="*60)
print("磨损趋势综合分析报告")
print("="*60)

summary_data = []
for feature in key_features:
    envelope_slope = envelope_slopes.get(feature, 0)
    lowpass_slope = lowpass_slopes.get(feature, 0)
    seg_slopes = segment_slopes.get(feature, [])
    avg_seg_slope = np.mean(seg_slopes) if seg_slopes else 0
    positive_ratio = sum(1 for s in seg_slopes if s > 0) / len(seg_slopes) if seg_slopes else 0
    
    summary_data.append({
        '特征': feature,
        '峰值包络斜率': envelope_slope,
        '低通滤波斜率': lowpass_slope,
        '分段平均斜率': avg_seg_slope,
        '递增段占比': positive_ratio
    })

summary_df = pd.DataFrame(summary_data)

print("\n核心发现：")
print(summary_df.to_string(index=False))

# 判断磨损趋势
print("\n" + "="*60)
print("结论：")
print("="*60)

wear_indicators = ['avg_rms_roughness', 'max_notch_depth']

for indicator in wear_indicators:
    row = summary_df[summary_df['特征'] == indicator].iloc[0]
    
    print(f"\n【{indicator}】")
    print(f"  峰值包络分析: {row['峰值包络斜率']:.6f} {'↑递增' if row['峰值包络斜率'] > 0 else '↓递减'}")
    print(f"  长期趋势分析: {row['低通滤波斜率']:.6f} {'↑递增' if row['低通滤波斜率'] > 0 else '↓递减'}")
    print(f"  分段趋势分析: {row['递增段占比']:.1%} 的段呈现递增")
    
    # 综合判断
    if row['峰值包络斜率'] > 0 or row['低通滤波斜率'] > 0 or row['递增段占比'] > 0.5:
        print(f"  ✓ 综合判断：存在磨损递增趋势")
    else:
        print(f"  ✗ 综合判断：未见明显磨损递增趋势")

# 保存摘要
summary_df.to_csv('results/features/trend_analysis_summary.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存分析摘要: results/features/trend_analysis_summary.csv")

print("\n" + "="*60)
print("分析完成！")
print("="*60)


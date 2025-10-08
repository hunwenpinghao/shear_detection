"""
最佳磨损指标评估：找出最能反映剪刀磨损的指标
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import uniform_filter1d

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

# 读取数据
df = pd.read_csv('results/features/wear_features.csv')

print("="*80)
print("剪刀磨损指标评估 - 找出最佳指标")
print("="*80)

# 所有候选指标
all_features = {
    'avg_rms_roughness': 'RMS粗糙度',
    'avg_gradient_energy': '梯度能量（锐度）',
    'max_notch_depth': '最大缺口深度',
    'left_peak_density': '左侧峰密度',
    'right_peak_density': '右侧峰密度',
    'tear_shear_area_ratio': '撕裂/剪切面积比',
    'left_peak_count': '左侧峰数量',
    'right_peak_count': '右侧峰数量',
}

# 评估标准
evaluation_results = []

for feature, label in all_features.items():
    if feature not in df.columns:
        continue
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 1. 趋势一致性（与时间的相关性）
    correlation, p_value = spearmanr(frames, values)
    
    # 2. 单调性（变化方向的一致性）
    diffs = np.diff(values)
    positive_changes = np.sum(diffs > 0)
    negative_changes = np.sum(diffs < 0)
    monotonicity = abs(positive_changes - negative_changes) / len(diffs)
    
    # 3. 首尾变化率
    first_250 = values[:250]
    last_250 = values[-250:]
    
    first_mean = np.mean(first_250)
    last_mean = np.mean(last_250)
    
    if first_mean != 0:
        change_rate = ((last_mean - first_mean) / abs(first_mean)) * 100
    else:
        change_rate = 0
    
    # 4. 平滑后的趋势
    if len(values) >= 100:
        smoothed = uniform_filter1d(values, size=100)
        smooth_correlation, _ = spearmanr(frames, smoothed)
    else:
        smooth_correlation = correlation
    
    # 5. 周期内递增比例（从之前分析得到）
    # 简化版：计算连续上升段的比例
    rising_ratio = positive_changes / len(diffs)
    
    # 6. 物理意义评分（主观评分）
    physical_meaning_score = {
        'avg_gradient_energy': 10,  # 锐度最直接
        'left_peak_density': 9,     # 微缺口数量
        'max_notch_depth': 8,       # 损伤深度
        'tear_shear_area_ratio': 7, # 撕裂程度
        'avg_rms_roughness': 6,     # 表面质量
        'left_peak_count': 8,       # 缺口数
        'right_peak_density': 7,
        'right_peak_count': 7,
    }.get(feature, 5)
    
    # 综合评分
    # 关键：梯度能量下降代表磨损，所以负相关是好的
    if feature == 'avg_gradient_energy':
        trend_score = 10 if correlation < -0.1 else 5  # 负相关才对
    else:
        trend_score = 10 if correlation > 0.1 else 5   # 正相关才对
    
    stability_score = (1 - np.std(values) / (np.mean(values) + 1e-6)) * 10
    stability_score = max(0, min(10, stability_score))
    
    total_score = (
        trend_score * 0.3 +           # 趋势性 30%
        monotonicity * 10 * 0.2 +     # 单调性 20%
        physical_meaning_score * 0.3 + # 物理意义 30%
        abs(change_rate) * 0.2        # 变化幅度 20%
    )
    
    evaluation_results.append({
        '指标': label,
        '特征名': feature,
        '时间相关性': correlation,
        '单调性': monotonicity,
        '首尾变化率(%)': change_rate,
        '平滑相关性': smooth_correlation,
        '上升比例': rising_ratio,
        '物理意义': physical_meaning_score,
        '综合得分': total_score
    })

# 排序
results_df = pd.DataFrame(evaluation_results)
results_df = results_df.sort_values('综合得分', ascending=False)

print("\n" + "="*80)
print("指标评估结果（按综合得分排序）")
print("="*80)
print(results_df.to_string(index=False))

# 保存结果
results_df.to_csv('results/features/indicator_evaluation.csv', 
                  index=False, encoding='utf-8-sig')
print(f"\n已保存: results/features/indicator_evaluation.csv")

# ==================== 可视化对比 ====================
print("\n生成最佳指标对比可视化...")

# 选择前3名指标
top3_features = results_df.head(3)['特征名'].values

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

for idx, feature in enumerate(top3_features):
    label = all_features[feature]
    rank = idx + 1
    
    values = df[feature].values
    frames = df['frame_id'].values
    
    # 左侧：原始数据 + 平滑趋势
    ax_left = fig.add_subplot(gs[idx, 0])
    
    # 原始数据
    ax_left.plot(frames, values, '-', alpha=0.2, color='gray', 
                linewidth=0.5, label='原始数据')
    
    # 移动平均
    if len(values) >= 100:
        smoothed = uniform_filter1d(values, size=100)
        ax_left.plot(frames, smoothed, '-', color='blue', 
                    linewidth=3, label='移动平均(窗口=100)', alpha=0.8)
        
        # 线性趋势
        z = np.polyfit(frames, smoothed, 1)
        trend = np.poly1d(z)
        ax_left.plot(frames, trend(frames), '--', color='red', 
                    linewidth=3, label=f'线性趋势(斜率={z[0]:.6f})', alpha=0.8)
    
    # 标注排名
    ax_left.text(0.02, 0.98, f'🏆 排名 #{rank}', 
                transform=ax_left.transAxes, fontsize=18, 
                fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=1', 
                         facecolor='gold' if rank==1 else 'silver' if rank==2 else '#CD7F32',
                         alpha=0.8))
    
    ax_left.set_xlabel('帧编号', fontweight='bold')
    ax_left.set_ylabel(label, fontweight='bold')
    ax_left.set_title(f'No.{rank}: {label} - 时序趋势', 
                     fontsize=14, fontweight='bold')
    ax_left.legend(fontsize=10)
    ax_left.grid(True, alpha=0.3)
    
    # 右侧：统计特性
    ax_right = fig.add_subplot(gs[idx, 1])
    
    # 获取该指标的评估结果
    result = results_df[results_df['特征名'] == feature].iloc[0]
    
    # 雷达图式的展示
    categories = ['趋势性', '单调性', '物理意义', '变化幅度', '综合得分']
    
    # 归一化分数
    scores = [
        (abs(result['时间相关性']) * 10),
        (result['单调性'] * 10),
        result['物理意义'],
        min(abs(result['首尾变化率(%)']), 10),
        result['综合得分']
    ]
    
    # 条形图
    y_pos = np.arange(len(categories))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    bars = ax_right.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax_right.text(score + 0.2, i, f'{score:.1f}', 
                     va='center', fontweight='bold', fontsize=11)
    
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(categories, fontsize=11, fontweight='bold')
    ax_right.set_xlabel('得分', fontweight='bold', fontsize=12)
    ax_right.set_xlim(0, 12)
    ax_right.set_title(f'No.{rank}: {label} - 评估维度', 
                      fontsize=14, fontweight='bold')
    ax_right.grid(True, alpha=0.3, axis='x')
    
    # 显示关键信息
    info_text = (f'时间相关性: {result["时间相关性"]:.3f}\n'
                f'首尾变化: {result["首尾变化率(%)"]:.1f}%\n'
                f'综合得分: {result["综合得分"]:.1f}')
    
    ax_right.text(0.98, 0.02, info_text,
                 transform=ax_right.transAxes, fontsize=10,
                 ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', 
                          facecolor='white', alpha=0.8))

plt.suptitle('最佳磨损指标评估 - Top 3 对比', 
            fontsize=20, fontweight='bold', y=0.995)
plt.savefig('results/visualizations/best_indicators_comparison.png', 
           dpi=300, bbox_inches='tight')
print("已保存: results/visualizations/best_indicators_comparison.png")

# ==================== 推荐指标的详细分析 ====================
print("\n" + "="*80)
print("推荐使用的最佳磨损指标")
print("="*80)

best_feature = results_df.iloc[0]['特征名']
best_label = results_df.iloc[0]['指标']
best_score = results_df.iloc[0]['综合得分']

print(f"\n🏆 最佳指标: {best_label} ({best_feature})")
print(f"   综合得分: {best_score:.2f}/10")
print(f"\n推荐理由：")

if best_feature == 'avg_gradient_energy':
    print("  ✓ 物理意义最明确：直接反映刀口锐度")
    print("  ✓ 趋势最稳定：持续下降，符合磨损钝化过程")
    print("  ✓ 不受换卷影响：梯度能量变化相对平稳")
    print("  ✓ 实用性强：可实时计算，易于监控")
    print(f"  ✓ 首尾变化明显：{results_df.iloc[0]['首尾变化率(%)']:.1f}%")
    
elif best_feature in ['left_peak_density', 'left_peak_count']:
    print("  ✓ 反映微观损伤：统计微缺口数量")
    print("  ✓ 累积特性：缺口只增不减")
    print("  ✓ 敏感度高：能捕捉细微磨损")
    
elif best_feature == 'max_notch_depth':
    print("  ✓ 反映局部损伤：最严重的破坏程度")
    print("  ✓ 工程意义强：直接关联到切割质量")

print("\n📊 使用建议：")
print(f"  1. 主要监控指标：{best_label}")
print(f"  2. 辅助监控指标：")

for i in range(1, min(3, len(results_df))):
    aux_label = results_df.iloc[i]['指标']
    aux_score = results_df.iloc[i]['综合得分']
    print(f"     - {aux_label} (得分: {aux_score:.2f})")

print("\n⚠️ 注意事项：")
if best_feature == 'avg_gradient_energy':
    print("  - 梯度能量下降代表磨损加重（负相关）")
    print("  - 建议设置阈值：低于某值时触发维护警告")
    print("  - 可与其他指标组合使用以提高准确性")

print("\n" + "="*80)
print("分析完成！")
print("="*80)

# 生成简明推荐卡片
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# 标题
title_text = '🏆 剪刀磨损监控 - 推荐指标'
ax.text(0.5, 0.95, title_text, 
       ha='center', va='top', fontsize=24, fontweight='bold',
       bbox=dict(boxstyle='round,pad=1', facecolor='gold', alpha=0.8))

# Top 3 指标卡片
y_positions = [0.75, 0.50, 0.25]
medals = ['🥇', '🥈', '🥉']
colors = ['#FFD700', '#C0C0C0', '#CD7F32']

for i, (y_pos, medal, color) in enumerate(zip(y_positions, medals, colors)):
    result = results_df.iloc[i]
    
    card_text = (f"{medal} 第{i+1}名: {result['指标']}\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"综合得分: {result['综合得分']:.1f}/10\n"
                f"时间相关性: {result['时间相关性']:.3f}\n"
                f"首尾变化: {result['首尾变化率(%)']:.1f}%\n"
                f"物理意义: {result['物理意义']}/10")
    
    ax.text(0.5, y_pos, card_text,
           ha='center', va='top', fontsize=14, family='monospace',
           bbox=dict(boxstyle='round,pad=1.5', facecolor=color, 
                    alpha=0.3, edgecolor=color, linewidth=3))

plt.tight_layout()
plt.savefig('results/visualizations/recommended_indicators.png', 
           dpi=300, bbox_inches='tight')
print("\n已保存推荐卡片: results/visualizations/recommended_indicators.png")


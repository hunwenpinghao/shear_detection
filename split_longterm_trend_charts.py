#!/usr/bin/env python3
"""
拆分长期趋势图脚本

功能：
    将包含多个子图的 longterm_trend.png 拆分为单独的图表文件，
    x轴拉长以便更清楚地查看随时间的变化曲线。
    
    本脚本复用 coil_wear_analysis.py 中的核心逻辑（包络线、鲁棒拟合等），
    避免维护两套代码。

用法：
    # 处理单个目录（使用默认路径）
    python split_longterm_trend_charts.py --input_dir data/coil_wear_analysis
    
    # 处理多个目录
    python split_longterm_trend_charts.py --input_dir data/coil_wear_analysis data_video7_20250909110956225
    
    # 自定义输入输出路径
    python split_longterm_trend_charts.py --input_dir data --csv_path features/wear_features.csv --output_subdir visualizations/individual_trends

作者: wphu
日期: 2025-10-13
更新: 2025-10-14 - 重构以复用 coil_wear_analysis.py 中的逻辑
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import matplotlib

# 导入主分析脚本中的分析器（复用其静态方法）
try:
    from coil_wear_analysis import UniversalWearAnalyzer
except ImportError:
    print("错误: 无法导入 coil_wear_analysis.py")
    print("请确保 coil_wear_analysis.py 在同一目录下")
    sys.exit(1)

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12


def process_directory(input_dir: str, csv_path: str, output_subdir: str, dpi: int = 200):
    """
    处理单个目录
    
    Args:
        input_dir: 输入主目录（如 data/coil_wear_analysis）
        csv_path: CSV文件相对路径（相对于input_dir）
        output_subdir: 输出子目录相对路径（相对于input_dir）
        dpi: 输出图片分辨率
    """
    print(f"\n{'='*80}")
    print(f"处理目录: {input_dir}")
    print(f"{'='*80}")
    
    # 构建完整路径
    csv_full_path = os.path.join(input_dir, csv_path)
    output_base_dir = os.path.join(input_dir, output_subdir)
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_full_path):
        print(f"❌ 错误: CSV文件不存在: {csv_full_path}")
        return False
    
    # 加载数据
    try:
        df = pd.read_csv(csv_full_path)
        print(f"📊 加载数据: {len(df)} 条记录")
        print(f"   帧号范围: {df['frame_id'].min()} - {df['frame_id'].max()}")
        
        # 验证和修正撕裂面占比数据
        if 'tear_shear_area_ratio' in df.columns:
            original_ratio = df['tear_shear_area_ratio']
            invalid_count = (original_ratio < 0).sum() + (original_ratio > 1).sum()
            
            if invalid_count > 0:
                print(f"⚠️  发现 {invalid_count} 个撕裂面占比值超出0-1范围，正在修正...")
                
                # 如果有很多值>1，可能是比值形式，使用转换公式
                if (original_ratio > 1).sum() > len(original_ratio) * 0.1:
                    print("   使用转换公式: new_ratio = old_ratio / (old_ratio + 1)")
                    df['tear_shear_area_ratio'] = original_ratio / (original_ratio + 1)
                else:
                    print("   直接截断到0-1范围")
                    df['tear_shear_area_ratio'] = np.clip(original_ratio, 0.0, 1.0)
                
                # 验证修正结果
                corrected_ratio = df['tear_shear_area_ratio']
                print(f"✅ 修正完成: 最小值={corrected_ratio.min():.4f}, 最大值={corrected_ratio.max():.4f}")
                print(f"   所有值在0-1范围内: {(corrected_ratio >= 0).all() and (corrected_ratio <= 1).all()}")
            else:
                print("✅ 撕裂面占比数据正常（0-1范围）")
        
    except Exception as e:
        print(f"❌ 错误: 无法读取CSV文件: {e}")
        return False
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"📁 输出目录: {output_base_dir}")
    
    # 定义特征及其对应的标签和颜色
    features_to_plot = [
        ('avg_rms_roughness', '平均RMS粗糙度', 'blue'),
        ('max_notch_depth', '最大缺口深度', 'red'),
        ('right_peak_density', '剪切面峰密度', 'green'),
        ('avg_gradient_energy', '平均梯度能量', 'purple'),
        ('tear_shear_area_ratio', '撕裂面占比', 'orange'),
    ]
    
    print("\n开始绘制各个特征图表...")
    
    success_count = 0
    skip_count = 0
    
    for feat, label, color in features_to_plot:
        if feat not in df.columns:
            print(f"⚠️  跳过: 特征 '{feat}' 不存在于数据中")
            skip_count += 1
            continue
        
        try:
            # 创建单独的图表，x轴拉长（60英寸）
            fig, ax = plt.subplots(figsize=(60, 6))
            
            # 获取数据
            y_values = df[feat].values
            
            # 🔄 复用主脚本逻辑：计算包络线
            upper_env, lower_env = UniversalWearAnalyzer.compute_envelope(
                y_values, window=min(31, len(y_values)//10)
            )
            
            # 🔄 复用主脚本逻辑：计算鲁棒拟合曲线
            fitted_curve, inlier_mask = UniversalWearAnalyzer.robust_curve_fit(
                y_values, percentile_range=(5, 95)
            )
            
            # 绘制包络范围（填充）
            ax.fill_between(df['frame_id'], lower_env, upper_env,
                           alpha=0.15, color='gray', label='包络范围', zorder=1)
            
            # 绘制上下包络线
            ax.plot(df['frame_id'], upper_env, ':', linewidth=1.5, 
                   color='red', alpha=0.6, label='上包络', zorder=2)
            ax.plot(df['frame_id'], lower_env, ':', linewidth=1.5, 
                   color='green', alpha=0.6, label='下包络', zorder=2)
            
            # 原始数据连线（半透明）
            ax.plot(df['frame_id'], y_values,
                   alpha=0.3, linewidth=1.2, color=color,
                   zorder=3, label='逐帧曲线')
            
            # 散点标记
            ax.scatter(df['frame_id'], y_values,
                      alpha=0.4, s=15, color=color, zorder=4)
            
            # 标注离群点
            outlier_indices = np.where(~inlier_mask)[0]
            if len(outlier_indices) > 0:
                ax.scatter(df['frame_id'].iloc[outlier_indices], 
                          y_values[outlier_indices],
                          c='orange', s=30, marker='x', alpha=0.7, 
                          label=f'离群点({len(outlier_indices)}个)', zorder=5)
            
            # 鲁棒拟合曲线（主趋势）
            ax.plot(df['frame_id'], fitted_curve,
                   color='purple', linewidth=3, linestyle='-',
                   alpha=0.8, zorder=6, label='鲁棒拟合')
            
            # 线性拟合趋势线（使用内点）
            x_inliers = df['frame_id'][inlier_mask]
            y_inliers = y_values[inlier_mask]
            if len(x_inliers) >= 2:
                z = np.polyfit(x_inliers, y_inliers, 1)
                p = np.poly1d(z)
                ax.plot(df['frame_id'], p(df['frame_id']),
                       color='darkred', linewidth=2.5, linestyle='--',
                       zorder=7, label=f'线性趋势: y={z[0]:.6f}x+{z[1]:.2f}')
            else:
                # 如果内点太少，使用所有数据点
                z = np.polyfit(df['frame_id'], y_values, 1)
                p = np.poly1d(z)
                ax.plot(df['frame_id'], p(df['frame_id']),
                       color='darkred', linewidth=2.5, linestyle='--',
                       zorder=7, label=f'线性趋势: y={z[0]:.6f}x+{z[1]:.2f}')
            
            # 计算趋势方向和内点率
            trend = "增加" if z[0] > 0 else "减少"
            inlier_ratio = inlier_mask.sum() / len(inlier_mask) * 100
            ax.text(0.05, 0.95, f'趋势: {trend}\n内点率: {inlier_ratio:.1f}%',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('帧编号', fontsize=12, fontweight='bold')
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(f'{label} 长期趋势', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 调整x轴范围，确保拉长效果
            ax.set_xlim(df['frame_id'].min(), df['frame_id'].max())
            
            # 保存图表
            individual_save_path = os.path.join(output_base_dir, f'{feat}_trend.png')
            plt.savefig(individual_save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✅ 已保存: {feat}_trend.png")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 错误: 绘制 '{feat}' 时出错: {e}")
            skip_count += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"处理完成: {input_dir}")
    print(f"  成功: {success_count} 个图表")
    print(f"  跳过: {skip_count} 个特征")
    print(f"{'='*80}")
    
    # 生成7x1总图（包含综合指标和斑块分析）
    print("\n生成7×1总图（包含综合指标和斑块分析）...")
    try:
        _generate_combined_plot_7x1(df, features_to_plot, output_base_dir, dpi)
        print("✅ 已保存: all_trends_7x1.png")
    except Exception as e:
        print(f"❌ 生成总图失败: {e}")
    
    return success_count > 0


def _generate_combined_plot_7x1(df: pd.DataFrame, features_to_plot: list, output_dir: str, dpi: int):
    """
    生成7×1组合图（综合指标 + 5个特征 + 白色斑块分析上下罗列）
    
    Args:
        df: 数据DataFrame
        features_to_plot: 特征列表 [(特征名, 标签, 颜色), ...]
        output_dir: 输出目录
        dpi: 图片分辨率
    """
    # 创建7×1子图布局，x轴缩小为80英寸（第1个为综合指标，中间5个为各特征，最后1个为斑块分析）
    fig, axes = plt.subplots(7, 1, figsize=(80, 34))
    
    # ========== 第1个子图：综合指标（4个特征归一化后叠加，不含梯度能量） ==========
    ax_composite = axes[0]
    
    # 计算综合指标 - 基于多剪刀周期磨损理论的融合策略
    composite_score = np.zeros(len(df))
    valid_features = []
    
    # 定义磨损指标及其方向性（基于多周期分析结果）
    wear_indicators = {
        # 正向指标：值越大，磨损越严重（在多周期内验证有效）
        'tear_shear_area_ratio': {'weight': 0.35, 'direction': 'positive', 'name': '撕裂面占比'},
        
        # 反向指标：值越小，磨损越严重（在多周期内验证有效）
        'right_peak_density': {'weight': 0.25, 'direction': 'negative', 'name': '剪切面峰密度'},
        'avg_gradient_energy': {'weight': 0.25, 'direction': 'negative', 'name': '平均梯度能量'},
        
        # 需要进一步验证的指标（在多周期内表现不一致）
        'avg_rms_roughness': {'weight': 0.10, 'direction': 'positive', 'name': '平均RMS粗糙度'},
        'max_notch_depth': {'weight': 0.05, 'direction': 'positive', 'name': '最大缺口深度'}
    }
    
    # 添加白色斑块特征（如果存在）
    white_patch_feature = None
    if 'white_composite_index_m1' in df.columns:
        white_patch_feature = 'white_composite_index_m1'
        wear_indicators[white_patch_feature] = {'weight': 0.10, 'direction': 'positive', 'name': '白色斑块综合指标'}
    elif 'white_area_ratio_m1' in df.columns:
        white_patch_feature = 'white_area_ratio_m1'
        wear_indicators[white_patch_feature] = {'weight': 0.10, 'direction': 'positive', 'name': '白色斑块面积占比'}
    
    # 重新归一化权重，确保总和为1
    total_weight = sum(indicator['weight'] for indicator in wear_indicators.values())
    for indicator in wear_indicators.values():
        indicator['weight'] = indicator['weight'] / total_weight
    
    # 计算磨损综合指标
    for feat, config in wear_indicators.items():
        if feat in df.columns:
            values = df[feat].values
            if values.max() > values.min():
                # 使用鲁棒归一化（基于百分位数，减少极值影响）
                p5, p95 = np.percentile(values, [5, 95])
                normalized = np.clip((values - p5) / (p95 - p5), 0, 1)
                
                # 根据方向性调整
                if config['direction'] == 'negative':
                    # 反向指标：取反，使值越大表示磨损越严重
                    wear_contribution = (1 - normalized) * config['weight']
                else:
                    # 正向指标：直接使用
                    wear_contribution = normalized * config['weight']
                
                composite_score += wear_contribution
                valid_features.append((feat, config['name']))
    
    # 确保综合指标在0-1范围内
    composite_score = np.clip(composite_score, 0, 1)
    
    # 添加周期信息到特征说明
    try:
        from scipy.signal import find_peaks
        # 使用撕裂面占比识别剪刀周期
        tear_ratio = df['tear_shear_area_ratio'].values
        frames = df['frame_id'].values
        
        # 平滑数据识别周期
        window_size = min(50, len(tear_ratio) // 20)
        smoothed = np.convolve(tear_ratio, np.ones(window_size)/window_size, mode='valid')
        peaks, _ = find_peaks(smoothed, height=np.mean(smoothed), distance=100)
        
        cycle_count = len(peaks)
        avg_cycle_length = len(frames) / cycle_count if cycle_count > 0 else 0
        
        # 将周期信息添加到特征说明中
        cycle_info = f' | 识别到{cycle_count}个剪刀周期(平均{avg_cycle_length:.0f}帧/周期)'
    except ImportError:
        cycle_info = ' | 周期识别需要scipy'
    
    # 🔄 复用主脚本逻辑：计算包络线和鲁棒拟合
    upper_env_comp, lower_env_comp = UniversalWearAnalyzer.compute_envelope(
        composite_score, window=min(31, len(composite_score)//10)
    )
    fitted_curve_comp, inlier_mask_comp = UniversalWearAnalyzer.robust_curve_fit(
        composite_score, percentile_range=(5, 95)
    )
    
    # 绘制包络范围
    ax_composite.fill_between(df['frame_id'], lower_env_comp, upper_env_comp,
                             alpha=0.15, color='gray', label='包络范围', zorder=1)
    
    # 绘制包络线
    ax_composite.plot(df['frame_id'], upper_env_comp, ':', linewidth=1.5, 
                     color='red', alpha=0.6, label='上包络', zorder=2)
    ax_composite.plot(df['frame_id'], lower_env_comp, ':', linewidth=1.5, 
                     color='green', alpha=0.6, label='下包络', zorder=2)
    
    # 绘制综合指标（半透明）
    ax_composite.plot(df['frame_id'], composite_score,
                     alpha=0.3, linewidth=1.5, color='darkblue',
                     zorder=3, label='综合磨损指标')
    
    ax_composite.scatter(df['frame_id'], composite_score,
                        alpha=0.4, s=20, color='darkblue', zorder=4)
    
    # 标注离群点
    outlier_indices_comp = np.where(~inlier_mask_comp)[0]
    if len(outlier_indices_comp) > 0:
        ax_composite.scatter(df['frame_id'].iloc[outlier_indices_comp], 
                            composite_score[outlier_indices_comp],
                            c='orange', s=35, marker='x', alpha=0.7, 
                            label=f'离群点({len(outlier_indices_comp)}个)', zorder=5)
    
    # 鲁棒拟合曲线
    ax_composite.plot(df['frame_id'], fitted_curve_comp,
                     color='purple', linewidth=3.5, linestyle='-',
                     alpha=0.8, zorder=6, label='鲁棒拟合')
    
    # 线性拟合（使用内点）
    x_inliers_comp = df['frame_id'][inlier_mask_comp]
    y_inliers_comp = composite_score[inlier_mask_comp]
    if len(x_inliers_comp) >= 2:
        z_comp = np.polyfit(x_inliers_comp, y_inliers_comp, 1)
        p_comp = np.poly1d(z_comp)
        ax_composite.plot(df['frame_id'], p_comp(df['frame_id']),
                         color='red', linewidth=3, linestyle='--',
                         zorder=7, label=f'线性趋势: y={z_comp[0]:.6f}x+{z_comp[1]:.2f}')
    else:
        # 如果内点太少，使用所有数据点
        z_comp = np.polyfit(df['frame_id'], composite_score, 1)
        p_comp = np.poly1d(z_comp)
        ax_composite.plot(df['frame_id'], p_comp(df['frame_id']),
                         color='red', linewidth=3, linestyle='--',
                         zorder=7, label=f'线性趋势: y={z_comp[0]:.6f}x+{z_comp[1]:.2f}')
    
    # 趋势标注（包含内点率）
    trend_comp = "增加" if z_comp[0] > 0 else "减少"
    trend_color_comp = 'lightgreen' if z_comp[0] > 0 else 'lightcoral'
    inlier_ratio_comp = inlier_mask_comp.sum() / len(inlier_mask_comp) * 100
    ax_composite.text(0.02, 0.98, f'趋势: {trend_comp}\n内点率: {inlier_ratio_comp:.1f}%',
                     transform=ax_composite.transAxes, fontsize=12,
                     verticalalignment='top', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor=trend_color_comp, alpha=0.7,
                              edgecolor='black', linewidth=2))
    
    # 统计信息
    mean_comp = composite_score.mean()
    std_comp = composite_score.std()
    min_comp = composite_score.min()
    max_comp = composite_score.max()
    
    stats_text_comp = f'均值: {mean_comp:.3f}\n标准差: {std_comp:.3f}\n范围: [{min_comp:.3f}, {max_comp:.3f}]'
    ax_composite.text(0.98, 0.98, stats_text_comp,
                     transform=ax_composite.transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8,
                              edgecolor='gray', linewidth=1))
    
    # 添加特征说明
    indicator_info = []
    for feat, config in wear_indicators.items():
        if feat in df.columns:
            direction_text = '正向' if config['direction'] == 'positive' else '反向'
            indicator_info.append(f'{config["name"]}({direction_text},{config["weight"]:.2f})')
    
    features_text = '磨损指标: ' + ' | '.join(indicator_info) + cycle_info
    ax_composite.text(0.02, 0.02, features_text,
                     transform=ax_composite.transAxes, fontsize=8,
                     verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                              edgecolor='gray', linewidth=0.5))
    
    ax_composite.set_xlabel('帧编号', fontsize=13, fontweight='bold')
    ax_composite.set_ylabel('综合磨损指标 (归一化)', fontsize=13, fontweight='bold')
    # 生成动态标题
    included_features = []
    for feat, config in wear_indicators.items():
        if feat in df.columns:
            direction_symbol = '↑' if config['direction'] == 'positive' else '↓'
            included_features.append(f'{config["name"]}{direction_symbol}')
    
    feature_text = '+'.join(included_features)
    ax_composite.set_title(f'综合磨损指标 (基于磨损理论融合: {feature_text})', fontsize=16, fontweight='bold', pad=15, 
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax_composite.grid(True, alpha=0.3)
    ax_composite.legend(loc='upper left', fontsize=11)
    ax_composite.set_xlim(df['frame_id'].min(), df['frame_id'].max())
    ax_composite.set_ylim(-0.05, 1.05)
    
    # ========== 后5个子图：各个特征 ==========
    
    for idx, (feat, label, color) in enumerate(features_to_plot):
        ax = axes[idx + 1]  # 因为第0个位置被综合指标占用
        
        if feat not in df.columns:
            ax.text(0.5, 0.5, f'特征 "{feat}" 不存在', 
                   ha='center', va='center', fontsize=14, color='red')
            ax.set_title(f'{label} - 数据缺失', fontsize=14, fontweight='bold')
            continue
        
        # 获取数据
        y_values = df[feat].values
        
        # 🔄 复用主脚本逻辑：计算包络线和鲁棒拟合
        upper_env, lower_env = UniversalWearAnalyzer.compute_envelope(
            y_values, window=min(31, len(y_values)//10)
        )
        fitted_curve, inlier_mask = UniversalWearAnalyzer.robust_curve_fit(
            y_values, percentile_range=(5, 95)
        )
        
        # 绘制包络范围
        ax.fill_between(df['frame_id'], lower_env, upper_env,
                       alpha=0.15, color='gray', label='包络范围', zorder=1)
        
        # 绘制包络线
        ax.plot(df['frame_id'], upper_env, ':', linewidth=1.5, 
               color='red', alpha=0.6, label='上包络', zorder=2)
        ax.plot(df['frame_id'], lower_env, ':', linewidth=1.5, 
               color='green', alpha=0.6, label='下包络', zorder=2)
        
        # 原始数据连线（半透明）
        ax.plot(df['frame_id'], y_values,
               alpha=0.3, linewidth=1.2, color=color,
               zorder=3, label='逐帧曲线')
        
        # 散点标记
        ax.scatter(df['frame_id'], y_values,
                  alpha=0.4, s=15, color=color, zorder=4)
        
        # 标注离群点
        outlier_indices = np.where(~inlier_mask)[0]
        if len(outlier_indices) > 0:
            ax.scatter(df['frame_id'].iloc[outlier_indices], 
                      y_values[outlier_indices],
                      c='orange', s=30, marker='x', alpha=0.7, 
                      label=f'离群点({len(outlier_indices)}个)', zorder=5)
        
        # 鲁棒拟合曲线
        ax.plot(df['frame_id'], fitted_curve,
               color='purple', linewidth=2.5, linestyle='-',
               alpha=0.8, zorder=6, label='鲁棒拟合')
        
        # 线性拟合趋势线（使用内点）
        x_inliers = df['frame_id'][inlier_mask]
        y_inliers = y_values[inlier_mask]
        if len(x_inliers) >= 2:
            z = np.polyfit(x_inliers, y_inliers, 1)
            p = np.poly1d(z)
            ax.plot(df['frame_id'], p(df['frame_id']),
                   color='darkred', linewidth=2.5, linestyle='--',
                   zorder=7, label=f'线性趋势: y={z[0]:.6f}x+{z[1]:.2f}')
        else:
            # 如果内点太少，使用所有数据点
            z = np.polyfit(df['frame_id'], y_values, 1)
            p = np.poly1d(z)
            ax.plot(df['frame_id'], p(df['frame_id']),
                   color='darkred', linewidth=2.5, linestyle='--',
                   zorder=7, label=f'线性趋势: y={z[0]:.6f}x+{z[1]:.2f}')
        
        # 计算趋势方向和内点率
        trend = "增加" if z[0] > 0 else "减少"
        trend_color = 'lightgreen' if z[0] > 0 else 'lightcoral'
        inlier_ratio = inlier_mask.sum() / len(inlier_mask) * 100
        ax.text(0.02, 0.98, f'趋势: {trend}\n内点率: {inlier_ratio:.1f}%',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=trend_color, alpha=0.6,
                        edgecolor='black', linewidth=1))
        
        # 添加统计信息
        mean_val = df[feat].mean()
        std_val = df[feat].std()
        min_val = df[feat].min()
        max_val = df[feat].max()
        
        stats_text = f'均值: {mean_val:.2f}\n标准差: {std_val:.2f}\n范围: [{min_val:.2f}, {max_val:.2f}]'
        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                        edgecolor='gray', linewidth=0.5))
        
        ax.set_xlabel('帧编号', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} 长期趋势', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # 调整x轴范围，确保拉长效果
        ax.set_xlim(df['frame_id'].min(), df['frame_id'].max())
    
    # ========== 第7个子图：白色斑块分析 ==========
    ax_patch = axes[6]  # 第7个子图（索引为6）
    
    # 检查是否有白色斑块相关数据
    patch_metrics = ['white_area_ratio_m1', 'white_patch_count_m1', 'white_composite_index_m1', 'white_brightness_entropy_m1']
    available_patch_metrics = [metric for metric in patch_metrics if metric in df.columns]
    
    if available_patch_metrics:
        # 选择最佳指标：优先使用综合指标，其次面积占比
        if 'white_composite_index_m1' in available_patch_metrics:
            patch_feature = 'white_composite_index_m1'
            patch_label = '白色斑块综合指标'
            patch_color = 'darkcyan'
        elif 'white_area_ratio_m1' in available_patch_metrics:
            patch_feature = 'white_area_ratio_m1'
            patch_label = '白色斑块面积占比(%)'
            patch_color = 'darkcyan'
        else:
            patch_feature = available_patch_metrics[0]
            patch_label = f'白色斑块{available_patch_metrics[0].replace("white_", "").replace("_m1", "")}'
            patch_color = 'darkcyan'
        
        # 获取斑块数据
        patch_values = df[patch_feature].values
        
        # 🔄 复用主脚本逻辑：计算包络线和鲁棒拟合
        upper_env_patch, lower_env_patch = UniversalWearAnalyzer.compute_envelope(
            patch_values, window=min(31, len(patch_values)//10)
        )
        fitted_curve_patch, inlier_mask_patch = UniversalWearAnalyzer.robust_curve_fit(
            patch_values, percentile_range=(5, 95)
        )
        
        # 绘制包络范围
        ax_patch.fill_between(df['frame_id'], lower_env_patch, upper_env_patch,
                             alpha=0.15, color='gray', label='包络范围', zorder=1)
        
        # 绘制包络线
        ax_patch.plot(df['frame_id'], upper_env_patch, ':', linewidth=1.5, 
                     color='red', alpha=0.6, label='上包络', zorder=2)
        ax_patch.plot(df['frame_id'], lower_env_patch, ':', linewidth=1.5, 
                     color='green', alpha=0.6, label='下包络', zorder=2)
        
        # 原始数据连线（半透明）
        ax_patch.plot(df['frame_id'], patch_values,
                     alpha=0.3, linewidth=1.2, color=patch_color,
                     zorder=3, label='逐帧曲线')
        
        # 散点标记
        ax_patch.scatter(df['frame_id'], patch_values,
                        alpha=0.4, s=15, color=patch_color, zorder=4)
        
        # 标注离群点
        outlier_indices_patch = np.where(~inlier_mask_patch)[0]
        if len(outlier_indices_patch) > 0:
            ax_patch.scatter(df['frame_id'].iloc[outlier_indices_patch], 
                            patch_values[outlier_indices_patch],
                            c='orange', s=30, marker='x', alpha=0.7, 
                            label=f'离群点({len(outlier_indices_patch)}个)', zorder=5)
        
        # 鲁棒拟合曲线
        ax_patch.plot(df['frame_id'], fitted_curve_patch,
                     color='purple', linewidth=2.5, linestyle='-',
                     alpha=0.8, zorder=6, label='鲁棒拟合')
        
        # 线性拟合趋势线（使用内点）
        x_inliers_patch = df['frame_id'][inlier_mask_patch]
        y_inliers_patch = patch_values[inlier_mask_patch]
        if len(x_inliers_patch) >= 2:
            z_patch = np.polyfit(x_inliers_patch, y_inliers_patch, 1)
            p_patch = np.poly1d(z_patch)
            ax_patch.plot(df['frame_id'], p_patch(df['frame_id']),
                         color='darkred', linewidth=2.5, linestyle='--',
                         zorder=7, label=f'线性趋势: y={z_patch[0]:.6f}x+{z_patch[1]:.2f}')
        else:
            # 如果内点太少，使用所有数据点
            z_patch = np.polyfit(df['frame_id'], patch_values, 1)
            p_patch = np.poly1d(z_patch)
            ax_patch.plot(df['frame_id'], p_patch(df['frame_id']),
                         color='darkred', linewidth=2.5, linestyle='--',
                         zorder=7, label=f'线性趋势: y={z_patch[0]:.6f}x+{z_patch[1]:.2f}')
        
        # 计算趋势方向和内点率
        trend_patch = "增加" if z_patch[0] > 0 else "减少"
        trend_color_patch = 'lightgreen' if z_patch[0] > 0 else 'lightcoral'
        inlier_ratio_patch = inlier_mask_patch.sum() / len(inlier_mask_patch) * 100
        ax_patch.text(0.02, 0.98, f'趋势: {trend_patch}\n内点率: {inlier_ratio_patch:.1f}%',
                     transform=ax_patch.transAxes, fontsize=11,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor=trend_color_patch, alpha=0.6,
                              edgecolor='black', linewidth=1))
        
        # 添加统计信息
        mean_patch = df[patch_feature].mean()
        std_patch = df[patch_feature].std()
        min_patch = df[patch_feature].min()
        max_patch = df[patch_feature].max()
        
        stats_text_patch = f'均值: {mean_patch:.2f}\n标准差: {std_patch:.2f}\n范围: [{min_patch:.2f}, {max_patch:.2f}]'
        ax_patch.text(0.98, 0.98, stats_text_patch,
                     transform=ax_patch.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                              edgecolor='gray', linewidth=0.5))
        
        ax_patch.set_xlabel('帧编号', fontsize=12, fontweight='bold')
        ax_patch.set_ylabel(patch_label, fontsize=12, fontweight='bold')
        ax_patch.set_title(f'{patch_label} 长期趋势', fontsize=14, fontweight='bold', pad=10)
        ax_patch.grid(True, alpha=0.3)
        ax_patch.legend(loc='upper left', fontsize=10)
        
        # 调整x轴范围，确保拉长效果
        ax_patch.set_xlim(df['frame_id'].min(), df['frame_id'].max())
        
    else:
        # 如果没有斑块数据，显示提示信息
        ax_patch.text(0.5, 0.5, '白色斑块数据不可用\n请运行白色斑块分析器生成数据', 
                     ha='center', va='center', fontsize=14, color='red',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax_patch.set_title('白色斑块分析 - 数据缺失', fontsize=14, fontweight='bold')
        ax_patch.set_xlim(0, 1)
        ax_patch.set_ylim(0, 1)
    
    # 设置总标题
    fig.suptitle('剪刀磨损长期趋势综合分析（综合指标[4特征] + 5特征详情 + 白色斑块分析）', 
                fontsize=18, fontweight='bold', y=0.996)
    
    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.996])
    
    # 保存
    combined_save_path = os.path.join(output_dir, 'all_trends_7x1.png')
    plt.savefig(combined_save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='拆分长期趋势图：将多子图拆分为单独图表文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个目录
  python split_longterm_trend_charts.py --input_dir data/coil_wear_analysis
  
  # 处理多个目录
  python split_longterm_trend_charts.py --input_dir data/coil_wear_analysis data_video7_20250909110956225 data_video5_20250909110956225_2025091310250004
  
  # 自定义路径
  python split_longterm_trend_charts.py --input_dir data --csv_path features/wear_features.csv --output_subdir viz/trends
  
  # 自定义分辨率
  python split_longterm_trend_charts.py --input_dir data/coil_wear_analysis --dpi 300
        """
    )
    
    parser.add_argument(
        '--input_dir', 
        nargs='+',
        required=True,
        help='输入主目录路径（可指定多个，如: data/coil_wear_analysis data_video7_20250909110956225）'
    )
    
    parser.add_argument(
        '--csv_path',
        default='features/wear_features.csv',
        help='CSV文件相对路径（相对于input_dir，默认: features/wear_features.csv）'
    )
    
    parser.add_argument(
        '--output_subdir',
        default='visualizations/individual_trends',
        help='输出子目录相对路径（相对于input_dir，默认: visualizations/individual_trends）'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='输出图片分辨率（默认: 200）'
    )
    
    args = parser.parse_args()
    
    # 处理所有目录
    total_dirs = len(args.input_dir)
    success_dirs = 0
    failed_dirs = 0
    
    print(f"\n{'#'*80}")
    print(f"# 拆分长期趋势图工具")
    print(f"# 待处理目录数: {total_dirs}")
    print(f"{'#'*80}")
    
    for input_dir in args.input_dir:
        # 检查目录是否存在
        if not os.path.exists(input_dir):
            print(f"\n❌ 错误: 目录不存在: {input_dir}")
            failed_dirs += 1
            continue
        
        # 处理目录
        success = process_directory(
            input_dir=input_dir,
            csv_path=args.csv_path,
            output_subdir=args.output_subdir,
            dpi=args.dpi
        )
        
        if success:
            success_dirs += 1
        else:
            failed_dirs += 1
    
    # 打印总结
    print(f"\n{'#'*80}")
    print(f"# 处理完成")
    print(f"# 总计: {total_dirs} 个目录")
    print(f"# 成功: {success_dirs} 个")
    print(f"# 失败: {failed_dirs} 个")
    print(f"{'#'*80}\n")
    
    return 0 if failed_dirs == 0 else 1


if __name__ == '__main__':
    sys.exit(main())


#!/usr/bin/env python3
"""
拆分长期趋势图脚本

功能：
    将包含多个子图的 longterm_trend.png 拆分为单独的图表文件，
    x轴拉长以便更清楚地查看随时间的变化曲线。

用法：
    # 处理单个目录（使用默认路径）
    python split_longterm_trend_charts.py --input_dir data/coil_wear_analysis
    
    # 处理多个目录
    python split_longterm_trend_charts.py --input_dir data/coil_wear_analysis data_video7_20250909110956225
    
    # 自定义输入输出路径
    python split_longterm_trend_charts.py --input_dir data --csv_path features/wear_features.csv --output_subdir visualizations/individual_trends

作者: wphu
日期: 2025-10-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import matplotlib

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
        ('tear_shear_area_ratio', '撕裂/剪切面积比', 'orange'),
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
            # 创建单独的图表，x轴拉长（调整为80英寸 ÷ 6×1的高度比例 = 60英寸）
            fig, ax = plt.subplots(figsize=(60, 6))
            
            # 原始数据连线（半透明）
            ax.plot(df['frame_id'], df[feat],
                   alpha=0.3, linewidth=1.2, color=color,
                   zorder=1, label='逐帧曲线')
            
            # 散点标记
            ax.scatter(df['frame_id'], df[feat],
                      alpha=0.4, s=15, color=color, zorder=2)
            
            # 线性拟合趋势线
            z = np.polyfit(df['frame_id'], df[feat], 1)
            p = np.poly1d(z)
            ax.plot(df['frame_id'], p(df['frame_id']),
                   color='darkred', linewidth=3, linestyle='--',
                   zorder=3, label=f'线性趋势: y={z[0]:.6f}x+{z[1]:.2f}')
            
            # 计算趋势方向
            trend = "增加" if z[0] > 0 else "减少"
            ax.text(0.05, 0.95, f'趋势: {trend}',
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
    
    # 生成6x1总图（包含综合指标）
    print("\n生成6×1总图（包含综合指标）...")
    try:
        _generate_combined_plot_6x1(df, features_to_plot, output_base_dir, dpi)
        print("✅ 已保存: all_trends_6x1.png")
    except Exception as e:
        print(f"❌ 生成总图失败: {e}")
    
    return success_count > 0


def _generate_combined_plot_6x1(df: pd.DataFrame, features_to_plot: list, output_dir: str, dpi: int):
    """
    生成6×1组合图（综合指标 + 5个特征上下罗列）
    
    Args:
        df: 数据DataFrame
        features_to_plot: 特征列表 [(特征名, 标签, 颜色), ...]
        output_dir: 输出目录
        dpi: 图片分辨率
    """
    # 创建6×1子图布局，x轴缩小为80英寸（第1个为综合指标，后5个为各特征）
    fig, axes = plt.subplots(6, 1, figsize=(80, 29))
    
    # ========== 第1个子图：综合指标（4个特征归一化后叠加，不含梯度能量） ==========
    ax_composite = axes[0]
    
    # 计算综合指标 - 排除 avg_gradient_energy
    composite_score = np.zeros(len(df))
    valid_features = []
    excluded_features = ['avg_gradient_energy']  # 排除的特征
    
    for feat, label, color in features_to_plot:
        if feat in df.columns and feat not in excluded_features:
            # 归一化到0-1
            values = df[feat].values
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
                composite_score += normalized
                valid_features.append((feat, label))
    
    # 平均化（避免简单求和导致值过大）
    if len(valid_features) > 0:
        composite_score = composite_score / len(valid_features)
    
    # 绘制综合指标
    ax_composite.plot(df['frame_id'], composite_score,
                     alpha=0.3, linewidth=1.5, color='darkblue',
                     zorder=1, label='综合磨损指标')
    
    ax_composite.scatter(df['frame_id'], composite_score,
                        alpha=0.4, s=20, color='darkblue', zorder=2)
    
    # 线性拟合
    z_comp = np.polyfit(df['frame_id'], composite_score, 1)
    p_comp = np.poly1d(z_comp)
    ax_composite.plot(df['frame_id'], p_comp(df['frame_id']),
                     color='red', linewidth=4, linestyle='--',
                     zorder=3, label=f'线性趋势: y={z_comp[0]:.6f}x+{z_comp[1]:.2f}')
    
    # 趋势标注
    trend_comp = "增加" if z_comp[0] > 0 else "减少"
    trend_color_comp = 'lightgreen' if z_comp[0] > 0 else 'lightcoral'
    ax_composite.text(0.02, 0.98, f'趋势: {trend_comp}',
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
    features_text = '包含特征: ' + ', '.join([label for _, label in valid_features])
    ax_composite.text(0.02, 0.02, features_text,
                     transform=ax_composite.transAxes, fontsize=9,
                     verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                              edgecolor='gray', linewidth=0.5))
    
    ax_composite.set_xlabel('帧编号', fontsize=13, fontweight='bold')
    ax_composite.set_ylabel('综合磨损指标 (归一化)', fontsize=13, fontweight='bold')
    ax_composite.set_title('综合磨损指标 (4特征归一化叠加: 不含梯度能量)', fontsize=16, fontweight='bold', pad=15, 
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
        
        # 原始数据连线（半透明）
        ax.plot(df['frame_id'], df[feat],
               alpha=0.3, linewidth=1.2, color=color,
               zorder=1, label='逐帧曲线')
        
        # 散点标记
        ax.scatter(df['frame_id'], df[feat],
                  alpha=0.4, s=15, color=color, zorder=2)
        
        # 线性拟合趋势线
        z = np.polyfit(df['frame_id'], df[feat], 1)
        p = np.poly1d(z)
        ax.plot(df['frame_id'], p(df['frame_id']),
               color='darkred', linewidth=3, linestyle='--',
               zorder=3, label=f'线性趋势: y={z[0]:.6f}x+{z[1]:.2f}')
        
        # 计算趋势方向
        trend = "增加" if z[0] > 0 else "减少"
        trend_color = 'lightgreen' if z[0] > 0 else 'lightcoral'
        ax.text(0.02, 0.98, f'趋势: {trend}',
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
    
    # 设置总标题
    fig.suptitle('剪刀磨损长期趋势综合分析（综合指标[4特征] + 5特征详情）', 
                fontsize=18, fontweight='bold', y=0.996)
    
    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.996])
    
    # 保存
    combined_save_path = os.path.join(output_dir, 'all_trends_6x1.png')
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


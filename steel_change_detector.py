#!/usr/bin/env python3
"""
换钢材时间段检测器
在高斯平滑基础上增强梯度来识别换钢材时间段
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

def setup_chinese_font():
    """设置中文字体"""
    import matplotlib.font_manager as fm
    import platform
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = []
        
        preferred_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
        
        for font in preferred_fonts:
            if font in available_fonts:
                chinese_fonts.append(font)
        
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"设置中文字体: {chinese_fonts[0]}")
            return True
    
    print("无法设置中文字体，将使用英文标签")
    return False

def detect_steel_change_periods(data_file="output/temporal_analysis/spot_temporal_data.csv", 
                               smoothing_sigma=10.0, 
                               gradient_strength=3.0, 
                               threshold=2.0):
    """
    检测换钢材时间段
    
    Args:
        data_file: 数据文件路径
        smoothing_sigma: 高斯平滑sigma值
        gradient_strength: 梯度增强强度
        threshold: 变化检测阈值
        
    Returns:
        检测结果
    """
    
    print("=== 换钢材时间段检测 ===")
    
    # 读取数据
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    data = df.to_dict('records')
    
    print(f"读取了 {len(data)} 个数据点")
    
    # 设置中文字体
    font_success = setup_chinese_font()
    
    # 提取数据
    time_seconds = np.array([d['time_seconds'] for d in data])
    spot_counts = np.array([d['spot_count'] for d in data])
    spot_densities = np.array([d['spot_density'] for d in data])
    
    # 第一步：高斯平滑
    smoothed_counts = gaussian_filter1d(spot_counts, sigma=smoothing_sigma)
    smoothed_densities = gaussian_filter1d(spot_densities, sigma=smoothing_sigma)
    
    # 第二步：计算梯度（一阶导数）
    gradient_counts = np.gradient(smoothed_counts) * gradient_strength
    gradient_densities = np.gradient(smoothed_densities) * gradient_strength
    
    # 检测变化时间段
    def find_change_periods(time_seconds, gradient_data, threshold):
        above_threshold = np.abs(gradient_data) > threshold
        
        change_periods = []
        in_change = False
        start_time = None
        
        for i, is_change in enumerate(above_threshold):
            if is_change and not in_change:
                # 开始变化
                in_change = True
                start_time = time_seconds[i]
            elif not is_change and in_change:
                # 结束变化
                end_time = time_seconds[i-1]
                duration = end_time - start_time
                
                if duration >= 50.0:  # 最小持续时间50秒
                    # 计算最大变化值
                    start_idx = max(0, i - int(duration/5))
                    end_idx = min(len(gradient_data), i)
                    max_change = np.max(np.abs(gradient_data[start_idx:end_idx]))
                    
                    change_periods.append({
                        'start_time': float(start_time),
                        'end_time': float(end_time),
                        'duration': float(duration),
                        'max_change': float(max_change)
                    })
                
                in_change = False
        
        # 处理最后一个变化段
        if in_change:
            end_time = time_seconds[-1]
            duration = end_time - start_time
            if duration >= 50.0:
                start_idx = max(0, len(time_seconds) - int(duration/5))
                max_change = np.max(np.abs(gradient_data[start_idx:]))
                
                change_periods.append({
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'duration': float(duration),
                    'max_change': float(max_change)
                })
        
        return change_periods
    
    # 检测变化时间段
    change_periods_counts = find_change_periods(time_seconds, gradient_counts, threshold)
    change_periods_densities = find_change_periods(time_seconds, gradient_densities, threshold)
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # 原始数据 vs 平滑数据 - 斑块数量
    ax1.plot(time_seconds, spot_counts, 'b-', linewidth=0.8, alpha=0.3, label='原始数据')
    ax1.plot(time_seconds, smoothed_counts, 'b-', linewidth=2.5, alpha=0.9, label='高斯平滑')
    ax1.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax1.set_ylabel('斑块数量' if font_success else 'Spot Count')
    ax1.set_title('斑块数量 - 高斯平滑' if font_success else 'Spot Count - Gaussian Smoothed')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(time_seconds))
    ax1.legend()
    
    # 梯度增强 - 斑块数量
    ax2.plot(time_seconds, gradient_counts, 'r-', linewidth=2, alpha=0.9)
    ax2.fill_between(time_seconds, gradient_counts, alpha=0.3, color='red')
    
    # 标记变化时间段
    for i, period in enumerate(change_periods_counts):
        ax2.axvspan(period['start_time'], period['end_time'], alpha=0.2, color='yellow',
                   label='疑似换钢材时间段' if i == 0 else "")
    
    ax2.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, label=f'阈值: ±{threshold}')
    ax2.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax2.set_ylabel('梯度增强斑块数量' if font_success else 'Enhanced Gradient Spot Count')
    ax2.set_title('斑块数量梯度增强 - 换钢材检测' if font_success else 'Spot Count Gradient Enhancement - Steel Change Detection')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(time_seconds))
    ax2.legend()
    
    # 原始数据 vs 平滑数据 - 斑块密度
    ax3.plot(time_seconds, spot_densities, 'g-', linewidth=0.8, alpha=0.3, label='原始数据')
    ax3.plot(time_seconds, smoothed_densities, 'g-', linewidth=2.5, alpha=0.9, label='高斯平滑')
    ax3.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax3.set_ylabel('斑块密度' if font_success else 'Spot Density')
    ax3.set_title('斑块密度 - 高斯平滑' if font_success else 'Spot Density - Gaussian Smoothed')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(time_seconds))
    ax3.legend()
    
    # 梯度增强 - 斑块密度
    ax4.plot(time_seconds, gradient_densities, 'm-', linewidth=2, alpha=0.9)
    ax4.fill_between(time_seconds, gradient_densities, alpha=0.3, color='magenta')
    
    # 标记变化时间段
    for i, period in enumerate(change_periods_densities):
        ax4.axvspan(period['start_time'], period['end_time'], alpha=0.2, color='yellow',
                   label='疑似换钢材时间段' if i == 0 else "")
    
    ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, label=f'阈值: ±{threshold}')
    ax4.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
    ax4.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax4.set_ylabel('梯度增强斑块密度' if font_success else 'Enhanced Gradient Spot Density')
    ax4.set_title('斑块密度梯度增强 - 换钢材检测' if font_success else 'Spot Density Gradient Enhancement - Steel Change Detection')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max(time_seconds))
    ax4.legend()
    
    # 添加方法说明
    fig.suptitle(f'换钢材时间段检测 (σ={smoothing_sigma}, 梯度强度={gradient_strength}, 阈值={threshold})', 
                fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # 创建输出目录
    output_dir = "output/steel_change_detection"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图表
    plot_path = os.path.join(output_dir, "steel_change_detection.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"换钢材检测图表已保存: {plot_path}")
    
    # 输出检测结果
    print(f"\n=== 换钢材时间段检测结果 ===")
    print(f"基于斑块数量的检测结果 (共{len(change_periods_counts)}个时间段):")
    for i, period in enumerate(change_periods_counts):
        print(f"  时间段 {i+1}: {period['start_time']:.1f}s - {period['end_time']:.1f}s "
              f"(持续 {period['duration']:.1f}s, 最大变化: {period['max_change']:.2f})")
    
    print(f"\n基于斑块密度的检测结果 (共{len(change_periods_densities)}个时间段):")
    for i, period in enumerate(change_periods_densities):
        print(f"  时间段 {i+1}: {period['start_time']:.1f}s - {period['end_time']:.1f}s "
              f"(持续 {period['duration']:.1f}s, 最大变化: {period['max_change']:.4f})")
    
    # 保存检测结果
    results = {
        'detection_parameters': {
            'smoothing_sigma': smoothing_sigma,
            'gradient_strength': gradient_strength,
            'threshold': threshold
        },
        'change_periods_counts': change_periods_counts,
        'change_periods_densities': change_periods_densities,
        'summary': {
            'total_data_points': len(data),
            'time_span_seconds': float(max(time_seconds)),
            'count_periods': len(change_periods_counts),
            'density_periods': len(change_periods_densities)
        }
    }
    
    results_path = os.path.join(output_dir, "steel_change_detection_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"检测结果已保存: {results_path}")
    
    return results

def main():
    """主函数"""
    # 测试不同的参数组合
    test_configs = [
        {'sigma': 10.0, 'strength': 3.0, 'threshold': 2.0, 'name': '标准参数'},
        {'sigma': 10.0, 'strength': 5.0, 'threshold': 3.0, 'name': '强梯度检测'},
        {'sigma': 8.0, 'strength': 4.0, 'threshold': 2.5, 'name': '中等平滑'},
        {'sigma': 12.0, 'strength': 2.5, 'threshold': 1.8, 'name': '平滑检测'}
    ]
    
    for config in test_configs:
        print(f"\n{'='*50}")
        print(f"测试配置: {config['name']}")
        print(f"参数: σ={config['sigma']}, 强度={config['strength']}, 阈值={config['threshold']}")
        print('='*50)
        
        try:
            results = detect_steel_change_periods(
                smoothing_sigma=config['sigma'],
                gradient_strength=config['strength'],
                threshold=config['threshold']
            )
        except Exception as e:
            print(f"检测失败: {e}")
    
    print(f"\n✅ 换钢材时间段检测完成！")
    print("📁 检测图表保存在 output/steel_change_detection/ 目录下")
    print("\n🎯 使用说明：")
    print("1. 查看生成的图表，黄色区域为疑似换钢材时间段")
    print("2. 对比不同参数配置的检测结果")
    print("3. 结合斑块数量和密度两种指标进行综合判断")
    print("4. 根据实际换钢材时间验证检测精度")

if __name__ == "__main__":
    main()

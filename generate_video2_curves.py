#!/usr/bin/env python3
"""
为第二个视频生成时间序列曲线图
包括平滑滤波和梯度增强分析
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

def setup_chinese_font():
    """设置中文字体"""
    import matplotlib.font_manager as fm
    
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
    
    elif system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("设置Windows中文字体")
        return True
    
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("设置Linux中文字体")
        return True
    
    print("无法设置中文字体，将使用英文标签")
    return False

def apply_smoothing_filters(data, smoothing_method='gaussian', window_size=50, sigma=10.0):
    """
    对时间序列数据应用平滑滤波
    """
    time_seconds = np.array([d['time_seconds'] for d in data])
    spot_counts = np.array([d['spot_count'] for d in data])
    spot_densities = np.array([d['spot_density'] for d in data])
    
    if smoothing_method == 'gaussian':
        smoothed_counts = gaussian_filter1d(spot_counts, sigma=sigma)
        smoothed_densities = gaussian_filter1d(spot_densities, sigma=sigma)
    elif smoothing_method == 'moving_avg':
        smoothed_counts = np.convolve(spot_counts, np.ones(window_size)/window_size, mode='same')
        smoothed_densities = np.convolve(spot_densities, np.ones(window_size)/window_size, mode='same')
    elif smoothing_method == 'savgol':
        from scipy.signal import savgol_filter
        window_length = min(window_size, len(spot_counts))
        if window_length % 2 == 0:
            window_length -= 1
        smoothed_counts = savgol_filter(spot_counts, window_length, 3)
        smoothed_densities = savgol_filter(spot_densities, window_length, 3)
    elif smoothing_method == 'median':
        from scipy.signal import medfilt
        smoothed_counts = medfilt(spot_counts, kernel_size=window_size)
        smoothed_densities = medfilt(spot_densities, kernel_size=window_size)
    else:
        smoothed_counts = gaussian_filter1d(spot_counts, sigma=sigma)
        smoothed_densities = gaussian_filter1d(spot_densities, sigma=sigma)
    
    return time_seconds, smoothed_counts, smoothed_densities

def apply_gradient_enhancement(smoothed_counts, smoothed_densities, gradient_strength=3.0):
    """
    对平滑数据应用梯度增强
    """
    # 计算一阶导数
    gradient_counts = np.gradient(smoothed_counts) * gradient_strength
    gradient_densities = np.gradient(smoothed_densities) * gradient_strength
    
    return gradient_counts, gradient_densities

def create_comprehensive_plots(data, output_dir="data_Video_20250821140339629/analysis"):
    """
    创建全面的时间序列分析图表
    """
    # 设置中文字体
    font_success = setup_chinese_font()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取原始数据
    time_seconds = np.array([d['time_seconds'] for d in data])
    spot_counts = np.array([d['spot_count'] for d in data])
    spot_densities = np.array([d['spot_density'] for d in data])
    
    # 应用不同的平滑方法
    smoothing_configs = [
        {'method': 'gaussian', 'params': {'sigma': 10.0}, 'name': '高斯滤波 (σ=10)'},
        {'method': 'gaussian', 'params': {'sigma': 15.0}, 'name': '高斯滤波 (σ=15)'},
        {'method': 'moving_avg', 'params': {'window_size': 50}, 'name': '移动平均 (窗口=50)'},
        {'method': 'savgol', 'params': {'window_size': 51}, 'name': 'Savitzky-Golay滤波'},
    ]
    
    for i, config in enumerate(smoothing_configs):
        print(f"生成平滑图表 {i+1}/{len(smoothing_configs)}: {config['name']}")
        
        # 应用平滑滤波
        _, smoothed_counts, smoothed_densities = apply_smoothing_filters(
            data, config['method'], **config['params'])
        
        # 应用梯度增强
        gradient_counts, gradient_densities = apply_gradient_enhancement(
            smoothed_counts, smoothed_densities, gradient_strength=3.0)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # 原始数据 vs 平滑数据 - 斑块数量
        ax1.plot(time_seconds, spot_counts, 'b-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax1.plot(time_seconds, smoothed_counts, 'b-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax1.fill_between(time_seconds, smoothed_counts, alpha=0.3, color='blue')
        ax1.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax1.set_ylabel('斑块数量' if font_success else 'Spot Count')
        ax1.set_title('斑块数量 - 平滑分析' if font_success else 'Spot Count - Smoothed')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(time_seconds))
        ax1.legend()
        
        # 梯度增强 - 斑块数量
        ax2.plot(time_seconds, gradient_counts, 'r-', linewidth=2, alpha=0.9)
        ax2.fill_between(time_seconds, gradient_counts, alpha=0.3, color='red')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax2.set_ylabel('梯度增强斑块数量' if font_success else 'Enhanced Gradient Spot Count')
        ax2.set_title('斑块数量梯度增强' if font_success else 'Spot Count Gradient Enhancement')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(time_seconds))
        
        # 原始数据 vs 平滑数据 - 斑块密度
        ax3.plot(time_seconds, spot_densities, 'g-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax3.plot(time_seconds, smoothed_densities, 'g-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax3.fill_between(time_seconds, smoothed_densities, alpha=0.3, color='green')
        ax3.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax3.set_ylabel('斑块密度' if font_success else 'Spot Density')
        ax3.set_title('斑块密度 - 平滑分析' if font_success else 'Spot Density - Smoothed')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max(time_seconds))
        ax3.legend()
        
        # 梯度增强 - 斑块密度
        ax4.plot(time_seconds, gradient_densities, 'm-', linewidth=2, alpha=0.9)
        ax4.fill_between(time_seconds, gradient_densities, alpha=0.3, color='magenta')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax4.set_ylabel('梯度增强斑块密度' if font_success else 'Enhanced Gradient Spot Density')
        ax4.set_title('斑块密度梯度增强' if font_success else 'Spot Density Gradient Enhancement')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, max(time_seconds))
        
        # 添加方法说明
        fig.suptitle(f'Video_20250821140339629 - {config["name"]} (梯度强度=3.0)', 
                    fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 保存图表
        safe_name = config['method'] + '_' + '_'.join([f"{k}_{v}" for k, v in config['params'].items()])
        plot_path = os.path.join(output_dir, f"video2_temporal_analysis_{safe_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  图表已保存: {plot_path}")
    
    # 创建换钢材检测图表
    print("\n生成换钢材检测图表...")
    create_steel_change_detection_plots(data, output_dir)
    
    # 创建统计摘要图表
    print("\n生成统计摘要图表...")
    create_statistics_summary_plots(data, output_dir)

def create_steel_change_detection_plots(data, output_dir):
    """
    创建换钢材时间段检测图表
    """
    font_success = setup_chinese_font()
    
    # 应用平滑和梯度增强
    time_seconds, smoothed_counts, smoothed_densities = apply_smoothing_filters(
        data, 'gaussian', sigma=10.0)
    gradient_counts, gradient_densities = apply_gradient_enhancement(
        smoothed_counts, smoothed_densities, gradient_strength=3.0)
    
    # 检测变化时间段
    threshold = 2.0
    change_periods_counts = detect_change_periods(time_seconds, gradient_counts, threshold)
    change_periods_densities = detect_change_periods(time_seconds, gradient_densities, threshold)
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # 原始数据 vs 平滑数据
    ax1.plot(time_seconds, [d['spot_count'] for d in data], 'b-', linewidth=0.8, alpha=0.3, label='原始数据')
    ax1.plot(time_seconds, smoothed_counts, 'b-', linewidth=2.5, alpha=0.9, label='高斯平滑')
    ax1.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax1.set_ylabel('斑块数量' if font_success else 'Spot Count')
    ax1.set_title('斑块数量 - 高斯平滑' if font_success else 'Spot Count - Gaussian Smoothed')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(time_seconds))
    ax1.legend()
    
    # 梯度增强 - 斑块数量 + 变化时间段
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
    
    # 原始数据 vs 平滑数据 - 密度
    ax3.plot(time_seconds, [d['spot_density'] for d in data], 'g-', linewidth=0.8, alpha=0.3, label='原始数据')
    ax3.plot(time_seconds, smoothed_densities, 'g-', linewidth=2.5, alpha=0.9, label='高斯平滑')
    ax3.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax3.set_ylabel('斑块密度' if font_success else 'Spot Density')
    ax3.set_title('斑块密度 - 高斯平滑' if font_success else 'Spot Density - Gaussian Smoothed')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(time_seconds))
    ax3.legend()
    
    # 梯度增强 - 斑块密度 + 变化时间段
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
    fig.suptitle(f'Video_20250821140339629 - 换钢材时间段检测 (σ=10.0, 梯度强度=3.0, 阈值={threshold})', 
                fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # 保存图表
    plot_path = os.path.join(output_dir, "video2_steel_change_detection.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  换钢材检测图表已保存: {plot_path}")
    
    # 输出检测结果
    print(f"\n=== Video_20250821140339629 换钢材时间段检测结果 ===")
    print(f"基于斑块数量的检测结果 (共{len(change_periods_counts)}个时间段):")
    for i, period in enumerate(change_periods_counts):
        print(f"  时间段 {i+1}: {period['start_time']:.1f}s - {period['end_time']:.1f}s "
              f"(持续 {period['duration']:.1f}s, 最大变化: {period['max_change']:.2f})")
    
    print(f"\n基于斑块密度的检测结果 (共{len(change_periods_densities)}个时间段):")
    for i, period in enumerate(change_periods_densities):
        print(f"  时间段 {i+1}: {period['start_time']:.1f}s - {period['end_time']:.1f}s "
              f"(持续 {period['duration']:.1f}s, 最大变化: {period['max_change']:.4f})")

def detect_change_periods(time_seconds, gradient_data, threshold, min_duration=50.0):
    """
    检测变化时间段
    """
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
            
            if duration >= min_duration:
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
        if duration >= min_duration:
            start_idx = max(0, len(time_seconds) - int(duration/5))
            max_change = np.max(np.abs(gradient_data[start_idx:]))
            
            change_periods.append({
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(duration),
                'max_change': float(max_change)
            })
    
    return change_periods

def create_statistics_summary_plots(data, output_dir):
    """
    创建统计摘要图表
    """
    font_success = setup_chinese_font()
    
    # 提取数据
    time_seconds = np.array([d['time_seconds'] for d in data])
    spot_counts = np.array([d['spot_count'] for d in data])
    spot_densities = np.array([d['spot_density'] for d in data])
    
    # 创建统计图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # 斑块数量统计
    ax1.hist(spot_counts, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(spot_counts), color='red', linestyle='--', linewidth=2, label=f'平均值: {np.mean(spot_counts):.1f}')
    ax1.axvline(np.median(spot_counts), color='green', linestyle='--', linewidth=2, label=f'中位数: {np.median(spot_counts):.1f}')
    ax1.set_xlabel('斑块数量' if font_success else 'Spot Count')
    ax1.set_ylabel('频次' if font_success else 'Frequency')
    ax1.set_title('斑块数量分布直方图' if font_success else 'Spot Count Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 斑块密度统计
    ax2.hist(spot_densities, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(spot_densities), color='red', linestyle='--', linewidth=2, label=f'平均值: {np.mean(spot_densities):.4f}')
    ax2.axvline(np.median(spot_densities), color='blue', linestyle='--', linewidth=2, label=f'中位数: {np.median(spot_densities):.4f}')
    ax2.set_xlabel('斑块密度' if font_success else 'Spot Density')
    ax2.set_ylabel('频次' if font_success else 'Frequency')
    ax2.set_title('斑块密度分布直方图' if font_success else 'Spot Density Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 时间序列散点图
    ax3.scatter(time_seconds, spot_counts, alpha=0.6, s=10, color='blue')
    ax3.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax3.set_ylabel('斑块数量' if font_success else 'Spot Count')
    ax3.set_title('斑块数量时间序列散点图' if font_success else 'Spot Count Time Series Scatter')
    ax3.grid(True, alpha=0.3)
    
    # 斑块数量 vs 密度散点图
    ax4.scatter(spot_counts, spot_densities, alpha=0.6, s=10, color='purple')
    ax4.set_xlabel('斑块数量' if font_success else 'Spot Count')
    ax4.set_ylabel('斑块密度' if font_success else 'Spot Density')
    ax4.set_title('斑块数量 vs 密度散点图' if font_success else 'Spot Count vs Density Scatter')
    ax4.grid(True, alpha=0.3)
    
    # 计算相关系数
    correlation = np.corrcoef(spot_counts, spot_densities)[0, 1]
    ax4.text(0.05, 0.95, f'相关系数: {correlation:.3f}', transform=ax4.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Video_20250821140339629 - 统计摘要分析', fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # 保存图表
    plot_path = os.path.join(output_dir, "video2_statistics_summary.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  统计摘要图表已保存: {plot_path}")
    
    # 输出统计信息
    print(f"\n=== Video_20250821140339629 统计摘要 ===")
    print(f"数据点总数: {len(data)}")
    print(f"时间跨度: {time_seconds[0]:.1f}s - {time_seconds[-1]:.1f}s (总计 {time_seconds[-1]-time_seconds[0]:.1f}s)")
    print(f"\n斑块数量统计:")
    print(f"  平均值: {np.mean(spot_counts):.2f}")
    print(f"  标准差: {np.std(spot_counts):.2f}")
    print(f"  最小值: {np.min(spot_counts):.0f}")
    print(f"  最大值: {np.max(spot_counts):.0f}")
    print(f"  中位数: {np.median(spot_counts):.2f}")
    print(f"\n斑块密度统计:")
    print(f"  平均值: {np.mean(spot_densities):.6f}")
    print(f"  标准差: {np.std(spot_densities):.6f}")
    print(f"  最小值: {np.min(spot_densities):.6f}")
    print(f"  最大值: {np.max(spot_densities):.6f}")
    print(f"  中位数: {np.median(spot_densities):.6f}")
    print(f"\n相关性分析:")
    print(f"  斑块数量与密度相关系数: {correlation:.3f}")

def main():
    """主函数"""
    print("=== 为第二个视频生成时间序列曲线图 ===")
    
    # 读取数据
    data_file = "data_Video_20250821140339629/analysis/spot_temporal_data.csv"
    
    if not os.path.exists(data_file):
        print(f"错误：数据文件不存在: {data_file}")
        print("请先运行完整的视频处理流水线")
        return
    
    # 读取数据
    df = pd.read_csv(data_file)
    data = df.to_dict('records')
    
    print(f"读取了 {len(data)} 个数据点")
    
    # 生成全面的图表
    create_comprehensive_plots(data)
    
    print(f"\n✅ 第二个视频的时间序列曲线图生成完成！")
    print("📁 所有图表保存在 data_Video_20250821140339629/analysis/ 目录下")
    print("\n生成的图表包括:")
    print("1. 多种平滑方法的对比分析")
    print("2. 梯度增强分析")
    print("3. 换钢材时间段检测")
    print("4. 统计摘要分析")

if __name__ == "__main__":
    main()

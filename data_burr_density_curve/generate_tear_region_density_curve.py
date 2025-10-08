#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from scipy import signal

# 设置中文字体
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
    """对时间序列数据应用平滑滤波"""
    time_seconds = data['time_seconds'].values
    burr_region_densities = data['burr_density'].values
    num_burrs = data['num_burrs'].values
    tear_region_densities = data['tear_region_density'].values
    
    if smoothing_method == 'gaussian':
        # 高斯滤波
        smoothed_densities = gaussian_filter1d(burr_region_densities, sigma=sigma)
        smoothed_burrs = gaussian_filter1d(num_burrs, sigma=sigma)
        smoothed_tear_densities = gaussian_filter1d(tear_region_densities, sigma=sigma)
        
    elif smoothing_method == 'moving_avg':
        # 移动平均滤波
        smoothed_densities = np.convolve(burr_region_densities, np.ones(window_size)/window_size, mode='same')
        smoothed_burrs = np.convolve(num_burrs, np.ones(window_size)/window_size, mode='same')
        smoothed_tear_densities = np.convolve(tear_region_densities, np.ones(window_size)/window_size, mode='same')
        
    elif smoothing_method == 'savgol':
        # Savitzky-Golay滤波
        window_length = min(window_size, len(burr_region_densities))
        if window_length % 2 == 0:
            window_length -= 1
        smoothed_densities = signal.savgol_filter(burr_region_densities, window_length, 3)
        smoothed_burrs = signal.savgol_filter(num_burrs, window_length, 3)
        smoothed_tear_densities = signal.savgol_filter(tear_region_densities, window_length, 3)
        
    else:
        # 默认使用高斯滤波
        smoothed_densities = gaussian_filter1d(burr_region_densities, sigma=sigma)
        smoothed_burrs = gaussian_filter1d(num_burrs, sigma=sigma)
        smoothed_tear_densities = gaussian_filter1d(tear_region_densities, sigma=sigma)
    
    return time_seconds, smoothed_densities, smoothed_burrs, smoothed_tear_densities

def create_visualizations(csv_path, output_dir):
    """创建可视化图表"""
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("没有分析结果，无法创建可视化")
        return
    
    # 按时间点排序
    df = df.sort_values('time_seconds')
    
    # 提取数据
    time_seconds = df['time_seconds'].values
    tear_region_densities = df['tear_region_density'].values
    num_burrs = df['num_burrs'].values
    burr_densities = df['burr_density'].values
    
    # 设置中文字体
    font_success = setup_chinese_font()
    
    # 应用平滑滤波
    _, smoothed_burr_densities, smoothed_burrs, smoothed_tear_densities = apply_smoothing_filters(
        df, smoothing_method='gaussian', window_size=50, sigma=10.0)
    
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
    # 撕裂面毛刺密度随时间变化（原始数据+平滑曲线）
    ax1.plot(time_seconds, burr_densities, 'b-', linewidth=0.8, alpha=0.3, label='Raw Data')
    ax1.plot(time_seconds, smoothed_burr_densities, 'b-', linewidth=2.5, alpha=0.9, label='Smoothed Curve')
    ax1.fill_between(time_seconds, smoothed_burr_densities, alpha=0.3, color='blue')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Burr Count Density (burrs/pixel)')
    ax1.set_title('Burr Count Density Over Time (Smoothed)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(time_seconds))
    
    # 添加统计信息
    mean_density = np.mean(burr_densities)
    ax1.axhline(y=mean_density, color='red', linestyle='--', alpha=0.7, 
               label=f'Mean: {mean_density:.4f}')
    ax1.legend()
    
    # 撕裂面毛刺数量随时间变化（原始数据+平滑曲线）
    ax2.plot(time_seconds, num_burrs, 'r-', linewidth=0.8, alpha=0.3, label='Raw Data')
    ax2.plot(time_seconds, smoothed_burrs, 'r-', linewidth=2.5, alpha=0.9, label='Smoothed Curve')
    ax2.fill_between(time_seconds, smoothed_burrs, alpha=0.3, color='red')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Number of Burrs in Tear Region')
    ax2.set_title('Number of Burrs in Tear Region Over Time (Smoothed)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(time_seconds))
    
    # 添加统计信息
    mean_burrs = np.mean(num_burrs)
    ax2.axhline(y=mean_burrs, color='blue', linestyle='--', alpha=0.7,
               label=f'Mean: {mean_burrs:.1f}')
    ax2.legend()
    
    # 撕裂面区域密度随时间变化（原始数据+平滑曲线）
    ax3.plot(time_seconds, tear_region_densities, 'g-', linewidth=0.8, alpha=0.3, label='Raw Data')
    ax3.plot(time_seconds, smoothed_tear_densities, 'g-', linewidth=2.5, alpha=0.9, label='Smoothed Curve')
    ax3.fill_between(time_seconds, smoothed_tear_densities, alpha=0.3, color='green')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Tear Region Density (%)')
    ax3.set_title('Tear Region Density Over Time (Smoothed)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(time_seconds))
    
    # 添加统计信息
    mean_tear_density = np.mean(tear_region_densities)
    ax3.axhline(y=mean_tear_density, color='red', linestyle='--', alpha=0.7,
               label=f'Mean: {mean_tear_density:.2f}%')
    ax3.legend()
    
    # 添加滤波方法说明
    fig.suptitle('Smoothing Method: Gaussian Filter (σ=10, window=50)', fontsize=12, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'burr_density_analysis_with_tear_region.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到: {plot_path}")

def main():
    """主函数"""
    csv_path = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_burr_density_curve/burr_density_analysis.csv"
    output_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_burr_density_curve"
    
    create_visualizations(csv_path, output_dir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
绘制撕裂面纹理密度随时间变化的曲线图
基于已有的纹理分析数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import platform

# 设置中文字体
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

def create_texture_density_curve():
    """创建纹理密度随时间变化的曲线图"""
    
    # 设置中文字体
    font_success = setup_chinese_font()
    
    # 读取纹理分析数据
    data_path = "output/tear_filter_temporal_analysis/step3_filtered_tear_texture/texture_analysis_data.csv"
    
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return
    
    # 读取CSV数据
    df = pd.read_csv(data_path)
    
    # 提取数据
    time_seconds = df['time_seconds'].values
    texture_densities = df['texture_density'].values
    spot_counts = df['spot_count'].values
    spot_densities = df['spot_density'].values
    
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('撕裂面纹理和斑块分析结果' if font_success else 'Tear Surface Texture and Patch Analysis Results', fontsize=16)
    
    # 纹理密度随时间变化
    ax1.plot(time_seconds, texture_densities, 'b-', linewidth=2, alpha=0.8, label='原始数据')
    ax1.fill_between(time_seconds, texture_densities, alpha=0.3, color='blue')
    ax1.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax1.set_ylabel('纹理密度' if font_success else 'Texture Density')
    ax1.set_title('撕裂面纹理密度随时间变化' if font_success else 'Tear Surface Texture Density Over Time')
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_texture_density = np.mean(texture_densities)
    std_texture_density = np.std(texture_densities)
    ax1.axhline(y=mean_texture_density, color='red', linestyle='--', alpha=0.7, 
               label=f'平均值: {mean_texture_density:.4f}')
    ax1.axhline(y=mean_texture_density + std_texture_density, color='orange', linestyle=':', alpha=0.7,
               label=f'+1σ: {mean_texture_density + std_texture_density:.4f}')
    ax1.axhline(y=mean_texture_density - std_texture_density, color='orange', linestyle=':', alpha=0.7,
               label=f'-1σ: {mean_texture_density - std_texture_density:.4f}')
    ax1.legend()
    
    # 斑块数量随时间变化
    ax2.plot(time_seconds, spot_counts, 'r-', linewidth=2, alpha=0.8, label='原始数据')
    ax2.fill_between(time_seconds, spot_counts, alpha=0.3, color='red')
    ax2.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax2.set_ylabel('斑块数量' if font_success else 'Patch Count')
    ax2.set_title('撕裂面斑块数量随时间变化' if font_success else 'Tear Surface Patch Count Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_spot_count = np.mean(spot_counts)
    std_spot_count = np.std(spot_counts)
    ax2.axhline(y=mean_spot_count, color='blue', linestyle='--', alpha=0.7,
               label=f'平均值: {mean_spot_count:.1f}')
    ax2.axhline(y=mean_spot_count + std_spot_count, color='green', linestyle=':', alpha=0.7,
               label=f'+1σ: {mean_spot_count + std_spot_count:.1f}')
    ax2.axhline(y=mean_spot_count - std_spot_count, color='green', linestyle=':', alpha=0.7,
               label=f'-1σ: {mean_spot_count - std_spot_count:.1f}')
    ax2.legend()
    
    # 斑块密度随时间变化
    ax3.plot(time_seconds, spot_densities, 'g-', linewidth=2, alpha=0.8, label='原始数据')
    ax3.fill_between(time_seconds, spot_densities, alpha=0.3, color='green')
    ax3.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax3.set_ylabel('斑块密度' if font_success else 'Patch Density')
    ax3.set_title('撕裂面斑块密度随时间变化' if font_success else 'Tear Surface Patch Density Over Time')
    ax3.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_spot_density = np.mean(spot_densities)
    std_spot_density = np.std(spot_densities)
    ax3.axhline(y=mean_spot_density, color='orange', linestyle='--', alpha=0.7,
               label=f'平均值: {mean_spot_density:.6f}')
    ax3.axhline(y=mean_spot_density + std_spot_density, color='purple', linestyle=':', alpha=0.7,
               label=f'+1σ: {mean_spot_density + std_spot_density:.6f}')
    ax3.axhline(y=mean_spot_density - std_spot_density, color='purple', linestyle=':', alpha=0.7,
               label=f'-1σ: {mean_spot_density - std_spot_density:.6f}')
    ax3.legend()
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = "output/tear_filter_temporal_analysis/step3_filtered_tear_texture"
    os.makedirs(output_dir, exist_ok=True)
    
    curve_path = os.path.join(output_dir, 'texture_density_detailed_curve.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"纹理密度详细曲线图已保存到: {curve_path}")
    
    # 打印统计摘要
    print("\n=== 纹理分析统计摘要 ===")
    print(f"数据点总数: {len(df)}")
    print(f"时间跨度: {time_seconds[0]:.1f} - {time_seconds[-1]:.1f} 秒")
    print(f"纹理密度 - 平均值: {mean_texture_density:.6f}, 标准差: {std_texture_density:.6f}")
    print(f"斑块数量 - 平均值: {mean_spot_count:.2f}, 标准差: {std_spot_count:.2f}")
    print(f"斑块密度 - 平均值: {mean_spot_density:.6f}, 标准差: {std_spot_density:.6f}")
    
    # 创建单独的纹理密度曲线图
    create_texture_only_curve(time_seconds, texture_densities, output_dir, font_success)

def create_texture_only_curve(time_seconds, texture_densities, output_dir, font_success):
    """创建单独的纹理密度曲线图"""
    
    # 创建单独的纹理密度图
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('撕裂面纹理密度随时间变化' if font_success else 'Tear Surface Texture Density Over Time', fontsize=16)
    
    # 绘制纹理密度曲线
    ax.plot(time_seconds, texture_densities, 'b-', linewidth=2.5, alpha=0.9, label='纹理密度')
    ax.fill_between(time_seconds, texture_densities, alpha=0.3, color='blue')
    
    # 计算并绘制平滑曲线
    from scipy.ndimage import gaussian_filter1d
    smoothed_densities = gaussian_filter1d(texture_densities, sigma=5)
    ax.plot(time_seconds, smoothed_densities, 'r-', linewidth=3, alpha=0.8, label='平滑曲线 (σ=5)')
    
    ax.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax.set_ylabel('纹理密度' if font_success else 'Texture Density')
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_texture_density = np.mean(texture_densities)
    std_texture_density = np.std(texture_densities)
    
    ax.axhline(y=mean_texture_density, color='red', linestyle='--', alpha=0.7, 
               label=f'平均值: {mean_texture_density:.4f}')
    ax.axhline(y=mean_texture_density + std_texture_density, color='orange', linestyle=':', alpha=0.7,
               label=f'+1σ: {mean_texture_density + std_texture_density:.4f}')
    ax.axhline(y=mean_texture_density - std_texture_density, color='orange', linestyle=':', alpha=0.7,
               label=f'-1σ: {mean_texture_density - std_texture_density:.4f}')
    
    ax.legend()
    
    plt.tight_layout()
    
    # 保存单独的纹理密度图
    texture_only_path = os.path.join(output_dir, 'texture_density_only_curve.png')
    plt.savefig(texture_only_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"单独纹理密度曲线图已保存到: {texture_only_path}")

if __name__ == "__main__":
    create_texture_density_curve()


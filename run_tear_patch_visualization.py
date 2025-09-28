#!/usr/bin/env python3
"""
单独运行撕裂面斑块可视化步骤
"""

import cv2
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))
# 添加data_shear_split目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_shear_split'))

from feature_extractor import FeatureExtractor
from config import PREPROCESS_CONFIG
from shear_tear_detector import ShearTearDetector
from spot_processor import SpotProcessor

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
            return True
    
    elif system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    
    return False

def extract_frame_info(filename: str) -> int:
    """从文件名提取帧号"""
    try:
        basename = os.path.basename(filename)
        # 对于 filtered_tear_region_frame_XXXXXX.png 格式
        if 'filtered_tear_region_frame_' in basename:
            # 提取 frame_XXXXXX 中的数字
            parts = basename.split('_')
            frame_part = parts[-1].replace('.png', '')  # 去掉 .png 扩展名
            frame_num = int(frame_part)
            return frame_num
        else:
            # 对于 frame_XXXXXX 格式
            frame_num = int(basename.split('_')[1])
            return frame_num
    except (IndexError, ValueError):
        return -1

def filter_tear_region_with_shear_mask(roi_image, segmented_image):
    """
    使用剪切面mask过滤出撕裂面区域
    
    Args:
        roi_image: 原始ROI图像
        segmented_image: 分割结果图像（128=撕裂面，255=剪切面）
        
    Returns:
        过滤后的撕裂面区域图像
    """
    # 从分割结果中提取剪切面mask（值为255的区域）
    shear_mask = (segmented_image == 255).astype(np.uint8) * 255
    
    # 创建撕裂面区域：原图 - 剪切面区域
    # 将剪切面区域设为黑色（0），保留撕裂面区域
    tear_region = roi_image.copy()
    tear_region[shear_mask > 0] = 0  # 将剪切面区域设为黑色
    
    return tear_region, shear_mask

def create_tear_patch_visualization(original_image, filtered_tear_region, spot_result, output_dir, frame_num):
    """
    创建撕裂面斑块可视化图
    
    Args:
        original_image: 原始ROI图像
        filtered_tear_region: 过滤后的撕裂面区域
        spot_result: 斑块检测结果
        output_dir: 输出目录
        frame_num: 帧号
    """
    # 设置中文字体
    font_success = setup_chinese_font()
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'撕裂面斑块可视化 - Frame {frame_num:06d}', fontsize=16)
    
    # 子图1: 原始ROI图像
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('原始ROI图像' if font_success else 'Original ROI Image')
    axes[0, 0].axis('off')
    
    # 子图2: 过滤后的撕裂面区域
    axes[0, 1].imshow(filtered_tear_region, cmap='gray')
    axes[0, 1].set_title('过滤后的撕裂面区域' if font_success else 'Filtered Tear Region')
    axes[0, 1].axis('off')
    
    # 子图3: 斑块检测结果
    if 'spot_image' in spot_result:
        axes[1, 0].imshow(spot_result['spot_image'], cmap='hot')
        axes[1, 0].set_title('斑块检测结果' if font_success else 'Patch Detection Result')
    else:
        axes[1, 0].imshow(filtered_tear_region, cmap='gray')
        axes[1, 0].set_title('无斑块检测结果' if font_success else 'No Patch Detection Result')
    axes[1, 0].axis('off')
    
    # 子图4: 叠加可视化
    # 将原始图像转换为RGB
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # 创建叠加图像
    overlay = original_rgb.copy()
    
    # 在过滤后的撕裂面区域上叠加红色
    tear_mask = filtered_tear_region > 0
    overlay[tear_mask] = [255, 0, 0]  # 红色表示撕裂面区域
    
    # 如果有斑块检测结果，在斑块上叠加绿色
    if 'spot_image' in spot_result:
        spot_mask = spot_result['spot_image'] > 0
        overlay[spot_mask] = [0, 255, 0]  # 绿色表示检测到的斑块
    
    # 混合显示
    alpha = 0.6
    blended = cv2.addWeighted(original_rgb, 1-alpha, overlay, alpha, 0)
    
    axes[1, 1].imshow(blended)
    axes[1, 1].set_title('叠加可视化\n(红: 撕裂面, 绿: 斑块)' if font_success else 'Overlay Visualization\n(Red: Tear, Green: Patches)')
    axes[1, 1].axis('off')
    
    # 添加统计信息
    patch_count = spot_result.get('all_spot_count', 0)
    patch_density = spot_result.get('all_spot_density', 0.0)
    
    stats_text = f"""
统计信息:
斑块数量: {patch_count}
斑块密度: {patch_density:.6f}
撕裂面像素: {np.sum(tear_mask)}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # 保存可视化图
    viz_filename = f"tear_patch_visualization_frame_{frame_num:06d}.png"
    viz_path = os.path.join(output_dir, viz_filename)
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_texture_map(image):
    """
    生成纹理图
    
    Args:
        image: 输入图像
        
    Returns:
        纹理图和纹理密度
    """
    # 使用局部二值模式 (LBP) 生成纹理图
    from skimage.feature import local_binary_pattern
    
    # LBP参数
    radius = 3
    n_points = 8 * radius
    
    # 计算LBP
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    # 计算纹理密度（纹理变化区域的像素比例）
    texture_threshold = np.percentile(lbp, 70)  # 使用70%分位数作为阈值
    texture_mask = lbp > texture_threshold
    texture_density = np.sum(texture_mask) / (image.shape[0] * image.shape[1])
    
    # 将LBP值映射到0-255范围
    texture_map = ((lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-7) * 255).astype(np.uint8)
    
    return texture_map, texture_density

def create_texture_visualization(original_image, texture_map, output_dir, frame_num):
    """
    创建纹理可视化图
    
    Args:
        original_image: 原始图像
        texture_map: 纹理图
        output_dir: 输出目录
        frame_num: 帧号
    """
    # 设置中文字体
    font_success = setup_chinese_font()
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'撕裂面纹理可视化 - Frame {frame_num:06d}', fontsize=16)
    
    # 子图1: 原始图像
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('原始撕裂面区域' if font_success else 'Original Tear Region')
    axes[0, 0].axis('off')
    
    # 子图2: 纹理图
    axes[0, 1].imshow(texture_map, cmap='hot')
    axes[0, 1].set_title('纹理图 (LBP)' if font_success else 'Texture Map (LBP)')
    axes[0, 1].axis('off')
    
    # 子图3: 纹理图叠加
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    texture_overlay = cv2.applyColorMap(texture_map, cv2.COLORMAP_HOT)
    blended = cv2.addWeighted(original_rgb, 0.7, texture_overlay, 0.3, 0)
    axes[1, 0].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('纹理叠加图' if font_success else 'Texture Overlay')
    axes[1, 0].axis('off')
    
    # 子图4: 纹理直方图
    axes[1, 1].hist(texture_map.ravel(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_xlabel('纹理值' if font_success else 'Texture Value')
    axes[1, 1].set_ylabel('频次' if font_success else 'Frequency')
    axes[1, 1].set_title('纹理值分布' if font_success else 'Texture Value Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存可视化图
    viz_filename = f"texture_visualization_frame_{frame_num:06d}.png"
    viz_path = os.path.join(output_dir, viz_filename)
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("=== 撕裂面斑块和纹理分析 ===")
    
    # 设置路径
    input_dir = "output/tear_filter_temporal_analysis/step2_filtered_tear_regions"
    bankuai_output_dir = "output/tear_filter_temporal_analysis/step3_filtered_tear_bankuai"
    texture_output_dir = "output/tear_filter_temporal_analysis/step3_filtered_tear_texture"
    
    # 确保输出目录存在
    os.makedirs(bankuai_output_dir, exist_ok=True)
    os.makedirs(texture_output_dir, exist_ok=True)
    
    # 初始化特征提取器和斑块处理器
    feature_extractor = FeatureExtractor(PREPROCESS_CONFIG)
    spot_processor = SpotProcessor()
    
    # 获取所有过滤后的撕裂面区域图像文件
    input_pattern = os.path.join(input_dir, "filtered_tear_region_frame_*.png")
    input_files = sorted(glob.glob(input_pattern), key=extract_frame_info)
    
    if not input_files:
        print(f"在目录 {input_dir} 中未找到过滤后的撕裂面区域图像文件")
        return
    
    print(f"找到 {len(input_files)} 个过滤后的撕裂面区域图像文件")
    
    # 存储分析结果
    analysis_results = []
    
    for i, input_file in enumerate(tqdm(input_files, desc="处理撕裂面斑块和纹理分析", unit="图像")):
        frame_num = extract_frame_info(input_file)
        if frame_num == -1:
            continue
            
        try:
            # 读取过滤后的撕裂面区域图像
            filtered_tear_region = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
            if filtered_tear_region is None:
                continue
            
            # 1. 生成斑块图（保留原有功能）
            patch_filename = f"tear_bankuai_frame_{frame_num:06d}.png"
            patch_path = os.path.join(bankuai_output_dir, patch_filename)
            spot_result = spot_processor.process_single_roi_spots(input_file, patch_path)
            
            # 2. 生成纹理图
            texture_map, texture_density = generate_texture_map(filtered_tear_region)
            
            # 保存纹理图
            texture_filename = f"tear_texture_frame_{frame_num:06d}.png"
            texture_path = os.path.join(texture_output_dir, texture_filename)
            cv2.imwrite(texture_path, texture_map)
            
            # 创建纹理可视化图
            create_texture_visualization(filtered_tear_region, texture_map, texture_output_dir, frame_num)
            
            # 存储分析结果
            result = {
                'frame_num': frame_num,
                'time_seconds': frame_num * 5,  # 假设每5秒一帧
                'texture_density': texture_density,
                'spot_count': spot_result.get('spot_count', 0) if spot_result['success'] else 0,
                'spot_density': spot_result.get('spot_density', 0.0) if spot_result['success'] else 0.0
            }
            analysis_results.append(result)
            
            # 调试信息（仅前几个图像）
            if frame_num <= 5:
                print(f"Frame {frame_num}: 纹理密度: {texture_density:.6f}, 斑块数量: {result['spot_count']}")
                
        except Exception as e:
            print(f"处理图像 {input_file} 时出错: {e}")
            continue
    
    # 生成纹理密度随时间变化的曲线图
    if analysis_results:
        create_texture_density_curve(analysis_results, texture_output_dir)
    
    print(f"撕裂面斑块图生成完成！结果已保存到: {bankuai_output_dir}")
    print(f"撕裂面纹理图生成完成！结果已保存到: {texture_output_dir}")

def create_texture_density_curve(analysis_results, output_dir):
    """
    创建纹理密度随时间变化的曲线图
    
    Args:
        analysis_results: 分析结果列表
        output_dir: 输出目录
    """
    # 设置中文字体
    font_success = setup_chinese_font()
    
    # 提取数据
    time_seconds = [r['time_seconds'] for r in analysis_results]
    texture_densities = [r['texture_density'] for r in analysis_results]
    spot_counts = [r['spot_count'] for r in analysis_results]
    spot_densities = [r['spot_density'] for r in analysis_results]
    
    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('撕裂面纹理和斑块分析结果' if font_success else 'Tear Surface Texture and Patch Analysis Results', fontsize=16)
    
    # 纹理密度随时间变化
    ax1.plot(time_seconds, texture_densities, 'b-', linewidth=2, alpha=0.8)
    ax1.fill_between(time_seconds, texture_densities, alpha=0.3, color='blue')
    ax1.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax1.set_ylabel('纹理密度' if font_success else 'Texture Density')
    ax1.set_title('纹理密度随时间变化' if font_success else 'Texture Density Over Time')
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_texture_density = np.mean(texture_densities)
    ax1.axhline(y=mean_texture_density, color='red', linestyle='--', alpha=0.7, 
               label=f'平均值: {mean_texture_density:.4f}')
    ax1.legend()
    
    # 斑块数量随时间变化
    ax2.plot(time_seconds, spot_counts, 'r-', linewidth=2, alpha=0.8)
    ax2.fill_between(time_seconds, spot_counts, alpha=0.3, color='red')
    ax2.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax2.set_ylabel('斑块数量' if font_success else 'Patch Count')
    ax2.set_title('斑块数量随时间变化' if font_success else 'Patch Count Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_spot_count = np.mean(spot_counts)
    ax2.axhline(y=mean_spot_count, color='blue', linestyle='--', alpha=0.7,
               label=f'平均值: {mean_spot_count:.1f}')
    ax2.legend()
    
    # 斑块密度随时间变化
    ax3.plot(time_seconds, spot_densities, 'g-', linewidth=2, alpha=0.8)
    ax3.fill_between(time_seconds, spot_densities, alpha=0.3, color='green')
    ax3.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
    ax3.set_ylabel('斑块密度' if font_success else 'Patch Density')
    ax3.set_title('斑块密度随时间变化' if font_success else 'Patch Density Over Time')
    ax3.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_spot_density = np.mean(spot_densities)
    ax3.axhline(y=mean_spot_density, color='orange', linestyle='--', alpha=0.7,
               label=f'平均值: {mean_spot_density:.6f}')
    ax3.legend()
    
    plt.tight_layout()
    
    # 保存图表
    curve_path = os.path.join(output_dir, 'texture_density_analysis_curve.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数据到CSV
    df = pd.DataFrame(analysis_results)
    csv_path = os.path.join(output_dir, 'texture_analysis_data.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"纹理密度分析曲线图已保存到: {curve_path}")
    print(f"分析数据已保存到: {csv_path}")
    
    # 打印统计摘要
    print("\n=== 纹理分析统计摘要 ===")
    print(f"数据点总数: {len(analysis_results)}")
    print(f"时间跨度: {time_seconds[0]:.1f} - {time_seconds[-1]:.1f} 秒")
    print(f"纹理密度 - 平均值: {mean_texture_density:.6f}, 标准差: {np.std(texture_densities):.6f}")
    print(f"斑块数量 - 平均值: {mean_spot_count:.2f}, 标准差: {np.std(spot_counts):.2f}")
    print(f"斑块密度 - 平均值: {mean_spot_density:.6f}, 标准差: {np.std(spot_densities):.6f}")

if __name__ == "__main__":
    main()

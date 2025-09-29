#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from scipy import ndimage
from skimage.measure import label, regionprops
import pandas as pd
from tqdm import tqdm
import sys
import platform
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_process'))

from feature_extractor import FeatureExtractor
from config import PREPROCESS_CONFIG

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

class TearDensityAnalyzer:
    """撕裂面密度分析器"""
    
    def __init__(self):
        self.results = []
        self.feature_extractor = FeatureExtractor(PREPROCESS_CONFIG)
        
    def analyze_tear_density(self, image_path, tear_mask):
        """分析撕裂面斑块数量和密度 - 先用撕裂面mask过滤原图，再检测斑块"""
        
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        
        # 确保mask是二值图像
        if tear_mask.dtype != np.uint8:
            tear_mask = (tear_mask > 0).astype(np.uint8) * 255
        
        # 使用撕裂面mask过滤原图，只保留撕裂面区域
        tear_region = cv2.bitwise_and(image, image, mask=tear_mask)
        
        # 在过滤后的撕裂面区域上使用FeatureExtractor检测斑块
        spot_result = self.feature_extractor.detect_all_white_spots(tear_region)
        
        # 计算撕裂面区域密度
        total_pixels = image.shape[0] * image.shape[1]
        tear_pixels = np.sum(tear_mask > 0)
        tear_region_density = (tear_pixels / total_pixels) * 100
        
        # 从文件名提取时间点
        frame_num = self.extract_frame_info(image_path)
        time_seconds = frame_num * 5 if frame_num > 0 else 0  # 假设每5秒一帧
        
        return {
            'frame_num': frame_num,
            'time_seconds': time_seconds,
            'tear_region_density': tear_region_density,  # 撕裂面区域占整个图像的比例
            'num_patches': spot_result.get('all_spot_count', 0),  # 撕裂面区域内的斑块数量
            'spot_density': spot_result.get('all_spot_density', 0.0),  # 撕裂面区域内的斑块密度
            'image_shape': image.shape,
            'image_path': image_path
        }
    
    def apply_smoothing_filters(self, data, 
                               smoothing_method: str = 'gaussian',
                               window_size: int = 50,
                               sigma: float = 10.0):
        """对时间序列数据应用平滑滤波"""
        time_seconds = np.array([d['time_seconds'] for d in data])
        tear_region_densities = np.array([d['tear_region_density'] for d in data])
        num_patches = np.array([d['num_patches'] for d in data])
        
        if smoothing_method == 'gaussian':
            # 高斯滤波
            smoothed_densities = gaussian_filter1d(tear_region_densities, sigma=sigma)
            smoothed_patches = gaussian_filter1d(num_patches, sigma=sigma)
            
        elif smoothing_method == 'moving_avg':
            # 移动平均滤波
            smoothed_densities = np.convolve(tear_region_densities, np.ones(window_size)/window_size, mode='same')
            smoothed_patches = np.convolve(num_patches, np.ones(window_size)/window_size, mode='same')
            
        elif smoothing_method == 'savgol':
            # Savitzky-Golay滤波
            window_length = min(window_size, len(tear_region_densities))
            if window_length % 2 == 0:
                window_length -= 1
            smoothed_densities = signal.savgol_filter(tear_region_densities, window_length, 3)
            smoothed_patches = signal.savgol_filter(num_patches, window_length, 3)
            
        else:
            # 默认使用高斯滤波
            smoothed_densities = gaussian_filter1d(tear_region_densities, sigma=sigma)
            smoothed_patches = gaussian_filter1d(num_patches, sigma=sigma)
        
        return time_seconds, smoothed_densities, smoothed_patches
    
    def extract_frame_info(self, filename: str) -> int:
        """从文件名提取帧号"""
        try:
            basename = os.path.basename(filename)
            # 提取frame_XXXXXX中的数字
            frame_num = int(basename.split('_')[1])
            return frame_num
        except (IndexError, ValueError):
            return -1
    
    def process_images(self, roi_dir, output_dir, use_contour_method=True):
        """按步骤处理所有图像"""
        
        print("开始分析撕裂面密度...")
        print("=" * 60)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建各步骤保存目录
        step1_dir = os.path.join(output_dir, 'step1_tear_masks')
        step2_dir = os.path.join(output_dir, 'step2_tear_regions')
        step3_dir = os.path.join(output_dir, 'step3_patch_analysis')
        
        os.makedirs(step1_dir, exist_ok=True)
        os.makedirs(step2_dir, exist_ok=True)
        os.makedirs(step3_dir, exist_ok=True)
        
        # 获取所有ROI图像文件
        import glob
        roi_pattern = os.path.join(roi_dir, "*_roi.png")
        image_files = sorted(glob.glob(roi_pattern), key=self.extract_frame_info)
        
        if not image_files:
            print(f"在目录 {roi_dir} 中未找到ROI图像文件")
            return
        
        print(f"找到 {len(image_files)} 个ROI图像文件")
        
        # 使用检测器获取撕裂面mask
        import sys
        sys.path.append('/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_split')
        from shear_tear_detector import ShearTearDetector
        # 默认使用新的等高线方法，可通过参数控制
        detector = ShearTearDetector(use_contour_method=use_contour_method)
        
        # 第一步：生成撕裂面mask（before fill + after fill）
        print("\n第一步：生成撕裂面mask...")
        detection_results = {}  # 缓存检测结果，避免重复计算
        
        for image_path in tqdm(image_files, desc="生成撕裂面mask", unit="图像"):
            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
                
            # 检测撕裂面
            result = detector.detect_surfaces(image, visualize=False)
            if result and 'segmented_image' in result:
                frame_num = self.extract_frame_info(image_path)
                
                # 缓存检测结果
                detection_results[frame_num] = result
                
                # 根据使用的方法保存不同的mask
                if use_contour_method:
                    # 新方法：使用等高线方法的结果
                    if 'tear_mask' in result.get('intermediate_results', {}):
                        tear_mask = result['intermediate_results']['tear_mask']
                        # 保存等高线方法生成的撕裂面mask
                        contour_mask = tear_mask.astype(np.uint8) * 255
                        contour_filename = f"tear_mask_contour_frame_{frame_num:06d}.png"
                        contour_path = os.path.join(step1_dir, contour_filename)
                        cv2.imwrite(contour_path, contour_mask)
                        
                        # 同时保存分割结果
                        segmented_image = result['segmented_image']
                        after_fill_mask = (segmented_image == 128).astype(np.uint8) * 255
                        after_fill_filename = f"tear_mask_after_fill_frame_{frame_num:06d}.png"
                        after_fill_path = os.path.join(step1_dir, after_fill_filename)
                        cv2.imwrite(after_fill_path, after_fill_mask)
                else:
                    # 老方法：保存before fill和after fill mask
                    if 'tear_mask_original' in result:
                        before_fill_mask = result['tear_mask_original'].astype(np.uint8) * 255
                        before_fill_filename = f"tear_mask_before_fill_frame_{frame_num:06d}.png"
                        before_fill_path = os.path.join(step1_dir, before_fill_filename)
                        cv2.imwrite(before_fill_path, before_fill_mask)
                    
                    # 保存after fill mask
                    segmented_image = result['segmented_image']
                    after_fill_mask = (segmented_image == 128).astype(np.uint8) * 255
                    after_fill_filename = f"tear_mask_after_fill_frame_{frame_num:06d}.png"
                    after_fill_path = os.path.join(step1_dir, after_fill_filename)
                    cv2.imwrite(after_fill_path, after_fill_mask)
        
        print(f"第一步完成，撕裂面mask已保存到: {step1_dir}")
        
        # 第二步：应用撕裂面mask过滤出原图的撕裂面区域，并提取斑块图
        print("\n第二步：提取撕裂面区域和斑块图...")
        for image_path in tqdm(image_files, desc="提取撕裂面区域", unit="图像"):
            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
                
            frame_num = self.extract_frame_info(image_path)
            
            # 检查是否已有检测结果
            if frame_num in detection_results:
                result = detection_results[frame_num]
                # 根据使用的方法提取撕裂面mask
                if use_contour_method:
                    # 新方法：从等高线结果中提取撕裂面mask
                    if 'tear_mask' in result.get('intermediate_results', {}):
                        tear_mask = result['intermediate_results']['tear_mask'].astype(np.uint8) * 255
                    else:
                        # 回退到分割结果
                        segmented_image = result['segmented_image']
                        tear_mask = (segmented_image == 128).astype(np.uint8) * 255
                else:
                    # 老方法：从分割结果中提取撕裂面mask
                    segmented_image = result['segmented_image']
                    tear_mask = (segmented_image == 128).astype(np.uint8) * 255
            else:
                # 如果没有缓存结果，则从第一步保存的mask文件中读取
                if use_contour_method:
                    # 新方法：优先读取等高线mask
                    contour_filename = f"tear_mask_contour_frame_{frame_num:06d}.png"
                    contour_path = os.path.join(step1_dir, contour_filename)
                    if os.path.exists(contour_path):
                        tear_mask = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        # 回退到after fill mask
                        after_fill_filename = f"tear_mask_after_fill_frame_{frame_num:06d}.png"
                        after_fill_path = os.path.join(step1_dir, after_fill_filename)
                        if os.path.exists(after_fill_path):
                            tear_mask = cv2.imread(after_fill_path, cv2.IMREAD_GRAYSCALE)
                        else:
                            print(f"警告：未找到帧 {frame_num} 的撕裂面mask，跳过处理")
                            continue
                else:
                    # 老方法：读取after fill mask
                    after_fill_filename = f"tear_mask_after_fill_frame_{frame_num:06d}.png"
                    after_fill_path = os.path.join(step1_dir, after_fill_filename)
                    if os.path.exists(after_fill_path):
                        tear_mask = cv2.imread(after_fill_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        print(f"警告：未找到帧 {frame_num} 的撕裂面mask，跳过处理")
                        continue
            
            # 应用mask过滤出撕裂面区域
            tear_region = cv2.bitwise_and(image, image, mask=tear_mask)
            
            # 保存撕裂面区域
            region_filename = f"tear_region_frame_{frame_num:06d}.png"
            region_path = os.path.join(step2_dir, region_filename)
            cv2.imwrite(region_path, tear_region)
            
            # 使用FeatureExtractor检测斑块
            spot_result = self.feature_extractor.detect_all_white_spots(tear_region)
            
            # 保存斑块检测结果图
            if 'spot_image' in spot_result:
                patch_filename = f"tear_patches_frame_{frame_num:06d}.png"
                patch_path = os.path.join(step2_dir, patch_filename)
                cv2.imwrite(patch_path, spot_result['spot_image'])
            
            # 分析撕裂面密度
            analysis_result = self.analyze_tear_density(image_path, tear_mask)
            if analysis_result:
                self.results.append(analysis_result)
        
        print(f"第二步完成，撕裂面区域和斑块图已保存到: {step2_dir}")
        
        # 第三步：计算斑块数量和密度，生成变化曲线图
        print("\n第三步：计算斑块数量和密度，生成变化曲线图...")
        
        # 保存结果
        self.save_results(output_dir)
        
        # 生成可视化
        self.create_visualizations(output_dir)
        
        print(f"第三步完成，分析结果和曲线图已保存到: {step3_dir}")
        
        print("=" * 60)
        print("撕裂面密度分析完成!")
        print(f"所有结果已保存到: {output_dir}")
    
    def save_results(self, output_dir):
        """保存分析结果"""
        
        # 保存JSON结果（转换numpy类型为Python原生类型）
        json_results = []
        for result in self.results:
            json_result = {}
            for key, value in result.items():
                if key == 'patch_areas':  # 跳过patch_areas列表
                    continue
                elif isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                elif hasattr(value, 'item'):  # numpy标量
                    json_result[key] = value.item()
                else:
                    json_result[key] = value
            json_results.append(json_result)
        
        json_path = os.path.join(output_dir, 'tear_density_analysis.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV结果
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = os.path.join(output_dir, 'tear_density_analysis.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"分析结果已保存到: {json_path}")
        print(f"CSV结果已保存到: {csv_path}")
    
    def create_visualizations(self, output_dir):
        """创建可视化图表"""
        
        if not self.results:
            print("没有分析结果，无法创建可视化")
            return
        
        # 按时间点排序
        sorted_results = sorted(self.results, key=lambda x: x['time_seconds'])
        
        # 提取数据
        time_seconds = np.array([r['time_seconds'] for r in sorted_results])
        tear_region_densities = np.array([r['tear_region_density'] for r in sorted_results])
        num_patches = np.array([r['num_patches'] for r in sorted_results])
        spot_densities = np.array([r['spot_density'] for r in sorted_results])
        
        # 设置中文字体
        font_success = setup_chinese_font()
        
        # 应用平滑滤波
        _, smoothed_region_densities, smoothed_patches = self.apply_smoothing_filters(
            sorted_results, smoothing_method='gaussian', window_size=50, sigma=10.0)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 撕裂面区域密度随时间变化（原始数据+平滑曲线）
        ax1.plot(time_seconds, tear_region_densities, 'b-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax1.plot(time_seconds, smoothed_region_densities, 'b-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax1.fill_between(time_seconds, smoothed_region_densities, alpha=0.3, color='blue')
        ax1.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax1.set_ylabel('撕裂面区域密度 (%)' if font_success else 'Tear Region Density (%)')
        ax1.set_title('撕裂面区域密度随时间变化 (平滑滤波)' if font_success else 'Tear Region Density Over Time (Smoothed)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_density = np.mean(tear_region_densities)
        ax1.axhline(y=mean_density, color='red', linestyle='--', alpha=0.7, 
                   label=f'平均值: {mean_density:.2f}%')
        ax1.legend()
        
        # 撕裂面斑块数量随时间变化（原始数据+平滑曲线）
        ax2.plot(time_seconds, num_patches, 'r-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax2.plot(time_seconds, smoothed_patches, 'r-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax2.fill_between(time_seconds, smoothed_patches, alpha=0.3, color='red')
        ax2.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax2.set_ylabel('撕裂面斑块数量' if font_success else 'Tear Patch Count')
        ax2.set_title('撕裂面斑块数量随时间变化 (平滑滤波)' if font_success else 'Tear Patch Count Over Time (Smoothed)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_patches = np.mean(num_patches)
        ax2.axhline(y=mean_patches, color='blue', linestyle='--', alpha=0.7,
                   label=f'平均值: {mean_patches:.1f}')
        ax2.legend()
        
        # 添加滤波方法说明
        fig.suptitle('平滑方法: 高斯滤波 (σ=10, 窗口=50)', fontsize=12, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 保存图表
        plot_path = os.path.join(output_dir, 'tear_density_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到: {plot_path}")
        
        # 创建综合统计图
        self.create_summary_plot(output_dir, sorted_results)
    
    def create_summary_plot(self, output_dir, sorted_results):
        """创建综合统计图"""
        
        # 提取数据
        time_points = [r['time_seconds'] for r in sorted_results]
        tear_densities = [r['tear_region_density'] for r in sorted_results]
        num_patches = [r['num_patches'] for r in sorted_results]
        
        # 创建综合图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Tear Surface Analysis Summary', fontsize=16)
        
        # 密度和斑块数量对比
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(time_points, tear_densities, 'b-o', linewidth=2, markersize=6, label='Density (%)')
        line2 = ax1_twin.plot(time_points, num_patches, 'r-s', linewidth=2, markersize=6, label='Number of Patches')
        
        ax1.set_xlabel('Time Point')
        ax1.set_ylabel('Tear Density (%)', color='b')
        ax1_twin.set_ylabel('Number of Patches', color='r')
        ax1.set_title('Tear Density vs Number of Patches')
        ax1.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 斑块面积分布
        patch_areas_all = []
        for r in sorted_results:
            if 'patch_areas' in r:
                patch_areas_all.extend(r['patch_areas'])
        
        if patch_areas_all:
            ax2.hist(patch_areas_all, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Patch Area (pixels)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Patch Areas')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存综合图
        summary_path = os.path.join(output_dir, 'tear_density_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"综合统计图已保存到: {summary_path}")

def main():
    """主函数"""
    import sys
    
    # 设置路径
    roi_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data/roi_imgs"
    output_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_tear_density_curve"
    
    if len(sys.argv) > 1:
        roi_dir = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    # 创建分析器
    analyzer = TearDensityAnalyzer()
    
    # 处理图像（默认使用新方法）
    analyzer.process_images(roi_dir, output_dir, use_contour_method=True)

if __name__ == "__main__":
    main()

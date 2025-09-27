#!/usr/bin/env python3
"""
撕裂面斑块数量和密度时间序列分析（使用剪切面mask过滤撕裂面）
主要功能：
1. 使用剪切面mask过滤出撕裂面区域
2. 从撕裂面区域中提取斑块数量和密度数据
3. 绘制斑块数量和密度随时间变化的曲线图
4. 生成时间序列分析报告
"""

import cv2
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
import platform
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))
# 添加data_shear_split目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_shear_split'))

from feature_extractor import FeatureExtractor
from config import PREPROCESS_CONFIG
from shear_tear_detector import ShearTearDetector

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

class TearFilterTemporalAnalyzer:
    """撕裂面过滤时间序列分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.feature_extractor = FeatureExtractor(PREPROCESS_CONFIG)
        self.shear_tear_detector = ShearTearDetector()
        self.data = []
        
    def extract_frame_info(self, filename: str) -> int:
        """从文件名提取帧号"""
        try:
            basename = os.path.basename(filename)
            # 提取frame_XXXXXX中的数字
            frame_num = int(basename.split('_')[1])
            return frame_num
        except (IndexError, ValueError):
            return -1
    
    def filter_tear_region_with_shear_mask(self, roi_image, segmented_image):
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
    
    def analyze_roi_spots_with_tear_filter(self, roi_dir: str, output_dir: str = "output") -> List[Dict[str, Any]]:
        """
        分析ROI图像的撕裂面斑块特征（使用剪切面过滤）
        
        Args:
            roi_dir: ROI图像目录路径
            output_dir: 输出目录
            
        Returns:
            斑块分析结果列表
        """
        print("开始分析ROI图像的撕裂面斑块特征（使用剪切面过滤）...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建各步骤保存目录
        step1_dir = os.path.join(output_dir, 'step1_shear_tear_masks')
        step2_dir = os.path.join(output_dir, 'step2_filtered_tear_regions')
        step3_dir = os.path.join(output_dir, 'step3_tear_patch_analysis')
        
        os.makedirs(step1_dir, exist_ok=True)
        os.makedirs(step2_dir, exist_ok=True)
        os.makedirs(step3_dir, exist_ok=True)
        
        # 获取所有ROI图像文件
        roi_pattern = os.path.join(roi_dir, "*_roi.png")
        roi_files = sorted(glob.glob(roi_pattern), key=self.extract_frame_info)
        
        if not roi_files:
            print(f"在目录 {roi_dir} 中未找到ROI图像文件")
            return []
        
        print(f"找到 {len(roi_files)} 个ROI图像文件")
        
        results = []
        
        # 第一步：生成剪切面和撕裂面mask
        print("\n第一步：生成剪切面和撕裂面mask...")
        for roi_file in tqdm(roi_files, desc="生成剪切面撕裂面mask", unit="图像"):
            frame_num = self.extract_frame_info(roi_file)
            if frame_num == -1:
                continue
                
            try:
                # 读取ROI图像
                roi_image = cv2.imread(roi_file, cv2.IMREAD_GRAYSCALE)
                if roi_image is None:
                    continue
                
                # 使用ShearTearDetector检测剪切面和撕裂面
                result = self.shear_tear_detector.detect_surfaces(roi_image, visualize=False)
                if result and 'segmented_image' in result:
                    segmented_image = result['segmented_image']
                    
                    # 保存剪切面mask
                    shear_mask = (segmented_image == 255).astype(np.uint8) * 255
                    shear_filename = f"shear_mask_frame_{frame_num:06d}.png"
                    shear_path = os.path.join(step1_dir, shear_filename)
                    cv2.imwrite(shear_path, shear_mask)
                    
                    # 保存撕裂面mask
                    tear_mask = (segmented_image == 128).astype(np.uint8) * 255
                    tear_filename = f"tear_mask_frame_{frame_num:06d}.png"
                    tear_path = os.path.join(step1_dir, tear_filename)
                    cv2.imwrite(tear_path, tear_mask)
                    
            except Exception as e:
                print(f"生成mask时出错 {roi_file}: {e}")
                continue
        
        print(f"第一步完成，剪切面和撕裂面mask已保存到: {step1_dir}")
        
        # 第二步：使用剪切面mask过滤出撕裂面区域
        print("\n第二步：过滤撕裂面区域...")
        for roi_file in tqdm(roi_files, desc="过滤撕裂面区域", unit="图像"):
            frame_num = self.extract_frame_info(roi_file)
            if frame_num == -1:
                continue
                
            try:
                # 读取ROI图像
                roi_image = cv2.imread(roi_file, cv2.IMREAD_GRAYSCALE)
                if roi_image is None:
                    continue
                
                # 使用ShearTearDetector检测剪切面和撕裂面
                result = self.shear_tear_detector.detect_surfaces(roi_image, visualize=False)
                if result and 'segmented_image' in result:
                    segmented_image = result['segmented_image']
                    
                    # 使用剪切面mask过滤出撕裂面区域
                    filtered_tear_region, shear_mask = self.filter_tear_region_with_shear_mask(roi_image, segmented_image)
                    
                    # 保存过滤后的撕裂面区域
                    filtered_filename = f"filtered_tear_region_frame_{frame_num:06d}.png"
                    filtered_path = os.path.join(step2_dir, filtered_filename)
                    cv2.imwrite(filtered_path, filtered_tear_region)
                    
                    # 检测斑块
                    spot_result = self.feature_extractor.detect_all_white_spots(filtered_tear_region)
                    
                    # 保存斑块检测结果图
                    if 'spot_image' in spot_result:
                        patch_filename = f"tear_patches_frame_{frame_num:06d}.png"
                        patch_path = os.path.join(step2_dir, patch_filename)
                        cv2.imwrite(patch_path, spot_result['spot_image'])
                    
                    # 提取关键信息
                    result_data = {
                        'frame_num': frame_num,
                        'time_seconds': frame_num * 5,  # 假设每5秒一帧
                        'spot_count': spot_result.get('all_spot_count', 0),
                        'spot_density': spot_result.get('all_spot_density', 0.0),
                        'image_shape': roi_image.shape,
                        'roi_file': roi_file
                    }
                    
                    results.append(result_data)
                    
            except Exception as e:
                print(f"分析ROI图像 {roi_file} 时出错: {e}")
                continue
        
        print(f"第二步完成，过滤后的撕裂面区域和斑块图已保存到: {step2_dir}")
        
        # 第三步：生成时间序列分析
        print("\n第三步：生成时间序列分析...")
        if results:
            # 创建时间序列图表
            plot_path = self.create_temporal_plots(results, step3_dir)
            
            # 保存数据
            self.save_data_to_csv(results, step3_dir)
            
            # 输出统计摘要
            self.print_statistics_summary(results)
            
            print(f"第三步完成，分析结果和曲线图已保存到: {step3_dir}")
        
        print(f"斑块分析完成，成功分析 {len(results)} 个ROI图像")
        return results
    
    def apply_smoothing_filters(self, data: List[Dict[str, Any]], 
                               smoothing_method: str = 'gaussian',
                               window_size: int = 50,
                               sigma: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对时间序列数据应用平滑滤波
        
        Args:
            data: 斑块分析数据
            smoothing_method: 平滑方法 ('gaussian', 'moving_avg', 'savgol', 'median')
            window_size: 滤波窗口大小
            sigma: 高斯滤波的标准差
            
        Returns:
            时间序列、平滑后的斑块数量、平滑后的斑块密度
        """
        time_seconds = np.array([d['time_seconds'] for d in data])
        spot_counts = np.array([d['spot_count'] for d in data])
        spot_densities = np.array([d['spot_density'] for d in data])
        
        if smoothing_method == 'gaussian':
            # 高斯滤波
            smoothed_counts = gaussian_filter1d(spot_counts, sigma=sigma)
            smoothed_densities = gaussian_filter1d(spot_densities, sigma=sigma)
            
        elif smoothing_method == 'moving_avg':
            # 移动平均滤波
            smoothed_counts = np.convolve(spot_counts, np.ones(window_size)/window_size, mode='same')
            smoothed_densities = np.convolve(spot_densities, np.ones(window_size)/window_size, mode='same')
            
        elif smoothing_method == 'savgol':
            # Savitzky-Golay滤波
            window_length = min(window_size, len(spot_counts))
            if window_length % 2 == 0:
                window_length -= 1
            smoothed_counts = signal.savgol_filter(spot_counts, window_length, 3)
            smoothed_densities = signal.savgol_filter(spot_densities, window_length, 3)
            
        elif smoothing_method == 'median':
            # 中值滤波
            smoothed_counts = signal.medfilt(spot_counts, kernel_size=window_size)
            smoothed_densities = signal.medfilt(spot_densities, kernel_size=window_size)
            
        else:
            # 默认使用高斯滤波
            smoothed_counts = gaussian_filter1d(spot_counts, sigma=sigma)
            smoothed_densities = gaussian_filter1d(spot_densities, sigma=sigma)
        
        return time_seconds, smoothed_counts, smoothed_densities
    
    def create_temporal_plots(self, data: List[Dict[str, Any]], output_dir: str = "output",
                            smoothing_method: str = 'gaussian', window_size: int = 50, sigma: float = 10.0):
        """
        创建时间序列图表（带平滑滤波）
        
        Args:
            data: 斑块分析数据
            output_dir: 输出目录
            smoothing_method: 平滑方法
            window_size: 滤波窗口大小
            sigma: 高斯滤波标准差
        """
        if not data:
            print("没有数据可以绘制")
            return
        
        # 设置中文字体
        font_success = setup_chinese_font()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取原始数据
        time_seconds = np.array([d['time_seconds'] for d in data])
        spot_counts = np.array([d['spot_count'] for d in data])
        spot_densities = np.array([d['spot_density'] for d in data])
        
        # 应用平滑滤波
        _, smoothed_counts, smoothed_densities = self.apply_smoothing_filters(
            data, smoothing_method, window_size, sigma)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 撕裂面斑块数量随时间变化（原始数据+平滑曲线）
        ax1.plot(time_seconds, spot_counts, 'b-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax1.plot(time_seconds, smoothed_counts, 'b-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax1.fill_between(time_seconds, smoothed_counts, alpha=0.3, color='blue')
        ax1.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax1.set_ylabel('撕裂面斑块数量' if font_success else 'Tear Spot Count')
        ax1.set_title('撕裂面斑块数量随时间变化 (平滑滤波)' if font_success else 'Tear Spot Count Over Time (Smoothed)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_count = np.mean(spot_counts)
        max_count = np.max(smoothed_counts)
        ax1.axhline(y=mean_count, color='red', linestyle='--', alpha=0.7, 
                   label=f'平均值: {mean_count:.1f}')
        ax1.legend()
        
        # 撕裂面斑块密度随时间变化（原始数据+平滑曲线）
        ax2.plot(time_seconds, spot_densities, 'r-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax2.plot(time_seconds, smoothed_densities, 'r-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax2.fill_between(time_seconds, smoothed_densities, alpha=0.3, color='red')
        ax2.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax2.set_ylabel('撕裂面斑块密度' if font_success else 'Tear Spot Density')
        ax2.set_title('撕裂面斑块密度随时间变化 (平滑滤波)' if font_success else 'Tear Spot Density Over Time (Smoothed)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_density = np.mean(spot_densities)
        max_density = np.max(smoothed_densities)
        ax2.axhline(y=mean_density, color='blue', linestyle='--', alpha=0.7,
                   label=f'平均值: {mean_density:.4f}')
        ax2.legend()
        
        # 添加滤波方法说明
        method_names = {
            'gaussian': '高斯滤波',
            'moving_avg': '移动平均',
            'savgol': 'Savitzky-Golay滤波',
            'median': '中值滤波'
        }
        method_name = method_names.get(smoothing_method, smoothing_method)
        fig.suptitle(f'平滑方法: {method_name} (σ={sigma}, 窗口={window_size})', fontsize=12, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 保存图表
        plot_path = os.path.join(output_dir, "tear_spot_temporal_analysis_smoothed.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"平滑时间序列图表已保存: {plot_path}")
        
        # 创建统计摘要图
        self.create_statistics_summary(data, output_dir)
        
        return plot_path
    
    def create_statistics_summary(self, data: List[Dict[str, Any]], output_dir: str):
        """
        创建统计摘要图表
        
        Args:
            data: 斑块分析数据
            output_dir: 输出目录
        """
        if not data:
            return
        
        # 设置中文字体
        font_success = setup_chinese_font()
        
        # 提取数据
        spot_counts = [d['spot_count'] for d in data]
        spot_densities = [d['spot_density'] for d in data]
        
        # 创建统计图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 撕裂面斑块数量分布直方图
        ax1.hist(spot_counts, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('撕裂面斑块数量' if font_success else 'Tear Spot Count')
        ax1.set_ylabel('频次' if font_success else 'Frequency')
        ax1.set_title('撕裂面斑块数量分布' if font_success else 'Tear Spot Count Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 撕裂面斑块密度分布直方图
        ax2.hist(spot_densities, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax2.set_xlabel('撕裂面斑块密度' if font_success else 'Tear Spot Density')
        ax2.set_ylabel('频次' if font_success else 'Frequency')
        ax2.set_title('撕裂面斑块密度分布' if font_success else 'Tear Spot Density Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 撕裂面斑块数量与密度的散点图
        ax3.scatter(spot_counts, spot_densities, alpha=0.6, color='green')
        ax3.set_xlabel('撕裂面斑块数量' if font_success else 'Tear Spot Count')
        ax3.set_ylabel('撕裂面斑块密度' if font_success else 'Tear Spot Density')
        ax3.set_title('撕裂面斑块数量与密度关系' if font_success else 'Tear Spot Count vs Density')
        ax3.grid(True, alpha=0.3)
        
        # 计算相关系数
        correlation = np.corrcoef(spot_counts, spot_densities)[0, 1]
        ax3.text(0.05, 0.95, f'相关系数: {correlation:.3f}', 
                transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 统计信息表格
        ax4.axis('off')
        stats_text = f"""
统计摘要:

撕裂面斑块数量:
  平均值: {np.mean(spot_counts):.2f}
  标准差: {np.std(spot_counts):.2f}
  最小值: {np.min(spot_counts)}
  最大值: {np.max(spot_counts)}
  中位数: {np.median(spot_counts):.2f}

撕裂面斑块密度:
  平均值: {np.mean(spot_densities):.6f}
  标准差: {np.std(spot_densities):.6f}
  最小值: {np.min(spot_densities):.6f}
  最大值: {np.max(spot_densities):.6f}
  中位数: {np.median(spot_densities):.6f}

数据点总数: {len(data)}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存统计图表
        stats_path = os.path.join(output_dir, "tear_spot_statistics_summary.png")
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"统计摘要图表已保存: {stats_path}")
    
    def save_data_to_csv(self, data: List[Dict[str, Any]], output_dir: str = "output"):
        """
        保存数据到CSV文件
        
        Args:
            data: 斑块分析数据
            output_dir: 输出目录
        """
        if not data:
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 保存CSV文件
        csv_path = os.path.join(output_dir, "tear_spot_temporal_data.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"数据已保存到CSV文件: {csv_path}")
        
        # 保存JSON格式
        json_path = os.path.join(output_dir, "tear_spot_temporal_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"数据已保存到JSON文件: {json_path}")
    
    def run_analysis(self, roi_dir: str = "data/roi_imgs", output_dir: str = "output",
                    smoothing_method: str = 'gaussian', window_size: int = 50, sigma: float = 10.0):
        """
        运行完整的撕裂面斑块时间序列分析（使用剪切面过滤）
        
        Args:
            roi_dir: ROI图像目录路径
            output_dir: 输出目录
            smoothing_method: 平滑方法 ('gaussian', 'moving_avg', 'savgol', 'median')
            window_size: 滤波窗口大小
            sigma: 高斯滤波标准差
        """
        print("=== 撕裂面斑块时间序列分析 (使用剪切面过滤) ===")
        
        # 分析ROI图像的撕裂面斑块特征
        data = self.analyze_roi_spots_with_tear_filter(roi_dir, output_dir)
        
        if not data:
            print("没有可分析的数据")
            return
        
        print(f"\n✅ 撕裂面斑块时间序列分析完成！")
        print(f"📊 分析结果保存位置: {output_dir}")
        print(f"🔧 平滑方法: {smoothing_method}, 窗口大小: {window_size}, σ: {sigma}")
        
        return data
    
    def print_statistics_summary(self, data: List[Dict[str, Any]]):
        """打印统计摘要"""
        if not data:
            return
        
        spot_counts = [d['spot_count'] for d in data]
        spot_densities = [d['spot_density'] for d in data]
        
        print("\n=== 撕裂面斑块时间序列统计摘要 ===")
        print(f"数据点总数: {len(data)}")
        print(f"时间跨度: {data[0]['time_seconds']:.1f} - {data[-1]['time_seconds']:.1f} 秒")
        print(f"帧数范围: {data[0]['frame_num']} - {data[-1]['frame_num']}")
        
        print("\n撕裂面斑块数量统计:")
        print(f"  平均值: {np.mean(spot_counts):.2f}")
        print(f"  标准差: {np.std(spot_counts):.2f}")
        print(f"  最小值: {np.min(spot_counts)}")
        print(f"  最大值: {np.max(spot_counts)}")
        print(f"  中位数: {np.median(spot_counts):.2f}")
        
        print("\n撕裂面斑块密度统计:")
        print(f"  平均值: {np.mean(spot_densities):.6f}")
        print(f"  标准差: {np.std(spot_densities):.6f}")
        print(f"  最小值: {np.min(spot_densities):.6f}")
        print(f"  最大值: {np.max(spot_densities):.6f}")
        print(f"  中位数: {np.median(spot_densities):.6f}")
        
        # 计算相关系数
        correlation = np.corrcoef(spot_counts, spot_densities)[0, 1]
        print(f"\n撕裂面斑块数量与密度相关系数: {correlation:.4f}")


def main():
    """主函数"""
    # 初始化分析器
    analyzer = TearFilterTemporalAnalyzer()
    
    # 运行分析（使用高斯滤波，σ=10，窗口大小50）
    data = analyzer.run_analysis(
        roi_dir="data/roi_imgs",
        output_dir="output/tear_filter_temporal_analysis",
        smoothing_method='gaussian',
        window_size=50,
        sigma=10.0
    )
    
    if data:
        print(f"\n🎯 分析完成！共分析了 {len(data)} 个时间点的撕裂面斑块数据")
        print("📈 生成了平滑时间序列曲线图和统计摘要")


if __name__ == "__main__":
    main()

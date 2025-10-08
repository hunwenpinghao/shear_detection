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

# 设置中文字体
def setup_chinese_font():
    """设置中文字体"""
    import matplotlib.font_manager as fm
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        available_fonts = ['Heiti TC', 'Songti SC', 'Microsoft Sans Serif', 'Kailasa', 'PingFang HK', 'Noto Sans Kaithi']
    elif system == "Windows":
        available_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun']
    else:  # Linux
        available_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
    
    for font_name in available_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"设置中文字体: {font_name}")
            return True
        except:
            continue
    
    print("警告：未找到合适的中文字体，将使用默认字体")
    return False

class ShearTextureEntropyAnalyzer:
    """剪切面纹理熵分析器"""
    
    def __init__(self):
        self.results = []
        setup_chinese_font()
    
    def extract_frame_info(self, filename: str) -> int:
        """从文件名提取帧号"""
        try:
            # 处理类似 "frame_000000_roi.png" 的格式
            basename = os.path.basename(filename)
            parts = basename.split('_')
            if len(parts) >= 2 and 'frame' in parts[0]:
                frame_str = parts[1]  # 获取 "000000"
                return int(frame_str)
            return -1
        except:
            return -1
    
    def compute_texture_entropy(self, shear_region):
        """计算纹理熵"""
        
        # 确保图像不为空
        if shear_region.size == 0 or np.all(shear_region == 0):
            return {
                'texture_entropy': 0.0,
                'normalized_entropy': 0.0
            }
        
        # 计算纹理熵（基于灰度直方图）
        hist, _ = np.histogram(shear_region, bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # 归一化为概率
        entropy = -np.sum(hist * np.log2(hist + 1e-7))  # 计算熵
        
        return {
            'texture_entropy': entropy,
            'normalized_entropy': entropy  # 熵本身就是归一化的
        }
    
    def analyze_shear_texture_entropy(self, image_path: str, shear_mask: np.ndarray) -> dict:
        """分析剪切面纹理熵"""
        
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        
        # 确保图像和mask尺寸一致
        if image.shape != shear_mask.shape:
            shear_mask = cv2.resize(shear_mask, (image.shape[1], image.shape[0]))
        
        # 应用mask过滤出剪切面区域
        shear_region = cv2.bitwise_and(image, image, mask=shear_mask)
        
        # 计算剪切面区域密度
        total_pixels = image.shape[0] * image.shape[1]
        shear_pixels = np.sum(shear_mask > 0)
        shear_region_density = (shear_pixels / total_pixels) * 100
        
        # 计算纹理熵特征
        entropy_features = self.compute_texture_entropy(shear_region)
        
        # 计算归一化纹理熵（熵 / 剪切面面积）
        shear_area_pixels = np.sum(shear_mask > 0)
        normalized_entropy = entropy_features['texture_entropy'] / max(shear_area_pixels, 1) * 10000  # 每万像素的熵
        
        # 从文件名提取时间点
        frame_num = self.extract_frame_info(image_path)
        time_seconds = frame_num * 5 if frame_num > 0 else 0  # 假设每5秒一帧
        
        return {
            'frame_num': frame_num,
            'time_seconds': time_seconds,
            'shear_region_density': shear_region_density,  # 剪切面区域占整个图像的比例
            'texture_entropy': entropy_features['texture_entropy'],  # 纹理熵
            'normalized_entropy': normalized_entropy,  # 归一化纹理熵（每万像素）
            'shear_area_pixels': shear_area_pixels,  # 剪切面面积（像素）
            'image_shape': image.shape,
            'image_path': image_path
        }
    
    def load_existing_shear_regions(self, image_files, step2_dir):
        """加载现有的剪切面区域数据"""
        
        print("加载现有剪切面区域数据...")
        existing_regions = []
        
        for image_path in image_files[:5]:  # 只检查前5个文件进行调试
            frame_num = self.extract_frame_info(image_path)
            print(f"处理文件: {os.path.basename(image_path)}, 提取的帧号: {frame_num}")
            if frame_num >= 0:
                # 查找对应的剪切面区域文件
                region_filename = f"shear_region_frame_{frame_num:06d}.png"
                region_path = os.path.join(step2_dir, region_filename)
                print(f"查找文件: {region_path}, 存在: {os.path.exists(region_path)}")
                
                if os.path.exists(region_path):
                    existing_regions.append((image_path, region_path, frame_num))
                    print(f"找到匹配: {os.path.basename(image_path)} -> {region_filename}")
        
        # 继续处理剩余文件
        for image_path in image_files[5:]:
            frame_num = self.extract_frame_info(image_path)
            if frame_num >= 0:
                region_filename = f"shear_region_frame_{frame_num:06d}.png"
                region_path = os.path.join(step2_dir, region_filename)
                if os.path.exists(region_path):
                    existing_regions.append((image_path, region_path, frame_num))
        
        print(f"找到 {len(existing_regions)} 个现有的剪切面区域文件")
        return existing_regions
    
    def process_existing_data(self, image_files, step2_dir, step3_dir):
        """处理现有数据，计算纹理熵"""
        
        # 加载现有剪切面区域
        existing_regions = self.load_existing_shear_regions(image_files, step2_dir)
        
        if not existing_regions:
            print("未找到现有的剪切面区域文件")
            return
        
        # 计算纹理熵
        print("计算纹理熵特征...")
        for image_path, region_path, frame_num in tqdm(existing_regions, desc="计算纹理熵", unit="图像"):
            try:
                # 读取剪切面区域
                shear_region = cv2.imread(region_path, cv2.IMREAD_GRAYSCALE)
                if shear_region is None:
                    continue
                
                # 创建简单的mask（非零区域）
                shear_mask = (shear_region > 0).astype(np.uint8) * 255
                
                # 分析纹理熵
                analysis_result = self.analyze_shear_texture_entropy(image_path, shear_mask)
                if analysis_result:
                    self.results.append(analysis_result)
                    
            except Exception as e:
                print(f"处理帧 {frame_num} 时出错: {e}")
                continue
    
    def apply_smoothing_filters(self, data, smoothing_method='gaussian', sigma=10, window_size=50):
        """应用平滑滤波器"""
        
        if not data:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        time_seconds = np.array([d['time_seconds'] for d in data])
        shear_region_densities = np.array([d['shear_region_density'] for d in data])
        texture_entropies = np.array([d['texture_entropy'] for d in data])
        normalized_entropies = np.array([d['normalized_entropy'] for d in data])
        
        if smoothing_method == 'gaussian':
            # 高斯滤波
            smoothed_densities = gaussian_filter1d(shear_region_densities, sigma=sigma)
            smoothed_entropies = gaussian_filter1d(texture_entropies, sigma=sigma)
            smoothed_normalized = gaussian_filter1d(normalized_entropies, sigma=sigma)
            
        elif smoothing_method == 'moving_avg':
            # 移动平均滤波
            smoothed_densities = np.convolve(shear_region_densities, np.ones(window_size)/window_size, mode='same')
            smoothed_entropies = np.convolve(texture_entropies, np.ones(window_size)/window_size, mode='same')
            smoothed_normalized = np.convolve(normalized_entropies, np.ones(window_size)/window_size, mode='same')
            
        elif smoothing_method == 'savgol':
            # Savitzky-Golay滤波
            window_length = min(window_size, len(shear_region_densities))
            if window_length % 2 == 0:
                window_length -= 1
            smoothed_densities = signal.savgol_filter(shear_region_densities, window_length, 3)
            smoothed_entropies = signal.savgol_filter(texture_entropies, window_length, 3)
            smoothed_normalized = signal.savgol_filter(normalized_entropies, window_length, 3)
            
        else:
            # 默认使用高斯滤波
            smoothed_densities = gaussian_filter1d(shear_region_densities, sigma=sigma)
            smoothed_entropies = gaussian_filter1d(texture_entropies, sigma=sigma)
            smoothed_normalized = gaussian_filter1d(normalized_entropies, sigma=sigma)
        
        return time_seconds, smoothed_densities, smoothed_entropies, smoothed_normalized
    
    def save_results(self, output_dir):
        """保存分析结果"""
        
        if not self.results:
            print("没有结果可保存")
            return
        
        # 转换为JSON可序列化格式
        json_results = []
        for result in self.results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    json_result[key] = value.item()
                else:
                    json_result[key] = value
            json_results.append(json_result)
        
        json_path = os.path.join(output_dir, 'shear_texture_entropy_analysis.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV结果
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = os.path.join(output_dir, 'shear_texture_entropy_analysis.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"CSV结果已保存到: {csv_path}")
        
        print(f"分析结果已保存到: {json_path}")
    
    def create_visualizations(self, output_dir):
        """创建可视化图表"""
        
        if not self.results:
            print("没有数据可可视化")
            return
        
        # 按时间排序
        sorted_results = sorted(self.results, key=lambda x: x['time_seconds'])
        
        # 应用平滑滤波
        time_seconds, smoothed_densities, smoothed_entropies, smoothed_normalized = self.apply_smoothing_filters(
            sorted_results, smoothing_method='gaussian', sigma=10, window_size=50
        )
        
        # 提取数据
        shear_region_densities = np.array([r['shear_region_density'] for r in sorted_results])
        texture_entropies = np.array([r['texture_entropy'] for r in sorted_results])
        normalized_entropies = np.array([r['normalized_entropy'] for r in sorted_results])
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # 剪切面区域密度随时间变化（原始数据+平滑曲线）
        ax1.plot(time_seconds, shear_region_densities, 'b-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax1.plot(time_seconds, smoothed_densities, 'b-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax1.fill_between(time_seconds, smoothed_densities, alpha=0.3, color='blue')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('剪切面区域密度 (%)')
        ax1.set_title('剪切面区域密度随时间变化 (平滑滤波)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_density = np.mean(shear_region_densities)
        ax1.axhline(y=mean_density, color='red', linestyle='--', alpha=0.7, 
                   label=f'平均值: {mean_density:.2f}%')
        ax1.legend()
        
        # 剪切面纹理熵随时间变化（原始数据+平滑曲线）
        ax2.plot(time_seconds, texture_entropies, 'r-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax2.plot(time_seconds, smoothed_entropies, 'r-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax2.fill_between(time_seconds, smoothed_entropies, alpha=0.3, color='red')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('剪切面纹理熵')
        ax2.set_title('剪切面纹理熵随时间变化 (平滑滤波)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_entropy = np.mean(texture_entropies)
        ax2.axhline(y=mean_entropy, color='blue', linestyle='--', alpha=0.7,
                   label=f'平均值: {mean_entropy:.2f}')
        ax2.legend()
        
        # 归一化纹理熵随时间变化（原始数据+平滑曲线）
        ax3.plot(time_seconds, normalized_entropies, 'g-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax3.plot(time_seconds, smoothed_normalized, 'g-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax3.fill_between(time_seconds, smoothed_normalized, alpha=0.3, color='green')
        ax3.set_xlabel('时间 (秒)')
        ax3.set_ylabel('归一化纹理熵 (每万像素)')
        ax3.set_title('归一化纹理熵随时间变化 (排除剪切面面积影响)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_normalized = np.mean(normalized_entropies)
        ax3.axhline(y=mean_normalized, color='orange', linestyle='--', alpha=0.7,
                   label=f'平均值: {mean_normalized:.2f}')
        ax3.legend()
        
        # 添加滤波方法说明
        fig.suptitle('平滑方法: 高斯滤波 (σ=10, 窗口=50)', fontsize=12, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 保存图表
        plot_path = os.path.join(output_dir, 'shear_texture_entropy_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到: {plot_path}")
        
        # 创建综合统计图
        self.create_summary_plot(output_dir, sorted_results)
    
    def create_summary_plot(self, output_dir, sorted_results):
        """创建综合统计图"""
        
        # 提取数据
        time_points = [r['time_seconds'] for r in sorted_results]
        shear_densities = [r['shear_region_density'] for r in sorted_results]
        texture_entropies = [r['texture_entropy'] for r in sorted_results]
        normalized_entropies = [r['normalized_entropy'] for r in sorted_results]
        
        # 创建综合图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('剪切面纹理熵分析总结', fontsize=16)
        
        # 密度和纹理熵对比
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(time_points, shear_densities, 'b-o', linewidth=2, markersize=6, label='剪切面密度 (%)')
        line2 = ax1_twin.plot(time_points, texture_entropies, 'r-s', linewidth=2, markersize=6, label='纹理熵')
        
        ax1.set_xlabel('时间点')
        ax1.set_ylabel('剪切面密度 (%)', color='b')
        ax1_twin.set_ylabel('纹理熵', color='r')
        ax1.set_title('剪切面密度 vs 纹理熵')
        ax1.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 归一化纹理熵
        ax2.plot(time_points, normalized_entropies, 'g-^', linewidth=2, markersize=6, label='归一化纹理熵')
        ax2.set_xlabel('时间点')
        ax2.set_ylabel('归一化纹理熵 (每万像素)', color='g')
        ax2.set_title('归一化纹理熵随时间变化 (面积无关)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存综合图
        summary_path = os.path.join(output_dir, 'shear_texture_entropy_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"综合统计图已保存到: {summary_path}")

def main():
    """主函数"""
    import sys
    
    print("开始分析剪切面纹理熵...")
    print("=" * 60)
    
    # 设置路径
    roi_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data/roi_imgs"
    output_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_texture_density_curve"
    step2_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_texture_density_curve/step2_shear_regions"
    step3_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_texture_density_curve/step3_entropy_analysis"
    
    # 确保输出目录存在
    os.makedirs(step3_dir, exist_ok=True)
    
    # 获取ROI图像文件
    if not os.path.exists(roi_dir):
        print(f"错误：ROI目录不存在: {roi_dir}")
        return
    
    image_files = [os.path.join(roi_dir, f) for f in os.listdir(roi_dir) 
                   if f.endswith('.png') and 'roi' in f]
    image_files.sort()
    
    print(f"找到 {len(image_files)} 个ROI图像文件")
    
    # 创建分析器
    analyzer = ShearTextureEntropyAnalyzer()
    
    # 处理现有数据
    analyzer.process_existing_data(image_files, step2_dir, step3_dir)
    
    if not analyzer.results:
        print("没有分析结果，请先运行剪切面检测")
        return
    
    # 保存结果
    analyzer.save_results(step3_dir)
    
    # 生成可视化
    analyzer.create_visualizations(step3_dir)
    
    print(f"分析完成，结果已保存到: {step3_dir}")
    print("=" * 60)
    print("剪切面纹理熵分析完成!")

if __name__ == "__main__":
    main()

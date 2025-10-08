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

class BurrDensityAnalyzer:
    """毛刺密度分析器"""
    
    def __init__(self):
        self.results = []
        self.feature_extractor = FeatureExtractor(PREPROCESS_CONFIG)
        
    def create_burr_visualization(self, background_image: np.ndarray, 
                                burr_binary: np.ndarray) -> np.ndarray:
        """
        创建毛刺可视化图像
        """
        try:
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            
            # 确保输入为灰度图
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 使用matplotlib生成可视化
            fig, ax = plt.subplots(1, 1, figsize=(6, 12))
            ax.imshow(gray_background, cmap='gray', alpha=0.7)
            ax.imshow(burr_binary, cmap='Blues', alpha=0.8)  # 使用蓝色表示毛刺
            ax.axis('off')
            
            # 保存到内存中的字节流
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # 读取图像数组并转换
            image = Image.open(buf)
            image_array = np.array(image)
            
            # 转换为OpenCV BGR格式
            if len(image_array.shape) == 3:
                bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            buf.close()
            plt.close(fig)
            return bgr
            
        except Exception as e:
            print(f"matplotlib方法失败，使用OpenCV回退: {e}")
            
            # OpenCV回退方法
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 背景alpha
            background_result = (gray_background * 0.7).astype(np.uint8)
            
            # 蓝色图层（毛刺）
            blue_result = np.zeros(gray_background.shape + (3,), dtype=np.uint8)
            burr_pixels = burr_binary > 0
            blue_result[burr_pixels, 2] = 255  # 蓝色通道
            
            # alpha混合
            blue_layer = blue_result.astype(np.float32)
            bg_layer = cv2.cvtColor(background_result, cv2.COLOR_GRAY2RGB).astype(np.float32)
            
            alpha_burr = 0.8
            alpha_bg = 0.7
            overlay = alpha_burr * blue_layer + alpha_bg * (1 - alpha_burr) * bg_layer
            return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def analyze_tear_burr_density(self, image_path, tear_mask, tear_burr_mask):
        """分析撕裂面毛刺密度 - 基于已抠出的撕裂面毛刺图"""
        
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        
        # 计算撕裂面区域密度
        total_pixels = image.shape[0] * image.shape[1]
        tear_pixels = np.sum(tear_mask > 0)
        tear_region_density = (tear_pixels / total_pixels) * 100
        
        # 计算毛刺数量（连通域分析）
        num_labels, labels = cv2.connectedComponents(tear_burr_mask)
        num_burrs = num_labels - 1  # 减去背景标签
        
        # 计算撕裂面区域内的毛刺统计
        burr_pixels = np.sum(tear_burr_mask > 0)
        # 修改密度定义：Density = num_burrs / 撕裂面区域内所有毛刺区域内的所有pixel的数量
        burr_density_in_tear = (num_burrs / burr_pixels) if burr_pixels > 0 else 0
        
        # 从文件名提取时间点
        frame_num = self.extract_frame_info(image_path)
        time_seconds = frame_num * 5 if frame_num > 0 else 0  # 假设每5秒一帧
        
        return {
            'frame_num': frame_num,
            'time_seconds': time_seconds,
            'tear_region_density': tear_region_density,  # 撕裂面区域占整个图像的比例
            'num_burrs': num_burrs,  # 撕裂面区域内的毛刺数量
            'burr_density': burr_density_in_tear,  # 毛刺数量密度：num_burrs / burr_pixels
            'burr_total_area': burr_pixels,  # 毛刺总面积
            'image_shape': image.shape,
            'image_path': image_path
        }
    
    def apply_smoothing_filters(self, data, 
                               smoothing_method: str = 'gaussian',
                               window_size: int = 50,
                               sigma: float = 10.0):
        """对时间序列数据应用平滑滤波"""
        time_seconds = np.array([d['time_seconds'] for d in data])
        burr_region_densities = np.array([d['burr_density'] for d in data])
        num_burrs = np.array([d['num_burrs'] for d in data])
        tear_region_densities = np.array([d['tear_region_density'] for d in data])
        
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
    
    def extract_frame_info(self, filename: str) -> int:
        """从文件名提取帧号"""
        try:
            basename = os.path.basename(filename)
            # 提取frame_XXXXXX中的数字
            frame_num = int(basename.split('_')[1])
            return frame_num
        except (IndexError, ValueError):
            return -1
    
    def process_images(self, roi_dir, output_dir):
        """按步骤处理所有图像"""
        
        print("开始分析毛刺密度...")
        print("=" * 60)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建各步骤保存目录
        step1_dir = os.path.join(output_dir, 'step1_tear_masks')
        step2_dir = os.path.join(output_dir, 'step2_tear_regions')
        step3_dir = os.path.join(output_dir, 'step3_burr_analysis')
        
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
        use_contour_method = True
        detector = ShearTearDetector(use_contour_method=use_contour_method)
        
        skip_step1 = True
        if not skip_step1:
            # 第一步：生成撕裂面mask（before fill + after fill）
            print("\n第一步：生成撕裂面mask...")
            for image_path in tqdm(image_files, desc="生成撕裂面mask", unit="图像"):
                # 读取图像
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                    
                # 检测撕裂面
                result = detector.detect_surfaces(image, visualize=False)
                if result and 'segmented_image' in result:
                    frame_num = self.extract_frame_info(image_path)
                    
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
        
        skip_step2 = True
        if not skip_step2:
            # 第二步：基于原始图片生成毛刺图
            print("\n第二步：基于原始图片生成毛刺图...")
            for image_path in tqdm(image_files, desc="生成原始毛刺图", unit="图像"):
                # 读取图像
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                    
                frame_num = self.extract_frame_info(image_path)
                
                # 基于原始图片检测毛刺
                burr_result = self.feature_extractor.detect_burs(image, mask=None)
                
                # 保存原始毛刺图（可视化版本）
                if 'burs_binary_mask' in burr_result and burr_result['burs_binary_mask'] is not None:
                    burr_binary = burr_result['burs_binary_mask']
                    
                    # 生成毛刺可视化图像
                    burr_visualization = self.create_burr_visualization(image, burr_binary)
                    
                    # 保存原始毛刺图（可视化版本）
                    original_burr_filename = f"original_burrs_frame_{frame_num:06d}.png"
                    original_burr_path = os.path.join(step2_dir, original_burr_filename)
                    cv2.imwrite(original_burr_path, burr_visualization)
            
            print(f"第二步完成，原始毛刺图已保存到: {step2_dir}")
        
        # 第三步：根据撕裂面mask抠出撕裂面区域的毛刺图
        print("\n第三步：根据撕裂面mask抠出撕裂面区域的毛刺图...")
        for image_path in tqdm(image_files, desc="抠出撕裂面毛刺图", unit="图像"):
            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
                
            # 检测撕裂面
            result = detector.detect_surfaces(image, visualize=False)
            if result and 'segmented_image' in result:
                frame_num = self.extract_frame_info(image_path)
                
                # 从分割结果中提取撕裂面mask
                segmented_image = result['segmented_image']
                tear_mask = (segmented_image == 128).astype(np.uint8) * 255
                
                # 应用mask过滤出撕裂面区域
                tear_region = cv2.bitwise_and(image, image, mask=tear_mask)
                
                # 保存撕裂面区域
                region_filename = f"tear_region_frame_{frame_num:06d}.png"
                region_path = os.path.join(step3_dir, region_filename)
                cv2.imwrite(region_path, tear_region)
                
                # 读取原始毛刺图
                original_burr_filename = f"original_burrs_frame_{frame_num:06d}.png"
                original_burr_path = os.path.join(step3_dir, original_burr_filename)
                
                if os.path.exists(original_burr_path):
                    # 重新检测毛刺以获得二值掩码
                    burr_result = self.feature_extractor.detect_burs(image, mask=None)
                    
                    if 'burs_binary_mask' in burr_result and burr_result['burs_binary_mask'] is not None:
                        burr_binary = burr_result['burs_binary_mask']
                        
                        # 使用撕裂面mask抠出撕裂面区域的毛刺二值掩码
                        tear_burr_binary = cv2.bitwise_and(burr_binary, burr_binary, mask=tear_mask)
                        
                        # 生成撕裂面区域的毛刺可视化图像
                        tear_burr_visualization = self.create_burr_visualization(tear_region, tear_burr_binary)
                        
                        # 保存撕裂面区域的毛刺图（可视化版本）
                        tear_burr_filename = f"tear_burrs_frame_{frame_num:06d}.png"
                        tear_burr_path = os.path.join(step3_dir, tear_burr_filename)
                        cv2.imwrite(tear_burr_path, tear_burr_visualization)
                        
                        # 分析撕裂面毛刺密度（使用二值掩码）
                        analysis_result = self.analyze_tear_burr_density(image_path, tear_mask, tear_burr_binary)
                        if analysis_result:
                            self.results.append(analysis_result)
        
        print(f"第三步完成，撕裂面毛刺图已保存到: {step3_dir}")
        
        # 第四步：计算撕裂面毛刺数量和密度，生成变化曲线图
        print("\n第四步：计算撕裂面毛刺数量和密度，生成变化曲线图...")
        
        # 保存结果
        self.save_results(output_dir)
        
        # 生成可视化
        self.create_visualizations(output_dir)
        
        print(f"第四步完成，分析结果和曲线图已保存到: {step3_dir}")
        
        print("=" * 60)
        print("撕裂面毛刺密度分析完成!")
        print(f"所有结果已保存到: {output_dir}")
    
    def create_burr_visualization(self, background_image: np.ndarray, 
                                burr_binary: np.ndarray) -> np.ndarray:
        """创建毛刺可视化图像"""
        try:
            import io
            from PIL import Image
            
            # 确保输入为灰度图
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 使用matplotlib生成毛刺可视化图像
            fig, ax = plt.subplots(1, 1, figsize=(6, 12))
            ax.imshow(gray_background, cmap='gray', alpha=0.7)
            ax.imshow(burr_binary, cmap='Oranges', alpha=0.8)  # 使用橙色表示毛刺
            ax.axis('off')
            
            # 保存到内存中的字节流
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # 读取图像数组并转换
            image = Image.open(buf)
            image_array = np.array(image)
            
            # 转换为OpenCV BGR格式
            if len(image_array.shape) == 3:
                bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR) 
                
            buf.close()
            plt.close(fig)
            return bgr
            
        except Exception as e:
            print(f"matplotlib方法失败，使用OpenCV回退: {e}")
            
            # OpenCV回退方法 - 模拟橙色毛刺可视化
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 背景透明度处理
            background_result = (gray_background * 0.7).astype(np.uint8)
            
            # 橙色毛刺图层 - 模拟橙色通道
            orange_result = np.zeros(gray_background.shape + (3,), dtype=np.uint8)
            burr_pixels = burr_binary > 0
            orange_result[burr_pixels, 0] = 255  # 红色通道 
            orange_result[burr_pixels, 1] = 165  # 绿色通道 (橙色=红+部分绿)
            orange_result[burr_pixels, 2] = 0    # 蓝色通道
            
            # Alpha混合
            orange_layer = orange_result.astype(np.float32)
            bg_layer = cv2.cvtColor(background_result, cv2.COLOR_GRAY2RGB).astype(np.float32)
            
            # 进行alpha混合
            alpha_burr = 0.8
            alpha_bg = 0.7
            overlay = alpha_burr * orange_layer + alpha_bg * (1 - alpha_burr) * bg_layer
            
            return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    def save_results(self, output_dir):
        """保存分析结果"""
        
        # 保存JSON结果（转换numpy类型为Python原生类型）
        json_results = []
        for result in self.results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                elif hasattr(value, 'item'):  # numpy标量
                    json_result[key] = value.item()
                else:
                    json_result[key] = value
            json_results.append(json_result)
        
        json_path = os.path.join(output_dir, 'burr_density_analysis.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV结果
        csv_path = None
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = os.path.join(output_dir, 'burr_density_analysis.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"分析结果已保存到: {json_path}")
        if csv_path:
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
        num_burrs = np.array([r['num_burrs'] for r in sorted_results])
        burr_densities = np.array([r['burr_density'] for r in sorted_results])
        
        # 设置中文字体
        font_success = setup_chinese_font()
        
        # 应用平滑滤波
        _, smoothed_burr_densities, smoothed_burrs, smoothed_tear_densities = self.apply_smoothing_filters(
            sorted_results, smoothing_method='gaussian', window_size=50, sigma=10.0)
        
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
        plot_path = os.path.join(output_dir, 'burr_density_analysis.png')
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
        num_burrs = [r['num_burrs'] for r in sorted_results]
        
        # 创建综合图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Tear Region Burr Analysis Summary', fontsize=16)
        
        # 密度和毛刺数量对比
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(time_points, tear_densities, 'b-o', linewidth=2, markersize=6, label='Tear Region Density (%)')
        line2 = ax1_twin.plot(time_points, num_burrs, 'r-s', linewidth=2, markersize=6, label='Number of Burrs')
        
        ax1.set_xlabel('Time Point')
        ax1.set_ylabel('Tear Region Density (%)', color='b')
        ax1_twin.set_ylabel('Number of Burrs', color='r')
        ax1.set_title('Tear Region Density vs Number of Burrs')
        ax1.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 毛刺密度分布
        burr_densities = [r['burr_density'] for r in sorted_results]
        ax2.hist(burr_densities, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Burr Count Density (burrs/pixel)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Burr Count Densities in Tear Region')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存综合图
        summary_path = os.path.join(output_dir, 'burr_density_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"综合统计图已保存到: {summary_path}")

def main():
    """主函数"""
    import sys
    
    # 设置路径
    roi_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data/roi_imgs"
    output_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_burr_density_curve"
    
    if len(sys.argv) > 1:
        roi_dir = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    # 创建分析器
    analyzer = BurrDensityAnalyzer()
    
    # 处理图像
    analyzer.process_images(roi_dir, output_dir)

if __name__ == "__main__":
    main()

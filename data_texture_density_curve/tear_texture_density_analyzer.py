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
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

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

class TearTextureDensityAnalyzer:
    """撕裂面纹理密度分析器"""
    
    def __init__(self):
        self.results = []
        self.feature_extractor = FeatureExtractor(PREPROCESS_CONFIG)
    
    def compute_texture_features(self, tear_region):
        """计算纹理特征"""
        
        # 确保图像不为空
        if tear_region.size == 0 or np.all(tear_region == 0):
            return {
                'texture_strength': 0.0,
                'texture_contrast': 0.0,
                'texture_energy': 0.0,
                'texture_homogeneity': 0.0,
                'texture_entropy': 0.0
            }
        
        # 1. 计算LBP纹理特征
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(tear_region, n_points, radius, method='uniform')
        
        # 2. 计算GLCM纹理特征
        # 将图像量化为8个灰度级
        gray_quantized = (tear_region // 32).astype(np.uint8)
        
        # 计算GLCM
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        try:
            glcm = graycomatrix(gray_quantized, distances=distances, angles=angles, 
                              levels=8, symmetric=True, normed=True)
            
            # 计算GLCM特征
            contrast = graycoprops(glcm, 'contrast').mean()
            energy = graycoprops(glcm, 'energy').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            
        except Exception as e:
            print(f"GLCM计算出错: {e}")
            contrast = energy = homogeneity = 0.0
        
        # 3. 计算梯度纹理特征
        grad_x = cv2.Sobel(tear_region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(tear_region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 4. 计算拉普拉斯纹理特征
        laplacian = cv2.Laplacian(tear_region, cv2.CV_64F)
        
        # 5. 综合纹理强度（多种纹理特征的加权组合）
        texture_strength = (
            0.3 * np.mean(gradient_magnitude) +      # 梯度强度
            0.2 * np.mean(np.abs(laplacian)) +       # 拉普拉斯强度
            0.2 * np.std(tear_region) +              # 灰度标准差
            0.15 * np.mean(lbp) +                    # LBP强度
            0.15 * contrast                          # GLCM对比度
        )
        
        # 6. 计算纹理熵（基于灰度直方图）
        hist, _ = np.histogram(tear_region, bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        return {
            'texture_strength': texture_strength,
            'texture_contrast': contrast,
            'texture_energy': energy,
            'texture_homogeneity': homogeneity,
            'texture_entropy': entropy
        }
    
    def analyze_tear_texture_density(self, image_path, tear_mask):
        """分析撕裂面纹理密度"""
        
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        
        # 确保mask是二值图像
        if tear_mask.dtype != np.uint8:
            tear_mask = (tear_mask > 0).astype(np.uint8) * 255
        
        # 使用撕裂面mask过滤原图，只保留撕裂面区域
        tear_region = cv2.bitwise_and(image, image, mask=tear_mask)
        
        # 计算撕裂面区域密度
        total_pixels = image.shape[0] * image.shape[1]
        tear_pixels = np.sum(tear_mask > 0)
        tear_region_density = (tear_pixels / total_pixels) * 100
        
        # 计算纹理特征
        texture_features = self.compute_texture_features(tear_region)
        
        # 计算归一化纹理密度（纹理强度 / 撕裂面面积）
        tear_area_pixels = np.sum(tear_mask > 0)
        normalized_texture_density = texture_features['texture_strength'] / max(tear_area_pixels, 1) * 10000  # 每万像素的纹理强度
        
        # 从文件名提取时间点
        frame_num = self.extract_frame_info(image_path)
        time_seconds = frame_num * 5 if frame_num > 0 else 0  # 假设每5秒一帧
        
        return {
            'frame_num': frame_num,
            'time_seconds': time_seconds,
            'tear_region_density': tear_region_density,  # 撕裂面区域占整个图像的比例
            'texture_strength': texture_features['texture_strength'],  # 综合纹理强度
            'texture_contrast': texture_features['texture_contrast'],  # GLCM对比度
            'texture_energy': texture_features['texture_energy'],  # GLCM能量
            'texture_homogeneity': texture_features['texture_homogeneity'],  # GLCM均匀性
            'texture_entropy': texture_features['texture_entropy'],  # 纹理熵
            'normalized_texture_density': normalized_texture_density,  # 归一化纹理密度（每万像素）
            'tear_area_pixels': tear_area_pixels,  # 撕裂面面积（像素）
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
        texture_strengths = np.array([d['texture_strength'] for d in data])
        normalized_texture_densities = np.array([d['normalized_texture_density'] for d in data])
        
        if smoothing_method == 'gaussian':
            # 高斯滤波
            smoothed_densities = gaussian_filter1d(tear_region_densities, sigma=sigma)
            smoothed_textures = gaussian_filter1d(texture_strengths, sigma=sigma)
            smoothed_normalized = gaussian_filter1d(normalized_texture_densities, sigma=sigma)
            
        elif smoothing_method == 'moving_avg':
            # 移动平均滤波
            smoothed_densities = np.convolve(tear_region_densities, np.ones(window_size)/window_size, mode='same')
            smoothed_textures = np.convolve(texture_strengths, np.ones(window_size)/window_size, mode='same')
            smoothed_normalized = np.convolve(normalized_texture_densities, np.ones(window_size)/window_size, mode='same')
            
        elif smoothing_method == 'savgol':
            # Savitzky-Golay滤波
            window_length = min(window_size, len(tear_region_densities))
            if window_length % 2 == 0:
                window_length -= 1
            smoothed_densities = signal.savgol_filter(tear_region_densities, window_length, 3)
            smoothed_textures = signal.savgol_filter(texture_strengths, window_length, 3)
            smoothed_normalized = signal.savgol_filter(normalized_texture_densities, window_length, 3)
            
        else:
            # 默认使用高斯滤波
            smoothed_densities = gaussian_filter1d(tear_region_densities, sigma=sigma)
            smoothed_textures = gaussian_filter1d(texture_strengths, sigma=sigma)
            smoothed_normalized = gaussian_filter1d(normalized_texture_densities, sigma=sigma)
        
        return time_seconds, smoothed_densities, smoothed_textures, smoothed_normalized
    
    def load_and_enhance_existing_results(self, csv_path):
        """加载现有分析结果并补充归一化字段"""
        import pandas as pd
        
        # 读取现有CSV文件
        df = pd.read_csv(csv_path)
        print(f"加载了 {len(df)} 条现有分析结果")
        
        # 转换为结果列表格式
        self.results = []
        for _, row in df.iterrows():
            # 计算撕裂面面积（从撕裂面区域密度反推）
            total_pixels = 512 * 128  # ROI图像尺寸
            tear_area_pixels = int(row['tear_region_density'] / 100.0 * total_pixels)
            
            # 计算归一化纹理密度
            normalized_texture_density = row['texture_strength'] / max(tear_area_pixels, 1) * 10000
            
            result = {
                'frame_num': int(row['frame_num']),
                'time_seconds': int(row['time_seconds']),
                'tear_region_density': row['tear_region_density'],
                'texture_strength': row['texture_strength'],
                'texture_contrast': row['texture_contrast'],
                'texture_energy': row['texture_energy'],
                'texture_homogeneity': row['texture_homogeneity'],
                'texture_entropy': row['texture_entropy'],
                'normalized_texture_density': normalized_texture_density,
                'tear_area_pixels': tear_area_pixels,
                'image_shape': eval(row['image_shape']),
                'image_path': row['image_path']
            }
            self.results.append(result)
        
        print(f"补充了归一化字段，共 {len(self.results)} 条结果")
        
        # 重新保存包含归一化字段的结果
        self.save_results(os.path.dirname(csv_path))
    
    def extract_frame_info(self, filename: str) -> int:
        """从文件名提取帧号"""
        try:
            basename = os.path.basename(filename)
            # 提取frame_XXXXXX中的数字
            frame_num = int(basename.split('_')[1])
            return frame_num
        except (IndexError, ValueError):
            return -1
    
    def process_images(self, roi_dir, output_dir, use_contour_method=True, visualization_interval=100):
        """按步骤处理所有图像
        
        Args:
            roi_dir: ROI图像目录
            output_dir: 输出目录
            use_contour_method: 是否使用等高线方法生成撕裂面mask
            visualization_interval: 可视化采样间隔，每隔多少帧生成一次可视化图像（默认100）
        """
        print("开始分析撕裂面纹理密度...")
        print("=" * 60)
        print(f"可视化采样间隔: 每 {visualization_interval} 帧")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建各步骤保存目录
        step1_dir = os.path.join(output_dir, 'step1_tear_masks')
        step2_dir = os.path.join(output_dir, 'step2_tear_regions')
        step3_dir = os.path.join(output_dir, 'step3_texture_analysis')
        
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
        
        skip_step1 = True
        if not skip_step1:
            # 第一步：生成撕裂面mask（before fill + after fill）
            print("\n第一步：生成撕裂面mask...")
            detection_results = {}  # 缓存检测结果，避免重复计算
            step1_count = 0
            
            for idx, image_path in enumerate(tqdm(image_files, desc="生成撕裂面mask", unit="图像")):
                # 读取图像
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                frame_num = self.extract_frame_info(image_path)
                
                # 所有帧都检测撕裂面并缓存结果
                result = detector.detect_surfaces(image, visualize=False)
                if result and 'segmented_image' in result:
                    detection_results[frame_num] = result
                    
                    # 只对采样帧保存mask（用于可视化检查）
                    if idx % visualization_interval == 0:
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
                        
                        step1_count += 1
            
            print(f"第一步完成，撕裂面mask已保存到: {step1_dir}")
            print(f"  - 共计算 {len(detection_results)} 个撕裂面（所有帧）")
            print(f"  - 共保存 {step1_count} 个可视化mask（采样间隔: {visualization_interval}）")
        
        skip_step2 = False  # 第一次运行，需要生成撕裂面区域和纹理特征
        if not skip_step2:
            # 第二步：应用撕裂面mask过滤出原图的撕裂面区域，并计算纹理特征
            print("\n第二步：提取撕裂面区域和纹理特征...")
            visualization_count = 0
            for idx, image_path in enumerate(tqdm(image_files, desc="提取撕裂面区域", unit="图像")):
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
                
                # 根据采样间隔保存撕裂面区域
                if idx % visualization_interval == 0:
                    region_filename = f"tear_region_frame_{frame_num:06d}.png"
                    region_path = os.path.join(step2_dir, region_filename)
                    cv2.imwrite(region_path, tear_region)
                    visualization_count += 1
                
                # 所有帧都执行撕裂面纹理密度分析
                analysis_result = self.analyze_tear_texture_density(image_path, tear_mask)
                if analysis_result:
                    self.results.append(analysis_result)
            
            print(f"第二步完成，撕裂面区域和纹理特征已保存到: {step2_dir}")
            print(f"  - 共保存 {visualization_count} 张撕裂面区域图像（采样间隔: {visualization_interval}）")
            print(f"  - 所有帧的纹理特征提取和密度分析已完成")
        
        # 第三步：计算纹理密度，生成变化曲线图
        print("\n第三步：计算纹理密度，生成变化曲线图...")
        
        # 检查是否已有分析结果，如果有则加载并补充归一化字段
        csv_path = os.path.join(step3_dir, 'tear_texture_density_analysis.csv')
        if os.path.exists(csv_path) and len(self.results) == 0:
            print("发现现有分析结果，正在加载并补充归一化字段...")
            self.load_and_enhance_existing_results(csv_path)
        elif len(self.results) == 0:
            # 尝试从现有的撕裂面mask文件重新计算纹理特征
            print("尝试从现有撕裂面mask文件重新计算纹理特征...")
            self.recompute_texture_from_existing_masks(image_files, step1_dir, step2_dir)
            
            if len(self.results) == 0:
                print("警告：无法找到现有的撕裂面mask文件，请先运行撕裂面检测")
                return
            
            # 保存结果
            self.save_results(step3_dir)
        else:
            # 保存结果
            self.save_results(step3_dir)
        
        # 生成可视化
        self.create_visualizations(step3_dir)
        
        print(f"第三步完成，分析结果和曲线图已保存到: {step3_dir}")
        
        print("=" * 60)
        print("撕裂面纹理密度分析完成!")
        print(f"所有结果已保存到: {output_dir}")
    
    def recompute_texture_from_existing_masks(self, image_files, step1_dir, step2_dir):
        """从现有的撕裂面mask文件重新计算纹理特征"""
        
        # 检查是否有现有的撕裂面mask文件
        existing_masks = []
        
        # 首先尝试从data_tear_density_curve目录查找现有的mask文件
        existing_tear_masks_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_tear_density_curve/step1_tear_masks"
        
        for image_path in image_files:
            frame_num = self.extract_frame_info(image_path)
            if frame_num >= 0:
                # 尝试查找现有的mask文件
                mask_paths = [
                    os.path.join(existing_tear_masks_dir, f"tear_mask_after_fill_frame_{frame_num:06d}.png"),
                    os.path.join(existing_tear_masks_dir, f"tear_mask_contour_frame_{frame_num:06d}.png"),
                    os.path.join(step1_dir, f"tear_mask_after_fill_frame_{frame_num:06d}.png"),
                    os.path.join(step1_dir, f"tear_mask_contour_frame_{frame_num:06d}.png"),
                ]
                
                for mask_path in mask_paths:
                    if os.path.exists(mask_path):
                        existing_masks.append((image_path, mask_path, frame_num))
                        break
        
        if not existing_masks:
            print("未找到现有的撕裂面mask文件")
            return
        
        print(f"找到 {len(existing_masks)} 个现有的撕裂面mask文件")
        
        # 重新计算纹理特征
        for image_path, mask_path, frame_num in tqdm(existing_masks, desc="重新计算纹理特征", unit="图像"):
            try:
                # 读取图像和mask
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                tear_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None or tear_mask is None:
                    continue
                
                # 分析纹理密度
                analysis_result = self.analyze_tear_texture_density(image_path, tear_mask)
                if analysis_result:
                    self.results.append(analysis_result)
                    
            except Exception as e:
                print(f"处理帧 {frame_num} 时出错: {e}")
                continue
        
        print(f"重新计算纹理特征完成，共 {len(self.results)} 条结果")
    
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
        
        json_path = os.path.join(output_dir, 'tear_texture_density_analysis.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV结果
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = os.path.join(output_dir, 'tear_texture_density_analysis.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"CSV结果已保存到: {csv_path}")
        
        print(f"分析结果已保存到: {json_path}")
    
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
        texture_strengths = np.array([r['texture_strength'] for r in sorted_results])
        
        # 设置中文字体
        font_success = setup_chinese_font()
        
        # 应用平滑滤波
        _, smoothed_region_densities, smoothed_texture, smoothed_normalized = self.apply_smoothing_filters(
            sorted_results, smoothing_method='gaussian', window_size=50, sigma=10.0)
        
        # 提取归一化纹理密度数据
        normalized_texture_densities = np.array([r['normalized_texture_density'] for r in sorted_results])
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
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
        
        # 撕裂面纹理强度随时间变化（原始数据+平滑曲线）
        ax2.plot(time_seconds, texture_strengths, 'r-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax2.plot(time_seconds, smoothed_texture, 'r-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax2.fill_between(time_seconds, smoothed_texture, alpha=0.3, color='red')
        ax2.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax2.set_ylabel('撕裂面纹理强度' if font_success else 'Tear Texture Strength')
        ax2.set_title('撕裂面纹理强度随时间变化 (平滑滤波)' if font_success else 'Tear Texture Strength Over Time (Smoothed)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_texture = np.mean(texture_strengths)
        ax2.axhline(y=mean_texture, color='blue', linestyle='--', alpha=0.7,
                   label=f'平均值: {mean_texture:.2f}')
        ax2.legend()
        
        # 归一化纹理密度随时间变化（原始数据+平滑曲线）
        ax3.plot(time_seconds, normalized_texture_densities, 'g-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax3.plot(time_seconds, smoothed_normalized, 'g-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax3.fill_between(time_seconds, smoothed_normalized, alpha=0.3, color='green')
        ax3.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax3.set_ylabel('归一化纹理密度 (每万像素)' if font_success else 'Normalized Texture Density (per 10k pixels)')
        ax3.set_title('归一化纹理密度随时间变化 (排除撕裂面面积影响)' if font_success else 'Normalized Texture Density Over Time (Area-Independent)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_normalized = np.mean(normalized_texture_densities)
        ax3.axhline(y=mean_normalized, color='orange', linestyle='--', alpha=0.7,
                   label=f'平均值: {mean_normalized:.2f}')
        ax3.legend()
        
        # 添加滤波方法说明
        fig.suptitle('平滑方法: 高斯滤波 (σ=10, 窗口=50)', fontsize=12, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 保存图表
        plot_path = os.path.join(output_dir, 'tear_texture_density_analysis.png')
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
        texture_strengths = [r['texture_strength'] for r in sorted_results]
        normalized_densities = [r['normalized_texture_density'] for r in sorted_results]
        
        # 创建综合图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Tear Texture Analysis Summary', fontsize=16)
        
        # 密度和纹理强度对比
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(time_points, tear_densities, 'b-o', linewidth=2, markersize=6, label='Tear Density (%)')
        line2 = ax1_twin.plot(time_points, texture_strengths, 'r-s', linewidth=2, markersize=6, label='Texture Strength')
        
        ax1.set_xlabel('Time Point')
        ax1.set_ylabel('Tear Density (%)', color='b')
        ax1_twin.set_ylabel('Texture Strength', color='r')
        ax1.set_title('Tear Density vs Texture Strength')
        ax1.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 归一化纹理密度
        ax2.plot(time_points, normalized_densities, 'g-^', linewidth=2, markersize=6, label='Normalized Texture Density')
        ax2.set_xlabel('Time Point')
        ax2.set_ylabel('Normalized Texture Density (per 10k pixels)', color='g')
        ax2.set_title('Normalized Texture Density Over Time (Area-Independent)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存综合图
        summary_path = os.path.join(output_dir, 'tear_texture_density_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"综合统计图已保存到: {summary_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='撕裂面纹理密度分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python tear_texture_density_analyzer.py --roi_dir data/roi_imgs --output_dir data_texture_density_curve
  
  # 指定可视化采样间隔
  python tear_texture_density_analyzer.py --roi_dir data/roi_imgs --output_dir data_texture_density_curve --viz_interval 50
  
  # 使用原始方法（不使用等高线方法）
  python tear_texture_density_analyzer.py --roi_dir data/roi_imgs --output_dir data_texture_density_curve --no_contour_method
        """
    )
    
    parser.add_argument('--roi_dir', type=str,
                       default="/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data/roi_imgs",
                       help='ROI图像目录路径')
    parser.add_argument('--output_dir', type=str,
                       default="/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_texture_density_curve",
                       help='输出目录路径')
    parser.add_argument('--viz_interval', type=int, default=100,
                       help='可视化采样间隔，每隔多少帧生成一次可视化图像（默认100）')
    parser.add_argument('--no_contour_method', action='store_true',
                       help='不使用等高线方法生成撕裂面mask（使用原始方法）')
    
    args = parser.parse_args()

    # 创建分析器
    analyzer = TearTextureDensityAnalyzer()
    
    # 处理图像
    analyzer.process_images(
        roi_dir=args.roi_dir,
        output_dir=args.output_dir,
        use_contour_method=not args.no_contour_method,
        visualization_interval=args.viz_interval
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
撕裂面和剪切面分离检测器
基于传统图像处理方法的特征提取和分类
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, morphology, measure, feature
from skimage.segmentation import watershed
from skimage.feature import local_binary_pattern
import warnings
warnings.filterwarnings('ignore')

class ShearTearDetector:
    def __init__(self):
        """初始化检测器"""
        self.shear_features = []
        self.tear_features = []
        
    def preprocess_image(self, image):
        """图像预处理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 高斯滤波去噪
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 直方图均衡化增强对比度
        enhanced = cv2.equalizeHist(denoised)
        
        return enhanced
    
    def extract_continuity_features(self, image):
        """提取连续性特征"""
        # 边缘检测
        edges = cv2.Canny(image, 50, 150)
        
        # 形态学操作连接断裂的边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 计算连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_closed)
        
        # 计算最大连通组件的长度（连续性指标）
        if num_labels > 1:
            # 排除背景（标签0）
            component_sizes = stats[1:, cv2.CC_STAT_AREA]
            max_component_size = np.max(component_sizes)
            continuity_score = max_component_size / (image.shape[0] * image.shape[1])
        else:
            continuity_score = 0
            
        return continuity_score, edges_closed
    
    def extract_smoothness_features(self, image):
        """提取平滑度特征"""
        # 计算梯度
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 平滑度指标：梯度幅值的标准差（越小越平滑）
        smoothness_score = 1.0 / (1.0 + np.std(gradient_magnitude))
        
        # 计算拉普拉斯算子（二阶导数）
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        roughness_score = np.std(laplacian)
        
        return smoothness_score, roughness_score, gradient_magnitude
    
    def extract_brightness_features(self, image):
        """提取亮度特征"""
        # 平均亮度
        mean_brightness = np.mean(image)
        
        # 亮度标准差
        brightness_std = np.std(image)
        
        # 亮度分布偏度
        brightness_skew = np.mean((image - mean_brightness)**3) / (brightness_std**3)
        
        return mean_brightness, brightness_std, brightness_skew
    
    def extract_texture_features(self, image):
        """提取纹理特征"""
        # 局部二值模式 (LBP)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # LBP直方图
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        # 纹理均匀性（熵的倒数）
        entropy = -np.sum(hist * np.log(hist + 1e-7))
        texture_uniformity = 1.0 / (1.0 + entropy)
        
        # 灰度共生矩阵特征
        from skimage.feature import graycomatrix, graycoprops
        
        # 计算GLCM
        glcm = graycomatrix(image, distances=[1], angles=[0, 45, 90, 135], 
                           levels=256, symmetric=True, normed=True)
        
        # 对比度
        contrast = graycoprops(glcm, 'contrast').mean()
        
        # 相关性
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # 能量
        energy = graycoprops(glcm, 'energy').mean()
        
        # 同质性
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        
        return {
            'texture_uniformity': texture_uniformity,
            'contrast': contrast,
            'correlation': correlation,
            'energy': energy,
            'homogeneity': homogeneity
        }
    
    def extract_geometric_features(self, image):
        """提取几何特征"""
        # 二值化
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {
                'aspect_ratio': 0,
                'solidity': 0,
                'convexity': 0,
                'rectangularity': 0
            }
        
        # 选择最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 长宽比
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-7)
        
        # 实心度
        area = cv2.contourArea(largest_contour)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-7)
        
        # 凸性
        perimeter = cv2.arcLength(largest_contour, True)
        hull_perimeter = cv2.arcLength(hull, True)
        convexity = hull_perimeter / (perimeter + 1e-7)
        
        # 矩形度
        rect_area = w * h
        rectangularity = area / (rect_area + 1e-7)
        
        return {
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'convexity': convexity,
            'rectangularity': rectangularity
        }
    
    def extract_wave_features(self, image):
        """提取波浪特征（剪切面特有）"""
        # 垂直投影
        vertical_projection = np.sum(image, axis=0)
        
        # 计算投影的周期性
        from scipy.fft import fft, fftfreq
        
        # 去除趋势
        detrended = vertical_projection - np.mean(vertical_projection)
        
        # FFT分析
        fft_result = fft(detrended)
        freqs = fftfreq(len(detrended))
        
        # 找到主要频率
        power_spectrum = np.abs(fft_result)**2
        # 排除直流分量
        power_spectrum[0] = 0
        
        if len(power_spectrum) > 1:
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_freq = freqs[dominant_freq_idx]
            wave_strength = power_spectrum[dominant_freq_idx] / np.sum(power_spectrum)
        else:
            dominant_freq = 0
            wave_strength = 0
        
        return {
            'wave_strength': wave_strength,
            'dominant_frequency': abs(dominant_freq),
            'projection_variance': np.var(vertical_projection)
        }
    
    def extract_fiber_features(self, image):
        """提取纤维特征（撕裂面特有）"""
        # 水平方向的结构张量
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 结构张量
        Ixx = grad_x * grad_x
        Ixy = grad_x * grad_y
        Iyy = grad_y * grad_y
        
        # 高斯平滑
        sigma = 2.0
        Ixx = cv2.GaussianBlur(Ixx, (0, 0), sigma)
        Ixy = cv2.GaussianBlur(Ixy, (0, 0), sigma)
        Iyy = cv2.GaussianBlur(Iyy, (0, 0), sigma)
        
        # 计算特征值
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        
        # 避免除零
        discriminant = trace**2 - 4 * det
        discriminant = np.maximum(discriminant, 0)
        
        lambda1 = (trace + np.sqrt(discriminant)) / 2
        lambda2 = (trace - np.sqrt(discriminant)) / 2
        
        # 各向异性
        anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-7)
        
        # 方向一致性
        orientation = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
        orientation_consistency = np.std(orientation)
        
        return {
            'anisotropy': np.mean(anisotropy),
            'orientation_consistency': orientation_consistency,
            'fiber_strength': np.mean(lambda1 - lambda2)
        }
    
    def extract_all_features(self, image):
        """提取所有特征"""
        # 预处理
        processed_image = self.preprocess_image(image)
        
        # 提取各类特征
        continuity_score, edges = self.extract_continuity_features(processed_image)
        smoothness_score, roughness_score, gradient_mag = self.extract_smoothness_features(processed_image)
        mean_brightness, brightness_std, brightness_skew = self.extract_brightness_features(processed_image)
        texture_features = self.extract_texture_features(processed_image)
        geometric_features = self.extract_geometric_features(processed_image)
        wave_features = self.extract_wave_features(processed_image)
        fiber_features = self.extract_fiber_features(processed_image)
        
        # 组合所有特征
        features = {
            'continuity_score': continuity_score,
            'smoothness_score': smoothness_score,
            'roughness_score': roughness_score,
            'mean_brightness': mean_brightness,
            'brightness_std': brightness_std,
            'brightness_skew': brightness_skew,
            **texture_features,
            **geometric_features,
            **wave_features,
            **fiber_features
        }
        
        return features, {
            'processed_image': processed_image,
            'edges': edges,
            'gradient_magnitude': gradient_mag
        }
    
    def classify_surface_type(self, features):
        """基于特征分类表面类型"""
        # 定义特征权重（基于观察到的特征差异）
        weights = {
            'continuity_score': 0.25,      # 连续性：剪切面高，撕裂面低
            'smoothness_score': 0.20,      # 平滑度：剪切面高，撕裂面低
            'roughness_score': -0.15,      # 粗糙度：剪切面低，撕裂面高
            'mean_brightness': 0.10,       # 亮度：剪切面高，撕裂面低
            'wave_strength': 0.15,         # 波浪强度：剪切面高，撕裂面低
            'anisotropy': -0.10,           # 各向异性：剪切面低，撕裂面高
            'texture_uniformity': 0.05,    # 纹理均匀性：剪切面高，撕裂面低
        }
        
        # 计算综合得分
        score = 0
        for feature_name, weight in weights.items():
            if feature_name in features:
                # 归一化特征值到[0,1]范围
                normalized_value = min(max(features[feature_name], 0), 1)
                score += weight * normalized_value
        
        # 分类阈值
        threshold = 0.3
        
        if score > threshold:
            return 'shear', score
        else:
            return 'tear', score
    
    def segment_surfaces(self, image):
        """分割撕裂面和剪切面区域"""
        # 预处理
        processed = self.preprocess_image(image)
        
        # 创建Otsu二值化mask
        _, otsu_mask = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 创建内部区域mask（排除白色区域的边缘）
        # 使用形态学腐蚀操作缩小白色区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner_mask = cv2.erode(otsu_mask, kernel, iterations=2)
        
        # 进一步从左右方向收缩，减少边缘干扰
        # 创建水平方向的腐蚀核，专门从左右收缩
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))  # 水平方向收缩
        inner_mask = cv2.erode(inner_mask, horizontal_kernel, iterations=3)  # 从左右各收缩约10像素
        
        # 只计算水平梯度（左右方向）
        grad_x = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
        
        # 应用内部mask：只在白色区域内部计算梯度
        grad_x_masked = grad_x.copy()
        grad_x_masked[inner_mask == 0] = 0  # 将非内部区域的梯度设为0
        
        # 对梯度图进行去噪处理
        # 1. 高斯滤波去噪
        grad_x_smooth = cv2.GaussianBlur(grad_x_masked, (5, 5), 1.0)
        
        # 2. 双边滤波保持边缘
        grad_x_bilateral = cv2.bilateralFilter(grad_x_smooth.astype(np.float32), 9, 75, 75)
        
        # 3. 形态学操作去除小噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad_x_clean = cv2.morphologyEx(np.abs(grad_x_bilateral), cv2.MORPH_OPEN, kernel)
        
        # 4. 非局部均值去噪
        grad_x_denoised = cv2.fastNlMeansDenoising(grad_x_clean.astype(np.uint8), None, 10, 7, 21)
        
        # 再次应用内部mask确保只在目标区域内部
        grad_x_denoised[inner_mask == 0] = 0
        
        # 使用去噪后的水平梯度
        horizontal_gradient = grad_x_denoised.astype(np.float64)
        
        # 计算梯度幅值（仅基于水平梯度）
        gradient_magnitude = np.abs(horizontal_gradient)
        
        # 计算局部二值模式纹理（只在mask区域内）
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(processed, n_points, radius, method='uniform')
        
        # 计算局部标准差（纹理复杂度）
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(processed.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D((processed.astype(np.float32))**2, -1, kernel)
        local_std = np.sqrt(local_sq_mean - local_mean**2)
        
        # 应用内部mask到纹理特征
        local_std_masked = local_std.copy()
        local_std_masked[inner_mask == 0] = 0
        
        # 基于多个特征创建分割图
        segmented_image = np.zeros_like(image, dtype=np.uint8)
        
        # 只在内部mask区域内计算阈值
        inner_region = inner_mask > 0
        if np.any(inner_region):
            gradient_threshold = np.percentile(gradient_magnitude[inner_region], 70)
            texture_threshold = np.percentile(local_std_masked[inner_region], 60)
            horizontal_gradient_threshold = np.percentile(horizontal_gradient[inner_region], 60)
        else:
            gradient_threshold = 0
            texture_threshold = 0
            horizontal_gradient_threshold = 0
        
        # 创建特征图（只在内部mask区域内）
        # 撕裂面特征：高水平梯度（左右方向的纤维状纹理）+ 高纹理复杂度
        tear_mask_original = (horizontal_gradient > horizontal_gradient_threshold) & (local_std_masked > texture_threshold) & (inner_mask > 0)
        
        # 剪切面特征：低水平梯度 + 低纹理复杂度（相对平滑的剪切面）
        shear_mask_original = (horizontal_gradient < horizontal_gradient_threshold * 0.7) & (local_std_masked < texture_threshold * 0.8) & (inner_mask > 0)
        
        # 复制原始mask用于填充
        tear_mask = tear_mask_original.copy()
        shear_mask = shear_mask_original.copy()
        
        # 步骤1：利用收缩前的mask通过梯度计算得到白色区域的左右两个边界线
        # 使用原始Otsu mask计算边界
        grad_x_full = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
        grad_x_full_masked = np.where(otsu_mask > 0, grad_x_full, 0)
        
        # 对每行找到左右边界
        left_boundaries = []
        right_boundaries = []
        
        for row in range(otsu_mask.shape[0]):
            row_mask = otsu_mask[row, :]
            if np.sum(row_mask) > 0:  # 如果这一行有白色区域
                # 找到左右边界
                white_pixels = np.where(row_mask > 0)[0]
                if len(white_pixels) > 0:
                    left_boundary = white_pixels[0]
                    right_boundary = white_pixels[-1]
                    left_boundaries.append(left_boundary)
                    right_boundaries.append(right_boundary)
                else:
                    left_boundaries.append(-1)
                    right_boundaries.append(-1)
            else:
                left_boundaries.append(-1)
                right_boundaries.append(-1)
        
        # 步骤2和3：区域填充
        # 对每行进行填充
        for row in range(otsu_mask.shape[0]):
            if left_boundaries[row] != -1 and right_boundaries[row] != -1:
                left_boundary = left_boundaries[row]
                right_boundary = right_boundaries[row]
                
                # 在这一行中找到撕裂面和剪切面的位置
                row_tear = tear_mask[row, :]
                row_shear = shear_mask[row, :]
                
            # 找到撕裂面最右边的位置
            tear_pixels = np.where(row_tear)[0]
            if len(tear_pixels) > 0:
                tear_rightmost = np.max(tear_pixels)
                # 从撕裂面最右边到左边界都算作撕裂面（取并集）
                tear_mask[row, left_boundary:tear_rightmost+1] = True
            
            # 找到剪切面最左边的位置
            shear_pixels = np.where(row_shear)[0]
            if len(shear_pixels) > 0:
                shear_leftmost = np.min(shear_pixels)
                # 从剪切面最左边到右边界都算作剪切面（取并集）
                shear_mask[row, shear_leftmost:right_boundary+1] = True
        
        # 确保最终mask包含原始检测区域（取并集）
        tear_mask = tear_mask | tear_mask_original
        shear_mask = shear_mask | shear_mask_original
        
        # 应用形态学操作平滑区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tear_mask = cv2.morphologyEx(tear_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        shear_mask = cv2.morphologyEx(shear_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # 去除重叠区域
        overlap = tear_mask & shear_mask
        tear_mask = tear_mask - overlap
        shear_mask = shear_mask - overlap
        
        # 填充分割结果
        segmented_image[tear_mask > 0] = 128  # 灰色表示撕裂面
        segmented_image[shear_mask > 0] = 255  # 白色表示剪切面
        
        # 创建标签图
        labels = np.zeros_like(image, dtype=np.int32)
        labels[tear_mask > 0] = 1
        labels[shear_mask > 0] = 2
        
        return segmented_image, labels, otsu_mask, inner_mask
    
    def detect_surfaces(self, image, visualize=True):
        """检测撕裂面和剪切面"""
        # 提取特征
        features, intermediate_results = self.extract_all_features(image)
        
        # 分类
        surface_type, confidence = self.classify_surface_type(features)
        
        # 区域分割
        segmented_image, labels, otsu_mask, inner_mask = self.segment_surfaces(image)
        
        if visualize:
            self.visualize_results(image, features, surface_type, confidence, intermediate_results, segmented_image, otsu_mask, inner_mask)
        
        return {
            'surface_type': surface_type,
            'confidence': confidence,
            'features': features,
            'intermediate_results': intermediate_results,
            'segmented_image': segmented_image,
            'labels': labels,
            'otsu_mask': otsu_mask,
            'inner_mask': inner_mask
        }
    
    def visualize_results(self, original_image, features, surface_type, confidence, intermediate_results, segmented_image=None, otsu_mask=None, inner_mask=None):
        """可视化检测结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Surface Type Detection Result: {surface_type} (Confidence: {confidence:.3f})', fontsize=16)
        
        # 原始图像
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 预处理后图像
        axes[0, 1].imshow(intermediate_results['processed_image'], cmap='gray')
        axes[0, 1].set_title('Preprocessed Image')
        axes[0, 1].axis('off')
        
        # 内部区域mask
        if inner_mask is not None:
            axes[0, 2].imshow(inner_mask, cmap='gray')
            axes[0, 2].set_title('Horizontally Shrunk\nRegion Mask')
            axes[0, 2].axis('off')
        elif otsu_mask is not None:
            axes[0, 2].imshow(otsu_mask, cmap='gray')
            axes[0, 2].set_title('Otsu Binarization Mask')
            axes[0, 2].axis('off')
        else:
            axes[0, 2].imshow(intermediate_results['edges'], cmap='gray')
            axes[0, 2].set_title('Edge Detection')
            axes[0, 2].axis('off')
        
        # 区域分割结果叠加可视化
        if segmented_image is not None:
            # 分割结果叠加可视化
            original_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            combined_overlay = original_rgb.copy()
            combined_overlay[intermediate_results['tear_mask'] > 0] = [255, 0, 0]  # 撕裂面：红色
            combined_overlay[intermediate_results['shear_mask'] > 0] = [0, 0, 255]  # 剪切面：蓝色
            
            axes[1, 0].imshow(original_rgb)
            axes[1, 0].imshow(combined_overlay, alpha=0.5)
            axes[1, 0].set_title('Surface Segmentation Overlay\n(Red: Tear, Blue: Shear)')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].imshow(intermediate_results['gradient_magnitude'], cmap='hot')
            axes[1, 0].set_title('Gradient Magnitude')
            axes[1, 0].axis('off')
        
        # 特征雷达图
        self.plot_feature_radar(axes[1, 1], features)
        
        # 特征条形图
        self.plot_feature_bars(axes[1, 2], features)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_radar(self, ax, features):
        """绘制特征雷达图"""
        # 选择关键特征
        key_features = ['continuity_score', 'smoothness_score', 'mean_brightness', 
                       'wave_strength', 'texture_uniformity']
        
        values = [features.get(f, 0) for f in key_features]
        labels = ['连续性', '平滑度', '亮度', '波浪强度', '纹理均匀性']
        
        # 归一化到[0,1]
        values = [min(max(v, 0), 1) for v in values]
        
        # 雷达图
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]  # 闭合
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label='特征值')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title('关键特征雷达图')
        ax.grid(True)
    
    def plot_feature_bars(self, ax, features):
        """绘制特征条形图"""
        # 选择数值特征
        numeric_features = {k: v for k, v in features.items() 
                           if isinstance(v, (int, float)) and not np.isnan(v)}
        
        # 排序并选择前10个
        sorted_features = sorted(numeric_features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        names, values = zip(*sorted_features)
        names = [name.replace('_', '\n') for name in names]
        
        bars = ax.barh(range(len(names)), values)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('特征值')
        ax.set_title('主要特征值')
        
        # 颜色编码
        for i, (name, value) in enumerate(sorted_features):
            if 'continuity' in name or 'smoothness' in name or 'brightness' in name:
                bars[i].set_color('blue')  # 剪切面特征
            elif 'roughness' in name or 'anisotropy' in name:
                bars[i].set_color('red')   # 撕裂面特征
            else:
                bars[i].set_color('gray')  # 中性特征

def main():
    """主函数 - 测试检测器"""
    detector = ShearTearDetector()
    
    # 测试图像路径
    test_images = [
        "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_Video_20250821140339629/roi_imgs/frame_006523_roi.png",
        "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_Video_20250821140339629/roi_imgs/frame_000044_roi.png",
        "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_Video_20250821140339629/roi_imgs/frame_000066_roi.png"
    ]
    
    for img_path in test_images:
        try:
            # 读取图像
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"无法读取图像: {img_path}")
                continue
                
            print(f"\n处理图像: {img_path}")
            print("=" * 50)
            
            # 检测表面类型
            result = detector.detect_surfaces(image, visualize=True)
            
            print(f"检测结果: {result['surface_type']}")
            print(f"置信度: {result['confidence']:.3f}")
            print("\n主要特征:")
            for key, value in result['features'].items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    print(f"  {key}: {value:.3f}")
                    
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")

if __name__ == "__main__":
    main()

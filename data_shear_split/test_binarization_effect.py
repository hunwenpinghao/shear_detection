#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试二值化预处理效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 确保可以从项目根目录导入包（例如 data_process）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 兼容 feature_extractor.py 中的顶层 import config
DATA_PROCESS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'data_process'))
if DATA_PROCESS_DIR not in sys.path:
    sys.path.insert(0, DATA_PROCESS_DIR)

from shear_tear_detector import ShearTearDetector
from data_process.feature_extractor import FeatureExtractor
from lbp_texture_processor import LBPTextureProcessor
from burr_processor import BurrProcessor
from data_burr_density_curve.burr_density_analyzer import BurrDensityAnalyzer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_binarization_effects(image_path, output_path):
    """测试不同二值化方法的效果"""
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 原始预处理（无二值化）
    detector = ShearTearDetector()
    processed_original = detector.preprocess_image(image)
    
    # 添加二值化的预处理
    def preprocess_with_binarization(image, binarization_method='otsu'):
        """带二值化的预处理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 高斯滤波去噪
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 直方图均衡化增强对比度
        enhanced = cv2.equalizeHist(denoised)
        
        # 二值化
        if binarization_method == 'otsu':
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif binarization_method == 'adaptive_mean':
            binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        elif binarization_method == 'adaptive_gaussian':
            binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif binarization_method == 'fixed':
            _, binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = enhanced
            
        return binary
    
    # 生成不同二值化方法的结果
    processed_otsu = preprocess_with_binarization(image, 'otsu')
    # 基于原始图像直接进行 Otsu（二值化直接作用于原图，不经高斯与直方图均衡）
    _, processed_otsu_on_original = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 基于Otsu on Original结果检测左边缘和右边缘线
    def detect_left_right_edges(binary_image):
        """检测二值化图像的左边缘和右边缘线"""
        left_edges = []
        right_edges = []
        
        for row in range(binary_image.shape[0]):
            row_pixels = binary_image[row, :]
            # 找到白色像素（255）的位置
            white_pixels = np.where(row_pixels == 255)[0]
            
            if len(white_pixels) > 0:
                left_edge = white_pixels[0]  # 最左边的白色像素
                right_edge = white_pixels[-1]  # 最右边的白色像素
                left_edges.append((row, left_edge))
                right_edges.append((row, right_edge))
            else:
                # 如果没有白色像素，标记为-1
                left_edges.append((row, -1))
                right_edges.append((row, -1))
        
        return left_edges, right_edges
    
    # 检测边缘线
    left_edges, right_edges = detect_left_right_edges(processed_otsu_on_original)
    
    # 创建填充mask：将左右边缘线之间的区域全部填充为白色
    def create_filled_mask(left_edges, right_edges, image_shape):
        """基于左右边缘线创建填充的mask"""
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        for i, (left_row, left_col) in enumerate(left_edges):
            if left_col != -1 and i < len(right_edges):
                right_row, right_col = right_edges[i]
                if right_col != -1:
                    # 填充从左边缘到右边缘之间的区域
                    mask[left_row, left_col:right_col+1] = 255
        
        return mask
    
    # 创建填充mask
    filled_mask = create_filled_mask(left_edges, right_edges, processed_otsu_on_original.shape)
    
    # 应用mask到Adaptive Mean和Adaptive Gaussian结果上
    def apply_mask_to_binarization(binary_result, mask):
        """将mask应用到二值化结果上，只保留mask区域内的内容"""
        filtered_result = binary_result.copy()
        # 将mask区域外的像素设为黑色（0）
        filtered_result[mask == 0] = 0
        return filtered_result
    
    # 生成不同二值化方法的结果
    processed_adaptive_mean = preprocess_with_binarization(image, 'adaptive_mean')
    processed_adaptive_gaussian = preprocess_with_binarization(image, 'adaptive_gaussian')
    processed_fixed = preprocess_with_binarization(image, 'fixed')
    
    # 生成过滤后的结果
    filtered_adaptive_mean = apply_mask_to_binarization(processed_adaptive_mean, filled_mask)
    filtered_adaptive_gaussian = apply_mask_to_binarization(processed_adaptive_gaussian, filled_mask)
    
    # 创建可视化（扩展为 3x4，第二行仍为二值化对比，第三行加入“Final Filled Mask 过滤后的 斑块/毛刺/纹理”）
    fig, axes = plt.subplots(3, 4, figsize=(22, 18))
    fig.suptitle(f'Binarization Effects Comparison (with Final Mask Filtering)\n{os.path.basename(image_path)}', fontsize=16)
    
    # 第一行：原始图像和预处理
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(processed_original, cmap='gray')
    axes[0, 1].set_title('Original Preprocessing\n(Gaussian + Histogram Equalization)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(processed_otsu, cmap='gray')
    axes[0, 2].set_title('Otsu on Preprocessed')
    axes[0, 2].axis('off')

    # 创建带边缘线的Otsu on Original图像
    otsu_with_edges = cv2.cvtColor(processed_otsu_on_original, cv2.COLOR_GRAY2RGB)
    
    # 绘制左边缘线（红色）
    for row, col in left_edges:
        if col != -1:
            cv2.circle(otsu_with_edges, (col, row), 1, (255, 0, 0), -1)  # 红色点
    
    # 绘制右边缘线（蓝色）
    for row, col in right_edges:
        if col != -1:
            cv2.circle(otsu_with_edges, (col, row), 1, (0, 0, 255), -1)  # 蓝色点
    
    axes[0, 3].imshow(otsu_with_edges)
    axes[0, 3].set_title('Otsu on Original\nwith Edge Detection')
    axes[0, 3].axis('off')
    
    # 第二行：过滤后的二值化方法
    axes[1, 0].imshow(filtered_adaptive_mean, cmap='gray')
    axes[1, 0].set_title('Filtered Adaptive Mean\n(Mask Applied)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(filtered_adaptive_gaussian, cmap='gray')
    axes[1, 1].set_title('Filtered Adaptive Gaussian\n(Mask Applied)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(processed_fixed, cmap='gray')
    axes[1, 2].set_title('Fixed Threshold\n(127)')
    axes[1, 2].axis('off')

    # 显示填充后的最终mask图
    axes[1, 3].imshow(filled_mask, cmap='gray')
    axes[1, 3].set_title('Final Filled Mask\n(Left to Right Fill)')
    axes[1, 3].axis('off')

    # =========================
    # 第三行：Final Filled Mask 过滤后的斑块/毛刺/纹理（引用 FeatureExtractor 逻辑）
    # =========================
    extractor = FeatureExtractor()

    # 斑块（使用 FeatureExtractor.detect_all_white_spots 得到整体白斑掩码后再套用 Final Mask）
    all_spots_info = extractor.detect_all_white_spots(image)
    spot_mask = all_spots_info.get('all_white_binary_mask', np.zeros_like(image))
    spot_mask_filtered = spot_mask.copy()
    spot_mask_filtered[filled_mask == 0] = 0
    axes[2, 0].imshow(image, cmap='gray', alpha=0.7)
    axes[2, 0].imshow(spot_mask_filtered, cmap='Reds', alpha=0.8)
    axes[2, 0].set_title('Final Mask 过滤后的斑块图')
    axes[2, 0].axis('off')

    # 毛刺（参考 burr_processor：用 FeatureExtractor.detect_burs，并采用橙色可视化）
    analyzer = BurrDensityAnalyzer()
    burs_info = analyzer.feature_extractor.detect_burs(image, filled_mask)
    burr_binary = burs_info.get('burs_binary_mask', np.zeros_like(image))
    # 使用 analyzer 的可视化以保持与参考实现一致（橙色叠加）
    burr_vis_bgr = analyzer.create_burr_visualization(image, burr_binary)
    burr_vis_rgb = cv2.cvtColor(burr_vis_bgr, cv2.COLOR_BGR2RGB)
    axes[2, 1].imshow(burr_vis_rgb)
    axes[2, 1].set_title('Final Mask 过滤后的毛刺图')
    axes[2, 1].axis('off')

    # 纹理（使用 lbp_texture_processor 的 LBPTextureProcessor 计算并掩膜后显示）
    try:
        lbp_processor = LBPTextureProcessor(radius=3, n_points=24)
        lbp_texture, _ = lbp_processor.compute_lbp_texture(image)
        lbp_masked = lbp_texture.copy()
        lbp_masked[filled_mask == 0] = 0
        axes[2, 2].imshow(lbp_masked, cmap='hot')
        axes[2, 2].set_title('Final Mask 过滤后的纹理图(LBP)')
        axes[2, 2].axis('off')
    except Exception:
        # 回退显示：被掩膜的原始灰度
        img_masked = image.copy()
        img_masked[filled_mask == 0] = 0
        axes[2, 2].imshow(img_masked, cmap='gray')
        axes[2, 2].set_title('Final Mask 过滤后的纹理图(灰度)')
        axes[2, 2].axis('off')

    # 显示去掉黑色纹理后的LBP纹理图（只保留橙色/红色/黄色/白色部分）
    try:
        # 使用与axes[2,2]相同的LBP纹理数据，确保颜色一致
        lbp_processor = LBPTextureProcessor(radius=3, n_points=24)
        lbp_texture, _ = lbp_processor.compute_lbp_texture(image)
        
        # 应用Final Mask
        lbp_masked = lbp_texture.copy()
        lbp_masked[filled_mask == 0] = 0
        
        # 去掉黑色纹理：将LBP值为0的像素设为NaN，这样在显示时会被忽略
        lbp_no_black = lbp_masked.copy()
        lbp_no_black[lbp_no_black == 0] = np.nan
        
        # 使用与axes[2,2]完全相同的颜色映射范围，确保颜色完全一致
        # 直接使用axes[2,2]的vmin和vmax值
        vmin = 0
        vmax = np.max(lbp_texture) if np.max(lbp_texture) > 0 else 255
        
        im = axes[2, 3].imshow(lbp_no_black, cmap='hot', vmin=vmin, vmax=vmax)
        axes[2, 3].set_title('去掉黑色纹理后的\nFinal Mask 过滤LBP纹理图')
        axes[2, 3].axis('off')
    except Exception as e:
        # 回退显示：被掩膜的原始灰度图，去掉黑色像素，但仍使用hot颜色映射
        img_masked = image.copy()
        img_masked[filled_mask == 0] = 0
        # 将黑色像素设为NaN
        img_no_black = img_masked.astype(np.float64)
        img_no_black[img_no_black == 0] = np.nan
        
        # 使用与原图相同的颜色映射范围
        vmin = 0
        vmax = 255
        
        axes[2, 3].imshow(img_no_black, cmap='hot', vmin=vmin, vmax=vmax)
        axes[2, 3].set_title('去掉黑色纹理后的\nFinal Mask 过滤纹理图')
        axes[2, 3].axis('off')
    
    # 添加统计信息
    stats_text = f"""Binarization Statistics:

Original Preprocessing:
Mean: {np.mean(processed_original):.1f}
Std: {np.std(processed_original):.1f}

Otsu on Preprocessed:
Mean: {np.mean(processed_otsu):.1f}
White Pixels: {np.sum(processed_otsu == 255):,}
Black Pixels: {np.sum(processed_otsu == 0):,}

Otsu on Original:
Mean: {np.mean(processed_otsu_on_original):.1f}
White Pixels: {np.sum(processed_otsu_on_original == 255):,}
Black Pixels: {np.sum(processed_otsu_on_original == 0):,}

Edge Detection:
Left Edge Points: {len([e for e in left_edges if e[1] != -1]):,}
Right Edge Points: {len([e for e in right_edges if e[1] != -1]):,}

Final Filled Mask:
White Pixels: {np.sum(filled_mask == 255):,}
Black Pixels: {np.sum(filled_mask == 0):,}
Fill Ratio: {np.sum(filled_mask == 255) / (filled_mask.shape[0] * filled_mask.shape[1]) * 100:.1f}%

Filtered Adaptive Mean:
Mean: {np.mean(filtered_adaptive_mean):.1f}
White Pixels: {np.sum(filtered_adaptive_mean == 255):,}
Black Pixels: {np.sum(filtered_adaptive_mean == 0):,}

Filtered Adaptive Gaussian:
Mean: {np.mean(filtered_adaptive_gaussian):.1f}
White Pixels: {np.sum(filtered_adaptive_gaussian == 255):,}
Black Pixels: {np.sum(filtered_adaptive_gaussian == 0):,}

Fixed Threshold:
Mean: {np.mean(processed_fixed):.1f}
White Pixels: {np.sum(processed_fixed == 255):,}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom', 
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"二值化效果对比图已保存: {output_path}")

def test_binarization_with_gradient(image_path, output_path):
    """测试二值化对梯度计算的影响"""
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 原始预处理
    detector = ShearTearDetector()
    processed_original = detector.preprocess_image(image)
    
    # 带二值化的预处理
    def preprocess_with_binarization(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 高斯滤波去噪
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 直方图均衡化增强对比度
        enhanced = cv2.equalizeHist(denoised)
        
        # Otsu二值化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    processed_binary = preprocess_with_binarization(image)
    
    # 计算梯度
    grad_x_original = cv2.Sobel(processed_original, cv2.CV_64F, 1, 0, ksize=3)
    grad_x_binary = cv2.Sobel(processed_binary, cv2.CV_64F, 1, 0, ksize=3)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Gradient Analysis: Original vs Binarized\n{os.path.basename(image_path)}', fontsize=16)
    
    # 第一行：预处理结果
    axes[0, 0].imshow(processed_original, cmap='gray')
    axes[0, 0].set_title('Original Preprocessing')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(processed_binary, cmap='gray')
    axes[0, 1].set_title('With Otsu Binarization')
    axes[0, 1].axis('off')
    
    # 差异图
    diff = cv2.absdiff(processed_original, processed_binary)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Difference Map')
    axes[0, 2].axis('off')
    
    # 第二行：梯度对比
    axes[1, 0].imshow(np.abs(grad_x_original), cmap='hot')
    axes[1, 0].set_title('Gradient from Original\nPreprocessing')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.abs(grad_x_binary), cmap='hot')
    axes[1, 1].set_title('Gradient from Binarized\nPreprocessing')
    axes[1, 1].axis('off')
    
    # 梯度差异
    grad_diff = np.abs(grad_x_original) - np.abs(grad_x_binary)
    axes[1, 2].imshow(grad_diff, cmap='RdBu_r')
    axes[1, 2].set_title('Gradient Difference\n(Red: Original > Binary)')
    axes[1, 2].axis('off')
    
    # 添加统计信息
    stats_text = f"""Gradient Analysis Statistics:

Original Preprocessing:
Gradient Mean: {np.mean(np.abs(grad_x_original)):.1f}
Gradient Std: {np.std(np.abs(grad_x_original)):.1f}
Gradient Max: {np.max(np.abs(grad_x_original)):.1f}

Binarized Preprocessing:
Gradient Mean: {np.mean(np.abs(grad_x_binary)):.1f}
Gradient Std: {np.std(np.abs(grad_x_binary)):.1f}
Gradient Max: {np.max(np.abs(grad_x_binary)):.1f}

Improvement:
Mean Change: {((np.mean(np.abs(grad_x_binary)) - np.mean(np.abs(grad_x_original))) / np.mean(np.abs(grad_x_original)) * 100):+.1f}%
Std Change: {((np.std(np.abs(grad_x_binary)) - np.std(np.abs(grad_x_original))) / np.std(np.abs(grad_x_original)) * 100):+.1f}%"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom', 
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"二值化梯度分析图已保存: {output_path}")

def main():
    """主函数"""
    # 图像目录
    roi_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_split/roi_images"
    output_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_split/output"
    
    # 获取第一个PNG文件进行测试
    png_files = [f for f in os.listdir(roi_dir) if f.endswith('.png')]
    if not png_files:
        print("没有找到PNG文件")
        return
    
    for test_file in png_files:
        img_path = os.path.join(roi_dir, test_file)
        
        print("测试二值化预处理效果...")
        print("=" * 60)
        
        # 测试二值化效果对比
        output_path1 = os.path.join(output_dir, f"{os.path.splitext(test_file)[0]}_binarization_comparison.png")
        test_binarization_effects(img_path, output_path1)
        
        # 测试二值化对梯度的影响
        output_path2 = os.path.join(output_dir, f"{os.path.splitext(test_file)[0]}_binarization_gradient_analysis.png")
        test_binarization_with_gradient(img_path, output_path2)
    
    print("\n二值化效果测试完成!")

if __name__ == "__main__":
    main()

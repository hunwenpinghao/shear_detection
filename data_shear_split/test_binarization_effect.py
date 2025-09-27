#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试二值化预处理效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from shear_tear_detector import ShearTearDetector

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
    processed_adaptive_mean = preprocess_with_binarization(image, 'adaptive_mean')
    processed_adaptive_gaussian = preprocess_with_binarization(image, 'adaptive_gaussian')
    processed_fixed = preprocess_with_binarization(image, 'fixed')
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Binarization Effects Comparison\n{os.path.basename(image_path)}', fontsize=16)
    
    # 第一行：原始图像和预处理
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(processed_original, cmap='gray')
    axes[0, 1].set_title('Original Preprocessing\n(Gaussian + Histogram Equalization)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(processed_otsu, cmap='gray')
    axes[0, 2].set_title('Otsu Binarization\n(Adaptive Threshold)')
    axes[0, 2].axis('off')
    
    # 第二行：不同二值化方法
    axes[1, 0].imshow(processed_adaptive_mean, cmap='gray')
    axes[1, 0].set_title('Adaptive Mean\nBinarization')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(processed_adaptive_gaussian, cmap='gray')
    axes[1, 1].set_title('Adaptive Gaussian\nBinarization')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(processed_fixed, cmap='gray')
    axes[1, 2].set_title('Fixed Threshold\n(127)')
    axes[1, 2].axis('off')
    
    # 添加统计信息
    stats_text = f"""Binarization Statistics:

Original Preprocessing:
Mean: {np.mean(processed_original):.1f}
Std: {np.std(processed_original):.1f}

Otsu Binarization:
Mean: {np.mean(processed_otsu):.1f}
White Pixels: {np.sum(processed_otsu == 255):,}
Black Pixels: {np.sum(processed_otsu == 0):,}

Adaptive Mean:
Mean: {np.mean(processed_adaptive_mean):.1f}
White Pixels: {np.sum(processed_adaptive_mean == 255):,}

Adaptive Gaussian:
Mean: {np.mean(processed_adaptive_gaussian):.1f}
White Pixels: {np.sum(processed_adaptive_gaussian == 255):,}

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
    
    test_file = png_files[0]
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细的撕裂面和剪切面分割可视化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from shear_tear_detector import ShearTearDetector

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_detailed_segmentation_visualization(image_path, output_path):
    """创建详细的分割可视化"""
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 初始化检测器
    detector = ShearTearDetector()
    
    # 预处理
    processed = detector.preprocess_image(image)
    
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
    gradient_magnitude = np.abs(horizontal_gradient)
    
    # 计算局部标准差
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    local_mean = cv2.filter2D(processed.astype(np.float32), -1, kernel)
    local_sq_mean = cv2.filter2D((processed.astype(np.float32))**2, -1, kernel)
    local_std = np.sqrt(local_sq_mean - local_mean**2)
    
    # 应用内部mask到纹理特征
    local_std_masked = local_std.copy()
    local_std_masked[inner_mask == 0] = 0
    
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
    
    # 创建分割掩码（只在内部mask区域内）
    tear_mask_original = (horizontal_gradient > horizontal_gradient_threshold) & (local_std_masked > texture_threshold) & (inner_mask > 0)
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
    
    # 形态学操作
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tear_mask = cv2.morphologyEx(tear_mask.astype(np.uint8), cv2.MORPH_CLOSE, morph_kernel)
    shear_mask = cv2.morphologyEx(shear_mask.astype(np.uint8), cv2.MORPH_CLOSE, morph_kernel)
    
    # 去除重叠
    overlap = tear_mask & shear_mask
    tear_mask = tear_mask - overlap
    shear_mask = shear_mask - overlap
    
    # 创建分割结果
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    segmented_image[tear_mask > 0] = 128  # 灰色表示撕裂面
    segmented_image[shear_mask > 0] = 255  # 白色表示剪切面
    
    # 创建叠加可视化
    # 将原图转换为RGB
    original_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 创建原始检测区域的彩色mask
    # 撕裂面：红色
    tear_overlay_original = original_rgb.copy()
    tear_overlay_original[tear_mask_original > 0] = [255, 0, 0]  # 红色
    
    # 剪切面：蓝色
    shear_overlay_original = original_rgb.copy()
    shear_overlay_original[shear_mask_original > 0] = [0, 0, 255]  # 蓝色
    
    # 创建原始组合叠加图
    combined_overlay_original = original_rgb.copy()
    combined_overlay_original[tear_mask_original > 0] = [255, 0, 0]  # 撕裂面：红色
    combined_overlay_original[shear_mask_original > 0] = [0, 0, 255]  # 剪切面：蓝色
    
    # 创建填充后的彩色mask
    # 撕裂面：红色
    tear_overlay = original_rgb.copy()
    tear_overlay[tear_mask > 0] = [255, 0, 0]  # 红色
    
    # 剪切面：蓝色
    shear_overlay = original_rgb.copy()
    shear_overlay[shear_mask > 0] = [0, 0, 255]  # 蓝色
    
    # 创建填充后组合叠加图
    combined_overlay = original_rgb.copy()
    combined_overlay[tear_mask > 0] = [255, 0, 0]  # 撕裂面：红色
    combined_overlay[shear_mask > 0] = [0, 0, 255]  # 剪切面：蓝色
    
    # 创建可视化
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f'Horizontally Shrunk Region Gradient Analysis for Surface Segmentation\n{os.path.basename(image_path)}', fontsize=16)
    
    # 第一行：原始图像和预处理
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(processed, cmap='gray')
    axes[0, 1].set_title('Preprocessed Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(inner_mask, cmap='gray')
    axes[0, 2].set_title('Inner Region Mask\n(Horizontally Shrunk)')
    axes[0, 2].axis('off')
    
    # 第二行：梯度处理过程
    axes[1, 0].imshow(np.abs(grad_x), cmap='hot')
    axes[1, 0].set_title('Raw Horizontal Gradient\n(Left-Right Direction)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.abs(grad_x_masked), cmap='hot')
    axes[1, 1].set_title('Horizontally Shrunk\nRegion Gradient')
    axes[1, 1].axis('off')
    
    # 边界线可视化
    boundary_image = original_rgb.copy()
    # 绘制边界线
    for row in range(len(left_boundaries)):
        if left_boundaries[row] != -1 and right_boundaries[row] != -1:
            # 左边界线（绿色）
            boundary_image[row, left_boundaries[row]] = [0, 255, 0]
            # 右边界线（黄色）
            boundary_image[row, right_boundaries[row]] = [255, 255, 0]
    
    axes[1, 2].imshow(boundary_image)
    axes[1, 2].set_title('Boundary Lines\n(Green: Left, Yellow: Right)')
    axes[1, 2].axis('off')
    
    # 第三行：特征图和叠加可视化
    axes[2, 0].imshow(local_std_masked, cmap='hot')
    axes[2, 0].set_title(f'Shrunk Region Texture Complexity\n(Threshold: {texture_threshold:.1f})')
    axes[2, 0].axis('off')
    
    # 原始检测区域叠加可视化
    axes[2, 1].imshow(original_rgb)
    # 创建不透明的mask
    tear_mask_alpha = np.zeros((*tear_mask_original.shape, 4))
    tear_mask_alpha[tear_mask_original > 0] = [0, 0, 1, 1]  # 蓝色，不透明
    shear_mask_alpha = np.zeros((*shear_mask_original.shape, 4))
    shear_mask_alpha[shear_mask_original > 0] = [1, 0, 0, 1]  # 红色，不透明
    combined_mask_alpha = np.maximum(tear_mask_alpha, shear_mask_alpha)
    axes[2, 1].imshow(combined_mask_alpha)
    axes[2, 1].set_title('Original Detection Overlay\n(Before Fill)')
    axes[2, 1].axis('off')
    
    # 填充后结果可视化
    axes[2, 2].imshow(original_rgb)
    # 创建不透明的mask
    tear_mask_alpha_filled = np.zeros((*tear_mask.shape, 4))
    tear_mask_alpha_filled[tear_mask > 0] = [0, 0, 1, 1]  # 蓝色，不透明
    shear_mask_alpha_filled = np.zeros((*shear_mask.shape, 4))
    shear_mask_alpha_filled[shear_mask > 0] = [1, 0, 0, 1]  # 红色，不透明
    combined_mask_alpha_filled = np.maximum(tear_mask_alpha_filled, shear_mask_alpha_filled)
    axes[2, 2].imshow(combined_mask_alpha_filled)
    axes[2, 2].set_title('Filled Surface Overlay\n(After Boundary Fill)')
    axes[2, 2].axis('off')
    
    # 第四行：对比可视化
    # 创建对比叠加图（同时显示填充前和填充后）
    comparison_overlay = original_rgb.copy()
    
    # 填充前区域用较浅的颜色
    comparison_overlay[tear_mask_original > 0] = [255, 100, 100]  # 浅红色
    comparison_overlay[shear_mask_original > 0] = [100, 100, 255]  # 浅蓝色
    
    # 填充后区域用较深的颜色
    comparison_overlay[tear_mask > 0] = [255, 0, 0]  # 深红色
    comparison_overlay[shear_mask > 0] = [0, 0, 255]  # 深蓝色
    
    axes[3, 0].imshow(original_rgb)
    # 创建对比mask（不透明）
    comparison_mask = np.zeros((*tear_mask.shape, 4))
    # 填充前区域用不透明颜色
    comparison_mask[tear_mask_original > 0] = [0, 0, 1, 1]  # 蓝色，不透明
    comparison_mask[shear_mask_original > 0] = [1, 0, 0, 1]  # 红色，不透明
    # 填充后区域用不透明颜色
    comparison_mask[tear_mask > 0] = [0, 0, 1, 1]  # 蓝色，不透明
    comparison_mask[shear_mask > 0] = [1, 0, 0, 1]  # 红色，不透明
    axes[3, 0].imshow(comparison_mask)
    axes[3, 0].set_title('Before vs After Fill Comparison\n(Blue: Tear, Red: Shear)')
    axes[3, 0].axis('off')
    
    # 填充区域可视化（只显示新增的填充部分）
    axes[3, 1].imshow(original_rgb)
    # 只显示填充区域（排除原始检测区域）
    tear_fill_only = tear_mask & (~tear_mask_original)
    shear_fill_only = shear_mask & (~shear_mask_original)
    
    # 创建填充区域mask（不透明）
    fill_mask = np.zeros((*tear_mask.shape, 4))
    fill_mask[tear_fill_only > 0] = [0, 0, 1, 1]  # 蓝色，不透明
    fill_mask[shear_fill_only > 0] = [1, 0, 0, 1]  # 红色，不透明
    
    axes[3, 1].imshow(fill_mask)
    axes[3, 1].set_title('Fill Regions Only\n(Blue: Tear Fill, Red: Shear Fill)')
    axes[3, 1].axis('off')
    
    # 统计信息
    total_pixels = image.size
    tear_pixels_original = np.sum(tear_mask_original)
    shear_pixels_original = np.sum(shear_mask_original)
    tear_pixels = np.sum(tear_mask)
    shear_pixels = np.sum(shear_mask)
    background_pixels = total_pixels - tear_pixels - shear_pixels
    
    # 最终结果统计
    axes[3, 2].text(0.1, 0.9, f'Final Results Summary:', fontsize=12, fontweight='bold', transform=axes[3, 2].transAxes)
    axes[3, 2].text(0.1, 0.8, f'Original Tear: {tear_pixels_original:,} pixels', fontsize=10, transform=axes[3, 2].transAxes)
    axes[3, 2].text(0.1, 0.7, f'Original Shear: {shear_pixels_original:,} pixels', fontsize=10, transform=axes[3, 2].transAxes)
    axes[3, 2].text(0.1, 0.6, f'Final Tear: {tear_pixels:,} pixels', fontsize=10, transform=axes[3, 2].transAxes)
    axes[3, 2].text(0.1, 0.5, f'Final Shear: {shear_pixels:,} pixels', fontsize=10, transform=axes[3, 2].transAxes)
    axes[3, 2].text(0.1, 0.4, f'Tear Added: {tear_pixels - tear_pixels_original:,} pixels', fontsize=10, transform=axes[3, 2].transAxes)
    axes[3, 2].text(0.1, 0.3, f'Shear Added: {shear_pixels - shear_pixels_original:,} pixels', fontsize=10, transform=axes[3, 2].transAxes)
    axes[3, 2].text(0.1, 0.2, f'Total Coverage: {(tear_pixels + shear_pixels)/total_pixels*100:.1f}%', fontsize=10, transform=axes[3, 2].transAxes)
    axes[3, 2].set_title('Statistics Summary')
    axes[3, 2].axis('off')
    
    # 在图像上添加统计信息
    otsu_pixels = np.sum(otsu_mask > 0)
    inner_pixels = np.sum(inner_mask > 0)
    # 计算边界线统计
    valid_boundaries = sum(1 for i in range(len(left_boundaries)) if left_boundaries[i] != -1)
    avg_left_boundary = np.mean([left_boundaries[i] for i in range(len(left_boundaries)) if left_boundaries[i] != -1]) if valid_boundaries > 0 else 0
    avg_right_boundary = np.mean([right_boundaries[i] for i in range(len(right_boundaries)) if right_boundaries[i] != -1]) if valid_boundaries > 0 else 0
    
    stats_text = f"""Horizontally Shrunk Region Gradient Analysis with Boundary Fill:
    
Total Pixels: {total_pixels:,}
Otsu Mask Region: {otsu_pixels:,} ({otsu_pixels/total_pixels*100:.1f}%)
Inner Region: {inner_pixels:,} ({inner_pixels/total_pixels*100:.1f}%)

Original Detection:
Tear Surface: {tear_pixels_original:,} ({tear_pixels_original/total_pixels*100:.1f}%)
Shear Surface: {shear_pixels_original:,} ({shear_pixels_original/total_pixels*100:.1f}%)

After Boundary Fill:
Tear Surface: {tear_pixels:,} ({tear_pixels/total_pixels*100:.1f}%)
Shear Surface: {shear_pixels:,} ({shear_pixels/total_pixels*100:.1f}%)
Background: {background_pixels:,} ({background_pixels/total_pixels*100:.1f}%)

Fill Enhancement:
Tear Pixels Added: {tear_pixels - tear_pixels_original:,}
Shear Pixels Added: {shear_pixels - shear_pixels_original:,}

Boundary Analysis:
Valid Boundary Rows: {valid_boundaries}
Avg Left Boundary: {avg_left_boundary:.1f}
Avg Right Boundary: {avg_right_boundary:.1f}
Boundary Width: {avg_right_boundary - avg_left_boundary:.1f}

Thresholds (in inner region):
Horizontal Gradient: {horizontal_gradient_threshold:.1f}
Texture Complexity: {texture_threshold:.1f}

Gradient Statistics:
Raw Gradient Mean: {np.mean(np.abs(grad_x)):.1f}
Inner Region Gradient Mean: {np.mean(np.abs(grad_x_masked)):.1f}
Denoised Gradient Mean: {np.mean(horizontal_gradient):.1f}
Shrunk Region Effectiveness: {((np.mean(np.abs(grad_x)) - np.mean(np.abs(grad_x_masked))) / np.mean(np.abs(grad_x)) * 100):.1f}%"""
    
    # 在右下角添加统计信息
    fig.text(0.98, 0.02, stats_text, fontsize=8, verticalalignment='bottom', horizontalalignment='right',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"详细分割可视化已保存: {output_path}")
    
    # 创建单独的颜色调换对比图
    create_color_swap_comparison(image, tear_mask_original, shear_mask_original, tear_mask, shear_mask, output_path)

def create_color_swap_comparison(image, tear_mask_original, shear_mask_original, tear_mask, shear_mask, output_path):
    """创建单独的颜色调换对比图"""
    
    # 将原图转换为RGB
    original_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 创建单独的颜色调换对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Color Swap Comparison for Surface Segmentation\n{os.path.basename(output_path).replace("_detailed_segmentation.png", "")}', fontsize=16)
    
    # 创建颜色调换的对比mask
    color_swap_mask = np.zeros((*tear_mask.shape, 4))
    
    # Before fill: 颜色调换（撕裂面用红色，剪切面用蓝色）
    color_swap_mask[tear_mask_original > 0] = [1, 0, 0, 1]  # 撕裂面用红色，不透明
    color_swap_mask[shear_mask_original > 0] = [0, 0, 1, 1]  # 剪切面用蓝色，不透明
    
    # After fill: 保持原色（撕裂面用蓝色，剪切面用红色）
    color_swap_mask[tear_mask > 0] = [0, 0, 1, 1]  # 撕裂面用蓝色，不透明
    color_swap_mask[shear_mask > 0] = [1, 0, 0, 1]  # 剪切面用红色，不透明
    
    axes[0].imshow(original_rgb)
    axes[0].imshow(color_swap_mask)
    axes[0].set_title('Color Swap Comparison\n(Before: Red=Tear, Blue=Shear)\n(After: Blue=Tear, Red=Shear)')
    axes[0].axis('off')
    
    # 只显示Before fill的调换颜色
    before_swap_mask = np.zeros((*tear_mask.shape, 4))
    before_swap_mask[tear_mask_original > 0] = [1, 0, 0, 1]  # 撕裂面用红色，不透明
    before_swap_mask[shear_mask_original > 0] = [0, 0, 1, 1]  # 剪切面用蓝色，不透明
    
    axes[1].imshow(original_rgb)
    axes[1].imshow(before_swap_mask)
    axes[1].set_title('Before Fill (Color Swapped)\n(Red: Tear, Blue: Shear)')
    axes[1].axis('off')
    
    # 只显示After fill的原始颜色
    after_original_mask = np.zeros((*tear_mask.shape, 4))
    after_original_mask[tear_mask > 0] = [0, 0, 1, 1]  # 撕裂面用蓝色，不透明
    after_original_mask[shear_mask > 0] = [1, 0, 0, 1]  # 剪切面用红色，不透明
    
    axes[2].imshow(original_rgb)
    axes[2].imshow(after_original_mask)
    axes[2].set_title('After Fill (Original Colors)\n(Blue: Tear, Red: Shear)')
    axes[2].axis('off')
    
    # 保存单独的颜色调换对比图
    color_swap_output_path = output_path.replace('_detailed_segmentation.png', '_color_swap_comparison.png')
    plt.tight_layout()
    plt.savefig(color_swap_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"颜色调换对比图已保存: {color_swap_output_path}")

def main():
    """主函数"""
    # 图像目录
    roi_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_split/roi_images"
    output_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_split/output"
    
    # 获取所有PNG文件
    png_files = [f for f in os.listdir(roi_dir) if f.endswith('.png')]
    png_files.sort()
    
    print("创建详细的分割可视化...")
    print("=" * 60)
    
    for filename in png_files:
        img_path = os.path.join(roi_dir, filename)
        output_filename = f"{os.path.splitext(filename)[0]}_detailed_segmentation.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"处理: {filename}")
        create_detailed_segmentation_visualization(img_path, output_path)
    
    print("\n详细分割可视化完成!")

if __name__ == "__main__":
    main()

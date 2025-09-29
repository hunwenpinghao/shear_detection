#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
梯度等高线检测功能演示脚本

这个脚本演示了如何使用新增的梯度等高线检测功能：
1. 检测final mask右边缘线往左第一条梯度等高线
2. 对等高线进行上下方向插值和平滑滤波
3. 生成可视化结果

使用方法：
python test_gradient_contour_demo.py [image_path]
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from test_count_black_strips import (
    detect_gradient_contour_from_right_edge,
    interpolate_and_smooth_contour,
    visualize_gradient_contour_detection,
    ensure_binary,
    setup_chinese_font
)

def demo_gradient_contour_detection(image_path: str, output_dir: str = None):
    """
    演示梯度等高线检测功能
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录，如果为None则使用图像所在目录
    """
    print("=" * 60)
    print("梯度等高线检测功能演示")
    print("=" * 60)
    
    # 设置中文字体
    setup_chinese_font()

    # 读取图像
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"无法读取图像: {image_path}")
        return
    
    print(f"输入图像: {image_path}")
    print(f"图像尺寸: {gray.shape}")
    
    # 确保为二值图像
    binary = ensure_binary(gray)
    print("图像已转换为二值图")
    
    # 设置输出目录：固定到 data_adagaus_density_curve/contour_ 前缀
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    module_dir = os.path.dirname(__file__)
    output_dir = os.path.join(module_dir, 'gradient_contour_demo_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 参数设置
    gradient_threshold = 0.2  # 梯度阈值
    search_width = 60         # 搜索宽度
    edge_sigma = 5.0          # 边缘平滑sigma
    smoothing_sigma = 2.0     # 等高线平滑sigma
    contraction_ratio = 0.1   # 边缘收缩比例
    
    print(f"\n检测参数:")
    print(f"  梯度阈值: {gradient_threshold}")
    print(f"  搜索宽度: {search_width} 像素")
    print(f"  边缘平滑sigma: {edge_sigma}")
    print(f"  等高线平滑sigma: {smoothing_sigma}")
    print(f"  边缘收缩比例: {contraction_ratio}")
    
    # 1. 检测梯度等高线
    print("\n1. 开始检测梯度等高线...")
    contour_x, contour_y, right_edges, gradient_map = detect_gradient_contour_from_right_edge(
        binary,
        gradient_threshold=gradient_threshold,
        search_width=search_width,
        edge_sigma=edge_sigma,
        contraction_ratio=contraction_ratio
    )
    
    print(f"   检测到 {len(contour_x)} 个等高线点")
    print(f"   右边缘线长度: {len(right_edges)}")
    
    # 2. 插值和平滑
    print("\n2. 对等高线进行插值和平滑...")
    smooth_x, smooth_y = interpolate_and_smooth_contour(
        contour_x,
        contour_y,
        image_height=binary.shape[0],
        image_width=binary.shape[1],
        right_edges=right_edges,
        closing_size=11,
        smoothing_sigma=smoothing_sigma,
    )
    
    print(f"   平滑后等高线长度: {len(smooth_x)}")
    
    # 3. 计算收缩区域用于可视化
    print("\n3. 计算收缩区域...")
    _, otsu_binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = binary.shape
    left_edges = np.full(h, w, dtype=int)
    for y in range(h):
        xs = np.where(otsu_binary[y] == 255)[0]
        if xs.size:
            left_edges[y] = xs.min()
    
    # 平滑左右边缘
    yy = np.arange(h)
    left_valid = left_edges < w
    right_valid = right_edges >= 0
    
    if np.any(left_valid):
        left_interp = np.interp(yy, yy[left_valid], left_edges[left_valid].astype(float))
        left_sm = gaussian_filter1d(left_interp, sigma=edge_sigma, mode='nearest')
        left_sm = np.clip(np.rint(left_sm), 0, w - 1).astype(int)
    else:
        left_sm = left_edges
        
    if np.any(right_valid):
        right_interp = np.interp(yy, yy[right_valid], right_edges[right_valid].astype(float))
        right_sm = gaussian_filter1d(right_interp, sigma=edge_sigma, mode='nearest')
        right_sm = np.clip(np.rint(right_sm), 0, w - 1).astype(int)
    else:
        right_sm = right_edges
    
    # 计算收缩区域
    contracted_left = left_sm + np.round((right_sm - left_sm) * contraction_ratio).astype(int)
    contracted_right = right_sm - np.round((right_sm - left_sm) * contraction_ratio).astype(int)
    contracted_left = np.clip(contracted_left, 0, w - 1)
    contracted_right = np.clip(contracted_right, 0, w - 1)
    
    # 4. 生成可视化
    print("\n4. 生成可视化结果...")
    output_path = os.path.join(output_dir, f"contour_{base_name}_demo.png")
    
    visualize_gradient_contour_detection(
        binary, contour_x, contour_y, right_edges, gradient_map,
        smooth_x, smooth_y, output_path,
        left_edges=left_sm, contracted_left=contracted_left, contracted_right=contracted_right
    )
    
    print(f"   可视化结果已保存: {output_path}")
    
    # 5. 统计信息
    print("\n5. 统计信息:")
    if len(contour_x) > 0:
        print(f"   等高线点分布:")
        print(f"     X坐标范围: {np.min(contour_x):.1f} - {np.max(contour_x):.1f}")
        print(f"     Y坐标范围: {np.min(contour_y):.1f} - {np.max(contour_y):.1f}")
        
        # 计算右边缘到等高线的平均距离
        if len(smooth_x) == len(right_edges):
            distances = right_edges - smooth_x
            valid_distances = distances[distances > 0]
            if len(valid_distances) > 0:
                print(f"   右边缘到等高线的距离:")
                print(f"     平均距离: {np.mean(valid_distances):.1f} 像素")
                print(f"     最小距离: {np.min(valid_distances):.1f} 像素")
                print(f"     最大距离: {np.max(valid_distances):.1f} 像素")

    # 6. 新指标：等高线占左右边界宽度的平均占比
    print("\n6. 等高线占比指标:")
    h = binary.shape[0]
    # 重新计算左右边界平滑（与上文保持一致）
    _, otsu_binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    left_edges = np.full(h, binary.shape[1], dtype=int)
    for y in range(h):
        xs = np.where(otsu_binary[y] == 255)[0]
        if xs.size:
            left_edges[y] = xs.min()
    yy = np.arange(h)
    left_valid = left_edges < binary.shape[1]
    right_valid = right_edges >= 0
    if np.any(left_valid):
        left_interp = np.interp(yy, yy[left_valid], left_edges[left_valid].astype(float))
        left_sm = gaussian_filter1d(left_interp, sigma=edge_sigma, mode='nearest')
        left_sm = np.clip(np.rint(left_sm), 0, binary.shape[1] - 1).astype(int)
    else:
        left_sm = left_edges
    if np.any(right_valid):
        right_interp = np.interp(yy, yy[right_valid], right_edges[right_valid].astype(float))
        right_sm = gaussian_filter1d(right_interp, sigma=edge_sigma, mode='nearest')
        right_sm = np.clip(np.rint(right_sm), 0, binary.shape[1] - 1).astype(int)
    else:
        right_sm = right_edges
    width_lr = (right_sm - left_sm).astype(float)
    if len(smooth_x) == h:
        safe_width = np.where(width_lr == 0, 1.0, width_lr)
        ratios = (smooth_x - left_sm) / safe_width
        ratios = np.clip(ratios, 0.0, 1.0)
        valid_rows = width_lr > 0
        if np.any(valid_rows):
            avg_ratio = float(np.mean(ratios[valid_rows]))
            print(f"   平均占比: {avg_ratio:.4f}")
        else:
            print("   平均占比: 无有效行")
    
    print("\n" + "=" * 60)
    print("梯度等高线检测演示完成!")
    print("=" * 60)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        # 使用默认图像
        default_image = os.path.join(os.path.dirname(__file__), 'adagaus_imgs', 'frame_000000_adagaus.png')
        if os.path.exists(default_image):
            image_path = default_image
        else:
            print("请提供图像路径作为参数")
            print("使用方法: python test_gradient_contour_demo.py <image_path>")
            sys.exit(1)
    else:
        image_path = sys.argv[1]
    
    # 设置输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'gradient_contour_demo_output')
    
    # 运行演示
    demo_gradient_contour_detection(image_path, output_dir)

if __name__ == "__main__":
    main()

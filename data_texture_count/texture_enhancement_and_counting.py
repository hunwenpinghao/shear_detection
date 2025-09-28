#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纹理增强和计数脚本
用于增强纹理图像的清晰度并统计纹理条纹或块的数量

模型信息：
- 模型名称: GPT-4
- 模型大小: 大模型
- 模型类型: 多模态大语言模型
- 更新日期: 2024年12月
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation
from skimage.feature import local_binary_pattern
import argparse
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TextureEnhancer:
    """纹理增强器类"""
    
    def __init__(self):
        self.enhanced_image = None
        self.binary_image = None
        
    def load_image(self, image_path):
        """加载图像"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        return gray
    
    def enhance_texture(self, image, method='combined'):
        """
        增强纹理清晰度
        
        Args:
            image: 输入灰度图像
            method: 增强方法 ('clahe', 'unsharp', 'gradient', 'combined')
        
        Returns:
            增强后的图像
        """
        enhanced = image.copy()
        
        if method == 'clahe' or method == 'combined':
            # CLAHE (对比度限制自适应直方图均衡化)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(enhanced)
        
        if method == 'unsharp' or method == 'combined':
            # 反锐化掩模
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        if method == 'gradient' or method == 'combined':
            # 梯度增强
            grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_magnitude = np.uint8(255 * gradient_magnitude / gradient_magnitude.max())
            
            # 将梯度信息融合到原图像
            enhanced = cv2.addWeighted(enhanced, 0.7, gradient_magnitude, 0.3, 0)
        
        # 形态学操作增强纹理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        self.enhanced_image = enhanced
        return enhanced
    
    def create_binary_mask(self, image, method='adaptive'):
        """
        创建二值化掩码用于纹理分割
        
        Args:
            image: 输入图像
            method: 二值化方法 ('otsu', 'adaptive', 'threshold')
        
        Returns:
            二值化图像
        """
        if method == 'otsu':
            # Otsu阈值
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            # 自适应阈值
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        else:
            # 固定阈值
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        self.binary_image = binary
        return binary

class TextureCounter:
    """纹理计数器类"""
    
    def __init__(self):
        self.texture_count = 0
        self.texture_regions = []
        
    def count_horizontal_stripes(self, image, min_width=5, min_height=10):
        """
        统计水平条纹数量
        
        Args:
            image: 二值化图像
            min_width: 最小条纹宽度
            min_height: 最小条纹高度
        
        Returns:
            条纹数量
        """
        # 水平投影
        horizontal_projection = np.sum(image, axis=1)
        
        # 找到条纹边界
        threshold = np.mean(horizontal_projection) * 0.3
        stripes = horizontal_projection > threshold
        
        # 计算连续区域
        stripe_regions = []
        in_stripe = False
        start = 0
        
        for i, is_stripe in enumerate(stripes):
            if is_stripe and not in_stripe:
                start = i
                in_stripe = True
            elif not is_stripe and in_stripe:
                if i - start >= min_height:
                    stripe_regions.append((start, i))
                in_stripe = False
        
        # 处理最后一个条纹
        if in_stripe and len(image) - start >= min_height:
            stripe_regions.append((start, len(image)))
        
        self.texture_count = len(stripe_regions)
        self.texture_regions = stripe_regions
        
        return len(stripe_regions)
    
    def count_vertical_stripes(self, image, min_width=10, min_height=5):
        """
        统计垂直条纹数量
        
        Args:
            image: 二值化图像
            min_width: 最小条纹宽度
            min_height: 最小条纹高度
        
        Returns:
            条纹数量
        """
        # 垂直投影
        vertical_projection = np.sum(image, axis=0)
        
        # 找到条纹边界
        threshold = np.mean(vertical_projection) * 0.3
        stripes = vertical_projection > threshold
        
        # 计算连续区域
        stripe_regions = []
        in_stripe = False
        start = 0
        
        for i, is_stripe in enumerate(stripes):
            if is_stripe and not in_stripe:
                start = i
                in_stripe = True
            elif not is_stripe and in_stripe:
                if i - start >= min_width:
                    stripe_regions.append((start, i))
                in_stripe = False
        
        # 处理最后一个条纹
        if in_stripe and len(image[0]) - start >= min_width:
            stripe_regions.append((start, len(image[0])))
        
        self.texture_count = len(stripe_regions)
        self.texture_regions = stripe_regions
        
        return len(stripe_regions)
    
    def count_texture_blocks(self, image, min_area=100):
        """
        统计纹理块数量（连通组件分析）
        
        Args:
            image: 二值化图像
            min_area: 最小块面积
        
        Returns:
            块数量
        """
        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        
        # 过滤小区域
        valid_blocks = []
        for i in range(1, num_labels):  # 跳过背景标签0
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                valid_blocks.append(i)
        
        self.texture_count = len(valid_blocks)
        self.texture_regions = valid_blocks
        
        return len(valid_blocks)
    
    def count_using_watershed(self, image, min_distance=20):
        """
        使用分水岭算法统计纹理区域
        
        Args:
            image: 输入图像
            min_distance: 最小距离参数
        
        Returns:
            区域数量
        """
        # 距离变换
        dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        
        # 寻找局部最大值作为种子点
        from scipy import ndimage
        # 使用形态学重建寻找局部最大值
        kernel = np.ones((min_distance, min_distance))
        local_maxima = ndimage.maximum_filter(dist_transform, size=min_distance) == dist_transform
        local_maxima = local_maxima & (dist_transform > 0.5 * dist_transform.max())
        
        # 创建标记
        markers = np.zeros_like(image, dtype=np.int32)
        labeled_maxima, num_maxima = ndimage.label(local_maxima)
        markers[labeled_maxima > 0] = labeled_maxima[labeled_maxima > 0]
        
        # 分水岭算法
        labels = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
        
        # 统计区域数量（排除背景和边界）
        unique_labels = np.unique(labels)
        region_count = len(unique_labels) - 2  # 减去背景(-1)和边界(0)
        
        self.texture_count = region_count
        self.texture_regions = unique_labels[1:-1]  # 排除背景和边界
        
        return region_count

def process_single_image(image_path, output_dir=None, show_results=True):
    """
    处理单张图像
    
    Args:
        image_path: 图像路径
        output_dir: 输出目录
        show_results: 是否显示结果
    
    Returns:
        处理结果字典
    """
    print(f"正在处理图像: {image_path}")
    
    # 创建输出目录
    if output_dir is None:
        # 如果在data_texture_count目录内，直接在当前目录创建输出
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if "data_texture_count" in current_dir:
            output_dir = os.path.join(current_dir, "texture_analysis_output")
        else:
            output_dir = os.path.join(os.path.dirname(image_path), "texture_analysis_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化处理器
    enhancer = TextureEnhancer()
    counter = TextureCounter()
    
    try:
        # 加载图像
        original_image = enhancer.load_image(image_path)
        
        # 增强纹理
        enhanced_image = enhancer.enhance_texture(original_image, method='combined')
        
        # 创建二值化掩码
        binary_image = enhancer.create_binary_mask(enhanced_image, method='adaptive')
        
        # 统计纹理数量（多种方法）
        results = {}
        
        # 水平条纹计数
        horizontal_count = counter.count_horizontal_stripes(binary_image)
        results['horizontal_stripes'] = horizontal_count
        
        # 垂直条纹计数
        vertical_count = counter.count_vertical_stripes(binary_image)
        results['vertical_stripes'] = vertical_count
        
        # 纹理块计数
        block_count = counter.count_texture_blocks(binary_image)
        results['texture_blocks'] = block_count
        
        # 分水岭算法计数
        watershed_count = counter.count_using_watershed(binary_image)
        results['watershed_regions'] = watershed_count
        
        # 保存结果图像
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存增强图像
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_enhanced.png"), enhanced_image)
        
        # 保存二值化图像
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary.png"), binary_image)
        
        # 创建可视化结果
        if show_results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Texture Analysis Results - {os.path.basename(image_path)}', fontsize=16)
            
            # 原图
            axes[0, 0].imshow(original_image, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # 增强图像
            axes[0, 1].imshow(enhanced_image, cmap='gray')
            axes[0, 1].set_title('Enhanced Image')
            axes[0, 1].axis('off')
            
            # 二值化图像
            axes[1, 0].imshow(binary_image, cmap='gray')
            axes[1, 0].set_title('Binary Image')
            axes[1, 0].axis('off')
            
            # 结果统计
            axes[1, 1].text(0.1, 0.8, f'Horizontal Stripes: {horizontal_count}', fontsize=12, transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.6, f'Vertical Stripes: {vertical_count}', fontsize=12, transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.4, f'Texture Blocks: {block_count}', fontsize=12, transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.2, f'Watershed Regions: {watershed_count}', fontsize=12, transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Statistics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_name}_analysis.png"), dpi=300, bbox_inches='tight')
        
        # 保存结果到文件
        result_file = os.path.join(output_dir, f"{base_name}_results.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"Texture Analysis Results - {os.path.basename(image_path)}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Horizontal Stripes: {horizontal_count}\n")
            f.write(f"Vertical Stripes: {vertical_count}\n")
            f.write(f"Texture Blocks: {block_count}\n")
            f.write(f"Watershed Regions: {watershed_count}\n")
            f.write("\nRecommendation: Choose the most appropriate method based on image characteristics\n")
        
        print(f"处理完成，结果保存到: {output_dir}")
        return results
        
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None

def process_folder(folder_path, output_dir=None, show_results=False):
    """
    处理文件夹中的所有图像
    
    Args:
        folder_path: 文件夹路径
        output_dir: 输出目录
        show_results: 是否显示结果
    
    Returns:
        所有图像的处理结果
    """
    print(f"正在处理文件夹: {folder_path}")
    
    # 支持的图像格式
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    
    # 查找所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"在文件夹 {folder_path} 中未找到图像文件")
        return None
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(folder_path, "texture_analysis_output")
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for image_path in image_files:
        try:
            results = process_single_image(image_path, output_dir, show_results)
            if results:
                all_results[os.path.basename(image_path)] = results
        except Exception as e:
            print(f"处理 {image_path} 时出错: {e}")
    
    # 保存汇总结果
    summary_file = os.path.join(output_dir, "summary_results.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Texture Analysis Summary Results\n")
        f.write("=" * 50 + "\n\n")
        
        for filename, results in all_results.items():
            f.write(f"File: {filename}\n")
            f.write(f"  Horizontal Stripes: {results['horizontal_stripes']}\n")
            f.write(f"  Vertical Stripes: {results['vertical_stripes']}\n")
            f.write(f"  Texture Blocks: {results['texture_blocks']}\n")
            f.write(f"  Watershed Regions: {results['watershed_regions']}\n")
            f.write("\n")
    
    print(f"文件夹处理完成，汇总结果保存到: {summary_file}")
    return all_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='纹理增强和计数工具')
    parser.add_argument('input_path', help='输入图像文件或文件夹路径')
    parser.add_argument('--output', '-o', help='输出目录路径')
    parser.add_argument('--show', '-s', action='store_true', help='显示处理结果')
    parser.add_argument('--method', '-m', choices=['horizontal', 'vertical', 'blocks', 'watershed', 'all'], 
                       default='all', help='计数方法')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"错误: 输入路径不存在: {args.input_path}")
        return
    
    if os.path.isfile(args.input_path):
        # 处理单张图像
        results = process_single_image(args.input_path, args.output, args.show)
        if results:
            print("\n处理结果:")
            for method, count in results.items():
                print(f"  {method}: {count}")
    
    elif os.path.isdir(args.input_path):
        # 处理文件夹
        results = process_folder(args.input_path, args.output, args.show)
        if results:
            print(f"\n处理了 {len(results)} 个文件")
            print("详细结果请查看输出目录中的汇总文件")
    
    else:
        print(f"错误: 无效的输入路径: {args.input_path}")

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认测试路径
    import sys
    if len(sys.argv) == 1:
        # 测试单张图像
        test_image = "../data/纹理图示例.png"
        if os.path.exists(test_image):
            print("使用默认测试图像进行演示...")
            results = process_single_image(test_image, show_results=True)
            if results:
                print("\n处理结果:")
                for method, count in results.items():
                    print(f"  {method}: {count}")
        else:
            print(f"测试图像不存在: {test_image}")
            print("请提供正确的图像路径作为命令行参数")
    else:
        main()

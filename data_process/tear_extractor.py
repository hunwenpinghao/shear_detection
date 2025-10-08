#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
撕裂面提取器
从ROI图像中提取撕裂面mask，使用adaptive gaussian二值图加等高线检测方法
"""

import cv2
import numpy as np
import os
import sys
import glob
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, grey_closing
from scipy.interpolate import interp1d
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TearExtractor:
    """撕裂面提取器 - 使用等高线方法提取撕裂面mask"""
    
    def __init__(self, output_dir: str = None, visualize: bool = True):
        """
        初始化撕裂面提取器
        
        Args:
            output_dir: 输出目录路径
            visualize: 是否生成可视化结果
        """
        self.output_dir = output_dir if output_dir else "data_process/tear_masks"
        self.visualize = visualize
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        if visualize:
            self.vis_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.vis_dir, exist_ok=True)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像（灰度图）
            
        Returns:
            预处理后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯滤波去噪
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 直方图均衡化增强对比度
        enhanced = cv2.equalizeHist(denoised)
        
        return enhanced
    
    def extract_tear_mask_with_contour(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        使用等高线方法提取撕裂面mask
        
        Args:
            image: 输入图像（灰度图）
            
        Returns:
            tuple: (tear_mask, info_dict)
                - tear_mask: 撕裂面二值mask
                - info_dict: 包含中间结果的字典
        """
        # 预处理图像
        processed = self.preprocess_image(image)
        h, w = image.shape[:2]
        
        # 1. 计算Otsu二值化（在原始图像上）
        _, otsu_binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. 计算左右边缘（不收缩的原始边缘）
        left_edges = np.full(h, w, dtype=int)
        right_edges = np.full(h, -1, dtype=int)
        
        for y in range(h):
            cols = np.where(otsu_binary[y] > 0)[0]
            if cols.size:
                left_edges[y] = cols.min()
                right_edges[y] = cols.max()
        
        # 3. 平滑左右边缘
        edge_sigma = 5.0
        yy = np.arange(h)
        valid = (left_edges < w) & (right_edges >= 0) & (right_edges > left_edges)
        
        if np.any(valid):
            left_interp = np.interp(yy, yy[valid], left_edges[valid].astype(float))
            right_interp = np.interp(yy, yy[valid], right_edges[valid].astype(float))
            left_sm = gaussian_filter1d(left_interp, sigma=edge_sigma, mode='nearest')
            right_sm = gaussian_filter1d(right_interp, sigma=edge_sigma, mode='nearest')
            left_sm = np.clip(np.rint(left_sm), 0, w - 1).astype(int)
            right_sm = np.clip(np.rint(right_sm), 0, w - 1).astype(int)
        else:
            left_sm, right_sm = left_edges, right_edges
        
        # 4. 计算收缩后的边界用于等高线检测
        contraction_ratio = 0.1
        contracted_left = left_sm + np.round((right_sm - left_sm) * contraction_ratio).astype(int)
        contracted_right = right_sm - np.round((right_sm - left_sm) * contraction_ratio).astype(int)
        contracted_left = np.clip(contracted_left, 0, w - 1)
        contracted_right = np.clip(contracted_right, 0, w - 1)
        
        # 5. 计算梯度（从右到左白->黑取负号）
        sobel_x = cv2.Sobel(processed, cv2.CV_32F, 1, 0, ksize=3)
        gradient_map = -sobel_x
        
        # 6. 在收缩区域内搜索最大梯度位置得到等高线
        contour_x = np.full(h, -1, dtype=float)
        contour_y = []
        
        for y in range(h):
            lb = int(contracted_left[y])
            rb = int(contracted_right[y])
            if lb >= 0 and rb < w and rb > lb:
                row = np.abs(gradient_map[y, lb:rb + 1])
                if row.size > 0:
                    x_rel = int(np.argmax(row))
                    contour_x[y] = lb + x_rel
                    contour_y.append(y)
        
        contour_y = np.array(contour_y)
        
        # 7. 插值和平滑等高线
        if len(contour_x) > 1 and len(contour_y) > 0:
            y_full = np.arange(h)
            f_x = interp1d(contour_y, contour_x[contour_y], kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
            x_interp = f_x(y_full).astype(float)
            
            # 不超过右边界
            x_interp = np.minimum(x_interp, right_sm.astype(float))
            
            # 形态学闭运算填坑
            closing_size = 11
            if closing_size % 2 == 0:
                closing_size += 1
            x_closed = grey_closing(x_interp, size=closing_size, mode='nearest')
            
            # 高斯平滑
            x_smooth = gaussian_filter1d(x_closed, sigma=2.0, mode='nearest')
            x_smooth = np.minimum(x_smooth, right_sm.astype(float))
            x_smooth = np.clip(x_smooth, 0, w - 1)
        else:
            # 如果没有找到等高线，使用中间位置
            x_smooth = (left_sm + right_sm) / 2.0
        
        # 8. 基于等高线提取撕裂面（从左边缘到等高线）
        tear_mask = np.zeros((h, w), dtype=np.uint8)
        
        for y in range(h):
            if left_sm[y] < w and right_sm[y] >= 0 and right_sm[y] > left_sm[y]:
                # 撕裂面：从左边缘到等高线
                tear_start = left_sm[y]
                tear_end = int(np.round(x_smooth[y]))
                if tear_end > tear_start:
                    tear_mask[y, tear_start:tear_end] = 255
        
        # 9. 形态学操作优化mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tear_mask = cv2.morphologyEx(tear_mask, cv2.MORPH_CLOSE, kernel)
        tear_mask = cv2.morphologyEx(tear_mask, cv2.MORPH_OPEN, kernel)
        
        # 10. 返回结果和中间信息
        info_dict = {
            'processed_image': processed,
            'otsu_binary': otsu_binary,
            'left_edges': left_sm,
            'right_edges': right_sm,
            'contour_line': x_smooth,
            'gradient_map': gradient_map,
            'contraction_ratio': contraction_ratio
        }
        
        return tear_mask, info_dict
    
    def visualize_extraction(self, image: np.ndarray, tear_mask: np.ndarray, 
                           info_dict: Dict[str, Any], output_path: str):
        """
        可视化撕裂面提取过程
        
        Args:
            image: 原始图像
            tear_mask: 撕裂面mask
            info_dict: 中间结果信息
            output_path: 输出路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Tear Surface Extraction with Contour Method', fontsize=16)
        
        # 转换为RGB用于叠加显示
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
        
        # 第一行
        # 1. 原始图像
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2. 预处理图像
        axes[0, 1].imshow(info_dict['processed_image'], cmap='gray')
        axes[0, 1].set_title('Preprocessed Image\n(Gaussian Blur + Histogram Equalization)')
        axes[0, 1].axis('off')
        
        # 3. Otsu二值化
        axes[0, 2].imshow(info_dict['otsu_binary'], cmap='gray')
        axes[0, 2].set_title('Otsu Binarization')
        axes[0, 2].axis('off')
        
        # 第二行
        # 4. 边缘和等高线
        axes[1, 0].imshow(image, cmap='gray')
        y_coords = np.arange(len(info_dict['left_edges']))
        axes[1, 0].plot(info_dict['left_edges'], y_coords, 'g-', linewidth=2, label='Left Edge')
        axes[1, 0].plot(info_dict['right_edges'], y_coords, 'b-', linewidth=2, label='Right Edge')
        axes[1, 0].plot(info_dict['contour_line'], y_coords, 'r-', linewidth=2, label='Contour Line')
        axes[1, 0].set_title('Edges and Contour Line Detection')
        axes[1, 0].legend(loc='upper right')
        axes[1, 0].axis('off')
        
        # 5. 梯度图
        axes[1, 1].imshow(np.abs(info_dict['gradient_map']), cmap='hot')
        axes[1, 1].set_title('Horizontal Gradient Map\n(Used for Contour Detection)')
        axes[1, 1].axis('off')
        
        # 6. 撕裂面mask叠加显示
        tear_overlay = image_rgb.copy()
        tear_overlay[tear_mask > 0] = [255, 0, 0]  # 红色表示撕裂面
        
        axes[1, 2].imshow(image_rgb)
        axes[1, 2].imshow(tear_overlay, alpha=0.5)
        axes[1, 2].set_title('Tear Surface Mask Overlay\n(Red: Tear Surface)')
        axes[1, 2].axis('off')
        
        # 添加统计信息
        tear_pixels = np.sum(tear_mask > 0)
        total_pixels = tear_mask.size
        tear_ratio = tear_pixels / total_pixels * 100
        
        stats_text = f"""Extraction Statistics:
        
Total Pixels: {total_pixels:,}
Tear Surface Pixels: {tear_pixels:,}
Tear Surface Ratio: {tear_ratio:.2f}%
Contraction Ratio: {info_dict['contraction_ratio']:.2f}
"""
        
        fig.text(0.98, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                horizontalalignment='right', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_single_image(self, image_path: str, output_mask_path: str = None,
                           output_vis_path: str = None) -> Dict[str, Any]:
        """
        处理单张图像，提取撕裂面mask
        
        Args:
            image_path: 输入图像路径
            output_mask_path: 输出mask路径（可选）
            output_vis_path: 输出可视化路径（可选）
            
        Returns:
            处理结果字典
        """
        try:
            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return {'success': False, 'error': f'无法读取图像: {image_path}'}
            
            # 提取撕裂面mask
            tear_mask, info_dict = self.extract_tear_mask_with_contour(image)
            
            # 保存mask
            if output_mask_path:
                cv2.imwrite(output_mask_path, tear_mask)
            
            # 生成可视化
            if self.visualize and output_vis_path:
                self.visualize_extraction(image, tear_mask, info_dict, output_vis_path)
            
            # 统计信息
            tear_pixels = np.sum(tear_mask > 0)
            total_pixels = tear_mask.size
            tear_ratio = tear_pixels / total_pixels
            
            return {
                'success': True,
                'tear_mask': tear_mask,
                'tear_pixels': tear_pixels,
                'total_pixels': total_pixels,
                'tear_ratio': tear_ratio,
                'output_mask_path': output_mask_path,
                'output_vis_path': output_vis_path
            }
            
        except Exception as e:
            return {'success': False, 'error': f'处理图像时出错: {str(e)}'}
    
    def extract_tears_from_directory(self, input_dir: str, 
                                     output_mask_dir: str = None,
                                     pattern: str = "*.png") -> Dict[str, Any]:
        """
        批量处理目录中的图像
        
        Args:
            input_dir: 输入图像目录
            output_mask_dir: 输出mask目录（可选，默认使用self.output_dir）
            pattern: 图像文件匹配模式
            
        Returns:
            处理结果统计
        """
        print(f"=== 批量提取撕裂面mask ===")
        print(f"输入目录: {input_dir}")
        
        # 设置输出目录
        if output_mask_dir is None:
            output_mask_dir = self.output_dir
        
        os.makedirs(output_mask_dir, exist_ok=True)
        print(f"输出mask目录: {output_mask_dir}")
        
        if self.visualize:
            vis_dir = os.path.join(output_mask_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            print(f"可视化目录: {vis_dir}")
        
        try:
            # 获取所有图像文件
            search_pattern = os.path.join(input_dir, pattern)
            image_files = sorted(glob.glob(search_pattern))
            
            if not image_files:
                return {'success': False, 'error': f'未找到匹配的图像文件: {search_pattern}'}
            
            print(f"找到 {len(image_files)} 个图像文件")
            print(f"开始处理...")
            
            # 批量处理
            success_count = 0
            failed_count = 0
            results = []
            
            for img_path in tqdm(image_files, desc="提取撕裂面mask"):
                try:
                    # 生成输出文件名
                    basename = os.path.basename(img_path)
                    name, ext = os.path.splitext(basename)
                    
                    # mask输出路径
                    mask_filename = f"{name}_tear_mask.png"
                    mask_output_path = os.path.join(output_mask_dir, mask_filename)
                    
                    # 可视化输出路径
                    vis_output_path = None
                    if self.visualize:
                        vis_filename = f"{name}_tear_visualization.png"
                        vis_output_path = os.path.join(vis_dir, vis_filename)
                    
                    # 检查文件是否已存在
                    if os.path.exists(mask_output_path):
                        success_count += 1
                        continue
                    
                    # 处理单张图像
                    result = self.process_single_image(
                        img_path, 
                        mask_output_path,
                        vis_output_path
                    )
                    
                    if result['success']:
                        success_count += 1
                        results.append({
                            'filename': basename,
                            'tear_ratio': result['tear_ratio'],
                            'tear_pixels': result['tear_pixels']
                        })
                    else:
                        failed_count += 1
                        print(f"处理失败 {basename}: {result.get('error', '未知错误')}")
                
                except Exception as e:
                    failed_count += 1
                    print(f"处理图像时出错 {img_path}: {e}")
                    continue
            
            print(f"\n撕裂面提取完成！")
            print(f"总文件数: {len(image_files)}")
            print(f"成功: {success_count}")
            print(f"失败: {failed_count}")
            
            return {
                'success': True,
                'total_files': len(image_files),
                'success_count': success_count,
                'failed_count': failed_count,
                'output_mask_dir': output_mask_dir,
                'results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': f'批量处理时出错: {str(e)}'}


def main():
    """主函数"""
    # 默认值
    default_input_dir = "data_shear_split/roi_images"
    default_output_dir = "data_shear_split/tear_masks"
    default_pattern = "*.png"
    default_visualize = True
    
    # 使用sys.argv获取参数
    visualize = default_visualize
    pattern = default_pattern
    
    if len(sys.argv) >= 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        # 检查是否有额外参数
        if len(sys.argv) >= 4:
            pattern = sys.argv[3]
        if len(sys.argv) >= 5 and sys.argv[4] == "--no-visualize":
            visualize = False
    elif len(sys.argv) == 2:
        input_dir = sys.argv[1]
        output_dir = os.path.join(os.path.dirname(input_dir), "tear_masks")
        print(f"使用默认输出目录: {output_dir}")
    else:
        input_dir = default_input_dir
        output_dir = default_output_dir
        print(f"使用默认输入目录: {input_dir}")
        print(f"使用默认输出目录: {output_dir}")
        print(f"\n使用方法: python tear_extractor.py <input_dir> [output_dir] [pattern] [--no-visualize]")
        print(f"示例: python tear_extractor.py data/roi_imgs data/tear_masks *.png")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 - {input_dir}")
        return
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n撕裂面提取器")
    print(f"=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"匹配模式: {pattern}")
    print(f"可视化: {'是' if visualize else '否'}")
    print(f"=" * 60)
    
    # 创建提取器
    extractor = TearExtractor(
        output_dir=output_dir,
        visualize=visualize
    )
    
    # 执行批量提取
    result = extractor.extract_tears_from_directory(
        input_dir=input_dir,
        pattern=pattern
    )
    
    # 输出结果
    if result['success']:
        print(f"\n✅ 撕裂面提取成功！")
        print(f"总文件数: {result['total_files']}")
        print(f"成功处理: {result['success_count']}")
        print(f"失败: {result['failed_count']}")
        print(f"输出目录: {result['output_mask_dir']}")
        
        # 输出统计信息
        if result['results']:
            print(f"\n撕裂面占比统计:")
            tear_ratios = [r['tear_ratio'] for r in result['results']]
            print(f"  平均占比: {np.mean(tear_ratios)*100:.2f}%")
            print(f"  最小占比: {np.min(tear_ratios)*100:.2f}%")
            print(f"  最大占比: {np.max(tear_ratios)*100:.2f}%")
            print(f"  标准差: {np.std(tear_ratios)*100:.2f}%")
    else:
        print(f"❌ 撕裂面提取失败: {result['error']}")


if __name__ == "__main__":
    main()


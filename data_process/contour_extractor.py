#!/usr/bin/env python3
"""
等高线提取器
从ROI图像中提取等高线特征，计算等高线平均占比，保存到contour_imgs目录
"""

import cv2
import numpy as np
import os
import sys
import glob
from typing import List, Dict, Any, Optional, Tuple
import time
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d, grey_closing
from scipy.interpolate import interp1d

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))
# 添加根目录到路径，以便导入adagaus_extractor
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from adagaus_extractor import AdagausExtractor
from config import PREPROCESS_CONFIG


class ContourExtractor:
    """等高线提取器"""
    
    def __init__(self, config: Dict[str, Any] = None, base_output_dir: str = None):
        """
        初始化等高线提取器
        
        Args:
            config: 预处理配置
            base_output_dir: 基础输出目录
        """
        self.config = config if config is not None else PREPROCESS_CONFIG
        self.adagaus_extractor = AdagausExtractor()
        
        # 设置目录路径
        if base_output_dir is None:
            self.base_output_dir = "data_adagaus_density_curve"
        else:
            self.base_output_dir = base_output_dir
            
        self.roi_dir = os.path.join(self.base_output_dir, 'roi_imgs')
        self.contour_dir = os.path.join(self.base_output_dir, 'contour_imgs')
        
        # 创建等高线输出目录
        os.makedirs(self.contour_dir, exist_ok=True)
    
    def detect_gradient_contour_from_right_edge(self, final_mask: np.ndarray, 
                                               filtered_adagaus: np.ndarray,
                                               contraction_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从右边缘检测梯度等高线
        
        Args:
            final_mask: 最终掩码
            filtered_adagaus: 过滤后的自适应高斯二值图
            contraction_ratio: 收缩比例
            
        Returns:
            (left_edges, right_edges, contracted_left, contracted_right)
        """
        h, w = final_mask.shape
        
        # 计算左右边缘
        left_edges = np.full(h, w, dtype=int)
        right_edges = np.full(h, -1, dtype=int)
        
        for y in range(h):
            cols = np.where(final_mask[y] > 0)[0]
            if cols.size:
                left_edges[y] = cols.min()
                right_edges[y] = cols.max()
        
        # 平滑边缘
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
        
        # 计算收缩后的边界
        contracted_left = left_sm + np.round((right_sm - left_sm) * contraction_ratio).astype(int)
        contracted_right = right_sm - np.round((right_sm - left_sm) * contraction_ratio).astype(int)
        contracted_left = np.clip(contracted_left, 0, w - 1)
        contracted_right = np.clip(contracted_right, 0, w - 1)
        
        return left_sm, right_sm, contracted_left, contracted_right
    
    def interpolate_and_smooth_contour(self, contour_x: np.ndarray, contour_y: np.ndarray,
                                      image_height: int, image_width: int,
                                      right_edges: np.ndarray = None,
                                      closing_size: int = 11,
                                      smoothing_sigma: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        插值和平滑等高线
        
        Args:
            contour_x: 等高线x坐标
            contour_y: 等高线y坐标
            image_height: 图像高度
            image_width: 图像宽度
            right_edges: 右边缘
            closing_size: 形态学闭运算核大小
            smoothing_sigma: 平滑sigma
            
        Returns:
            (x_smooth, y_full)
        """
        y_full = np.arange(image_height)
        
        if len(contour_x) > 1:
            f_x = interp1d(contour_y, contour_x, kind='linear', bounds_error=False, fill_value='extrapolate')
            x_interp = f_x(y_full).astype(float)
            
            # 不超过右边界
            if right_edges is not None and right_edges.shape[0] == image_height:
                x_interp = np.minimum(x_interp, right_edges.astype(float))
            
            # 形态学闭运算填坑
            if closing_size < 1:
                closing_size = 1
            if closing_size % 2 == 0:
                closing_size += 1
            x_closed = grey_closing(x_interp, size=closing_size, mode='nearest')
            
            # 高斯平滑
            x_smooth = gaussian_filter1d(x_closed, sigma=smoothing_sigma, mode='nearest') if smoothing_sigma > 0 else x_closed
            
            # 再次限制在右边界内
            if right_edges is not None and right_edges.shape[0] == image_height:
                x_smooth = np.minimum(x_smooth, right_edges.astype(float))
            
            x_smooth = np.clip(x_smooth, 0, image_width - 1)
            return x_smooth, y_full
        else:
            return np.full(image_height, -1, dtype=float), y_full
    
    def calculate_contour_ratio(self, contour_x: np.ndarray, left_edges: np.ndarray, 
                                right_edges: np.ndarray) -> float:
        """
        计算等高线平均占比
        
        Args:
            contour_x: 等高线x坐标
            left_edges: 左边缘
            right_edges: 右边缘
            
        Returns:
            平均占比
        """
        width_lr = (right_edges - left_edges).astype(float)
        safe_width = np.where(width_lr == 0, 1.0, width_lr)
        ratios = (contour_x - left_edges) / safe_width
        ratios = np.clip(ratios, 0.0, 1.0)
        valid_rows = width_lr > 0
        avg_contour_ratio = float(np.mean(ratios[valid_rows])) if np.any(valid_rows) else 0.0
        return avg_contour_ratio
    
    def extract_contour_from_single_roi(self, roi_path: str, output_path: str) -> Dict[str, Any]:
        """
        从单个ROI图像提取等高线
        
        Args:
            roi_path: ROI图像路径
            output_path: 输出路径
            
        Returns:
            提取结果
        """
        try:
            # 读取ROI图像
            roi_image = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
            if roi_image is None:
                return {'success': False, 'error': f'无法读取ROI图像: {roi_path}'}
            
            h, w = roi_image.shape
            
            # 计算final mask和filtered adagaus
            final_mask = self.adagaus_extractor._compute_final_filled_mask(roi_image)
            adagaus = self.adagaus_extractor._adaptive_gaussian_binary(roi_image)
            filtered_adagaus = adagaus.copy()
            filtered_adagaus[final_mask == 0] = 0
            
            # 检测梯度等高线
            left_sm, right_sm, contracted_left, contracted_right = self.detect_gradient_contour_from_right_edge(
                final_mask, filtered_adagaus, contraction_ratio=0.1
            )
            
            # 计算梯度（从右到左白->黑取负号）
            sobel_x = cv2.Sobel(filtered_adagaus, cv2.CV_32F, 1, 0, ksize=3)
            gradient_map = -sobel_x
            
            # 在收缩区域内搜索最大梯度位置
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
            
            # 插值和平滑
            x_smooth, y_full = self.interpolate_and_smooth_contour(
                contour_x[contour_y], contour_y, h, w, right_sm, closing_size=11, smoothing_sigma=2.0
            )
            
            # 计算平均占比
            avg_contour_ratio = self.calculate_contour_ratio(x_smooth, left_sm, right_sm)
            
            # 创建可视化图像
            contour_vis = self.create_contour_visualization(
                roi_image, final_mask, left_sm, right_sm, contracted_left, contracted_right, x_smooth
            )
            
            # 保存可视化图像
            cv2.imwrite(output_path, contour_vis)
            
            return {
                'success': True,
                'avg_contour_ratio': avg_contour_ratio,
                'contour_x': x_smooth.tolist(),
                'left_edges': left_sm.tolist(),
                'right_edges': right_sm.tolist(),
                'contracted_left': contracted_left.tolist(),
                'contracted_right': contracted_right.tolist()
            }
            
        except Exception as e:
            return {'success': False, 'error': f'提取等高线时出错: {str(e)}'}
    
    def create_contour_visualization(self, roi_image: np.ndarray, final_mask: np.ndarray,
                                   left_edges: np.ndarray, right_edges: np.ndarray,
                                   contracted_left: np.ndarray, contracted_right: np.ndarray,
                                   contour_x: np.ndarray) -> np.ndarray:
        """
        创建等高线可视化图像
        
        Args:
            roi_image: ROI图像
            final_mask: 最终掩码
            left_edges: 左边缘
            right_edges: 右边缘
            contracted_left: 收缩左边缘
            contracted_right: 收缩右边缘
            contour_x: 等高线x坐标
            
        Returns:
            可视化图像
        """
        # 创建彩色可视化
        vis = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        
        # 绘制最终掩码区域（半透明红色）
        mask_colored = np.zeros_like(vis)
        mask_colored[final_mask > 0] = [0, 0, 255]  # 红色
        vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
        
        # 绘制边缘线
        h, w = roi_image.shape
        for y in range(h):
            if left_edges[y] < w and right_edges[y] >= 0:
                # 左边缘（绿色）
                if left_edges[y] < w:
                    vis[y, left_edges[y]] = [0, 255, 0]
                # 右边缘（蓝色）
                if right_edges[y] >= 0:
                    vis[y, right_edges[y]] = [255, 0, 0]
                # 收缩左边缘（黄色）
                if contracted_left[y] < w:
                    vis[y, contracted_left[y]] = [0, 255, 255]
                # 收缩右边缘（紫色）
                if contracted_right[y] >= 0:
                    vis[y, contracted_right[y]] = [255, 0, 255]
                # 等高线（白色）
                if 0 <= contour_x[y] < w:
                    vis[y, int(contour_x[y])] = [255, 255, 255]
        
        return vis
    
    def extract_contours_from_roi_images(self, roi_dir: str, contour_dir: str) -> Dict[str, Any]:
        """
        从ROI图像目录提取等高线特征
        
        Args:
            roi_dir: 输入ROI图像目录路径
            contour_dir: 输出等高线图像目录路径
            
        Returns:
            等高线提取结果统计
        """
        print(f"=== 从ROI图像提取等高线特征 ===")
        print(f"输入ROI目录: {roi_dir}")
        print(f"输出等高线目录: {contour_dir}")
        
        try:
            # 检查输入目录是否存在
            if not os.path.exists(roi_dir):
                return {'success': False, 'error': f'输入ROI目录不存在: {roi_dir}'}
            
            # 创建输出目录
            os.makedirs(contour_dir, exist_ok=True)
            
            # 获取所有ROI图像文件（按顺序）
            roi_pattern = os.path.join(roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not roi_files:
                return {'success': False, 'error': f'在目录 {roi_dir} 中未找到ROI图像文件'}
            
            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print(f"开始等高线特征提取...")
            
            # 批量处理ROI等高线检测
            success_count = 0
            failed_count = 0
            results = []
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="提取等高线特征")):
                try:
                    # 从文件名提取帧号
                    basename = os.path.basename(roi_path)
                    frame_number = int(basename.split('_')[1].split('.')[0])
                    
                    # 生成输出文件名
                    filename = f"frame_{frame_number:06d}_contour.png"
                    output_file = os.path.join(contour_dir, filename)
                    
                    # 检查文件是否已存在，如果存在则跳过
                    if os.path.exists(output_file):
                        success_count += 1
                        continue
                    
                    # 处理单个ROI等高线检测
                    result = self.extract_contour_from_single_roi(roi_path, output_file)
                    
                    if result['success']:
                        success_count += 1
                        # 保存结果数据
                        result['frame_number'] = frame_number
                        result['roi_path'] = roi_path
                        result['output_path'] = output_file
                        results.append(result)
                    else:
                        failed_count += 1
                        print(f"处理ROI图像失败 {roi_path}: {result.get('error', '未知错误')}")
                
                except Exception as e:
                    failed_count += 1
                    print(f"处理ROI图像时出错 {roi_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 帧图像，成功 {success_count} 帧，失败 {failed_count} 帧")
            
            print(f"\n等高线特征提取完成！")
            print(f"总帧数: {len(roi_files)}")
            print(f"成功提取: {success_count} 帧")
            print(f"失败: {failed_count} 帧")
            print(f"输出目录: {contour_dir}")
            
            # 计算统计信息
            if results:
                avg_ratios = [r['avg_contour_ratio'] for r in results]
                print(f"等高线平均占比统计:")
                print(f"  最小值: {min(avg_ratios):.4f}")
                print(f"  最大值: {max(avg_ratios):.4f}")
                print(f"  平均值: {np.mean(avg_ratios):.4f}")
                print(f"  标准差: {np.std(avg_ratios):.4f}")
            
            return {
                'success': True,
                'output_dir': contour_dir,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': failed_count,
                'results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': f'等高线特征提取时出错: {str(e)}'}
    
    def extract_contours_from_all_rois(self) -> Dict[str, Any]:
        """
        从所有ROI图像提取等高线特征
        
        Returns:
            等高线提取结果统计
        """
        print(f"\n=== 提取等高线特征 ===")
        print(f"输入ROI目录: {self.roi_dir}")
        print(f"输出等高线目录: {self.contour_dir}")
        
        try:
            # 获取所有ROI图像文件（按顺序）
            roi_pattern = os.path.join(self.roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not roi_files:
                return {'success': False, 'error': f'在目录 {self.roi_dir} 中未找到ROI图像文件'}
            
            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print(f"开始等高线特征提取...")
            
            # 批量处理ROI等高线检测
            success_count = 0
            failed_count = 0
            results = []
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="提取等高线特征")):
                try:
                    # 生成输出文件名
                    basename = os.path.basename(roi_path)
                    name, ext = os.path.splitext(basename)
                    contour_output_path = os.path.join(self.contour_dir, f"{name}_contour.png")
                    
                    # 检查等高线文件是否已存在，如果存在则跳过
                    if os.path.exists(contour_output_path):
                        success_count += 1
                        continue
                    
                    # 处理单个ROI等高线检测
                    result = self.extract_contour_from_single_roi(roi_path, contour_output_path)
                    
                    if result['success']:
                        success_count += 1
                        # 保存结果数据
                        frame_number = int(basename.split('_')[1].split('.')[0])
                        result['frame_number'] = frame_number
                        result['roi_path'] = roi_path
                        result['output_path'] = contour_output_path
                        results.append(result)
                    else:
                        failed_count += 1
                        print(f"处理ROI图像失败 {roi_path}: {result.get('error', '未知错误')}")
                
                except Exception as e:
                    failed_count += 1
                    print(f"处理ROI图像时出错 {roi_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 帧图像，成功 {success_count} 帧，失败 {failed_count} 帧")
            
            print(f"\n等高线特征提取完成！")
            print(f"总帧数: {len(roi_files)}")
            print(f"成功提取: {success_count} 帧")
            print(f"失败: {failed_count} 帧")
            print(f"跳过已存在: {len(roi_files) - success_count - failed_count} 帧")
            
            # 计算统计信息
            if results:
                avg_ratios = [r['avg_contour_ratio'] for r in results]
                print(f"等高线平均占比统计:")
                print(f"  最小值: {min(avg_ratios):.4f}")
                print(f"  最大值: {max(avg_ratios):.4f}")
                print(f"  平均值: {np.mean(avg_ratios):.4f}")
                print(f"  标准差: {np.std(avg_ratios):.4f}")
            
            return {
                'success': True,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': failed_count,
                'skipped_frames': len(roi_files) - success_count - failed_count,
                'output_dir': self.contour_dir,
                'results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': f'等高线特征提取时出错: {str(e)}'}
    
    def get_extraction_status(self) -> Dict[str, Any]:
        """
        获取等高线特征提取状态
        
        Returns:
            提取状态信息
        """
        try:
            # 获取所有ROI文件
            roi_pattern = os.path.join(self.roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            # 获取所有等高线文件
            contour_pattern = os.path.join(self.contour_dir, "*_contour.png")
            contour_files = glob.glob(contour_pattern)
            
            total_rois = len(roi_files)
            total_contours = len(contour_files)
            completion_rate = (total_contours / total_rois * 100) if total_rois > 0 else 0
            
            return {
                'success': True,
                'total_rois': total_rois,
                'total_contours': total_contours,
                'completion_rate': completion_rate,
                'roi_dir': self.roi_dir,
                'contour_dir': self.contour_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'获取状态时出错: {str(e)}'}


def main():
    """主函数"""
    # 默认值
    default_roi_dir = "data_adagaus_density_curve/roi_imgs"
    default_contour_dir = "data_adagaus_density_curve/contour_imgs"
    
    # 使用sys.argv获取参数
    if len(sys.argv) >= 3:
        roi_dir = sys.argv[1]
        contour_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        roi_dir = sys.argv[1]
        contour_dir = default_contour_dir
        print(f"使用默认输出目录: {contour_dir}")
    else:
        roi_dir = default_roi_dir
        contour_dir = default_contour_dir
        print(f"使用默认输入ROI目录: {roi_dir}")
        print(f"使用默认输出目录: {contour_dir}")

    # 判断roi_dir是否存在
    if not os.path.exists(roi_dir):
        raise ValueError(f"错误：ROI目录不存在 - {roi_dir}")
    
    # 判断contour_dir是否存在
    if not os.path.exists(contour_dir):
        os.makedirs(contour_dir, exist_ok=True)
    
    # 初始化等高线提取器
    extractor = ContourExtractor()
    
    # 执行等高线特征提取
    result = extractor.extract_contours_from_roi_images(
        roi_dir=roi_dir, 
        contour_dir=contour_dir
    )
    
    # 输出结果
    if result['success']:
        print(f"\n✅ 等高线特征提取完成！")
        print(f"输出目录: {result['output_dir']}")
        print(f"总帧数: {result['total_frames']}")
        print(f"成功提取: {result['success_frames']} 帧")
        print(f"失败: {result['failed_frames']} 帧")
    else:
        print(f"❌ 等高线特征提取失败: {result['error']}")


if __name__ == "__main__":
    main()

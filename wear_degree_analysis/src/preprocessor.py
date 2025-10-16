"""
预处理模块
实现图像去噪、中心线检测、ROI划分和法线采样
"""
import cv2
import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, List, Optional


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, gaussian_sigma: float = 1.0, savgol_window: int = 51, savgol_order: int = 3):
        """
        初始化预处理器
        
        Args:
            gaussian_sigma: 高斯滤波的标准差
            savgol_window: Savitzky-Golay滤波窗口大小（必须是奇数）
            savgol_order: Savitzky-Golay滤波多项式阶数
        """
        self.gaussian_sigma = gaussian_sigma
        self.savgol_window = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        self.savgol_order = savgol_order
        
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        图像去噪
        
        Args:
            image: 输入灰度图像
            
        Returns:
            去噪后的图像
        """
        kernel_size = int(6 * self.gaussian_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), self.gaussian_sigma)
    
    def detect_centerline(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测中心分割线
        在白色区域内检测不明显的分割线（局部最暗点）
        
        Args:
            image: 去噪后的灰度图像
            
        Returns:
            (xs, ys): 中心线的x坐标和y坐标数组
        """
        height, width = image.shape
        xs = []
        ys = []
        
        # 对每一行进行扫描
        for y in range(height):
            row = image[y, :]
            
            # 找到白色区域（假设白色区域亮度 > 100）
            white_mask = row > 100
            if not white_mask.any():
                continue
            
            # 在白色区域内找到最暗的点
            white_indices = np.where(white_mask)[0]
            if len(white_indices) < 2:
                continue
                
            # 在白色区域范围内找局部最暗点
            left_bound = white_indices[0]
            right_bound = white_indices[-1]
            
            # 在白色区域内部搜索最暗点（排除边界附近）
            search_start = left_bound + 5
            search_end = right_bound - 5
            
            if search_end <= search_start:
                # 如果区域太小，取中点
                x_center = (left_bound + right_bound) // 2
            else:
                # 在搜索范围内找最暗点
                search_region = row[search_start:search_end]
                local_min_idx = np.argmin(search_region)
                x_center = search_start + local_min_idx
            
            xs.append(x_center)
            ys.append(y)
        
        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        
        # 保存原始中心线（未平滑）
        xs_raw = xs.copy()
        
        # 使用Savitzky-Golay滤波平滑中心线
        if len(xs) > self.savgol_window:
            xs = savgol_filter(xs, self.savgol_window, self.savgol_order)
        
        return xs, ys, xs_raw
    
    def create_roi_masks(self, image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据中心线创建左右ROI掩码
        
        Args:
            image: 原始图像
            xs: 中心线x坐标
            ys: 中心线y坐标
            
        Returns:
            (left_mask, right_mask): 左侧（撕裂面）和右侧（剪切面）的掩码
        """
        height, width = image.shape
        left_mask = np.zeros((height, width), dtype=np.uint8)
        right_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 为每一行设置掩码
        for i, (x, y) in enumerate(zip(xs, ys)):
            y_int = int(round(y))
            x_int = int(round(x))
            
            if 0 <= y_int < height:
                # 左侧掩码（撕裂面）
                left_mask[y_int, :x_int] = 255
                # 右侧掩码（剪切面）
                right_mask[y_int, x_int:] = 255
        
        return left_mask, right_mask
    
    def sample_normals(self, image: np.ndarray, xs: np.ndarray, ys: np.ndarray, 
                      half_width: int = 30, step: int = 5) -> List[dict]:
        """
        沿中心线采样法线剖面
        
        Args:
            image: 去噪后的图像
            xs: 中心线x坐标
            ys: 中心线y坐标
            half_width: 法线两侧采样的半宽度（像素）
            step: 采样步长
            
        Returns:
            包含法线信息的字典列表，每个字典包含:
                - y: y坐标
                - x_center: 中心x坐标
                - profile: intensity剖面
                - x_coords: 剖面对应的x坐标
        """
        height, width = image.shape
        normals = []
        
        for i in range(0, len(xs), step):
            y = int(round(ys[i]))
            x_center = int(round(xs[i]))
            
            if not (0 <= y < height):
                continue
            
            # 对于竖直方向的图像，法线就是水平方向
            # 采样左右各half_width像素
            x_start = max(0, x_center - half_width)
            x_end = min(width, x_center + half_width + 1)
            
            profile = image[y, x_start:x_end].astype(np.float64)
            x_coords = np.arange(x_start, x_end)
            
            normals.append({
                'y': y,
                'x_center': x_center,
                'profile': profile,
                'x_coords': x_coords,
                'x_start': x_start,
                'x_end': x_end
            })
        
        return normals
    
    def detect_edge_positions(self, normals: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        从法线剖面中检测左右边界位置
        
        Args:
            normals: 法线剖面列表
            
        Returns:
            (left_edges, right_edges): 左右边界位置数组（相对于中心线）
        """
        left_edges = []
        right_edges = []
        
        for normal in normals:
            profile = normal['profile']
            x_center = normal['x_center']
            x_coords = normal['x_coords']
            
            # 计算梯度
            gradient = np.abs(np.gradient(profile))
            
            # 找到中心点在profile中的索引
            center_idx = np.argmin(np.abs(x_coords - x_center))
            
            # 左侧边界：从中心向左找最大梯度
            if center_idx > 0:
                left_grad = gradient[:center_idx]
                if len(left_grad) > 0:
                    left_idx = np.argmax(left_grad)
                    left_pos = x_coords[left_idx]
                    left_edges.append(left_pos - x_center)  # 相对位置
                else:
                    left_edges.append(0)
            else:
                left_edges.append(0)
            
            # 右侧边界：从中心向右找最大梯度
            if center_idx < len(profile) - 1:
                right_grad = gradient[center_idx:]
                if len(right_grad) > 0:
                    right_idx = center_idx + np.argmax(right_grad)
                    right_pos = x_coords[right_idx]
                    right_edges.append(right_pos - x_center)  # 相对位置
                else:
                    right_edges.append(0)
            else:
                right_edges.append(0)
        
        return np.array(left_edges), np.array(right_edges)
    
    def process(self, image: np.ndarray) -> dict:
        """
        完整的预处理流程
        
        Args:
            image: 输入灰度图像
            
        Returns:
            包含所有预处理结果的字典
        """
        # 1. 去噪
        denoised = self.denoise(image)
        
        # 2. 检测中心线（同时返回平滑和原始）
        xs, ys, xs_raw = self.detect_centerline(denoised)
        
        if len(xs) == 0:
            return {
                'success': False,
                'error': '无法检测到中心线'
            }
        
        # 3. 创建ROI掩码
        left_mask, right_mask = self.create_roi_masks(image, xs, ys)
        
        # 4. 采样法线
        normals = self.sample_normals(denoised, xs, ys)
        
        # 5. 检测边界位置
        left_edges, right_edges = self.detect_edge_positions(normals)
        
        return {
            'success': True,
            'denoised': denoised,
            'centerline_x': xs,  # 平滑后的中心线（用于ROI分割）
            'centerline_x_raw': xs_raw,  # 原始中心线（用于缺口/峰检测和可视化）
            'centerline_y': ys,
            'left_mask': left_mask,
            'right_mask': right_mask,
            'normals': normals,
            'left_edges': left_edges,
            'right_edges': right_edges
        }


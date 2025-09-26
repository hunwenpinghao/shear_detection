"""
SAM2 分割模块
增加SAM2.1模型支持以精确检测条状物体内部分割界面
"""
import cv2
import numpy as np
import torch
import os

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("[Warning] SAM2 is not installed, SAM2 segmentator disabled. Please install 'sam2' package")
    SAM2_AVAILABLE = False

from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from scipy import ndimage

from config import SAM2_CONFIG, VIS_CONFIG
from font_utils import setup_chinese_font


class SAM2Segmentator:
    """SAM2分割器 - 使用facebook/sam2.1模型进行条纹内部分割"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """ 
        初始化SAM2分割器
        
        Args:
            config: SAM2模型配置参数
        """
        self.config = config if config is not None else SAM2_CONFIG
        
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 is not installed. Please install sam2 package first.")
        
        # 自动设备检测
        if self.config.get('device') == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = self.config['device']
            
        try:
            self.model_name = self.config['model_name']
            self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
            print(f"[Info] SAM2 Model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2 model '{self.model_name}': {e}")
    
    def set_image(self, image: np.ndarray) -> None:
        """
        设置输入图像
        Args:
            image: 输入RGB图像(numpy.ndarray)。
        """
        # SAM2 需要 RGB 顺序
        if len(image.shape) == 2:  # 灰度图
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:  # BGR -> RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.predictor.set_image(rgb_image)
        
    def generate_prompt_points(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        生成提示点：基于灰度条纹宽度 
        
        Args: 
            image: 条纹图像 
        
        Returns:
            left_points, right_points: 两侧的提示点坐标xy形状(N,2)
        """
        # 1. 寻找白色区域掩码
        gray = image if len(image.shape)==2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold = np.mean(gray)+20
        mask = gray > threshold
        
        h, w = gray.shape
        density = self.config.get('prompt_point_density', 0.05)
        max_points_per_row = max(1, int(h * w * density) // h)
        
        # 2. 均匀生成提示点
        left_points = []
        right_points = []
        row_step = h // max_points_per_row if max_points_per_row>0 else h//5   # 至少5行点
        
        for i in range(0, h, row_step):
            row_mask = mask[i,:]
            white_region_indices = np.where(row_mask)[0]
            
            if len(white_region_indices) < 5:
                continue
                
            region_start = white_region_indices[0]
            region_end = white_region_indices[-1]
            region_center = (region_start + region_end) / 2
            
            # COG 重心位置
            moment = np.sum(white_region_indices * row_mask[white_region_indices])
            count = np.sum(row_mask[white_region_indices]) 
            cog = int(moment / count) if count>0 else int(region_center)
            
            # 左右侧点：基于重心位置分割
            if region_end - cog > 5:
                left_points.append( [int(region_start + (cog-region_start)/3), int(i)] ) 
                right_points.append( [int(cog + (region_end-cog)*2/3), int(i)] )
                
        left_points = np.array(left_points) if left_points else np.zeros((0,2))
        right_points = np.array(right_points) if right_points else np.zeros((0,2))
        
        return left_points, right_points
    
    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        使用SAM2对图像分割
        
        Args: 
            image: 输入灰度图像或RGB图像
        
        Returns:
            sample_left_mask: 左侧mask(np.ndarray h*w二值uint8 )  
            sample_right_mask: 右侧mask(np.ndarray h*w二值uint8 )
        """
        # 1. 设置图像到模型
        self.set_image(image)
        
        # 2. 生成自适应提示点
        left_pts, right_pts = self.generate_prompt_points(image)        
        h = image.shape[0]
        w = image.shape[1]
        
        if len(left_pts)==0 or len(right_pts) == 0:
            print("[Warning] 未检测到有效关键点，返回中点分割备选")
            left_center_col = w//4
            right_center_col = 3*w//4
            midpoint_rows = np.arange(0, h, h//6)
            mid_lefts = np.stack([np.full_like(midpoint_rows, left_center_col), midpoint_rows], axis=-1)
            mid_rights = np.stack([np.full_like(midpoint_rows, right_center_col), midpoint_rows], axis=-1)
            left_pts, right_pts = mid_lefts, mid_rights
        
        print(f"[Info] Generated {len(left_pts)} left_ref_points and {len(right_pts)} right_ref_points")
        
        # 3. SAM2 预测
        sample_left_mask = np.zeros(shape=(h,w), dtype=np.uint8)
        sample_right_mask = np.zeros(shape=(h,w), dtype=np.uint8)
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            
            # 左侧预测
            try:
                masks_i_l, logits_i_l, painter_i_l = self.predictor.predict( 
                    point_coords=left_pts, point_labels=np.ones(len(left_pts)), 
                    multimask_output=True)
                idx_i = 0
                sample_left_mask = masks_i_l[idx_i].astype(np.uint8)*255
            except:
                pass
                
            # 右侧预测
            try:
                masks_i_r, logits_i_r, painter_i_r = self.predictor.predict(
                    point_coords=right_pts, point_labels=np.ones(len(right_pts)), 
                    multimask_output=True)
                idx_i = 0
                sample_right_mask = masks_i_r[idx_i].astype(np.uint8)*255
            except:
                pass
        
        return sample_left_mask, sample_right_mask
    
    def segment_surface(self, image: np.ndarray, method: str = 'sam2') -> Tuple[np.ndarray, np.ndarray]:
        """
        执行表面分割，支持SAM2标签
        Args:
            image: 输入图像
            method: 分割方法，必使用'sam2'
        Returns:
            tear_mask, shear_mask
        """
        if method != 'sam2':
            raise ValueError(f"Unsupported method {method}, expected 'sam2'")
        h,w = image.shape[:2]
        
        # 处理SAM2分割
        sample_left_mask, sample_right_mask = self.segment_image(image)
        
        # 限制白色区域避免背景像素
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        threshold = np.mean(gray) + 30
        white_mask = gray > threshold

        sample_left_mask[~white_mask] = 0
        sample_right_mask[~white_mask] = 0
             
        tear_mask = sample_left_mask
        shear_mask = sample_right_mask
        
        return tear_mask, shear_mask
    
    def visualize_segmentation(self, image: np.ndarray,
                             tear_mask: np.ndarray, 
                             shear_mask: np.ndarray, 
                             save_path: str,
                             boundary_positions: np.ndarray = None) -> None:
        """
        可视化SAM2分割结果
        Args:
            image: 原图像
            tear_mask: 撕裂面mask
            shear_mask: 剪切面mask
            save_path: 保存路径
            boundary_positions: 未用的参数（兼容）
        """
        setup_chinese_font()
        fig, axes = plt.subplots(2, 2, figsize=VIS_CONFIG['figure_size'])
        axes = axes.flatten()
        
        # 原图与SAM2结果
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("原始图像")
        axes[0].axis('off')
        
        axes[1].imshow(tear_mask, cmap='Reds', alpha=0.7)
        axes[1].set_title("SAM2撕裂面检测结果")
        axes[1].axis('off') 
        
        axes[2].imshow(shear_mask, cmap='Blues', alpha=0.7)
        axes[2].set_title("SAM2剪切面检测结果")
        axes[2].axis('off')
        
        # 最终融合
        combined = np.zeros(image.shape+(3,), dtype=np.uint8)
        if len(image.shape)==2:
            combined[:,:,0] = image
            combined[:,:,1] = image
            combined[:,:,2] = image
            
        for u,v in np.stack(np.where(tear_mask), axis=1):
            combined[u,v] = (150, 0, 0) # 撕裂面红色
        for u,v in np.stack(np.where(shear_mask), axis=1):
            combined[u,v] = (0, 0, 150) # 剪切面蓝色
                
        axes[3].imshow(combined)
        axes[3].set_title("SAM2分割结果(最终融合)")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, format='png', dpi=VIS_CONFIG['dpi'],
                   bbox_inches='tight', pad_inches=0.1)
        plt.close()


if __name__ == '__main__':
    test_image = cv2.imread('test.png') 
    if test_image is not None:
        segmentor = SAM2Segmentator()
        tear, shear = segmentor.segment_surface(test_image, 'sam2')
        print("SAM2分割测试成功")
    else:
        print("请提供test.png图片进行SAM2分割测试")
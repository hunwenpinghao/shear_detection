"""
图像预处理模块
主要功能：
1. 从线阵相机图像中提取有效ROI区域（白色条状物）
2. 去除大面积黑色背景
3. 增强图像对比度和去噪
4. 标准化图像尺寸
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from config import PREPROCESS_CONFIG, VIS_CONFIG
from font_utils import setup_chinese_font


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化预处理器
        
        Args:
            config: 配置参数，如果为None则使用默认配置
        """
        self.config = config if config is not None else PREPROCESS_CONFIG
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            加载的图像数组
        """
        # 使用cv2加载图像，支持各种格式包括bmp
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        增强图像对比度
        
        Args:
            image: 输入图像
            
        Returns:
            对比度增强后的图像
        """
        # 使用CLAHE（对比度限制自适应直方图均衡化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像去噪
        
        Args:
            image: 输入图像
            
        Returns:
            去噪后的图像
        """
        kernel_size = self.config['gaussian_kernel']
        denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return denoised
    
    def extract_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        提取ROI区域（白色条状物）
        
        核心思路：
        1. 二值化分离白色条状物和黑色背景
        2. 形态学操作去噪和连接断开的部分
        3. 轮廓检测找到最大的条状物区域
        4. 提取包围盒作为ROI
        
        Args:
            image: 输入图像
            
        Returns:
            roi_image: 提取的ROI图像
            roi_info: ROI信息字典，包含边界框坐标等
        """
        # 步骤1: 二值化
        threshold = self.config['roi_threshold']
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        # 步骤2: 形态学操作
        kernel_size = self.config['morphology_kernel']
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # 闭操作：连接断开的白色区域
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        # 开操作：去除小噪点
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 步骤3: 轮廓检测
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("未检测到有效的白色条状物区域")
        
        # 找到最大的轮廓（假设为主要的条状物）
        max_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(max_contour)
        
        if contour_area < self.config['min_contour_area']:
            raise ValueError(f"检测到的区域太小: {contour_area} < {self.config['min_contour_area']}")
        
        # 步骤4: 计算包围盒
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 添加一些边距以确保完整包含条状物
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # 提取ROI
        roi_image = image[y:y+h, x:x+w]
        
        roi_info = {
            'bbox': (x, y, w, h),
            'contour_area': contour_area,
            'roi_shape': roi_image.shape,
            'original_shape': image.shape
        }
        
        return roi_image, roi_info
    
    def normalize_size(self, image: np.ndarray, target_size: Tuple[int, int] = (128, 512)) -> np.ndarray:
        """
        标准化图像尺寸
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)，线阵相机图像应为竖条状 shape=H>W
            
        Returns:
            调整尺寸后的图像
        """
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def preprocess_pipeline(self, image_path: str, 
                          target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        完整的预处理流水线
        
        Args:
            image_path: 图像路径
            target_size: 目标尺寸，如果为None则不调整尺寸
            
        Returns:
            processed_image: 处理后的图像
            processing_info: 处理信息
        """
        # 加载图像
        original_image = self.load_image(image_path)
        
        # 对比度增强
        enhanced_image = self.enhance_contrast(original_image)
        
        # 去噪
        denoised_image = self.denoise_image(enhanced_image)
        
        # 提取ROI
        roi_image, roi_info = self.extract_roi(denoised_image)
        
        # 可选的尺寸标准化
        if target_size is not None:
            final_image = self.normalize_size(roi_image, target_size)
        else:
            final_image = roi_image
        
        processing_info = {
            'original_shape': original_image.shape,
            'final_shape': final_image.shape,
            'roi_info': roi_info,
            'preprocessing_steps': [
                'contrast_enhancement',
                'denoising', 
                'roi_extraction',
                'size_normalization' if target_size else None
            ]
        }
        
        return final_image, processing_info
    
    def visualize_preprocessing(self, image_path: str, save_path: Optional[str] = None):
        """
        可视化预处理步骤
        
        Args:
            image_path: 图像路径
            save_path: 保存路径，如果为None则显示图像
        """
        # 设置中文字体
        setup_chinese_font()
        
        # 加载原始图像
        original = self.load_image(image_path)
        
        # 各个处理步骤
        enhanced = self.enhance_contrast(original)
        denoised = self.denoise_image(enhanced)
        roi, roi_info = self.extract_roi(denoised)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=VIS_CONFIG['figure_size'])
        
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(enhanced, cmap='gray') 
        axes[0, 1].set_title('对比度增强')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(denoised, cmap='gray')
        axes[1, 0].set_title('去噪处理')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(roi, cmap='gray')
        axes[1, 1].set_title(f'ROI提取 ({roi.shape[1]}x{roi.shape[0]})')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
            plt.close()


def test_preprocessor():
    """测试预处理器"""
    import os
    from config import DATA_DIR, OUTPUT_DIR
    
    # 初始化预处理器
    preprocessor = ImagePreprocessor()
    
    # 测试图像路径
    test_image = os.path.join(DATA_DIR, 'Image_20250710125452500.bmp')
    
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
    try:
        # 运行预处理流水线
        processed_image, info = preprocessor.preprocess_pipeline(test_image, target_size=(128, 512))
        
        print("预处理成功完成!")
        print(f"原始图像尺寸: {info['original_shape']}")
        print(f"最终图像尺寸: {info['final_shape']}")
        print(f"ROI信息: {info['roi_info']}")
        
        # 可视化结果
        vis_path = os.path.join(OUTPUT_DIR, 'preprocessing_visualization.png')
        preprocessor.visualize_preprocessing(test_image, vis_path)
        print(f"可视化结果已保存到: {vis_path}")
        
        # 保存处理后的图像
        processed_path = os.path.join(OUTPUT_DIR, 'processed_roi.png')
        cv2.imwrite(processed_path, processed_image)
        print(f"处理后的ROI图像已保存到: {processed_path}")
        
    except Exception as e:
        print(f"预处理过程中出现错误: {e}")


if __name__ == "__main__":
    test_preprocessor()

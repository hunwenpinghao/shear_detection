#!/usr/bin/env python3
"""
纹理转换器
主要功能：
1. 将ROI图像转换为纹理图
2. 使用多种纹理分析方法（LBP、GLCM、Gabor滤波器等）
3. 生成纹理特征可视化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, filters, measure
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import ndimage
import os
from typing import Dict, Any, Tuple, Optional
import json
import platform

# 设置中文字体
def setup_chinese_font():
    """设置中文字体"""
    import matplotlib.font_manager as fm
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # 查找可用的中文字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = []
        
        # 优先选择的中文字体
        preferred_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
        
        for font in preferred_fonts:
            if font in available_fonts:
                chinese_fonts.append(font)
        
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"设置中文字体: {chinese_fonts[0]}")
            return True
    
    elif system == "Windows":
        # Windows 中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("设置Windows中文字体")
        return True
    
    else:  # Linux
        # Linux 中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("设置Linux中文字体")
        return True
    
    # 如果都失败了，使用英文标签
    print("无法设置中文字体，将使用英文标签")
    return False

class TextureConverter:
    """纹理转换器"""
    
    def __init__(self):
        """初始化纹理转换器"""
        self.texture_methods = {
            'lbp': self._compute_lbp_texture,
            'glcm': self._compute_glcm_texture,
            'gabor': self._compute_gabor_texture,
            'gradient': self._compute_gradient_texture,
            'laplacian': self._compute_laplacian_texture,
            'sobel': self._compute_sobel_texture,
            'haralick': self._compute_haralick_texture
        }
    
    def convert_to_texture(self, image_path: str, output_dir: str = "output", 
                          methods: list = None) -> Dict[str, Any]:
        """
        将图像转换为纹理图
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            methods: 纹理分析方法列表
            
        Returns:
            纹理转换结果
        """
        if methods is None:
            methods = ['lbp', 'glcm', 'gabor', 'gradient', 'laplacian', 'sobel']
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        print(f"图像尺寸: {gray.shape}")
        print(f"像素值范围: {gray.min()} - {gray.max()}")
        
        results = {}
        texture_images = {}
        
        # 应用各种纹理分析方法
        for method in methods:
            if method in self.texture_methods:
                print(f"正在计算 {method} 纹理...")
                try:
                    texture_img, features = self.texture_methods[method](gray)
                    texture_images[method] = texture_img
                    results[method] = features
                    
                    # 保存纹理图像
                    output_path = os.path.join(output_dir, f"texture_{method}.png")
                    cv2.imwrite(output_path, texture_img)
                    print(f"保存纹理图像: {output_path}")
                    
                except Exception as e:
                    print(f"计算 {method} 纹理时出错: {e}")
                    continue
        
        # 生成综合纹理图
        if texture_images:
            combined_texture = self._create_combined_texture(texture_images)
            combined_path = os.path.join(output_dir, "texture_combined.png")
            cv2.imwrite(combined_path, combined_texture)
            print(f"保存综合纹理图: {combined_path}")
            results['combined'] = {'path': combined_path}
        
        # 生成纹理分析报告
        report_path = os.path.join(output_dir, "texture_analysis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"保存纹理分析报告: {report_path}")
        
        # 生成可视化
        self._create_visualization(gray, texture_images, output_dir)
        
        return {
            'success': True,
            'texture_images': texture_images,
            'results': results,
            'output_dir': output_dir
        }
    
    def _compute_lbp_texture(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算局部二值模式(LBP)纹理
        
        Args:
            gray: 灰度图像
            
        Returns:
            纹理图像和特征
        """
        # LBP参数
        radius = 3
        n_points = 8 * radius
        
        # 计算LBP
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # 归一化到0-255
        lbp_normalized = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)
        
        # 计算LBP直方图特征
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        features = {
            'lbp_histogram': hist.tolist(),
            'lbp_mean': float(np.mean(lbp)),
            'lbp_std': float(np.std(lbp)),
            'lbp_entropy': float(-np.sum(hist * np.log2(hist + 1e-7)))
        }
        
        return lbp_normalized, features
    
    def _compute_glcm_texture(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算灰度共生矩阵(GLCM)纹理
        
        Args:
            gray: 灰度图像
            
        Returns:
            纹理图像和特征
        """
        # 量化图像到较少灰度级
        gray_quantized = (gray // 32) * 32
        
        # 计算GLCM
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(gray_quantized, distances=distances, angles=angles, 
                           levels=8, symmetric=True, normed=True)
        
        # 计算GLCM属性
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # 创建纹理图像（使用对比度）
        texture_img = np.zeros_like(gray)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                # 计算局部对比度
                local_region = gray[max(0, i-2):min(gray.shape[0], i+3),
                                  max(0, j-2):min(gray.shape[1], j+3)]
                if local_region.size > 0:
                    texture_img[i, j] = np.std(local_region)
        
        # 归一化
        texture_img = ((texture_img - texture_img.min()) / 
                      (texture_img.max() - texture_img.min()) * 255).astype(np.uint8)
        
        features = {
            'glcm_contrast': float(contrast),
            'glcm_dissimilarity': float(dissimilarity),
            'glcm_homogeneity': float(homogeneity),
            'glcm_energy': float(energy),
            'glcm_correlation': float(correlation)
        }
        
        return texture_img, features
    
    def _compute_gabor_texture(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算Gabor滤波器纹理
        
        Args:
            gray: 灰度图像
            
        Returns:
            纹理图像和特征
        """
        # Gabor滤波器参数
        frequencies = [0.1, 0.2, 0.3]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        gabor_responses = []
        
        for freq in frequencies:
            for theta in orientations:
                # 创建Gabor滤波器
                kernel = cv2.getGaborKernel((21, 21), 5, theta, 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                
                # 应用滤波器
                filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
                gabor_responses.append(np.abs(filtered))
        
        # 组合所有响应
        combined_response = np.mean(gabor_responses, axis=0)
        
        # 归一化
        texture_img = ((combined_response - combined_response.min()) / 
                      (combined_response.max() - combined_response.min()) * 255).astype(np.uint8)
        
        features = {
            'gabor_mean': float(np.mean(combined_response)),
            'gabor_std': float(np.std(combined_response)),
            'gabor_energy': float(np.sum(combined_response**2))
        }
        
        return texture_img, features
    
    def _compute_gradient_texture(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算梯度纹理
        
        Args:
            gray: 灰度图像
            
        Returns:
            纹理图像和特征
        """
        # 计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅度
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化
        texture_img = ((gradient_magnitude - gradient_magnitude.min()) / 
                      (gradient_magnitude.max() - gradient_magnitude.min()) * 255).astype(np.uint8)
        
        features = {
            'gradient_mean': float(np.mean(gradient_magnitude)),
            'gradient_std': float(np.std(gradient_magnitude)),
            'gradient_max': float(np.max(gradient_magnitude))
        }
        
        return texture_img, features
    
    def _compute_laplacian_texture(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算拉普拉斯纹理
        
        Args:
            gray: 灰度图像
            
        Returns:
            纹理图像和特征
        """
        # 计算拉普拉斯
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian = np.abs(laplacian)
        
        # 归一化
        texture_img = ((laplacian - laplacian.min()) / 
                      (laplacian.max() - laplacian.min()) * 255).astype(np.uint8)
        
        features = {
            'laplacian_mean': float(np.mean(laplacian)),
            'laplacian_std': float(np.std(laplacian)),
            'laplacian_max': float(np.max(laplacian))
        }
        
        return texture_img, features
    
    def _compute_sobel_texture(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算Sobel纹理
        
        Args:
            gray: 灰度图像
            
        Returns:
            纹理图像和特征
        """
        # 计算Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算Sobel幅度
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 归一化
        texture_img = ((sobel_magnitude - sobel_magnitude.min()) / 
                      (sobel_magnitude.max() - sobel_magnitude.min()) * 255).astype(np.uint8)
        
        features = {
            'sobel_mean': float(np.mean(sobel_magnitude)),
            'sobel_std': float(np.std(sobel_magnitude)),
            'sobel_max': float(np.max(sobel_magnitude))
        }
        
        return texture_img, features
    
    def _compute_haralick_texture(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算Haralick纹理特征
        
        Args:
            gray: 灰度图像
            
        Returns:
            纹理图像和特征
        """
        # 量化图像
        gray_quantized = (gray // 32) * 32
        
        # 计算GLCM
        glcm = graycomatrix(gray_quantized, distances=[1], angles=[0], 
                           levels=8, symmetric=True, normed=True)
        
        # 计算Haralick特征
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        
        # 创建纹理图像（使用局部方差）
        texture_img = np.zeros_like(gray)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                local_region = gray[max(0, i-2):min(gray.shape[0], i+3),
                                  max(0, j-2):min(gray.shape[1], j+3)]
                if local_region.size > 0:
                    texture_img[i, j] = np.var(local_region)
        
        # 归一化
        texture_img = ((texture_img - texture_img.min()) / 
                      (texture_img.max() - texture_img.min()) * 255).astype(np.uint8)
        
        features = {
            'haralick_contrast': float(contrast),
            'haralick_dissimilarity': float(dissimilarity),
            'haralick_homogeneity': float(homogeneity),
            'haralick_energy': float(energy),
            'haralick_correlation': float(correlation)
        }
        
        return texture_img, features
    
    def _create_combined_texture(self, texture_images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        创建综合纹理图
        
        Args:
            texture_images: 各种纹理图像字典
            
        Returns:
            综合纹理图像
        """
        if not texture_images:
            return np.zeros((100, 100), dtype=np.uint8)
        
        # 将所有纹理图像平均
        combined = np.zeros_like(list(texture_images.values())[0], dtype=np.float32)
        
        for texture_img in texture_images.values():
            combined += texture_img.astype(np.float32)
        
        combined /= len(texture_images)
        
        return combined.astype(np.uint8)
    
    def _create_visualization(self, original: np.ndarray, texture_images: Dict[str, np.ndarray], 
                            output_dir: str):
        """
        创建纹理分析可视化
        
        Args:
            original: 原始图像
            texture_images: 纹理图像字典
            output_dir: 输出目录
        """
        # 设置中文字体并选择标签语言
        font_success = setup_chinese_font()
        
        if font_success:
            method_names = {
                'lbp': 'LBP纹理',
                'glcm': 'GLCM纹理',
                'gabor': 'Gabor纹理',
                'gradient': '梯度纹理',
                'laplacian': '拉普拉斯纹理',
                'sobel': 'Sobel纹理',
                'haralick': 'Haralick纹理'
            }
            original_title = '原始图像'
        else:
            method_names = {
                'lbp': 'LBP Texture',
                'glcm': 'GLCM Texture',
                'gabor': 'Gabor Texture',
                'gradient': 'Gradient Texture',
                'laplacian': 'Laplacian Texture',
                'sobel': 'Sobel Texture',
                'haralick': 'Haralick Texture'
            }
            original_title = 'Original Image'
        
        n_methods = len(texture_images)
        if n_methods == 0:
            return
        
        # 计算子图布局
        cols = min(4, n_methods + 1)  # +1 for original image
        rows = (n_methods + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # 显示原始图像
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title(original_title, fontsize=12)
        axes[0, 0].axis('off')
        
        idx = 1
        for method, texture_img in texture_images.items():
            row = idx // cols
            col = idx % cols
            
            if row < rows and col < cols:
                axes[row, col].imshow(texture_img, cmap='hot')
                title = method_names.get(method, method.upper())
                axes[row, col].set_title(title, fontsize=12)
                axes[row, col].axis('off')
            
            idx += 1
        
        # 隐藏多余的子图
        for i in range(idx, rows * cols):
            row = i // cols
            col = i % cols
            if row < rows and col < cols:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # 保存可视化图像
        vis_path = os.path.join(output_dir, "texture_visualization.png")
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"保存纹理可视化: {vis_path}")


def main():
    """主函数"""
    # 初始化纹理转换器
    converter = TextureConverter()
    
    # 输入图像路径
    input_image = "data/extracted_roi_image.png"
    
    # 检查输入图像是否存在
    if not os.path.exists(input_image):
        print(f"错误：输入图像不存在: {input_image}")
        return
    
    print("=== 纹理转换器 ===")
    print(f"输入图像: {input_image}")
    
    try:
        # 执行纹理转换
        result = converter.convert_to_texture(
            image_path=input_image,
            output_dir="output/texture_analysis",
            methods=['lbp', 'glcm', 'gabor', 'gradient', 'laplacian', 'sobel']
        )
        
        if result['success']:
            print("\n✅ 纹理转换完成！")
            print(f"📁 输出目录: {result['output_dir']}")
            print(f"🖼️ 生成的纹理图像数量: {len(result['texture_images'])}")
            
            # 显示主要特征
            print("\n=== 主要纹理特征 ===")
            for method, features in result['results'].items():
                if isinstance(features, dict):
                    print(f"\n{method.upper()} 特征:")
                    for key, value in features.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
        else:
            print("❌ 纹理转换失败")
            
    except Exception as e:
        print(f"❌ 纹理转换过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

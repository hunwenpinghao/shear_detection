#!/usr/bin/env python3
"""
LBP纹理图视频处理器
主要功能：
1. 从ROI图像中计算LBP纹理特征
2. 生成LBP纹理可视化图像
3. 批量处理并创建LBP纹理视频
"""

import cv2
import numpy as np
import os
import glob
import sys
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import platform

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

from skimage.feature import local_binary_pattern

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

class LBPTextureProcessor:
    """LBP纹理处理器"""
    
    def __init__(self, radius: int = 3, n_points: int = None):
        """
        初始化LBP纹理处理器
        
        Args:
            radius: LBP半径
            n_points: LBP采样点数，如果为None则使用8*radius
        """
        self.radius = radius
        self.n_points = n_points if n_points is not None else 8 * radius
        
    def compute_lbp_texture(self, gray_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        计算LBP纹理特征
        
        Args:
            gray_image: 灰度图像
            
        Returns:
            LBP纹理图像和特征
        """
        # 计算LBP
        lbp = local_binary_pattern(gray_image, self.n_points, self.radius, method='uniform')
        
        # 归一化到0-255
        lbp_normalized = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)
        
        # 计算LBP直方图特征
        hist, _ = np.histogram(lbp.ravel(), bins=self.n_points + 2, range=(0, self.n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        features = {
            'lbp_histogram': hist.tolist(),
            'lbp_mean': float(np.mean(lbp)),
            'lbp_std': float(np.std(lbp)),
            'lbp_entropy': float(-np.sum(hist * np.log2(hist + 1e-7))),
            'lbp_energy': float(np.sum(hist**2))
        }
        
        return lbp_normalized, features
    
    def create_lbp_visualization(self, original_image: np.ndarray, 
                               lbp_texture: np.ndarray, 
                               features: Dict[str, Any]) -> np.ndarray:
        """
        创建LBP纹理可视化图像
        
        Args:
            original_image: 原始图像
            lbp_texture: LBP纹理图像
            features: LBP特征
            
        Returns:
            可视化图像
        """
        try:
            # 设置中文字体
            font_success = setup_chinese_font()
            
            # 创建子图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始图像
            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('原始图像' if font_success else 'Original Image', fontsize=12)
            axes[0].axis('off')
            
            # LBP纹理图像
            axes[1].imshow(lbp_texture, cmap='hot')
            axes[1].set_title('LBP纹理' if font_success else 'LBP Texture', fontsize=12)
            axes[1].axis('off')
            
            # LBP直方图
            hist = features['lbp_histogram']
            axes[2].bar(range(len(hist)), hist)
            axes[2].set_title('LBP直方图' if font_success else 'LBP Histogram', fontsize=12)
            axes[2].set_xlabel('LBP值' if font_success else 'LBP Value')
            axes[2].set_ylabel('频率' if font_success else 'Frequency')
            
            # 添加特征信息
            info_text = f"均值: {features['lbp_mean']:.2f}\n"
            info_text += f"标准差: {features['lbp_std']:.2f}\n"
            info_text += f"熵: {features['lbp_entropy']:.2f}\n"
            info_text += f"能量: {features['lbp_energy']:.2f}"
            
            axes[2].text(0.02, 0.98, info_text, transform=axes[2].transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存到内存中的字节流
            import io
            from PIL import Image
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # 读取图像数组并转换
            image = Image.open(buf)
            image_array = np.array(image)
            
            # 转换为OpenCV BGR格式
            if len(image_array.shape) == 3:
                bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                
            buf.close()
            plt.close(fig)
            return bgr
            
        except Exception as e:
            print(f"matplotlib可视化失败，使用简单方法: {e}")
            
            # 简单回退方法：只显示LBP纹理
            if len(lbp_texture.shape) == 2:
                # 转换为彩色图像
                lbp_colored = cv2.applyColorMap(lbp_texture, cv2.COLORMAP_HOT)
            else:
                lbp_colored = lbp_texture
                
            return lbp_colored
    
    def process_single_roi_lbp(self, roi_image_path: str, output_path: str) -> Dict[str, Any]:
        """
        对ROI图像进行LBP纹理检测并生成可视化图
        
        Args:
            roi_image_path: ROI图像路径
            output_path: 输出LBP纹理图像路径
            
        Returns:
            LBP纹理检测结果
        """
        try:
            # 读取ROI图像
            roi_image = cv2.imread(roi_image_path, cv2.IMREAD_GRAYSCALE)
            if roi_image is None:
                return {'success': False, 'error': f'无法读取ROI图像: {roi_image_path}'}
            
            # 计算LBP纹理
            lbp_texture, features = self.compute_lbp_texture(roi_image)
            
            # 创建LBP纹理可视化图像
            lbp_visualization = self.create_lbp_visualization(roi_image, lbp_texture, features)
            
            # 保存LBP纹理可视化图像
            success = cv2.imwrite(output_path, lbp_visualization)
            if success:
                return {
                    'success': True,
                    'lbp_mean': features['lbp_mean'],
                    'lbp_std': features['lbp_std'],
                    'lbp_entropy': features['lbp_entropy'],
                    'lbp_energy': features['lbp_energy'],
                    'output_path': output_path
                }
            else:
                return {'success': False, 'error': f'无法保存LBP纹理图像到: {output_path}'}
                
        except Exception as e:
            return {'success': False, 'error': f'处理ROI图像时出错: {str(e)}'}
    
    def process_all_roi_lbp(self, roi_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        处理所有ROI图像生成LBP纹理图
        
        Args:
            roi_dir: ROI图像目录路径
            output_dir: 输出LBP纹理图目录路径
            
        Returns:
            处理结果统计
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取所有ROI图像文件（按顺序）
            roi_pattern = os.path.join(roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not roi_files:
                return {'success': False, 'error': f'在目录 {roi_dir} 中未找到ROI图像文件'}
            
            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print(f"开始计算LBP纹理图...")
            
            # 批量处理ROI LBP纹理检测
            results = []
            success_count = 0
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="计算LBP纹理图")):
                # 生成输出文件名
                basename = os.path.basename(roi_path)
                name_parts = basename.split('_')
                if len(name_parts) >= 2:
                    frame_name = "_".join(name_parts[:2])  # 提取frame_XXXXXX部分
                    output_path = os.path.join(output_dir, f"{frame_name}_lbp.png")
                else:
                    continue
                
                # 处理单个ROI LBP纹理检测
                result = self.process_single_roi_lbp(roi_path, output_path)
                results.append(result)
                
                if result['success']:
                    success_count += 1
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 个ROI图像，成功 {success_count} 个")
            
            print(f"\nLBP纹理图像处理完成！")
            print(f"成功处理: {success_count}/{len(roi_files)} 帧")
            
            return {
                'success': True,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': len(roi_files) - success_count,
                'results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': f'批量处理ROI LBP纹理时出错: {str(e)}'}
    
    def create_video_from_lbp(self, lbp_dir: str, output_video_path: str, 
                             fps: float = 2.39) -> Dict[str, Any]:
        """
        将LBP纹理图像序列组装成视频
        
        Args:
            lbp_dir: LBP纹理图像目录路径
            output_video_path: 输出视频路径
            fps: 视频帧率，默认为原始视频帧率
            
        Returns:
            视频生成结果
        """
        try:
            # 获取所有LBP纹理图像文件（按顺序）
            lbp_pattern = os.path.join(lbp_dir, "*_lbp.png")
            lbp_files = sorted(glob.glob(lbp_pattern), 
                              key=lambda x: int(os.path.basename(x).split('_')[1]))
            
            if not lbp_files:
                return {'success': False, 'error': f'在目录 {lbp_dir} 中未找到LBP纹理图像文件'}
            
            print(f"找到 {len(lbp_files)} 个LBP纹理图像文件")
            print(f"开始创建视频: {output_video_path}")
            
            # 读取第一张图像获取尺寸
            first_frame = cv2.imread(lbp_files[0])
            if first_frame is None:
                return {'success': False, 'error': '无法读取第一张LBP纹理图像'}
            
            height, width = first_frame.shape[:2]
            
            # 设置视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                return {'success': False, 'error': '无法创建视频写入器'}
            
            # 逐帧写入视频
            success_count = 0
            for i, lbp_file in enumerate(tqdm(lbp_files, desc="生成LBP纹理视频")):
                frame = cv2.imread(lbp_file)
                if frame is not None:
                    video_writer.write(frame)
                    success_count += 1
                else:
                    print(f"警告：无法读取LBP纹理图像: {lbp_file}")
            
            # 释放资源
            video_writer.release()
            
            # 检查输出文件
            if os.path.exists(output_video_path):
                file_size = os.path.getsize(output_video_path)
                if file_size > 0:
                    print(f"LBP纹理视频创建成功: {output_video_path}")
                    print(f"统计信息: 尺寸：{width}x{height}，帧数：{success_count}，大小：{file_size/1024/1024:.2f} MB")
                    
                    return {
                        'success': True,
                        'video_path': output_video_path,
                        'total_frames': len(lbp_files),
                        'written_frames': success_count,
                        'resolution': f"{width}x{height}",
                        'file_size_mb': file_size / 1024 / 1024,
                        'fps': fps
                    }
                else:
                    return {'success': False, 'error': '生成的视频文件为空'}
            else:
                return {'success': False, 'error': '视频文件未生成'}
                
        except Exception as e:
            return {'success': False, 'error': f'创建视频时出错: {str(e)}'}


def main():
    """主函数 - LBP纹理图视频生成流程"""
    # 设置路径
    roi_dir = "data/roi_imgs"
    lbp_dir = "data/lbp_imgs"
    output_video = "data/LBP纹理视频.mp4"
    
    # 初始化LBP纹理处理器
    processor = LBPTextureProcessor(radius=3, n_points=24)
    
    print("=== LBP纹理图视频生成流程 ===")
    
    # 步骤1: 对ROI图像计算LBP纹理图
    print("步骤1: 对ROI图像计算LBP纹理图...")
    result1 = processor.process_all_roi_lbp(roi_dir, lbp_dir)
    
    if not result1['success']:
        print(f"错误：{result1['error']}")
        return
    
    print(f"LBP纹理图生成结果: 成功 {result1['success_frames']}/{result1['total_frames']} 帧")
    
    # 步骤2: 将LBP纹理图组装成视频
    print("\n步骤2: 将LBP纹理图组装成视频...")
    result2 = processor.create_video_from_lbp(lbp_dir, output_video, fps=2.39)
    
    if not result2['success']:
        print(f"错误：{result2['error']}")
        return
    
    print(f"LBP纹理视频创建结果: {result2}")
    print(f"完成！LBP纹理视频保存到: {output_video}")


if __name__ == "__main__":
    main()

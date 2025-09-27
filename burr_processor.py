#!/usr/bin/env python3
"""
毛刺图像处理器
主要功能：
1. 从ROI图像中检测毛刺特征
2. 生成毛刺可视化图像
3. 批量处理并创建毛刺视频
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

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

from feature_extractor import FeatureExtractor
from preprocessor import ImagePreprocessor
from config import PREPROCESS_CONFIG, DATA_DIR

class BurrProcessor:
    """毛刺图像处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化毛刺处理器
        
        Args:
            config: 配置参数
        """
        self.config = config if config is not None else PREPROCESS_CONFIG
        self.feature_extractor = FeatureExtractor(self.config)
        
    def process_single_roi_burrs(self, roi_image_path: str, output_path: str) -> Dict[str, Any]:
        """
        对ROI图像进行毛刺检测并生成毛刺可视化图
        
        Args:
            roi_image_path: ROI图像路径
            output_path: 输出毛刺图像路径
            
        Returns:
            毛刺检测结果
        """
        try:
            # 读取ROI图像
            roi_image = cv2.imread(roi_image_path, cv2.IMREAD_GRAYSCALE)
            if roi_image is None:
                return {'success': False, 'error': f'无法读取ROI图像: {roi_image_path}'}
            
            # 使用特征提取器进行整体毛刺检测（mask=None让内部处理）
            burr_result = self.feature_extractor.detect_burs(roi_image, mask=None)  # 整图检测，mask=None时内部会创建
            
            # 获取毛刺二值掩码
            burr_binary = burr_result.get('burs_binary_mask', None)
            
            if burr_binary is not None:
                # 创建毛刺可视化图像
                burr_visualization = self.create_burr_visualization(roi_image, burr_binary)
                
                # 保存毛刺可视化图像
                success = cv2.imwrite(output_path, burr_visualization)
                if success:
                    return {
                        'success': True,
                        'burs_count': burr_result.get('burs_count', 0),
                        'burs_density': burr_result.get('burs_density', 0.0),
                        'burs_total_area': burr_result.get('burs_total_area', 0),
                        'output_path': output_path
                    }
                else:
                    return {'success': False, 'error': f'无法保存毛刺图像到: {output_path}'}
            else:
                return {'success': False, 'error': '未能生成毛刺二值图像'}
                
        except Exception as e:
            return {'success': False, 'error': f'处理ROI图像时出错: {str(e)}'}

    def create_burr_visualization(self, background_image: np.ndarray, 
                                burr_binary: np.ndarray) -> np.ndarray:
        """
        使用matplotlib方法生成毛刺可视化，类似斑块图的生成方式
        Args:
            background_image: 背景图像
            burr_binary: 毛刺二值掩码
            
        Returns:
            毛刺可视化图像
        """
        try:
            import io
            from PIL import Image
            
            # 确保输入为灰度图
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 使用matplotlib生成毛刺可视化图像
            fig, ax = plt.subplots(1, 1, figsize=(6, 12))
            ax.imshow(gray_background, cmap='gray', alpha=0.7)
            ax.imshow(burr_binary, cmap='Oranges', alpha=0.8)  # 使用橙色表示毛刺
            ax.axis('off')
            
            # 保存到内存中的字节流
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
            print(f"matplotlib方法失败，使用OpenCV回退: {e}")
            
            # OpenCV回退方法 - 模拟橙色毛刺可视化
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 背景透明度处理
            background_result = (gray_background * 0.7).astype(np.uint8)
            
            # 橙色毛刺图层 - 模拟橙色通道
            orange_result = np.zeros(gray_background.shape + (3,), dtype=np.uint8)
            burr_pixels = burr_binary > 0
            orange_result[burr_pixels, 0] = 255  # 红色通道 
            orange_result[burr_pixels, 1] = 165  # 绿色通道 (橙色=红+部分绿)
            orange_result[burr_pixels, 2] = 0    # 蓝色通道
            
            # Alpha混合
            orange_layer = orange_result.astype(np.float32)
            bg_layer = cv2.cvtColor(background_result, cv2.COLOR_GRAY2RGB).astype(np.float32)
            
            # 进行alpha混合
            alpha_burr = 0.8
            alpha_bg = 0.7
            overlay = alpha_burr * orange_layer + alpha_bg * (1 - alpha_burr) * bg_layer
            
            return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def process_all_roi_burrs(self, roi_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        处理所有ROI图像生成毛刺图
        
        Args:
            roi_dir: ROI图像目录路径
            output_dir: 输出毛刺图目录路径
            
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
            print(f"开始计算毛刺图...")
            
            # 批量处理ROI毛刺检测
            results = []
            success_count = 0
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="计算毛刺图")):
                # 生成输出文件名
                basename = os.path.basename(roi_path)
                name_parts = basename.split('_')
                if len(name_parts) >= 2:
                    frame_name = "_".join(name_parts[:2])  # 提取frame_XXXXXX部分
                    output_path = os.path.join(output_dir, f"{frame_name}_burr.png")
                else:
                    continue
                
                # 处理单个ROI毛刺检测
                result = self.process_single_roi_burrs(roi_path, output_path)
                results.append(result)
                
                if result['success']:
                    success_count += 1
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 个ROI图像，成功 {success_count} 个")
            
            print(f"\n毛刺图像处理完成！")
            print(f"成功处理: {success_count}/{len(roi_files)} 帧")
            
            return {
                'success': True,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': len(roi_files) - success_count,
                'results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': f'批量处理ROI毛刺时出错: {str(e)}'}
    
    def create_video_from_burrs(self, burrs_dir: str, output_video_path: str, 
                               fps: float = 2.39) -> Dict[str, Any]:
        """
        将毛刺图像序列组装成视频
        
        Args:
            burrs_dir: 毛刺图像目录路径
            output_video_path: 输出视频路径
            fps: 视频帧率，默认为原始视频帧率
            
        Returns:
            视频生成结果
        """
        try:
            # 获取所有毛刺图像文件（按顺序）
            burr_pattern = os.path.join(burrs_dir, "*_burr.png")
            burr_files = sorted(glob.glob(burr_pattern), 
                               key=lambda x: int(os.path.basename(x).split('_')[1]))
            
            if not burr_files:
                return {'success': False, 'error': f'在目录 {burrs_dir} 中未找到毛刺图像文件'}
            
            print(f"找到 {len(burr_files)} 个毛刺图像文件")
            print(f"开始创建毛刺视频: {output_video_path}")
            
            # 读取第一张图像获取尺寸
            first_frame = cv2.imread(burr_files[0], cv2.IMREAD_COLOR)
            if first_frame is None:
                return {'success': False, 'error': '无法读取第一张毛刺图像'}
            
            height, width, channels = first_frame.shape
            
            # 设置视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, 
                                         (width, height), isColor=True)
            
            if not video_writer.isOpened():
                return {'success': False, 'error': '无法创建视频写入器'}
            
            # 逐帧写入视频
            success_count = 0
            for i, burr_file in enumerate(tqdm(burr_files, desc="生成毛刺视频")):
                frame = cv2.imread(burr_file, cv2.IMREAD_COLOR)
                if frame is not None:
                    video_writer.write(frame)
                    success_count += 1
                else:
                    print(f"警告：无法读取毛刺图像: {burr_file}")
            
            # 释放资源
            video_writer.release()
            
            # 检查输出文件
            if os.path.exists(output_video_path):
                file_size = os.path.getsize(output_video_path)
                if file_size > 0:
                    print(f"毛刺视频创建成功: {output_video_path}")
                    print(f"统计 INFORMATION：尺寸：{width}x{height}，帧数：{success_count}，大小：{file_size/1024/1024:.2f} MB")
                    
                    return {
                        'success': True,
                        'video_path': output_video_path,
                        'total_frames': len(burr_files),
                        'written_frames': success_count,
                        'resolution': f"{width}x{height}",
                        'file_size_mb': file_size / 1024 / 1024,
                        'fps': fps
                    }
                else:
                    return {'success': False, 'error': '生成的视频文件为空'}
            else:
                return {'success': False, 'error': '毛刺视频文件未生成'}
                
        except Exception as e:
            return {'success': False, 'error': f'创建毛刺视频时出错: {str(e)}'}


def main():
    """主函数 - 毛刺视频生成流程：ROI图像 -> 毛刺检测 -> 毛刺视频"""
    
    # 设置路径
    roi_dir = os.path.join(DATA_DIR, 'roi_imgs')
    burr_dir = os.path.join(DATA_DIR, 'burr_imgs')
    output_video = os.path.join(DATA_DIR, '毛刺视频.mp4')
    
    # 初始化毛刺处理器
    processor = BurrProcessor()
    
    print("=== 毛刺图像处理流程 ===")
    print("方式：整体毛刺检测（mask=None）")
    
    # 步骤1: 对ROI图像进行毛刺检测
    print("步骤1: 对ROI图像进行毛刺检测...")
    result1 = processor.process_all_roi_burrs(roi_dir, burr_dir)
    
    if not result1['success']:
        print(f"错误：{result1['error']}")
        return
    
    print(f"毛刺图像生成结果: 成功 {result1['success_frames']}/{result1['total_frames']} 帧")
    
    # 步骤2: 将毛刺图组装成视频
    print("\n步骤2: 将毛刺图组装成视频...")
    result2 = processor.create_video_from_burrs(burr_dir, output_video, fps=2.39)
    
    if not result2['success']:
        print(f"错误：{result2['error']}")
        return
    
    print(f"毛刺视频创建结果: {result2}")
    print(f"完成！毛刺视频保存到: {output_video}")


if __name__ == "__main__":
    main()

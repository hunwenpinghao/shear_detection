#!/usr/bin/env python3
"""
处理第二个视频的完整流水线
Video_20250821140339629.avi -> 帧提取 -> ROI -> 斑块分析 -> LBP纹理 -> 毛刺检测 -> 组合视频
"""

import cv2
import numpy as np
import os
import sys
import glob
from typing import List, Dict, Any, Optional, Tuple
import time
from tqdm import tqdm

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

from feature_extractor import FeatureExtractor
from preprocessor import ImagePreprocessor
from config import PREPROCESS_CONFIG, DATA_DIR

class Video2Processor:
    """第二个视频处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化处理器"""
        self.config = config if config is not None else PREPROCESS_CONFIG
        self.preprocessor = ImagePreprocessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        
        # 设置输出目录
        self.base_output_dir = "data_video3_20250820124904881"
        self.images_dir = os.path.join(self.base_output_dir, 'images')
        self.roi_dir = os.path.join(self.base_output_dir, 'roi_imgs')
        self.spots_dir = os.path.join(self.base_output_dir, 'bankuai')
        self.burrs_dir = os.path.join(self.base_output_dir, 'burrs')
        self.texture_dir = os.path.join(self.base_output_dir, 'texture')
        self.analysis_dir = os.path.join(self.base_output_dir, 'analysis')
        
        # 创建所有目录
        for dir_path in [self.images_dir, self.roi_dir, self.spots_dir, 
                        self.burrs_dir, self.texture_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def extract_frames_from_video(self, video_path: str, interval: int = 5) -> Dict[str, Any]:
        """
        从视频中提取帧图像
        
        Args:
            video_path: 视频文件路径
            interval: 提取间隔（秒）
            
        Returns:
            提取结果统计
        """
        print(f"=== 从视频提取帧图像 ===")
        print(f"视频路径: {video_path}")
        print(f"提取间隔: {interval}秒")
        
        try:
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'success': False, 'error': f'无法打开视频: {video_path}'}
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}秒")
            
            # 计算提取帧数
            frame_interval = int(fps * interval)
            extracted_count = 0
            
            # 提取帧
            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按间隔提取帧
                if frame_number % frame_interval == 0:
                    # 生成文件名
                    timestamp = frame_number / fps
                    filename = f"frame_{frame_number:06d}.jpg"
                    filepath = os.path.join(self.images_dir, filename)
                    
                    # 保存帧
                    success = cv2.imwrite(filepath, frame)
                    if success:
                        extracted_count += 1
                    
                    # 每100帧输出一次进度
                    if extracted_count % 100 == 0:
                        print(f"已提取 {extracted_count} 帧")
                
                frame_number += 1
            
            cap.release()
            
            print(f"帧提取完成！共提取 {extracted_count} 帧")
            
            return {
                'success': True,
                'total_frames': total_frames,
                'extracted_frames': extracted_count,
                'fps': fps,
                'duration': duration,
                'interval': interval,
                'output_dir': self.images_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'提取帧时出错: {str(e)}'}
    
    def extract_roi_from_all_frames(self, target_size: Tuple[int, int] = (128, 512)) -> Dict[str, Any]:
        """
        从所有帧图像提取ROI区域
        
        Args:
            target_size: ROI标准化后的目标尺寸
            
        Returns:
            ROI提取结果统计
        """
        print(f"\n=== 提取ROI区域 ===")
        
        try:
            # 获取所有图像文件（按顺序）
            image_pattern = os.path.join(self.images_dir, "frame_*.jpg")
            image_files = sorted(glob.glob(image_pattern), 
                               key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not image_files:
                return {'success': False, 'error': f'在目录 {self.images_dir} 中未找到图像文件'}
            
            print(f"找到 {len(image_files)} 个图像文件")
            print(f"开始ROI提取...")
            
            # 批量处理图像ROI提取
            success_count = 0
            
            for i, image_path in enumerate(tqdm(image_files, desc="提取ROI区域")):
                # 生成输出文件名
                basename = os.path.basename(image_path)
                name, ext = os.path.splitext(basename)
                roi_output_path = os.path.join(self.roi_dir, f"{name}_roi.png")
                
                # 提取ROI
                roi_image, processing_info = self.preprocessor.preprocess_pipeline(
                    image_path, target_size=target_size)
                
                # 保存ROI图像
                success = cv2.imwrite(roi_output_path, roi_image)
                if success:
                    success_count += 1
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已提取ROI {i + 1}/{len(image_files)} 帧图像，成功 {success_count} 帧")
            
            print(f"\nROI提取完成！")
            print(f"成功提取: {success_count}/{len(image_files)} 帧")
            
            return {
                'success': True,
                'total_frames': len(image_files),
                'success_frames': success_count,
                'failed_frames': len(image_files) - success_count,
                'output_dir': self.roi_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'ROI提取时出错: {str(e)}'}
    
    def extract_roi_from_all_frames_with_config(self, preprocessor, target_size: Tuple[int, int] = (128, 512)) -> Dict[str, Any]:
        """
        从所有帧图像提取ROI区域（使用自定义预处理器配置）
        
        Args:
            preprocessor: 自定义预处理器实例
            target_size: ROI标准化后的目标尺寸
            
        Returns:
            ROI提取结果统计
        """
        print(f"\n=== 提取ROI区域（自定义配置） ===")
        
        try:
            # 获取所有图像文件（按顺序）
            image_pattern = os.path.join(self.images_dir, "frame_*.jpg")
            image_files = sorted(glob.glob(image_pattern), 
                               key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not image_files:
                return {'success': False, 'error': f'在目录 {self.images_dir} 中未找到图像文件'}
            
            print(f"找到 {len(image_files)} 个图像文件")
            print(f"开始ROI提取（自定义配置）...")
            
            # 批量处理图像ROI提取
            success_count = 0
            
            for i, image_path in enumerate(tqdm(image_files, desc="提取ROI区域（自定义配置）")):
                try:
                    # 生成输出文件名
                    basename = os.path.basename(image_path)
                    name, ext = os.path.splitext(basename)
                    roi_output_path = os.path.join(self.roi_dir, f"{name}_roi.png")
                    
                    # 检查ROI文件是否已存在，如果存在则跳过
                    if os.path.exists(roi_output_path):
                        success_count += 1
                        continue
                    
                    # 提取ROI
                    roi_image, processing_info = preprocessor.preprocess_pipeline(
                        image_path, target_size=target_size)
                    
                    # 保存ROI图像
                    success = cv2.imwrite(roi_output_path, roi_image)
                    if success:
                        success_count += 1
                
                except Exception as e:
                    # 跳过无法处理的图像，继续处理下一张
                    print(f"跳过无法处理的图像 {image_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已提取ROI {i + 1}/{len(image_files)} 帧图像，成功 {success_count} 帧")
            
            print(f"\nROI提取完成（自定义配置）！")
            print(f"成功提取: {success_count}/{len(image_files)} 帧")
            
            return {
                'success': True,
                'total_frames': len(image_files),
                'success_frames': success_count,
                'failed_frames': len(image_files) - success_count,
                'output_dir': self.roi_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'ROI提取时出错: {str(e)}'}
    
    def continue_roi_extraction(self, preprocessor, target_size: Tuple[int, int] = (128, 512)) -> Dict[str, Any]:
        """
        继续完成ROI提取（跳过已存在的文件）
        
        Args:
            preprocessor: 预处理器实例
            target_size: ROI标准化后的目标尺寸
            
        Returns:
            ROI提取结果统计
        """
        print(f"\n=== 继续完成ROI提取 ===")
        
        try:
            # 获取所有图像文件（按顺序）
            image_pattern = os.path.join(self.images_dir, "frame_*.jpg")
            image_files = sorted(glob.glob(image_pattern), 
                               key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not image_files:
                return {'success': False, 'error': f'在目录 {self.images_dir} 中未找到图像文件'}
            
            print(f"找到 {len(image_files)} 个图像文件")
            print(f"开始继续ROI提取...")
            
            # 批量处理图像ROI提取
            success_count = 0
            processed_count = 0
            
            for i, image_path in enumerate(tqdm(image_files, desc="继续ROI提取")):
                try:
                    # 生成输出文件名
                    basename = os.path.basename(image_path)
                    name, ext = os.path.splitext(basename)
                    roi_output_path = os.path.join(self.roi_dir, f"{name}_roi.png")
                    
                    # 检查ROI文件是否已存在，如果存在则跳过
                    if os.path.exists(roi_output_path):
                        success_count += 1
                        continue
                    
                    processed_count += 1
                    
                    # 提取ROI
                    roi_image, processing_info = preprocessor.preprocess_pipeline(
                        image_path, target_size=target_size)
                    
                    # 保存ROI图像
                    success = cv2.imwrite(roi_output_path, roi_image)
                    if success:
                        success_count += 1
                
                except Exception as e:
                    # 跳过无法处理的图像，继续处理下一张
                    print(f"跳过无法处理的图像 {image_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(image_files)} 帧图像，成功 {success_count} 帧，新处理 {processed_count} 帧")
            
            print(f"\nROI提取继续完成！")
            print(f"成功处理: {success_count}/{len(image_files)} 帧，新处理: {processed_count} 帧")
            
            return {
                'success': True,
                'total_frames': len(image_files),
                'success_frames': success_count,
                'failed_frames': len(image_files) - success_count,
                'newly_processed': processed_count,
                'output_dir': self.roi_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'继续ROI提取时出错: {str(e)}'}
    
    def generate_spot_analysis(self) -> Dict[str, Any]:
        """
        生成斑块分析（斑块图和统计数据）
        
        Returns:
            斑块分析结果统计
        """
        print(f"\n=== 生成斑块分析 ===")
        
        try:
            # 获取所有ROI图像文件（按顺序）
            roi_pattern = os.path.join(self.roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not roi_files:
                return {'success': False, 'error': f'在目录 {self.roi_dir} 中未找到ROI图像文件'}
            
            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print(f"开始斑块检测...")
            
            # 存储分析数据
            analysis_data = []
            success_count = 0
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="斑块检测")):
                try:
                    # 读取ROI图像
                    roi_image = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
                    if roi_image is None:
                        continue
                    
                    # 使用特征提取器检测斑块
                    spot_result = self.feature_extractor.detect_all_white_spots(roi_image)
                    
                    # 获取斑块二值掩码
                    spot_binary = spot_result.get('all_white_binary_mask', None)
                    
                    if spot_binary is not None:
                        # 生成斑块可视化图像
                        spot_visualization = self.create_spot_visualization(roi_image, spot_binary)
                        
                        # 生成输出文件名
                        basename = os.path.basename(roi_path)
                        name_parts = basename.split('_')
                        if len(name_parts) >= 2:
                            frame_name = "_".join(name_parts[:2])
                            spot_output_path = os.path.join(self.spots_dir, f"{frame_name}_bankuai.png")
                            
                            # 保存斑块图像
                            success = cv2.imwrite(spot_output_path, spot_visualization)
                            if success:
                                success_count += 1
                                
                                # 记录分析数据
                                frame_number = int(name_parts[1])
                                time_seconds = frame_number * 5  # 假设每5秒一帧
                                
                                analysis_data.append({
                                    'frame_number': frame_number,
                                    'time_seconds': time_seconds,
                                    'spot_count': spot_result['all_spot_count'],
                                    'spot_density': spot_result['all_spot_density'],
                                    'roi_path': roi_path,
                                    'spot_path': spot_output_path
                                })
                
                except Exception as e:
                    print(f"处理ROI图像时出错 {roi_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 个ROI图像，成功 {success_count} 个")
            
            print(f"\n斑块分析完成！")
            print(f"成功处理: {success_count}/{len(roi_files)} 帧")
            
            # 保存分析数据
            import pandas as pd
            df = pd.DataFrame(analysis_data)
            analysis_file = os.path.join(self.analysis_dir, 'spot_temporal_data.csv')
            df.to_csv(analysis_file, index=False)
            
            print(f"分析数据已保存: {analysis_file}")
            
            return {
                'success': True,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': len(roi_files) - success_count,
                'analysis_data': analysis_data,
                'analysis_file': analysis_file,
                'output_dir': self.spots_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'斑块分析时出错: {str(e)}'}
    
    def create_spot_visualization(self, background_image: np.ndarray, 
                                spot_binary: np.ndarray) -> np.ndarray:
        """
        创建斑块可视化图像（使用matplotlib方法）
        """
        try:
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            
            # 确保输入为灰度图
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 使用matplotlib生成可视化
            fig, ax = plt.subplots(1, 1, figsize=(6, 12))
            ax.imshow(gray_background, cmap='gray', alpha=0.7)
            ax.imshow(spot_binary, cmap='Reds', alpha=0.8)
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
            
            # OpenCV回退方法
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 背景alpha
            background_result = (gray_background * 0.7).astype(np.uint8)
            
            # 红色图层
            red_result = np.zeros(gray_background.shape + (3,), dtype=np.uint8)
            spot_pixels = spot_binary > 0
            red_result[spot_pixels, 0] = 255
            
            # alpha混合
            red_layer = red_result.astype(np.float32)
            bg_layer = cv2.cvtColor(background_result, cv2.COLOR_GRAY2RGB).astype(np.float32)
            
            alpha_spot = 0.8
            alpha_bg = 0.7
            overlay = alpha_spot * red_layer + alpha_bg * (1 - alpha_spot) * bg_layer
            return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    def generate_burr_analysis(self) -> Dict[str, Any]:
        """
        生成毛刺分析
        
        Returns:
            毛刺分析结果统计
        """
        print(f"\n=== 生成毛刺分析 ===")
        
        try:
            # 获取所有ROI图像文件（按顺序）
            roi_pattern = os.path.join(self.roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not roi_files:
                return {'success': False, 'error': f'在目录 {self.roi_dir} 中未找到ROI图像文件'}
            
            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print(f"开始毛刺检测...")
            
            success_count = 0
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="毛刺检测")):
                try:
                    # 读取ROI图像
                    roi_image = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
                    if roi_image is None:
                        continue
                    
                    # 使用特征提取器检测毛刺
                    burr_result = self.feature_extractor.detect_burs(roi_image)
                    
                    # 获取毛刺二值掩码 - 修复键名问题
                    burr_binary = burr_result.get('burs_binary_mask', None)
                    
                    if burr_binary is not None:
                        # 生成毛刺可视化图像
                        burr_visualization = self.create_burr_visualization(roi_image, burr_binary)
                        
                        # 生成输出文件名
                        basename = os.path.basename(roi_path)
                        name_parts = basename.split('_')
                        if len(name_parts) >= 2:
                            frame_name = "_".join(name_parts[:2])
                            burr_output_path = os.path.join(self.burrs_dir, f"{frame_name}_burr.png")
                            
                            # 保存毛刺图像
                            success = cv2.imwrite(burr_output_path, burr_visualization)
                            if success:
                                success_count += 1
                
                except Exception as e:
                    print(f"处理ROI图像时出错 {roi_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 个ROI图像，成功 {success_count} 个")
            
            print(f"\n毛刺分析完成！")
            print(f"成功处理: {success_count}/{len(roi_files)} 帧")
            
            return {
                'success': True,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': len(roi_files) - success_count,
                'output_dir': self.burrs_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'毛刺分析时出错: {str(e)}'}
    
    def generate_burr_analysis_fixed(self) -> Dict[str, Any]:
        """
        生成毛刺分析（修复版本）
        
        Returns:
            毛刺分析结果统计
        """
        print(f"\n=== 生成毛刺分析（修复版本） ===")
        
        try:
            # 获取所有ROI图像文件（按顺序）
            roi_pattern = os.path.join(self.roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not roi_files:
                return {'success': False, 'error': f'在目录 {self.roi_dir} 中未找到ROI图像文件'}
            
            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print(f"开始毛刺检测（修复版本）...")
            
            success_count = 0
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="毛刺检测（修复版本）")):
                try:
                    # 读取ROI图像
                    roi_image = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
                    if roi_image is None:
                        continue
                    
                    # 使用特征提取器检测毛刺
                    burr_result = self.feature_extractor.detect_burs(roi_image)
                    
                    # 获取毛刺二值掩码 - 使用正确的键名
                    burr_binary = burr_result.get('burs_binary_mask', None)
                    
                    if burr_binary is not None:
                        # 生成毛刺可视化图像
                        burr_visualization = self.create_burr_visualization(roi_image, burr_binary)
                        
                        # 生成输出文件名
                        basename = os.path.basename(roi_path)
                        name_parts = basename.split('_')
                        if len(name_parts) >= 2:
                            frame_name = "_".join(name_parts[:2])
                            burr_output_path = os.path.join(self.burrs_dir, f"{frame_name}_burr.png")
                            
                            # 保存毛刺图像
                            success = cv2.imwrite(burr_output_path, burr_visualization)
                            if success:
                                success_count += 1
                
                except Exception as e:
                    print(f"处理ROI图像时出错 {roi_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 个ROI图像，成功 {success_count} 个")
            
            print(f"\n毛刺分析完成（修复版本）！")
            print(f"成功处理: {success_count}/{len(roi_files)} 帧")
            
            return {
                'success': True,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': len(roi_files) - success_count,
                'output_dir': self.burrs_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'毛刺分析时出错: {str(e)}'}
    
    def create_burr_visualization(self, background_image: np.ndarray, 
                                burr_binary: np.ndarray) -> np.ndarray:
        """
        创建毛刺可视化图像
        """
        try:
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            
            # 确保输入为灰度图
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 使用matplotlib生成可视化
            fig, ax = plt.subplots(1, 1, figsize=(6, 12))
            ax.imshow(gray_background, cmap='gray', alpha=0.7)
            ax.imshow(burr_binary, cmap='Blues', alpha=0.8)  # 使用蓝色表示毛刺
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
            
            # OpenCV回退方法
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 背景alpha
            background_result = (gray_background * 0.7).astype(np.uint8)
            
            # 蓝色图层（毛刺）
            blue_result = np.zeros(gray_background.shape + (3,), dtype=np.uint8)
            burr_pixels = burr_binary > 0
            blue_result[burr_pixels, 2] = 255  # 蓝色通道
            
            # alpha混合
            blue_layer = blue_result.astype(np.float32)
            bg_layer = cv2.cvtColor(background_result, cv2.COLOR_GRAY2RGB).astype(np.float32)
            
            alpha_burr = 0.8
            alpha_bg = 0.7
            overlay = alpha_burr * blue_layer + alpha_bg * (1 - alpha_burr) * bg_layer
            return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    def generate_texture_analysis(self) -> Dict[str, Any]:
        """
        生成LBP纹理分析
        
        Returns:
            纹理分析结果统计
        """
        print(f"\n=== 生成LBP纹理分析 ===")
        
        try:
            # 获取所有ROI图像文件（按顺序）
            roi_pattern = os.path.join(self.roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not roi_files:
                return {'success': False, 'error': f'在目录 {self.roi_dir} 中未找到ROI图像文件'}
            
            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print(f"开始LBP纹理分析...")
            
            success_count = 0
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="LBP纹理分析")):
                try:
                    # 读取ROI图像
                    roi_image = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
                    if roi_image is None:
                        continue
                    
                    # 计算LBP纹理
                    lbp_result = self.compute_lbp_texture(roi_image)
                    
                    if lbp_result is not None:
                        # 生成输出文件名
                        basename = os.path.basename(roi_path)
                        name_parts = basename.split('_')
                        if len(name_parts) >= 2:
                            frame_name = "_".join(name_parts[:2])
                            texture_output_path = os.path.join(self.texture_dir, f"{frame_name}_texture.png")
                            
                            # 保存纹理图像
                            success = cv2.imwrite(texture_output_path, lbp_result)
                            if success:
                                success_count += 1
                
                except Exception as e:
                    print(f"处理ROI图像时出错 {roi_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 个ROI图像，成功 {success_count} 个")
            
            print(f"\nLBP纹理分析完成！")
            print(f"成功处理: {success_count}/{len(roi_files)} 帧")
            
            return {
                'success': True,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': len(roi_files) - success_count,
                'output_dir': self.texture_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'LBP纹理分析时出错: {str(e)}'}
    
    def compute_lbp_texture(self, gray_image: np.ndarray) -> Optional[np.ndarray]:
        """
        计算LBP纹理图
        """
        try:
            from skimage.feature import local_binary_pattern
            
            # 计算LBP
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
            
            # 转换为uint8
            lbp_normalized = ((lbp / lbp.max()) * 255).astype(np.uint8)
            
            # 应用颜色映射
            lbp_colored = cv2.applyColorMap(lbp_normalized, cv2.COLORMAP_JET)
            
            return lbp_colored
            
        except ImportError:
            print("警告：skimage未安装，使用简化LBP计算")
            return self.simple_lbp(gray_image)
        except Exception as e:
            print(f"LBP计算失败: {e}")
            return None
    
    def simple_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """
        简化的LBP计算
        """
        # 简化的LBP实现
        lbp = np.zeros_like(gray_image)
        
        for i in range(1, gray_image.shape[0] - 1):
            for j in range(1, gray_image.shape[1] - 1):
                center = gray_image[i, j]
                binary_string = 0
                
                # 8邻域比较
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        binary_string |= (1 << k)
                
                lbp[i, j] = binary_string
        
        # 归一化并应用颜色映射
        lbp_normalized = ((lbp / 255) * 255).astype(np.uint8)
        lbp_colored = cv2.applyColorMap(lbp_normalized, cv2.COLORMAP_JET)
        
        return lbp_colored
    
    def create_combined_video(self, fps: float = 2.39) -> Dict[str, Any]:
        """
        创建组合视频：ROI + 斑块 + 毛刺 + 纹理
        
        Args:
            fps: 视频帧率
            
        Returns:
            视频创建结果
        """
        print(f"\n=== 创建组合视频 ===")
        
        try:
            # 获取所有ROI图像文件（按顺序）
            roi_pattern = os.path.join(self.roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not roi_files:
                return {'success': False, 'error': f'在目录 {self.roi_dir} 中未找到ROI图像文件'}
            
            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print(f"开始创建组合视频...")
            
            # 读取第一张图像获取尺寸
            first_roi = cv2.imread(roi_files[0])
            if first_roi is None:
                return {'success': False, 'error': '无法读取第一张ROI图像'}
            
            roi_height, roi_width = first_roi.shape[:2]
            
            # 组合图像的尺寸（4列）
            combined_width = roi_width * 4
            combined_height = roi_height
            
            # 设置视频编码器
            output_video_path = os.path.join(self.base_output_dir, 'final_combined_video.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, 
                                         (combined_width, combined_height), isColor=True)
            
            if not video_writer.isOpened():
                return {'success': False, 'error': '无法创建视频写入器'}
            
            # 获取中文字体
            font = self.get_chinese_font(14)  # 进一步调整字体大小
            
            # 逐帧处理
            success_count = 0
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="创建组合视频")):
                try:
                    # 读取四种图像
                    roi_image = cv2.imread(roi_path)
                    if roi_image is None:
                        continue
                    
                    # 获取对应的斑块、毛刺、纹理图像路径
                    basename = os.path.basename(roi_path)
                    name_parts = basename.split('_')
                    if len(name_parts) >= 2:
                        frame_name = "_".join(name_parts[:2])
                        
                        spot_path = os.path.join(self.spots_dir, f"{frame_name}_bankuai.png")
                        burr_path = os.path.join(self.burrs_dir, f"{frame_name}_burr.png")
                        texture_path = os.path.join(self.texture_dir, f"{frame_name}_texture.png")
                        
                        # 读取其他图像
                        spot_image = cv2.imread(spot_path) if os.path.exists(spot_path) else np.zeros_like(roi_image)
                        burr_image = cv2.imread(burr_path) if os.path.exists(burr_path) else np.zeros_like(roi_image)
                        texture_image = cv2.imread(texture_path) if os.path.exists(texture_path) else np.zeros_like(roi_image)
                        
                        # 调整图像尺寸
                        roi_resized = cv2.resize(roi_image, (roi_width, roi_height))
                        spot_resized = cv2.resize(spot_image, (roi_width, roi_height))
                        burr_resized = cv2.resize(burr_image, (roi_width, roi_height))
                        texture_resized = cv2.resize(texture_image, (roi_width, roi_height))
                        
                        # 水平拼接
                        combined_frame = np.hstack([roi_resized, spot_resized, burr_resized, texture_resized])
                        
                        # 添加标签（支持中文字符）
                        combined_frame = self.add_labels_to_combined_frame(combined_frame, roi_width, font)
                        
                        # 写入视频
                        video_writer.write(combined_frame)
                        success_count += 1
                
                except Exception as e:
                    print(f"处理帧时出错 {roi_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 帧，成功 {success_count} 帧")
            
            # 释放资源
            video_writer.release()
            
            # 检查输出文件
            if os.path.exists(output_video_path):
                file_size = os.path.getsize(output_video_path)
                if file_size > 0:
                    print(f"组合视频创建成功: {output_video_path}")
                    print(f"统计信息: 尺寸：{combined_width}x{combined_height}，帧数：{success_count}，大小：{file_size/1024/1024:.2f} MB")
                    
                    return {
                        'success': True,
                        'video_path': output_video_path,
                        'total_frames': len(roi_files),
                        'written_frames': success_count,
                        'resolution': f"{combined_width}x{combined_height}",
                        'file_size_mb': file_size / 1024 / 1024,
                        'fps': fps
                    }
                else:
                    return {'success': False, 'error': '生成的视频文件为空'}
            else:
                return {'success': False, 'error': '视频文件未生成'}
                
        except Exception as e:
            return {'success': False, 'error': f'创建组合视频时出错: {str(e)}'}
    
    def get_chinese_font(self, font_size: int):
        """根据操作系统获取中文字体"""
        import platform
        from PIL import ImageFont
        
        system = platform.system()
        font_path = None
        
        if system == "Darwin":  # macOS
            font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc"
        elif system == "Windows":
            font_path = "C:/Windows/Fonts/simhei.ttf"
        else:  # Linux
            font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # 假设路径

        if font_path and os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception as e:
                print(f"加载字体 {font_path} 失败: {e}")
        
        print("警告：未找到中文字体，使用默认字体。")
        return ImageFont.load_default()

    def add_labels_to_combined_frame(self, combined_frame: np.ndarray, section_width: int, font=None) -> np.ndarray:
        """
        为组合帧添加标签（支持中文字符显示）
        """
        try:
            from PIL import Image, ImageDraw
            
            # 转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # 如果没有提供字体，使用默认字体
            if font is None:
                font = self.get_chinese_font(14)  # 进一步调整字体大小

            labels = [
                "ROI",
                "SPOTS (斑块)",
                "BURRS (毛刺)",
                "LBP TEXTURE (纹理)"
            ]

            for i, label in enumerate(labels):
                # 计算文本宽度和高度
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # 计算文本位置 (居中)
                x = i * section_width + (section_width - text_width) // 2
                y = 10  # 距离顶部10像素

                draw.text((x, y), label, font=font, fill=(255, 255, 255))  # 白色字体

            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"添加标签失败: {e}")
            return combined_frame
    
    def generate_temporal_plots(self) -> Dict[str, Any]:
        """
        生成时间序列分析图表
        
        Returns:
            图表生成结果
        """
        print(f"\n=== 生成时间序列分析图表 ===")
        
        try:
            # 读取分析数据
            analysis_file = os.path.join(self.analysis_dir, 'spot_temporal_data.csv')
            if not os.path.exists(analysis_file):
                return {'success': False, 'error': f'分析数据文件不存在: {analysis_file}'}
            
            import pandas as pd
            df = pd.read_csv(analysis_file)
            data = df.to_dict('records')
            
            print(f"读取了 {len(data)} 个数据点")
            
            # 生成图表
            self.create_temporal_plots(data, self.analysis_dir)
            
            return {
                'success': True,
                'data_points': len(data),
                'output_dir': self.analysis_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'生成时间序列图表时出错: {str(e)}'}
    
    def create_temporal_plots(self, data: List[Dict[str, Any]], output_dir: str):
        """
        创建时间序列图表
        """
        import matplotlib.pyplot as plt
        import platform
        
        # 设置中文字体
        system = platform.system()
        if system == "Darwin":  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'DejaVu Sans']
        elif system == "Windows":
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 提取数据
        time_seconds = [d['time_seconds'] for d in data]
        spot_counts = [d['spot_count'] for d in data]
        spot_densities = [d['spot_density'] for d in data]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 斑块数量
        ax1.plot(time_seconds, spot_counts, 'b-', linewidth=2, alpha=0.8)
        ax1.fill_between(time_seconds, spot_counts, alpha=0.3, color='blue')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('斑块数量')
        ax1.set_title('斑块数量随时间变化')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(time_seconds))
        
        # 斑块密度
        ax2.plot(time_seconds, spot_densities, 'r-', linewidth=2, alpha=0.8)
        ax2.fill_between(time_seconds, spot_densities, alpha=0.3, color='red')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('斑块密度')
        ax2.set_title('斑块密度随时间变化')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(time_seconds))
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(output_dir, 'spot_temporal_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"时间序列图表已保存: {plot_path}")
    
    def run_full_pipeline(self, video_path: str) -> Dict[str, Any]:
        """
        运行完整的处理流水线
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            完整处理结果
        """
        print("=" * 60)
        print("开始处理第三个视频的完整流水线")
        print("=" * 60)
        
        start_time = time.time()
        results = {}
        
        # 步骤1: 提取帧
        print("\n步骤1: 提取视频帧...")
        result1 = self.extract_frames_from_video(video_path, interval=12)
        results['frame_extraction'] = result1
        
        if not result1['success']:
            print(f"错误：{result1['error']}")
            return results
        
        # 步骤2: 提取ROI（调整配置以处理较小的区域）
        print("\n步骤2: 提取ROI区域...")
        # 临时调整配置以处理较小的ROI区域
        temp_config = self.config.copy()
        temp_config['min_roi_area'] = 500  # 降低最小ROI面积要求
        temp_preprocessor = ImagePreprocessor(temp_config)
        result2 = self.extract_roi_from_all_frames_with_config(temp_preprocessor, target_size=(128, 512))
        results['roi_extraction'] = result2
        
        # 检查是否需要继续完成ROI提取
        if result2['success'] and result2['success_frames'] < result2['total_frames']:
            print(f"\n检测到ROI提取不完整 ({result2['success_frames']}/{result2['total_frames']})，继续完成...")
            continue_result = self.continue_roi_extraction(temp_preprocessor, target_size=(128, 512))
            if continue_result['success']:
                result2['success_frames'] = continue_result['success_frames']
                result2['failed_frames'] = continue_result['failed_frames']
                print(f"ROI提取继续完成: {result2['success_frames']}/{result2['total_frames']} 帧")
        
        if not result2['success']:
            print(f"错误：{result2['error']}")
            return results
        
        # 步骤3: 斑块分析
        print("\n步骤3: 斑块分析...")
        result3 = self.generate_spot_analysis()
        results['spot_analysis'] = result3
        
        if not result3['success']:
            print(f"错误：{result3['error']}")
            return results
        
        # 步骤4: 毛刺分析（修复版本）
        print("\n步骤4: 毛刺分析（修复版本）...")
        result4 = self.generate_burr_analysis_fixed()
        results['burr_analysis'] = result4
        
        # 步骤5: LBP纹理分析
        print("\n步骤5: LBP纹理分析...")
        result5 = self.generate_texture_analysis()
        results['texture_analysis'] = result5
        
        # 步骤6: 时间序列图表
        print("\n步骤6: 生成时间序列图表...")
        result6 = self.generate_temporal_plots()
        results['temporal_plots'] = result6
        
        # 步骤7: 组合视频
        print("\n步骤7: 创建组合视频...")
        result7 = self.create_combined_video(fps=2.39)
        results['combined_video'] = result7
        
        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("完整流水线处理完成！")
        print("=" * 60)
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"输出目录: {self.base_output_dir}")
        
        # 输出结果摘要
        print("\n处理结果摘要:")
        for step, result in results.items():
            if result.get('success', False):
                if 'frames' in result:
                    print(f"  {step}: 成功处理 {result.get('success_frames', 0)} 帧")
                elif 'video_path' in result:
                    print(f"  {step}: 视频已保存到 {result['video_path']}")
                else:
                    print(f"  {step}: 成功完成")
            else:
                print(f"  {step}: 失败 - {result.get('error', '未知错误')}")
        
        return results


def main():
    """主函数"""
    # 视频路径
    video_path = "data/Video3_20250820124904881.avi"
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return
    
    # 初始化处理器
    processor = Video2Processor()
    
    # 运行完整流水线
    results = processor.run_full_pipeline(video_path)
    
    print("\n✅ 第三个视频处理完成！")

def continue_roi_only():
    """只继续完成ROI提取"""
    print("=== 继续完成ROI提取 ===")
    
    # 初始化处理器
    processor = Video2Processor()
    
    # 设置自定义配置
    temp_config = processor.config.copy()
    temp_config['min_roi_area'] = 500  # 降低最小ROI面积要求
    temp_preprocessor = ImagePreprocessor(temp_config)
    
    # 继续完成ROI提取
    result = processor.continue_roi_extraction(temp_preprocessor, target_size=(128, 512))
    
    if result['success']:
        print(f"✅ ROI提取继续完成！")
        print(f"总帧数: {result['total_frames']}")
        print(f"成功处理: {result['success_frames']}")
        print(f"新处理: {result['newly_processed']}")
        print(f"失败: {result['failed_frames']}")
    else:
        print(f"❌ ROI提取失败: {result['error']}")


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "continue_roi":
        # 只继续完成ROI提取
        continue_roi_only()
    else:
        # 运行完整流水线
        main()

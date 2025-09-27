"""
斑块图像处理器
主要功能：
1. 提取每一帧的ROI区域（统一尺寸）
2. 对ROI图像计算白色斑块图
3. 保存斑块图像到指定目录
4. 生成斑块视频
"""

import cv2
import numpy as np
import os
import glob
from typing import List, Dict, Any, Optional, Tuple
from feature_extractor import FeatureExtractor
from preprocessor import ImagePreprocessor
from config import PREPROCESS_CONFIG, DATA_DIR
from tqdm import tqdm
import time


class SpotProcessor:
    """斑块处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化斑块处理器
        
        Args:
            config: 配置参数
        """
        self.config = config if config is not None else PREPROCESS_CONFIG
        self.preprocessor = ImagePreprocessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        
    def extract_roi_from_frame(self, image_path: str, output_path: str, 
                               target_size: Tuple[int, int] = (128, 512)) -> Dict[str, Any]:
        """
        从单帧提取ROI区域
        
        Args:
            image_path: 输入图像路径
            output_path: 输出ROI图像路径
            target_size: ROI标准化后的目标尺寸
            
        Returns:
            ROI提取结果
        """
        try:
            # 使用预处理器完整流水线提取ROI
            roi_image, processing_info = self.preprocessor.preprocess_pipeline(
                image_path, target_size=target_size)
            
            # 保存ROI图像
            success = cv2.imwrite(output_path, roi_image)
            if success:
                return {
                    'success': True,
                    'roi_shape': roi_image.shape,
                    'original_shape': processing_info['original_shape'],
                    'roi_info': processing_info['roi_info'],
                    'output_path': output_path
                }
            else:
                return {'success': False, 'error': f'无法保存ROI图像到: {output_path}'}
                
        except Exception as e:
            return {'success': False, 'error': f'ROI提取时出错: {str(e)}'}

    def process_single_roi_spots(self, roi_image_path: str, output_path: str) -> Dict[str, Any]:
        """
        对ROI图像进行斑块检测并生成红色斑块可视化图
        
        Args:
            roi_image_path: ROI图像路径
            output_path: 输出斑块图像路径
            
        Returns:
            斑块检测结果
        """
        try:
            # 读取ROI图像
            roi_image = cv2.imread(roi_image_path, cv2.IMREAD_GRAYSCALE)
            if roi_image is None:
                return {'success': False, 'error': f'无法读取ROI图像: {roi_image_path}'}
            
            # 使用特征提取器检测斑块
            spot_result = self.feature_extractor.detect_all_white_spots(roi_image)
            
            # 获取斑块二值掩码
            spot_binary = spot_result.get('all_white_binary_mask', None)
            
            if spot_binary is not None:
                # 创建红色斑块可视化图像，类似可视化图中的样式
                spot_visualization = self.create_spot_visualization(roi_image, spot_binary)
                
                # 保存红色斑块可视化图像
                success = cv2.imwrite(output_path, spot_visualization)
                if success:
                    return {
                        'success': True,
                        'spot_count': spot_result['all_spot_count'],
                        'spot_density': spot_result['all_spot_density'],
                        'output_path': output_path
                    }
                else:
                    return {'success': False, 'error': f'无法保存斑块图像到: {output_path}'}
            else:
                return {'success': False, 'error': '未能生成斑块二值图像'}
                
        except Exception as e:
            return {'success': False, 'error': f'处理ROI图像时出错: {str(e)}'}

    def create_spot_visualization(self, background_image: np.ndarray, 
                                spot_binary: np.ndarray) -> np.ndarray:
        """
        使用与test_single_image_spots.py完全一致的matplotlib方法生成斑块可视化
        完全按照feature_extractor.py第1326-1327行和test_single_image_spots.py的逻辑
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
            
            # 完全按照test_single_image_spots.py中生成single_image_spots_feature_method.png的逻辑
            # 这是feature_extractor.py第1326-1327行的准确重现
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
            print(f"matplotlib method failed, using OpenCV fallback: {e}")
            
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
            red_result[spot_pixels, 0] = 255  # 红色通道  
            
            # alpha 混合
            red_layer = red_result.astype(np.float32)
            bg_layer = cv2.cvtColor(background_result, cv2.COLOR_GRAY2RGB).astype(np.float32)
            
            # Alpha blending
            alpha_spot = 0.8
            alpha_bg = 0.7
            overlay = alpha_spot * red_layer + alpha_bg * (1 - alpha_spot) * bg_layer
            return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    def process_all_frames(self, images_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        处理所有帧图像生成斑块图
        
        Args:
            images_dir: 输入图像目录路径
            output_dir: 输出斑块图目录路径
            
        Returns:
            处理结果统计
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取所有图像文件（按顺序）
            image_pattern = os.path.join(images_dir, "frame_*.jpg")
            image_files = sorted(glob.glob(image_pattern), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not image_files:
                return {'success': False, 'error': f'在目录 {images_dir} 中未找到图像文件'}
            
            print(f"找到 {len(image_files)} 个图像文件")
            print(f"开始分批处理图像...")
            
            # 批量处理图像
            results = []
            success_count = 0
            
            for i, image_path in enumerate(tqdm(image_files, desc="处理斑块图")):
                # 生成输出文件名
                basename = os.path.basename(image_path)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(output_dir, f"{name}_bankuai.png")
                
                # 处理单帧（保持原有逻辑，但调用已废弃）
                result = {'success': False, 'error': '请使用新的ROI流程'}
                results.append(result)
                
                if result['success']:
                    success_count += 1
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(image_files)} 帧图像，成功 {success_count} 帧")
            
            print(f"\n斑块图像处理完成！")
            print(f"成功处理: {success_count}/{len(image_files)} 帧")
            
            return {
                'success': True,
                'total_frames': len(image_files),
                'success_frames': success_count,
                'failed_frames': len(image_files) - success_count,
                'results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': f'批量处理时出错: {str(e)}'}
    
    def extract_roi_from_all_frames(self, images_dir: str, roi_output_dir: str, 
                                     target_size: Tuple[int, int] = (128, 512)) -> Dict[str, Any]:
        """
        从所有帧图像提取ROI区域
        
        Args:
            images_dir: 输入图像目录路径
            roi_output_dir: 输出ROI图像目录路径
            target_size: ROI标准化后的目标尺寸
            
        Returns:
            ROI提取结果统计
        """
        try:
            # 创建输出目录
            os.makedirs(roi_output_dir, exist_ok=True)
            
            # 获取所有图像文件（按顺序）
            image_pattern = os.path.join(images_dir, "frame_*.jpg")
            image_files = sorted(glob.glob(image_pattern), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not image_files:
                return {'success': False, 'error': f'在目录 {images_dir} 中未找到图像文件'}
            
            print(f"找到 {len(image_files)} 个图像文件")
            print(f"开始ROI提取...")
            
            # 批量处理图像ROI提取
            results = []
            success_count = 0
            
            for i, image_path in enumerate(tqdm(image_files, desc="提取ROI区域")):
                # 生成输出文件名
                basename = os.path.basename(image_path)
                name, ext = os.path.splitext(basename)
                roi_output_path = os.path.join(roi_output_dir, f"{name}_roi.png")
                
                # 提取ROI
                result = self.extract_roi_from_frame(image_path, roi_output_path, target_size)
                results.append(result)
                
                if result['success']:
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
                'results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': f'ROI提取时出错: {str(e)}'}
    
    def process_all_roi_spots(self, roi_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        处理所有ROI图像生成斑块图
        
        Args:
            roi_dir: ROI图像目录路径
            output_dir: 输出斑块图目录路径
            
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
            print(f"开始计算斑块图...")
            
            # 批量处理ROI斑块检测
            results = []
            success_count = 0
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="计算斑块图")):
                # 生成输出文件名
                basename = os.path.basename(roi_path)
                name_parts = basename.split('_')
                if len(name_parts) >= 2:
                    frame_name = "_".join(name_parts[:2])  # 提取frame_XXXXXX部分
                    output_path = os.path.join(output_dir, f"{frame_name}_bankuai.png")
                else:
                    continue
                
                # 处理单个ROI斑块检测
                result = self.process_single_roi_spots(roi_path, output_path)
                results.append(result)
                
                if result['success']:
                    success_count += 1
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 个ROI图像，成功 {success_count} 个")
            
            print(f"\n斑块图像处理完成！")
            print(f"成功处理: {success_count}/{len(roi_files)} 帧")
            
            return {
                'success': True,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': len(roi_files) - success_count,
                'results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': f'批量处理ROI斑块时出错: {str(e)}'}
    
    def create_video_from_spots(self, spots_dir: str, output_video_path: str, 
                                fps: float = 2.39) -> Dict[str, Any]:
        """
        将斑块图像序列组装成视频
        
        Args:
            spots_dir: 斑块图像目录路径
            output_video_path: 输出视频路径
            fps: 视频帧率，默认为原始视频帧率
            
        Returns:
            视频生成结果
        """
        try:
            # 获取所有斑块图像文件（按顺序）
            spot_pattern = os.path.join(spots_dir, "*_bankuai.png")
            spot_files = sorted(glob.glob(spot_pattern), 
                               key=lambda x: int(os.path.basename(x).split('_')[1]))
            
            if not spot_files:
                return {'success': False, 'error': f'在目录 {spots_dir} 中未找到斑块图像文件'}
            
            print(f"找到 {len(spot_files)} 个斑块图像文件")
            print(f"开始创建视频: {output_video_path}")
            
            # 读取第一张图像获取尺寸
            first_frame = cv2.imread(spot_files[0], cv2.IMREAD_GRAYSCALE)
            if first_frame is None:
                return {'success': False, 'error': '无法读取第一张斑块图像'}
            
            height, width = first_frame.shape
            
            # 设置视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, 
                                         (width, height), isColor=False)
            
            if not video_writer.isOpened():
                return {'success': False, 'error': '无法创建视频写入器'}
            
            # 逐帧写入视频
            success_count = 0
            for i, spot_file in enumerate(tqdm(spot_files, desc="生成斑块视频")):
                frame = cv2.imread(spot_file, cv2.IMREAD_GRAYSCALE)
                if frame is not None:
                    video_writer.write(frame)
                    success_count += 1
                else:
                    print(f"警告：无法读取斑块图像: {spot_file}")
            
            # 释放资源
            video_writer.release()
            
            # 检查输出文件
            if os.path.exists(output_video_path):
                file_size = os.path.getsize(output_video_path)
                if file_size > 0:
                    print(f"斑块视频创建成功: {output_video_path}")
                    print(f"统计信息: 尺寸：{width}x{height}，帧数：{success_count}，大小：{file_size/1024/1024:.2f} MB")
                    
                    return {
                        'success': True,
                        'video_path': output_video_path,
                        'total_frames': len(spot_files),
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
    """主函数 - 新的三步骤流程：原图像 -> ROI提取 -> 斑块检测 -> 视频"""
    # 设置路径
    images_dir = os.path.join(DATA_DIR, 'images')
    roi_dir = os.path.join(DATA_DIR, 'roi_imgs')
    bankuai_dir = os.path.join(DATA_DIR, 'bankuai')
    output_video = os.path.join(DATA_DIR, '银行斑块视频.mp4')
    
    # 初始化斑块处理器
    processor = SpotProcessor()
    
    print("=== 斑块图像处理流程（新增ROI步骤） ===")
    
    # 步骤1: ROI提取，将所有帧处理为ROI并保存
    print("步骤1: 提取ROI区域...")
    result1 = processor.extract_roi_from_all_frames(images_dir, roi_dir, target_size=(128, 512))
    
    if not result1['success']:
        print(f"错误：{result1['error']}")
        return
    
    print(f"ROI提取结果: 成功 {result1['success_frames']}/{result1['total_frames']} 帧")
    
    # 步骤2: 对ROI图像计算斑块图
    print("\n步骤2: 对ROI图像计算斑块图...")
    result2 = processor.process_all_roi_spots(roi_dir, bankuai_dir)
    
    if not result2['success']:
        print(f"错误：{result2['error']}")
        return
    
    print(f"斑块图生成结果: 成功 {result2['success_frames']}/{result2['total_frames']} 帧")
    
    # 步骤3: 将斑块图组装成视频
    print("\n步骤3: 将斑块图组装成视频...")
    result3 = processor.create_video_from_spots(bankuai_dir, output_video, fps=2.39)
    
    if not result3['success']:
        print(f"错误：{result3['error']}")
        return
    
    print(f"斑块视频创建结果: {result3}")
    print(f"完成！斑块视频保存到: {output_video}")


if __name__ == "__main__":
    main()

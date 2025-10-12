#!/usr/bin/env python3
"""
ROI提取器
从原始视频抽帧的图片images目录中提取ROI区域，保存到roi_imgs目录
"""

import cv2
import numpy as np
import os
import sys
import glob
from typing import List, Dict, Any, Optional, Tuple
import time
from tqdm import tqdm

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from preprocessor import ImagePreprocessor
from config import PREPROCESS_CONFIG


class ROIExtractor:
    """ROI提取器"""
    
    def __init__(self, config: Dict[str, Any] = None, base_output_dir: str = None):
        """
        初始化ROI提取器
        
        Args:
            config: 预处理配置
            base_output_dir: 基础输出目录
        """
        self.config = config if config is not None else PREPROCESS_CONFIG
        self.preprocessor = ImagePreprocessor(self.config)
        
        # 设置目录路径
        if base_output_dir is None:
            self.base_output_dir = "data_video_20250821152112032"
        else:
            self.base_output_dir = base_output_dir
            
        self.images_dir = os.path.join(self.base_output_dir, 'images')
        self.roi_dir = os.path.join(self.base_output_dir, 'roi_imgs')
        
        # 创建ROI输出目录
        os.makedirs(self.roi_dir, exist_ok=True)
    
    def extract_roi_from_video(self, video_file: str, output_dir: str, 
                              target_size: Tuple[int, int] = None,
                              skip_preprocessing: bool = True) -> Dict[str, Any]:
        """
        从视频文件提取ROI区域
        
        Args:
            video_file: 输入视频文件路径
            output_dir: 输出ROI目录路径
            target_size: ROI标准化后的目标尺寸，None表示保持原始尺寸
            skip_preprocessing: 是否跳过预处理（对比度增强和去噪），True保持原始质量
            
        Returns:
            ROI提取结果统计
        """
        print(f"=== 从视频提取ROI区域 ===")
        print(f"输入视频: {video_file}")
        print(f"输出目录: {output_dir}")
        
        # 检查输入文件是否存在
        if not os.path.exists(video_file):
            return {'success': False, 'error': f'输入视频文件不存在: {video_file}'}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开视频
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return {'success': False, 'error': f'无法打开视频文件: {video_file}'}
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 时长={duration:.2f}秒")
        
        # 逐帧提取ROI
        success_count = 0
        failed_count = 0
        
        for frame_number in tqdm(range(total_frames), desc="提取ROI区域"):
            ret, frame = cap.read()
            if not ret:
                failed_count += 1
                continue
            
            # 生成输出文件名
            filename = f"frame_{frame_number:06d}_roi.png"
            output_file = os.path.join(output_dir, filename)
            
            # 检查文件是否已存在，如果存在则跳过
            if os.path.exists(output_file):
                success_count += 1
                continue
            
            # 提取ROI
            roi_image, processing_info = self.preprocessor.preprocess_pipeline(
                frame, target_size=target_size, skip_preprocessing=skip_preprocessing)
            
            # 保存ROI图像
            success = cv2.imwrite(output_file, roi_image)
            if success:
                success_count += 1
            else:
                failed_count += 1
                print(f"保存ROI图像失败: {output_file}")
            
        cap.release()
        
        print(f"\nROI提取完成！")
        print(f"总帧数: {total_frames}")
        print(f"成功提取: {success_count} 帧")
        print(f"失败: {failed_count} 帧")
        print(f"输出目录: {output_dir}")
        
        return {
            'success': True,
            'output_dir': output_dir,
            'total_frames': total_frames,
            'success_frames': success_count,
            'failed_frames': failed_count,
            'target_size': target_size
        }
    
    def extract_roi_from_all_frames(self, target_size: Tuple[int, int] = None,
                                   skip_preprocessing: bool = True) -> Dict[str, Any]:
        """
        从所有帧图像提取ROI区域
        
        Args:
            target_size: ROI标准化后的目标尺寸，None表示保持原始尺寸
            skip_preprocessing: 是否跳过预处理（对比度增强和去噪），True保持原始质量
            
        Returns:
            ROI提取结果统计
        """
        print(f"\n=== 提取ROI区域 ===")
        print(f"输入目录: {self.images_dir}")
        print(f"输出目录: {self.roi_dir}")
        
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
            failed_count = 0
            
            for i, image_path in enumerate(tqdm(image_files, desc="提取ROI区域")):
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
                    roi_image, processing_info = self.preprocessor.preprocess_pipeline(
                        image_path, target_size=target_size, skip_preprocessing=skip_preprocessing)
                    
                    # 保存ROI图像
                    success = cv2.imwrite(roi_output_path, roi_image)
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        print(f"保存ROI图像失败: {roi_output_path}")
                
                except Exception as e:
                    failed_count += 1
                    print(f"处理图像时出错 {image_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(image_files)} 帧图像，成功 {success_count} 帧，失败 {failed_count} 帧")
            
            print(f"\nROI提取完成！")
            print(f"总帧数: {len(image_files)}")
            print(f"成功提取: {success_count} 帧")
            print(f"失败: {failed_count} 帧")
            print(f"跳过已存在: {len(image_files) - success_count - failed_count} 帧")
            
            return {
                'success': True,
                'total_frames': len(image_files),
                'success_frames': success_count,
                'failed_frames': failed_count,
                'skipped_frames': len(image_files) - success_count - failed_count,
                'output_dir': self.roi_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'ROI提取时出错: {str(e)}'}
    
    def extract_roi_with_custom_config(self, custom_config: Dict[str, Any], 
                                     target_size: Tuple[int, int] = None,
                                     skip_preprocessing: bool = True) -> Dict[str, Any]:
        """
        使用自定义配置提取ROI区域
        
        Args:
            custom_config: 自定义预处理配置
            target_size: ROI标准化后的目标尺寸，None表示保持原始尺寸
            skip_preprocessing: 是否跳过预处理（对比度增强和去噪），True保持原始质量
            
        Returns:
            ROI提取结果统计
        """
        print(f"\n=== 提取ROI区域（自定义配置） ===")
        print(f"输入目录: {self.images_dir}")
        print(f"输出目录: {self.roi_dir}")
        
        try:
            # 创建自定义预处理器
            custom_preprocessor = ImagePreprocessor(custom_config)
            
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
            failed_count = 0
            
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
                    roi_image, processing_info = custom_preprocessor.preprocess_pipeline(
                        image_path, target_size=target_size, skip_preprocessing=skip_preprocessing)
                    
                    # 保存ROI图像
                    success = cv2.imwrite(roi_output_path, roi_image)
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        print(f"保存ROI图像失败: {roi_output_path}")
                
                except Exception as e:
                    failed_count += 1
                    print(f"处理图像时出错 {image_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(image_files)} 帧图像，成功 {success_count} 帧，失败 {failed_count} 帧")
            
            print(f"\nROI提取完成（自定义配置）！")
            print(f"总帧数: {len(image_files)}")
            print(f"成功提取: {success_count} 帧")
            print(f"失败: {failed_count} 帧")
            print(f"跳过已存在: {len(image_files) - success_count - failed_count} 帧")
            
            return {
                'success': True,
                'total_frames': len(image_files),
                'success_frames': success_count,
                'failed_frames': failed_count,
                'skipped_frames': len(image_files) - success_count - failed_count,
                'output_dir': self.roi_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'ROI提取时出错: {str(e)}'}
    
    def continue_roi_extraction(self, target_size: Tuple[int, int] = None,
                               skip_preprocessing: bool = True) -> Dict[str, Any]:
        """
        继续完成ROI提取（跳过已存在的文件）
        
        Args:
            target_size: ROI标准化后的目标尺寸，None表示保持原始尺寸
            skip_preprocessing: 是否跳过预处理（对比度增强和去噪），True保持原始质量
            
        Returns:
            ROI提取结果统计
        """
        print(f"\n=== 继续完成ROI提取 ===")
        print(f"输入目录: {self.images_dir}")
        print(f"输出目录: {self.roi_dir}")
        
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
            failed_count = 0
            
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
                    roi_image, processing_info = self.preprocessor.preprocess_pipeline(
                        image_path, target_size=target_size, skip_preprocessing=skip_preprocessing)
                    
                    # 保存ROI图像
                    success = cv2.imwrite(roi_output_path, roi_image)
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        print(f"保存ROI图像失败: {roi_output_path}")
                
                except Exception as e:
                    failed_count += 1
                    print(f"处理图像时出错 {image_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(image_files)} 帧图像，成功 {success_count} 帧，新处理 {processed_count} 帧，失败 {failed_count} 帧")
            
            print(f"\nROI提取继续完成！")
            print(f"总帧数: {len(image_files)}")
            print(f"成功处理: {success_count} 帧")
            print(f"新处理: {processed_count} 帧")
            print(f"失败: {failed_count} 帧")
            print(f"跳过已存在: {len(image_files) - success_count - processed_count} 帧")
            
            return {
                'success': True,
                'total_frames': len(image_files),
                'success_frames': success_count,
                'failed_frames': failed_count,
                'newly_processed': processed_count,
                'skipped_frames': len(image_files) - success_count - processed_count,
                'output_dir': self.roi_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'继续ROI提取时出错: {str(e)}'}
    
    def get_extraction_status(self) -> Dict[str, Any]:
        """
        获取ROI提取状态
        
        Returns:
            提取状态信息
        """
        try:
            # 获取所有图像文件
            image_pattern = os.path.join(self.images_dir, "frame_*.jpg")
            image_files = sorted(glob.glob(image_pattern), 
                               key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            # 获取所有ROI文件
            roi_pattern = os.path.join(self.roi_dir, "*_roi.png")
            roi_files = glob.glob(roi_pattern)
            
            total_images = len(image_files)
            total_rois = len(roi_files)
            completion_rate = (total_rois / total_images * 100) if total_images > 0 else 0
            
            return {
                'success': True,
                'total_images': total_images,
                'total_rois': total_rois,
                'completion_rate': completion_rate,
                'images_dir': self.images_dir,
                'roi_dir': self.roi_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'获取状态时出错: {str(e)}'}
    
    def extract_roi_from_images(self, image_dir: str, output_dir: str, 
                               target_size: Tuple[int, int] = None,
                               skip_preprocessing: bool = True) -> Dict[str, Any]:
        """
        从图像目录提取ROI区域
        
        Args:
            image_dir: 输入图像目录路径
            output_dir: 输出ROI目录路径
            target_size: ROI标准化后的目标尺寸，None表示保持原始尺寸
            skip_preprocessing: 是否跳过预处理（对比度增强和去噪），True保持原始质量
            
        Returns:
            ROI提取结果统计
        """
        print(f"=== 从图像目录提取ROI区域 ===")
        print(f"输入图像目录: {image_dir}")
        print(f"输出目录: {output_dir}")
        
        try:
            # 检查输入目录是否存在
            if not os.path.exists(image_dir):
                return {'success': False, 'error': f'输入图像目录不存在: {image_dir}'}
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取所有图像文件（按顺序）
            image_pattern = os.path.join(image_dir, "frame_*.jpg")
            image_files = sorted(glob.glob(image_pattern), 
                               key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not image_files:
                return {'success': False, 'error': f'在目录 {image_dir} 中未找到图像文件'}
            
            print(f"找到 {len(image_files)} 个图像文件")
            print(f"开始ROI提取...")
            
            # 批量处理图像ROI提取
            success_count = 0
            failed_count = 0
            
            for i, image_path in enumerate(tqdm(image_files, desc="提取ROI区域")):
                try:
                    # 从文件名提取帧号
                    basename = os.path.basename(image_path)
                    frame_number = int(basename.split('_')[1].split('.')[0])
                    
                    # 生成输出文件名
                    filename = f"frame_{frame_number:06d}_roi.png"
                    output_file = os.path.join(output_dir, filename)
                    
                    # 检查文件是否已存在，如果存在则跳过
                    if os.path.exists(output_file):
                        success_count += 1
                        continue
                    
                    # 提取ROI
                    roi_image, processing_info = self.preprocessor.preprocess_pipeline(
                        image_path, target_size=target_size, skip_preprocessing=skip_preprocessing)
                    
                    # 保存ROI图像
                    success = cv2.imwrite(output_file, roi_image)
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        print(f"保存ROI图像失败: {output_file}")
                
                except Exception as e:
                    failed_count += 1
                    print(f"处理图像时出错 {image_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(image_files)} 帧图像，成功 {success_count} 帧，失败 {failed_count} 帧")
            
            print(f"\nROI提取完成！")
            print(f"总帧数: {len(image_files)}")
            print(f"成功提取: {success_count} 帧")
            print(f"失败: {failed_count} 帧")
            print(f"输出目录: {output_dir}")
            
            return {
                'success': True,
                'output_dir': output_dir,
                'total_frames': len(image_files),
                'success_frames': success_count,
                'failed_frames': failed_count,
                'target_size': target_size
            }
            
        except Exception as e:
            return {'success': False, 'error': f'ROI提取时出错: {str(e)}'}


def main():
    """主函数"""
    # 默认值
    default_image_dir = "data_video_20250821152112032/images"
    default_output_dir = "data_video_20250821152112032/roi_imgs"
    
    # 使用sys.argv获取参数
    if len(sys.argv) >= 3:
        image_dir = sys.argv[1]
        output_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        image_dir = sys.argv[1]
        output_dir = default_output_dir
        print(f"使用默认输出目录: {output_dir}")
    else:
        image_dir = default_image_dir
        output_dir = default_output_dir
        print(f"使用默认输入图像目录: {image_dir}")
        print(f"使用默认输出目录: {output_dir}")

    # 判断image_dir是否存在，
    if not os.path.exists(image_dir):
        raise ValueError(f"错误：图像目录不存在 - {image_dir}")
    
    # 判断output_dir是否存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 初始化ROI提取器
    extractor = ROIExtractor()
    
    # 执行ROI提取
    result = extractor.extract_roi_from_images(
        image_dir=image_dir, 
        output_dir=output_dir
    )
    
    # 输出结果
    if result['success']:
        print(f"\n✅ ROI提取完成！")
        print(f"输出目录: {result['output_dir']}")
        print(f"总帧数: {result['total_frames']}")
        print(f"成功提取: {result['success_frames']} 帧")
        print(f"失败: {result['failed_frames']} 帧")
        print(f"目标尺寸: {result['target_size']}")
    else:
        print(f"❌ ROI提取失败: {result['error']}")


if __name__ == "__main__":
    main()

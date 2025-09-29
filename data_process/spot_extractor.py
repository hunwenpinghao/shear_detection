#!/usr/bin/env python3
"""
斑块提取器
从ROI图像中提取斑块特征，保存到spot_imgs目录
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

from spot_processor import SpotProcessor
from config import PREPROCESS_CONFIG


class SpotExtractor:
    """斑块提取器"""
    
    def __init__(self, config: Dict[str, Any] = None, base_output_dir: str = None):
        """
        初始化斑块提取器
        
        Args:
            config: 预处理配置
            base_output_dir: 基础输出目录
        """
        self.config = config if config is not None else PREPROCESS_CONFIG
        self.spot_processor = SpotProcessor(self.config)
        
        # 设置目录路径
        if base_output_dir is None:
            self.base_output_dir = "data_video_20250821152112032"
        else:
            self.base_output_dir = base_output_dir
            
        self.roi_dir = os.path.join(self.base_output_dir, 'roi_imgs')
        self.spot_dir = os.path.join(self.base_output_dir, 'spot_imgs')
        
        # 创建斑块输出目录
        os.makedirs(self.spot_dir, exist_ok=True)
    
    def extract_spots_from_roi_images(self, roi_dir: str, spot_dir: str) -> Dict[str, Any]:
        """
        从ROI图像目录提取斑块特征
        
        Args:
            roi_dir: 输入ROI图像目录路径
            spot_dir: 输出斑块图像目录路径
            
        Returns:
            斑块提取结果统计
        """
        print(f"=== 从ROI图像提取斑块特征 ===")
        print(f"输入ROI目录: {roi_dir}")
        print(f"输出斑块目录: {spot_dir}")
        
        try:
            # 检查输入目录是否存在
            if not os.path.exists(roi_dir):
                return {'success': False, 'error': f'输入ROI目录不存在: {roi_dir}'}
            
            # 创建输出目录
            os.makedirs(spot_dir, exist_ok=True)
            
            # 获取所有ROI图像文件（按顺序）
            roi_pattern = os.path.join(roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not roi_files:
                return {'success': False, 'error': f'在目录 {roi_dir} 中未找到ROI图像文件'}
            
            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print(f"开始斑块特征提取...")
            
            # 批量处理ROI斑块检测
            success_count = 0
            failed_count = 0
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="提取斑块特征")):
                try:
                    # 从文件名提取帧号
                    basename = os.path.basename(roi_path)
                    frame_number = int(basename.split('_')[1].split('.')[0])
                    
                    # 生成输出文件名
                    filename = f"frame_{frame_number:06d}_spot.png"
                    output_file = os.path.join(spot_dir, filename)
                    
                    # 检查文件是否已存在，如果存在则跳过
                    if os.path.exists(output_file):
                        success_count += 1
                        continue
                    
                    # 处理单个ROI斑块检测
                    result = self.spot_processor.process_single_roi_spots(roi_path, output_file)
                    
                    if result['success']:
                        success_count += 1
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
            
            print(f"\n斑块特征提取完成！")
            print(f"总帧数: {len(roi_files)}")
            print(f"成功提取: {success_count} 帧")
            print(f"失败: {failed_count} 帧")
            print(f"输出目录: {spot_dir}")
            
            return {
                'success': True,
                'output_dir': spot_dir,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': failed_count
            }
            
        except Exception as e:
            return {'success': False, 'error': f'斑块特征提取时出错: {str(e)}'}
    
    def extract_spots_from_all_rois(self) -> Dict[str, Any]:
        """
        从所有ROI图像提取斑块特征
        
        Returns:
            斑块提取结果统计
        """
        print(f"\n=== 提取斑块特征 ===")
        print(f"输入ROI目录: {self.roi_dir}")
        print(f"输出斑块目录: {self.spot_dir}")
        
        try:
            # 获取所有ROI图像文件（按顺序）
            roi_pattern = os.path.join(self.roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not roi_files:
                return {'success': False, 'error': f'在目录 {self.roi_dir} 中未找到ROI图像文件'}
            
            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print(f"开始斑块特征提取...")
            
            # 批量处理ROI斑块检测
            success_count = 0
            failed_count = 0
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="提取斑块特征")):
                try:
                    # 生成输出文件名
                    basename = os.path.basename(roi_path)
                    name, ext = os.path.splitext(basename)
                    spot_output_path = os.path.join(self.spot_dir, f"{name}_spot.png")
                    
                    # 检查斑块文件是否已存在，如果存在则跳过
                    if os.path.exists(spot_output_path):
                        success_count += 1
                        continue
                    
                    # 处理单个ROI斑块检测
                    result = self.spot_processor.process_single_roi_spots(roi_path, spot_output_path)
                    
                    if result['success']:
                        success_count += 1
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
            
            print(f"\n斑块特征提取完成！")
            print(f"总帧数: {len(roi_files)}")
            print(f"成功提取: {success_count} 帧")
            print(f"失败: {failed_count} 帧")
            print(f"跳过已存在: {len(roi_files) - success_count - failed_count} 帧")
            
            return {
                'success': True,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': failed_count,
                'skipped_frames': len(roi_files) - success_count - failed_count,
                'output_dir': self.spot_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'斑块特征提取时出错: {str(e)}'}
    
    def continue_spot_extraction(self) -> Dict[str, Any]:
        """
        继续完成斑块特征提取（跳过已存在的文件）
        
        Returns:
            斑块提取结果统计
        """
        print(f"\n=== 继续完成斑块特征提取 ===")
        print(f"输入ROI目录: {self.roi_dir}")
        print(f"输出斑块目录: {self.spot_dir}")
        
        try:
            # 获取所有ROI图像文件（按顺序）
            roi_pattern = os.path.join(self.roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            if not roi_files:
                return {'success': False, 'error': f'在目录 {self.roi_dir} 中未找到ROI图像文件'}
            
            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print(f"开始继续斑块特征提取...")
            
            # 批量处理ROI斑块检测
            success_count = 0
            processed_count = 0
            failed_count = 0
            
            for i, roi_path in enumerate(tqdm(roi_files, desc="继续斑块特征提取")):
                try:
                    # 生成输出文件名
                    basename = os.path.basename(roi_path)
                    name, ext = os.path.splitext(basename)
                    spot_output_path = os.path.join(self.spot_dir, f"{name}_spot.png")
                    
                    # 检查斑块文件是否已存在，如果存在则跳过
                    if os.path.exists(spot_output_path):
                        success_count += 1
                        continue
                    
                    processed_count += 1
                    
                    # 处理单个ROI斑块检测
                    result = self.spot_processor.process_single_roi_spots(roi_path, spot_output_path)
                    
                    if result['success']:
                        success_count += 1
                    else:
                        failed_count += 1
                        print(f"处理ROI图像失败 {roi_path}: {result.get('error', '未知错误')}")
                
                except Exception as e:
                    failed_count += 1
                    print(f"处理ROI图像时出错 {roi_path}: {e}")
                    continue
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 帧图像，成功 {success_count} 帧，新处理 {processed_count} 帧，失败 {failed_count} 帧")
            
            print(f"\n斑块特征提取继续完成！")
            print(f"总帧数: {len(roi_files)}")
            print(f"成功处理: {success_count} 帧")
            print(f"新处理: {processed_count} 帧")
            print(f"失败: {failed_count} 帧")
            print(f"跳过已存在: {len(roi_files) - success_count - processed_count} 帧")
            
            return {
                'success': True,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': failed_count,
                'newly_processed': processed_count,
                'skipped_frames': len(roi_files) - success_count - processed_count,
                'output_dir': self.spot_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'继续斑块特征提取时出错: {str(e)}'}
    
    def get_extraction_status(self) -> Dict[str, Any]:
        """
        获取斑块特征提取状态
        
        Returns:
            提取状态信息
        """
        try:
            # 获取所有ROI文件
            roi_pattern = os.path.join(self.roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern), 
                             key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            # 获取所有斑块文件
            spot_pattern = os.path.join(self.spot_dir, "*_spot.png")
            spot_files = glob.glob(spot_pattern)
            
            total_rois = len(roi_files)
            total_spots = len(spot_files)
            completion_rate = (total_spots / total_rois * 100) if total_rois > 0 else 0
            
            return {
                'success': True,
                'total_rois': total_rois,
                'total_spots': total_spots,
                'completion_rate': completion_rate,
                'roi_dir': self.roi_dir,
                'spot_dir': self.spot_dir
            }
            
        except Exception as e:
            return {'success': False, 'error': f'获取状态时出错: {str(e)}'}


def main():
    """主函数"""
    # 默认值
    default_roi_dir = "data_video_20250821152112032/roi_imgs"
    default_spot_dir = "data_video_20250821152112032/spot_imgs"
    
    # 使用sys.argv获取参数
    if len(sys.argv) >= 3:
        roi_dir = sys.argv[1]
        spot_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        roi_dir = sys.argv[1]
        spot_dir = default_spot_dir
        print(f"使用默认输出目录: {spot_dir}")
    else:
        roi_dir = default_roi_dir
        spot_dir = default_spot_dir
        print(f"使用默认输入ROI目录: {roi_dir}")
        print(f"使用默认输出目录: {spot_dir}")

    # 判断roi_dir是否存在
    if not os.path.exists(roi_dir):
        raise ValueError(f"错误：ROI目录不存在 - {roi_dir}")
    
    # 判断spot_dir是否存在
    if not os.path.exists(spot_dir):
        os.makedirs(spot_dir, exist_ok=True)
    
    # 初始化斑块提取器
    extractor = SpotExtractor()
    
    # 执行斑块特征提取
    result = extractor.extract_spots_from_roi_images(
        roi_dir=roi_dir, 
        spot_dir=spot_dir
    )
    
    # 输出结果
    if result['success']:
        print(f"\n✅ 斑块特征提取完成！")
        print(f"输出目录: {result['output_dir']}")
        print(f"总帧数: {result['total_frames']}")
        print(f"成功提取: {result['success_frames']} 帧")
        print(f"失败: {result['failed_frames']} 帧")
    else:
        print(f"❌ 斑块特征提取失败: {result['error']}")


if __name__ == "__main__":
    main()

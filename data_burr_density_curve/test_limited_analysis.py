#!/usr/bin/env python3
"""
测试有限数量的毛刺密度分析
"""

import cv2
import numpy as np
import os
import sys
import glob
from tqdm import tqdm

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_process'))

from feature_extractor import FeatureExtractor
from config import PREPROCESS_CONFIG

# 添加撕裂面检测器路径
sys.path.append('/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_split')
from shear_tear_detector import ShearTearDetector

def test_limited_analysis():
    """测试有限数量的毛刺密度分析"""
    
    # 初始化
    feature_extractor = FeatureExtractor(PREPROCESS_CONFIG)
    detector = ShearTearDetector()
    
    # 设置路径
    roi_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data/roi_imgs"
    output_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_burr_density_curve"
    
    # 创建输出目录
    step2_dir = os.path.join(output_dir, 'step2_tear_regions')
    os.makedirs(step2_dir, exist_ok=True)
    
    # 获取前10个ROI图像
    roi_pattern = os.path.join(roi_dir, "*_roi.png")
    image_files = sorted(glob.glob(roi_pattern))[:10]
    
    print(f"测试 {len(image_files)} 个图像")
    
    results = []
    
    for image_path in tqdm(image_files, desc="处理图像"):
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
            
        # 检测撕裂面
        result = detector.detect_surfaces(image, visualize=False)
        if result and 'segmented_image' in result:
            frame_num = int(os.path.basename(image_path).split('_')[1])
            
            # 从分割结果中提取撕裂面mask
            segmented_image = result['segmented_image']
            tear_mask = (segmented_image == 128).astype(np.uint8) * 255
            
            # 应用mask过滤出撕裂面区域
            tear_region = cv2.bitwise_and(image, image, mask=tear_mask)
            
            # 保存撕裂面区域
            region_filename = f"tear_region_frame_{frame_num:06d}.png"
            region_path = os.path.join(step2_dir, region_filename)
            cv2.imwrite(region_path, tear_region)
            
            # 使用FeatureExtractor检测毛刺
            burr_result = feature_extractor.detect_burs(tear_region, mask=None)
            
            # 保存毛刺检测结果图
            if 'burs_binary_mask' in burr_result and burr_result['burs_binary_mask'] is not None:
                # 确保毛刺mask是正确的格式
                burr_mask = burr_result['burs_binary_mask']
                if burr_mask.dtype != np.uint8:
                    burr_mask = (burr_mask > 0).astype(np.uint8) * 255
                else:
                    burr_mask = burr_mask * 255
                
                burr_filename = f"tear_burrs_frame_{frame_num:06d}.png"
                burr_path = os.path.join(step2_dir, burr_filename)
                cv2.imwrite(burr_path, burr_mask)
                
                print(f"Frame {frame_num}: 检测到 {burr_result.get('burs_count', 0)} 个毛刺")
            else:
                print(f"Frame {frame_num}: 未检测到毛刺")
                
                # 保存空的黑图
                empty_mask = np.zeros_like(tear_region, dtype=np.uint8)
                burr_filename = f"tear_burrs_frame_{frame_num:06d}.png"
                burr_path = os.path.join(step2_dir, burr_filename)
                cv2.imwrite(burr_path, empty_mask)
    
    print(f"测试完成，结果已保存到: {step2_dir}")

if __name__ == "__main__":
    test_limited_analysis()

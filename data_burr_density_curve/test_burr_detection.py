#!/usr/bin/env python3
"""
测试毛刺检测功能
"""

import cv2
import numpy as np
import os
import sys
import glob

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_process'))

from feature_extractor import FeatureExtractor
from config import PREPROCESS_CONFIG

# 添加撕裂面检测器路径
sys.path.append('/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_split')
from shear_tear_detector import ShearTearDetector

def test_burr_detection():
    """测试毛刺检测功能"""
    
    # 初始化
    feature_extractor = FeatureExtractor(PREPROCESS_CONFIG)
    detector = ShearTearDetector()
    
    # 获取前5个ROI图像
    roi_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data/roi_imgs"
    roi_pattern = os.path.join(roi_dir, "*_roi.png")
    image_files = sorted(glob.glob(roi_pattern))[:5]
    
    print(f"测试 {len(image_files)} 个图像")
    
    for i, image_path in enumerate(image_files):
        print(f"\n=== 测试图像 {i+1}: {os.path.basename(image_path)} ===")
        
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("无法读取图像")
            continue
            
        print(f"原始图像形状: {image.shape}")
        print(f"原始图像非零像素: {np.sum(image > 0)}")
        
        # 检测撕裂面
        result = detector.detect_surfaces(image, visualize=False)
        if result and 'segmented_image' in result:
            segmented_image = result['segmented_image']
            tear_mask = (segmented_image == 128).astype(np.uint8) * 255
            
            print(f"撕裂面mask形状: {tear_mask.shape}")
            print(f"撕裂面mask非零像素: {np.sum(tear_mask > 0)}")
            
            # 应用mask过滤出撕裂面区域
            tear_region = cv2.bitwise_and(image, image, mask=tear_mask)
            
            print(f"撕裂面区域形状: {tear_region.shape}")
            print(f"撕裂面区域非零像素: {np.sum(tear_region > 0)}")
            
            # 检测毛刺
            burr_result = feature_extractor.detect_burs(tear_region, mask=None)
            
            print(f"毛刺检测结果键: {burr_result.keys()}")
            if 'burs_count' in burr_result:
                print(f"检测到的毛刺数量: {burr_result['burs_count']}")
            if 'burs_density' in burr_result:
                print(f"毛刺密度: {burr_result['burs_density']}")
            if 'burs_binary_mask' in burr_result and burr_result['burs_binary_mask'] is not None:
                burr_mask = burr_result['burs_binary_mask']
                print(f"毛刺mask形状: {burr_mask.shape}")
                print(f"毛刺mask非零像素: {np.sum(burr_mask > 0)}")
                
                # 保存调试图像
                debug_dir = "debug_images"
                os.makedirs(debug_dir, exist_ok=True)
                
                # 保存原始图像
                cv2.imwrite(f"{debug_dir}/original_{i:02d}.png", image)
                
                # 保存撕裂面mask
                cv2.imwrite(f"{debug_dir}/tear_mask_{i:02d}.png", tear_mask)
                
                # 保存撕裂面区域
                cv2.imwrite(f"{debug_dir}/tear_region_{i:02d}.png", tear_region)
                
                # 保存毛刺mask
                if np.sum(burr_mask > 0) > 0:
                    burr_mask_vis = (burr_mask * 255).astype(np.uint8)
                    cv2.imwrite(f"{debug_dir}/burr_mask_{i:02d}.png", burr_mask_vis)
                    print(f"毛刺mask已保存到: {debug_dir}/burr_mask_{i:02d}.png")
                else:
                    print("没有检测到毛刺")
            else:
                print("毛刺检测失败或返回空结果")
        else:
            print("撕裂面检测失败")

if __name__ == "__main__":
    test_burr_detection()

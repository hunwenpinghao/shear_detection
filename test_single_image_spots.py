#!/usr/bin/env python3
"""
使用SpotProcessor的逻辑处理单张图像生成斑块图
"""

import os
import sys
import cv2
import numpy as np

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

from spot_processor import SpotProcessor
from preprocessor import ImagePreprocessor
from feature_extractor import FeatureExtractor

def process_single_image():
    """处理单张图像生成斑块图"""
    
    # 文件路径
    input_image_path = "data/Image_20250710125452500.bmp"
    roi_output_path = "output/single_image_roi.png"
    spot_output_path = "output/single_image_spots.png"
    
    print("=== 单张图像斑块处理流程 ===")
    
    # 初始化处理器
    processor = SpotProcessor()
    preprocessor = ImagePreprocessor()
    
    # 步骤1: ROI提取
    print(f"步骤1: 从 {input_image_path} 提取ROI区域...")
    try:
        roi_image, processing_info = preprocessor.preprocess_pipeline(
            input_image_path, target_size=(128, 512))
        print(f"✓ ROI提取成功: {roi_image.shape}")
        print(f"  - 原始尺寸: {processing_info['original_shape']}")
        print(f"  - ROI信息: {processing_info['roi_info']}")
        
        # 保存ROI图像  
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(roi_output_path, roi_image)
        print(f"✓ ROI图像保存到: {roi_output_path}")
        
    except Exception as e:
        print(f"✗ ROI提取失败: {e}")
        return None
    
    # 步骤2: 斑块检测
    print(f"步骤2: 检测斑块并生成红色可视化图...")
    try:
        # 使用FeatureExtractor检测斑块
        feature_extractor = FeatureExtractor()
        spot_result = feature_extractor.detect_all_white_spots(roi_image)
        
        print(f"✓ 斑块检测成功:")
        print(f"  - 斑块数量: {spot_result['all_spot_count']}")
        print(f"  - 斑块密度: {spot_result['all_spot_density']:.3f}")
        print(f"  - 平均斑块大小: {spot_result['all_average_spot_size']:.1f}")
        print(f"  - 总斑块面积: {spot_result['all_total_spot_area']}")
        print(f"  - 分布均匀性: {spot_result['all_distribution_uniformity']:.3f}")
        
        # 获取斑块二值掩码
        all_white_spots = spot_result.get('all_white_binary_mask', None)
        if all_white_spots is not None:
            
            # 方法1: 使用OpenCV方法（当前spot_processor）
            spot_visualization = processor.create_spot_visualization(roi_image, all_white_spots)
            cv2.imwrite(spot_output_path, spot_visualization)
            print(f"✓ OpenCV版本斑块图保存到: {spot_output_path}")
            
            # 方法2: 使用feature_extractor.py的确切matplotlib逻辑生成对比图
            import matplotlib.pyplot as plt
            
            # 完全按照feature_extractor.py第1326-1327行
            fig, ax = plt.subplots(1, 1, figsize=(6, 12))
            ax.imshow(roi_image, cmap='gray', alpha=0.7)
            ax.imshow(all_white_spots, cmap='Reds', alpha=0.8)
            ax.set_title(f'feature_extractor方法斑块图\n数量: {spot_result["all_spot_count"]}')
            ax.axis('off')
            
            # 保存matplotlib版本
            feature_comparison_path = "output/single_image_spots_feature_method.png"
            plt.savefig(feature_comparison_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"✓ feature_extractor方法斑块图保存到: {feature_comparison_path}")
            
        else:
            print("✗ 未生成斑块二值掩码")
            return None
            
    except Exception as e:
        print(f"✗ 斑块检测失败: {e}")
        return None
    
    print("\n=== 处理完成 ===")
    print(f"输入的原始图像: {input_image_path}")
    print(f"ROI提取结果: {roi_output_path}")
    print(f"斑块可视化图: {spot_output_path}")
    
    return {
        'roi_path': roi_output_path,
        'spot_path': spot_output_path,
        'spot_count': spot_result['all_spot_count'],
        'spot_density': spot_result['all_spot_density']
    }

if __name__ == "__main__":
    result = process_single_image()
    if result:
        print(f"\n最终结果: ")
        print(f"- 检测到 {result['spot_count']} 个白色斑块")
        print(f"- 斑块密度: {result['spot_density']:.3f}")
    else:
        print("处理失败")

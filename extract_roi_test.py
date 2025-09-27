#!/usr/bin/env python3
"""
提取单张图像的ROI区域
"""

import os
import sys
import cv2
import numpy as np

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

from preprocessor import ImagePreprocessor

def extract_roi_from_image():
    """从单张图像提取ROI区域"""
    
    input_image_path = "data/Image_20250710125452500.bmp"
    roi_output_path = "output/extracted_roi_image.png"
    
    print("=== 提取ROI图像 ===")
    print(f"输入图像: {input_image_path}")
    
    # 检查输入图像是否存在
    if not os.path.exists(input_image_path):
        print(f"✗ 输入图像不存在: {input_image_path}")
        return False
    
    # 初始化预处理器
    preprocessor = ImagePreprocessor()
    
    try:
        print("正在提取ROI区域...")
        
        # 提取ROI区域，使用标准尺寸(128, 512)
        roi_image, processing_info = preprocessor.preprocess_pipeline(
            input_image_path, target_size=(128, 512))
        
        print("✅ ROI提取成功!")
        print(f"  - ROI尺寸: {roi_image.shape}")
        print(f"  - 原始尺寸: {processing_info['original_shape']}")
        print(f"  - ROI信息: {processing_info['roi_info']}")
        
        # 创建输出目录
        os.makedirs("output", exist_ok=True)
        
        # 保存ROI图像
        success = cv2.imwrite(roi_output_path, roi_image)
        
        if success:
            print(f"✅ ROI图像保存成功: {roi_output_path}")
            
            # 检查文件大小
            if os.path.exists(roi_output_path):
                file_size = os.path.getsize(roi_output_path)
                print(f"  - 文件大小: {file_size} bytes")
                return True
            else:
                print("✗ ROI图像保存失败")
                return False
        else:
            print("✗ ROI图像保存失败")
            return False
            
    except Exception as e:
        print(f"✗ ROI提取过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = extract_roi_from_image()
    if success:
        print("\n🎉 ROI提取完成!")
    else:
        print("\n❌ ROI提取失败!")

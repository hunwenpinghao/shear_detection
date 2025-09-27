#!/usr/bin/env python3
"""
测试spot_processor.py修改后的斑块图生成逻辑
使用data/roi_imgs中的一个ROI图像进行测试
"""

import os
import sys
import cv2
import numpy as np

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

from spot_processor import SpotProcessor

def test_roi_spot_generation():
    """测试ROI图像的斑块图生成逻辑"""
    
    # 选择一个ROI图像进行测试
    # roi_image_path = "data/roi_imgs/frame_000000_roi.png"
    roi_image_path = "data/extracted_roi_image.png"
    # test_output_path = "output/test_roi_spot_generation.png"
    test_output_path = "data/extracted_roi_image_bankuai.png"
    
    print("=== 测试spot_processor.py修改后的斑块图生成逻辑 ===")
    print(f"使用ROI图像: {roi_image_path}")
    
    # 检查ROI图像是否存在
    if not os.path.exists(roi_image_path):
        print(f"✗ ROI图像不存在: {roi_image_path}")
        return False
    
    # 初始化斑块处理器
    processor = SpotProcessor()
    
    # 测试处理单张ROI图像生成斑块图
    try:
        print(f"正在处理: {roi_image_path}")
        
        # 创建输出目录
        os.makedirs("output", exist_ok=True)
        
        # 调用spot_processor的斑块检测和可视化方法
        result = processor.process_single_roi_spots(roi_image_path, test_output_path)
        
        if result['success']:
            print("✅ 斑块图生成成功!")
            print(f"  - 斑块数量: {result['spot_count']}")
            print(f"  - 斑块密度: {result['spot_density']}")
            print(f"  - 输出路径: {test_output_path}")
            
            # 检查输出文件是否存在
            if os.path.exists(test_output_path):
                file_size = os.path.getsize(test_output_path)
                print(f"  - 文件大小: {file_size} bytes")
                return True
            else:
                print("✗ 输出文件未生成")
                return False
        else:
            print(f"✗ 斑块图生成失败: {result['error']}")
            return False
            
    except Exception as e:
        print(f"✗ 处理过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = test_roi_spot_generation()
    if success:
        print("\n🎉 测试成功! spot_processor.py的斑块图生成逻辑工作正常")
    else:
        print("\n❌ 测试失败! 需要检查修改后的代码")

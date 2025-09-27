#!/usr/bin/env python3
"""
测试毛刺处理器的毛刺检测功能
"""

import os
import sys
import cv2

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

from burr_processor import BurrProcessor

def test_burr_detection():
    """测试毛刺检测功能"""
    
    # 选择一个ROI图像进行测试
    roi_image_path = "data/roi_imgs/frame_000000_roi.png"
    test_output_path = "output/test_burr_detection.png"
    
    print("=== 测试毛刺检测功能 ===")
    print(f"使用ROI图像: {roi_image_path}")
    
    # 检查ROI图像是否存在
    if not os.path.exists(roi_image_path):
        print(f"✗ ROI图像不存在: {roi_image_path}")
        return False
    
    # 初始化毛刺处理器
    processor = BurrProcessor()
    
    try:
        print(f"正在处理: {roi_image_path}")
        
        # 创建输出目录
        os.makedirs("output", exist_ok=True)
        
        # 调用毛刺检测方法
        result = processor.process_single_roi_burrs(roi_image_path, test_output_path)
        
        if result['success']:
            print("✅ 毛刺图生成成功!")
            print(f"  - 毛刺数量: {result['burs_count']}")
            print(f"  - 毛刺密度: {result['burs_density']:.6f}")
            print(f"  - 毛刺总面积: {result['burs_total_area']}")
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
            print(f"✗ 毛刺图生成失败: {result['error']}")
            return False
            
    except Exception as e:
        print(f"✗ 处理过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = test_burr_detection()
    if success:
        print("\n🎉 毛刺检测测试成功!")
    else:
        print("\n❌ 毛刺检测测试失败!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
为已存在的剪切面区域生成缺失的斑块检测结果图
"""

import os
import sys
import glob
import cv2
from tqdm import tqdm

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_process'))

from shear_density_analyzer import ShearDensityAnalyzer
from feature_extractor import FeatureExtractor
from config import PREPROCESS_CONFIG

def generate_missing_patches():
    """为已存在的剪切面区域生成缺失的斑块检测结果图"""
    
    print("为已存在的剪切面区域生成缺失的斑块检测结果图...")
    print("=" * 60)
    
    # 设置路径
    step2_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_density_curve/step2_shear_regions"
    
    if not os.path.exists(step2_dir):
        print(f"❌ step2_shear_regions 目录不存在: {step2_dir}")
        return
    
    # 创建分析器和特征提取器
    analyzer = ShearDensityAnalyzer()
    feature_extractor = FeatureExtractor(PREPROCESS_CONFIG)
    
    # 获取所有剪切面区域文件
    region_pattern = os.path.join(step2_dir, "shear_region_frame_*.png")
    region_files = sorted(glob.glob(region_pattern), 
                         key=lambda x: int(os.path.basename(x).split('_')[3].split('.')[0]))
    
    if not region_files:
        print(f"❌ 在 {step2_dir} 中未找到剪切面区域文件")
        return
    
    print(f"找到 {len(region_files)} 个剪切面区域文件")
    
    # 检查哪些斑块检测结果图缺失
    missing_patches = []
    for region_file in region_files:
        frame_num = int(os.path.basename(region_file).split('_')[3].split('.')[0])
        patch_file = os.path.join(step2_dir, f"shear_patches_frame_{frame_num:06d}.png")
        
        if not os.path.exists(patch_file):
            missing_patches.append((region_file, patch_file, frame_num))
    
    if not missing_patches:
        print("✅ 所有斑块检测结果图已存在")
        return
    
    print(f"需要生成 {len(missing_patches)} 个缺失的斑块检测结果图")
    
    # 生成缺失的斑块检测结果图
    success_count = 0
    failed_count = 0
    
    for region_file, patch_file, frame_num in tqdm(missing_patches, desc="生成斑块检测结果图", unit="图像"):
        try:
            # 读取剪切面区域图像
            shear_region = cv2.imread(region_file, cv2.IMREAD_GRAYSCALE)
            if shear_region is None:
                print(f"❌ 无法读取剪切面区域图像: {region_file}")
                failed_count += 1
                continue
            
            # 检测斑块
            spot_result = feature_extractor.detect_all_white_spots(shear_region)
            
            # 创建斑块可视化
            if 'all_white_binary_mask' in spot_result:
                spot_binary = spot_result['all_white_binary_mask']
                spot_visualization = analyzer.create_spot_visualization(shear_region, spot_binary)
                
                # 保存斑块检测结果图
                success = cv2.imwrite(patch_file, spot_visualization)
                if success:
                    success_count += 1
                else:
                    print(f"❌ 无法保存斑块检测结果图: {patch_file}")
                    failed_count += 1
            else:
                print(f"❌ 斑块检测结果中缺少 'all_white_binary_mask' 键: {region_file}")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ 处理图像时出错 {region_file}: {e}")
            failed_count += 1
    
    print("=" * 60)
    print(f"✅ 斑块检测结果图生成完成！")
    print(f"成功生成: {success_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"总处理: {len(missing_patches)} 个")

if __name__ == "__main__":
    generate_missing_patches()

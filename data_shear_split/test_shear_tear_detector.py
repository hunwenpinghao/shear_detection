#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
撕裂面和剪切面检测器测试脚本
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from shear_tear_detector import ShearTearDetector

def test_detector_on_roi_images():
    """在ROI图像上测试检测器"""
    detector = ShearTearDetector()
    
    # ROI图像目录
    roi_dirs = [
        "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_Video_20250821140339629/roi_imgs",
        "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data/roi_imgs"
    ]
    
    results = []
    
    for roi_dir in roi_dirs:
        if not os.path.exists(roi_dir):
            print(f"目录不存在: {roi_dir}")
            continue
            
        print(f"\n处理目录: {roi_dir}")
        print("=" * 60)
        
        # 获取所有PNG文件
        png_files = [f for f in os.listdir(roi_dir) if f.endswith('.png')]
        png_files.sort()
        
        # 测试前10个图像
        test_files = png_files[:10]
        
        for filename in test_files:
            img_path = os.path.join(roi_dir, filename)
            
            try:
                # 读取图像
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"无法读取图像: {img_path}")
                    continue
                
                print(f"\n处理: {filename}")
                print("-" * 40)
                
                # 检测表面类型
                result = detector.detect_surfaces(image, visualize=False)
                
                # 保存结果
                results.append({
                    'filename': filename,
                    'path': img_path,
                    'surface_type': result['surface_type'],
                    'confidence': result['confidence'],
                    'features': result['features']
                })
                
                print(f"检测结果: {result['surface_type']}")
                print(f"置信度: {result['confidence']:.3f}")
                
                # 显示关键特征
                key_features = ['continuity_score', 'smoothness_score', 'roughness_score', 
                              'mean_brightness', 'wave_strength', 'anisotropy']
                print("关键特征:")
                for feature in key_features:
                    if feature in result['features']:
                        print(f"  {feature}: {result['features'][feature]:.3f}")
                        
            except Exception as e:
                print(f"处理图像 {filename} 时出错: {e}")
    
    return results

def analyze_results(results):
    """分析检测结果"""
    if not results:
        print("没有检测结果可分析")
        return
    
    print("\n" + "=" * 60)
    print("检测结果统计分析")
    print("=" * 60)
    
    # 统计表面类型分布
    surface_types = [r['surface_type'] for r in results]
    shear_count = surface_types.count('shear')
    tear_count = surface_types.count('tear')
    
    print(f"总检测图像数: {len(results)}")
    print(f"剪切面: {shear_count} ({shear_count/len(results)*100:.1f}%)")
    print(f"撕裂面: {tear_count} ({tear_count/len(results)*100:.1f}%)")
    
    # 置信度统计
    confidences = [r['confidence'] for r in results]
    print(f"\n置信度统计:")
    print(f"  平均置信度: {np.mean(confidences):.3f}")
    print(f"  置信度标准差: {np.std(confidences):.3f}")
    print(f"  最高置信度: {np.max(confidences):.3f}")
    print(f"  最低置信度: {np.min(confidences):.3f}")
    
    # 特征分析
    print(f"\n特征分析:")
    all_features = {}
    for result in results:
        for feature_name, feature_value in result['features'].items():
            if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                if feature_name not in all_features:
                    all_features[feature_name] = []
                all_features[feature_name].append(feature_value)
    
    # 计算每个特征的平均值和标准差
    feature_stats = {}
    for feature_name, values in all_features.items():
        if values:
            feature_stats[feature_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # 显示重要特征统计
    important_features = ['continuity_score', 'smoothness_score', 'roughness_score', 
                         'mean_brightness', 'wave_strength', 'anisotropy', 'texture_uniformity']
    
    print("重要特征统计:")
    for feature in important_features:
        if feature in feature_stats:
            stats = feature_stats[feature]
            print(f"  {feature}:")
            print(f"    平均值: {stats['mean']:.3f}")
            print(f"    标准差: {stats['std']:.3f}")
            print(f"    范围: [{stats['min']:.3f}, {stats['max']:.3f}]")

def create_feature_comparison_plot(results):
    """创建特征对比图"""
    if not results:
        return
    
    # 分离剪切面和撕裂面的结果
    shear_results = [r for r in results if r['surface_type'] == 'shear']
    tear_results = [r for r in results if r['surface_type'] == 'tear']
    
    if not shear_results or not tear_results:
        print("需要同时有剪切面和撕裂面的样本才能进行对比")
        return
    
    # 选择关键特征
    key_features = ['continuity_score', 'smoothness_score', 'roughness_score', 
                   'mean_brightness', 'wave_strength', 'anisotropy']
    
    # 提取特征值
    shear_features = {f: [] for f in key_features}
    tear_features = {f: [] for f in key_features}
    
    for result in shear_results:
        for feature in key_features:
            if feature in result['features']:
                shear_features[feature].append(result['features'][feature])
    
    for result in tear_results:
        for feature in key_features:
            if feature in result['features']:
                tear_features[feature].append(result['features'][feature])
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('剪切面 vs 撕裂面 特征对比', fontsize=16)
    
    axes = axes.flatten()
    
    for i, feature in enumerate(key_features):
        if i >= len(axes):
            break
            
        shear_values = shear_features[feature]
        tear_values = tear_features[feature]
        
        if shear_values and tear_values:
            # 箱线图
            data = [shear_values, tear_values]
            labels = ['剪切面', '撕裂面']
            
            bp = axes[i].boxplot(data, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            axes[i].set_title(f'{feature}')
            axes[i].grid(True, alpha=0.3)
            
            # 添加统计信息
            shear_mean = np.mean(shear_values)
            tear_mean = np.mean(tear_values)
            axes[i].text(0.5, 0.95, f'剪切面均值: {shear_mean:.3f}\n撕裂面均值: {tear_mean:.3f}', 
                        transform=axes[i].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()

def main():
    """主函数"""
    print("撕裂面和剪切面检测器测试")
    print("=" * 60)
    
    # 测试检测器
    results = test_detector_on_roi_images()
    
    # 分析结果
    analyze_results(results)
    
    # 创建特征对比图
    if results:
        create_feature_comparison_plot(results)
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()

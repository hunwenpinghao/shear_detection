#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
撕裂面和剪切面检测器测试脚本
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import sys
from datetime import datetime
from shear_tear_detector import ShearTearDetector

# 设置中文字体支持
import matplotlib
import matplotlib.font_manager as fm

# 检查系统可用的中文字体
def get_chinese_fonts():
    """获取系统可用的中文字体"""
    chinese_fonts = []
    for font in fm.fontManager.ttflist:
        font_name = font.name
        if any(keyword in font_name.lower() for keyword in ['simhei', 'microsoft', 'yahei', 'song', 'kai', 'fang', 'hei']):
            chinese_fonts.append(font_name)
    return list(set(chinese_fonts))

# 获取可用中文字体
available_fonts = get_chinese_fonts()
print(f"可用中文字体: {available_fonts}")

# 设置字体优先级
font_candidates = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS']
if available_fonts:
    font_candidates = available_fonts[:3] + font_candidates

matplotlib.rcParams['font.sans-serif'] = font_candidates
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

# 清除字体缓存
try:
    fm._rebuild()
except AttributeError:
    # 对于较新版本的matplotlib，使用不同的方法
    try:
        fm.fontManager.__init__()
    except:
        pass


def test_detector_on_roi_images(roi_imgs_dir="data/roi_imgs", output_dir="data_shear_split/split_results"):
    """在ROI图像上测试检测器"""
    detector = ShearTearDetector()
    
    # ROI图像目录
    roi_dirs = [
        #"/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_Video_20250821140339629/roi_imgs",
        roi_imgs_dir
    ]
    
    results = []
    
    for roi_dir in roi_dirs:
        if not os.path.exists(roi_dir):
            print(f"目录不存在: {roi_dir}")
            continue
            
        print(f"处理目录: {roi_dir}")
        
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
                    continue
                
                print(f"处理: {filename}")
                
                # 检测表面类型并生成可视化
                result = detector.detect_surfaces(image, visualize=True)
                
                # 保存可视化结果
                if 'visualization' in result:
                    vis_output_dir = os.path.join(output_dir, 'visualizations')
                    os.makedirs(vis_output_dir, exist_ok=True)
                    
                    # 保存可视化图像
                    vis_filename = f"{os.path.splitext(filename)[0]}_detection_result.png"
                    vis_path = os.path.join(vis_output_dir, vis_filename)
                    cv2.imwrite(vis_path, result['visualization'])
                
                # 保存结果
                results.append({
                    'filename': filename,
                    'path': img_path,
                    'surface_type': result['surface_type'],
                    'confidence': result['confidence'],
                    'features': result['features']
                })
                
                print(f"  -> {result['surface_type']} (置信度: {result['confidence']:.3f})")
                        
            except Exception as e:
                print(f"处理图像 {filename} 时出错: {e}")
    
    return results

def analyze_results(results):
    """分析检测结果"""
    if not results:
        print("没有检测结果可分析")
        return
    
    print("\n检测结果统计:")
    
    # 统计表面类型分布
    surface_types = [r['surface_type'] for r in results]
    shear_count = surface_types.count('shear')
    tear_count = surface_types.count('tear')
    
    print(f"  总图像数: {len(results)}")
    print(f"  剪切面: {shear_count} ({shear_count/len(results)*100:.1f}%)")
    print(f"  撕裂面: {tear_count} ({tear_count/len(results)*100:.1f}%)")
    
    # 置信度统计
    confidences = [r['confidence'] for r in results]
    print(f"  平均置信度: {np.mean(confidences):.3f}")
    print(f"  置信度范围: [{np.min(confidences):.3f}, {np.max(confidences):.3f}]")

def create_feature_comparison_plot(results, output_dir="output"):
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
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"feature_comparison_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    return plot_path

def save_results_to_files(results, output_dir="output"):
    """保存检测结果到文件"""
    if not results:
        print("没有结果可保存")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存为CSV格式（简化版结果）
    csv_data = []
    for result in results:
        csv_data.append({
            'filename': result['filename'],
            'path': result['path'],
            'surface_type': result['surface_type'],
            'confidence': result['confidence'],
            'continuity_score': result['features'].get('continuity_score', 0),
            'smoothness_score': result['features'].get('smoothness_score', 0),
            'roughness_score': result['features'].get('roughness_score', 0),
            'mean_brightness': result['features'].get('mean_brightness', 0),
            'wave_strength': result['features'].get('wave_strength', 0),
            'anisotropy': result['features'].get('anisotropy', 0)
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, f"detection_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 2. 保存为JSON格式（完整结果）
    json_path = os.path.join(output_dir, f"detection_results_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    # 3. 保存统计摘要
    summary = {
        'timestamp': timestamp,
        'total_images': len(results),
        'surface_type_distribution': {
            'shear': len([r for r in results if r['surface_type'] == 'shear']),
            'tear': len([r for r in results if r['surface_type'] == 'tear'])
        },
        'confidence_stats': {
            'mean': float(np.mean([r['confidence'] for r in results])),
            'std': float(np.std([r['confidence'] for r in results])),
            'min': float(np.min([r['confidence'] for r in results])),
            'max': float(np.max([r['confidence'] for r in results]))
        }
    }
    
    summary_path = os.path.join(output_dir, f"detection_summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return csv_path, json_path, summary_path


def main():
    """主函数"""
    roi_imgs = 'data/roi_imgs'
    output_dir = 'data_shear_split/split_results'
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        roi_imgs = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # 测试检测器
    results = test_detector_on_roi_images(roi_imgs, output_dir)
    
    # 分析结果
    analyze_results(results)
    
    # 保存结果到文件
    if results:
        print("\n保存结果...")
        
        # 保存检测结果
        csv_path, json_path, summary_path = save_results_to_files(results, output_dir)
        
        # 创建并保存特征对比图
        plot_path = create_feature_comparison_plot(results, output_dir)
        
        print(f"结果已保存到: {output_dir}")
    
    print("完成!")

if __name__ == "__main__":
    main()

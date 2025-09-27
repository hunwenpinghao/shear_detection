#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在data_shear_split/roi_images/目录下测试撕裂面和剪切面检测器
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from shear_tear_detector import ShearTearDetector

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_detector_on_shear_split_images():
    """在data_shear_split/roi_images/目录下测试检测器"""
    detector = ShearTearDetector()
    
    # 图像目录
    roi_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_split/roi_images"
    output_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_split/output"
    
    if not os.path.exists(roi_dir):
        print(f"目录不存在: {roi_dir}")
        return
    
    print(f"处理目录: {roi_dir}")
    print("=" * 60)
    
    # 获取所有PNG文件
    png_files = [f for f in os.listdir(roi_dir) if f.endswith('.png')]
    png_files.sort()
    
    results = []
    
    for filename in png_files:
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
            
            # 保存可视化结果
            save_visualization(image, result, filename, output_dir)
                    
        except Exception as e:
            print(f"处理图像 {filename} 时出错: {e}")
    
    # 保存结果到JSON文件
    save_results_to_json(results, output_dir)
    
    # 生成统计报告
    generate_statistics_report(results, output_dir)
    
    return results

def save_visualization(image, result, filename, output_dir):
    """保存单个图像的可视化结果"""
    try:
        # 提取特征
        features = result['features']
        surface_type = result['surface_type']
        confidence = result['confidence']
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{filename} - Detection Result: {surface_type} (Confidence: {confidence:.3f})', 
                    fontsize=14)
        
        # 原始图像
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 预处理后图像
        processed = cv2.equalizeHist(cv2.GaussianBlur(image, (3, 3), 0))
        axes[0, 1].imshow(processed, cmap='gray')
        axes[0, 1].set_title('Preprocessed')
        axes[0, 1].axis('off')
        
        # 边缘检测
        edges = cv2.Canny(processed, 50, 150)
        axes[0, 2].imshow(edges, cmap='gray')
        axes[0, 2].set_title('Edge Detection')
        axes[0, 2].axis('off')
        
        # 区域分割结果
        if 'segmented_image' in result:
            axes[1, 0].imshow(result['segmented_image'], cmap='viridis')
            axes[1, 0].set_title('Surface Segmentation\n(White: Shear, Gray: Tear)')
            axes[1, 0].axis('off')
        else:
            grad_x = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(processed, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            axes[1, 0].imshow(gradient_magnitude, cmap='hot')
            axes[1, 0].set_title('Gradient Magnitude')
            axes[1, 0].axis('off')
        
        # 特征雷达图
        plot_feature_radar(axes[1, 1], features)
        
        # 特征条形图
        plot_feature_bars(axes[1, 2], features)
        
        plt.tight_layout()
        
        # 保存图像
        output_filename = f"{os.path.splitext(filename)[0]}_detection_result.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化结果已保存: {output_path}")
        
    except Exception as e:
        print(f"保存可视化结果时出错: {e}")

def plot_feature_radar(ax, features):
    """绘制特征雷达图"""
    # 选择关键特征
    key_features = ['continuity_score', 'smoothness_score', 'mean_brightness', 
                   'wave_strength', 'texture_uniformity']
    
    values = [features.get(f, 0) for f in key_features]
    labels = ['Continuity', 'Smoothness', 'Brightness', 'Wave Strength', 'Texture Uniformity']
    
    # 归一化到[0,1]
    values = [min(max(v, 0), 1) for v in values]
    
    # 雷达图
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]  # 闭合
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label='Feature Values')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title('Key Features Radar Chart')
    ax.grid(True)

def plot_feature_bars(ax, features):
    """绘制特征条形图"""
    # 选择数值特征
    numeric_features = {k: v for k, v in features.items() 
                       if isinstance(v, (int, float)) and not np.isnan(v)}
    
    # 排序并选择前8个
    sorted_features = sorted(numeric_features.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    
    names, values = zip(*sorted_features)
    names = [name.replace('_', '\n') for name in names]
    
    bars = ax.barh(range(len(names)), values)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Feature Values')
    ax.set_title('Main Feature Values')
    
    # 颜色编码
    for i, (name, value) in enumerate(sorted_features):
        if 'continuity' in name or 'smoothness' in name or 'brightness' in name:
            bars[i].set_color('blue')  # Shear surface features
        elif 'roughness' in name or 'anisotropy' in name:
            bars[i].set_color('red')   # Tear surface features
        else:
            bars[i].set_color('gray')  # Neutral features

def save_results_to_json(results, output_dir):
    """保存结果到JSON文件"""
    try:
        # 准备JSON数据
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'results': []
        }
        
        for result in results:
            json_data['results'].append({
                'filename': result['filename'],
                'surface_type': result['surface_type'],
                'confidence': result['confidence'],
                'features': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in result['features'].items()}
            })
        
        # 保存到文件
        json_path = os.path.join(output_dir, 'detection_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {json_path}")
        
    except Exception as e:
        print(f"保存JSON结果时出错: {e}")

def generate_statistics_report(results, output_dir):
    """生成统计报告"""
    try:
        if not results:
            print("没有检测结果可分析")
            return
        
        # 统计表面类型分布
        surface_types = [r['surface_type'] for r in results]
        shear_count = surface_types.count('shear')
        tear_count = surface_types.count('tear')
        
        # 置信度统计
        confidences = [r['confidence'] for r in results]
        
        # 特征分析
        all_features = {}
        for result in results:
            for feature_name, feature_value in result['features'].items():
                if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                    if feature_name not in all_features:
                        all_features[feature_name] = []
                    all_features[feature_name].append(feature_value)
        
        # 生成报告
        report = f"""
# 撕裂面和剪切面检测统计报告

## 基本信息
- 检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 总检测图像数: {len(results)}

## 检测结果分布
- 剪切面: {shear_count} ({shear_count/len(results)*100:.1f}%)
- 撕裂面: {tear_count} ({tear_count/len(results)*100:.1f}%)

## 置信度统计
- 平均置信度: {np.mean(confidences):.3f}
- 置信度标准差: {np.std(confidences):.3f}
- 最高置信度: {np.max(confidences):.3f}
- 最低置信度: {np.min(confidences):.3f}

## 特征统计
"""
        
        # 重要特征统计
        important_features = ['continuity_score', 'smoothness_score', 'roughness_score', 
                             'mean_brightness', 'wave_strength', 'anisotropy', 'texture_uniformity']
        
        for feature in important_features:
            if feature in all_features:
                values = all_features[feature]
                report += f"""
### {feature}
- 平均值: {np.mean(values):.3f}
- 标准差: {np.std(values):.3f}
- 最小值: {np.min(values):.3f}
- 最大值: {np.max(values):.3f}
"""
        
        # 保存报告
        report_path = os.path.join(output_dir, 'detection_statistics_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"统计报告已保存到: {report_path}")
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("检测结果摘要")
        print("=" * 60)
        print(f"总检测图像数: {len(results)}")
        print(f"剪切面: {shear_count} ({shear_count/len(results)*100:.1f}%)")
        print(f"撕裂面: {tear_count} ({tear_count/len(results)*100:.1f}%)")
        print(f"平均置信度: {np.mean(confidences):.3f}")
        
    except Exception as e:
        print(f"生成统计报告时出错: {e}")

def main():
    """主函数"""
    print("在data_shear_split/roi_images/目录下测试撕裂面和剪切面检测器")
    print("=" * 60)
    
    # 测试检测器
    results = test_detector_on_shear_split_images()
    
    print("\n测试完成!")
    print(f"结果已保存到: /Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_split/output/")

if __name__ == "__main__":
    main()

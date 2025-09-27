#!/usr/bin/env python3
"""
生成陡峭梯度图表
使用最有效的梯度增强方法
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

from gradient_enhancement_analysis import GradientEnhancementAnalyzer
import pandas as pd

def generate_steep_gradient_plots():
    """生成陡峭梯度图表"""
    
    print("=== 生成陡峭梯度图表 ===")
    
    # 初始化分析器
    analyzer = GradientEnhancementAnalyzer()
    
    # 读取数据
    data_file = "output/temporal_analysis/spot_temporal_data.csv"
    
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    data = df.to_dict('records')
    
    print(f"读取了 {len(data)} 个数据点")
    
    # 最推荐的梯度增强方法
    recommended_methods = [
        {
            'method': 'derivative',
            'params': {},
            'name': '一阶导数 - 直接显示变化率',
            'description': '最直观地显示变化梯度，正值表示增加，负值表示减少'
        },
        {
            'method': 'high_pass',
            'params': {'cutoff': 0.15, 'order': 4},
            'name': '高通滤波 - 突出快速变化',
            'description': '滤除缓慢变化，突出快速变化和突变'
        },
        {
            'method': 'sharpening',
            'params': {'alpha': 0.6},
            'name': '锐化滤波 - 增强边缘',
            'description': '增强信号的边缘和突变，使变化更陡峭'
        },
        {
            'method': 'sobel',
            'params': {},
            'name': 'Sobel梯度 - 检测变化边界',
            'description': '检测变化的边界和转折点，突出变化区域'
        },
        {
            'method': 'difference',
            'params': {'window': 3},
            'name': '差分增强 - 局部变化检测',
            'description': '检测局部快速变化，窗口越小越敏感'
        }
    ]
    
    print(f"\n生成 {len(recommended_methods)} 种推荐的梯度增强效果...")
    
    # 为每种推荐方法生成图表
    for i, config in enumerate(recommended_methods):
        print(f"\n{i+1}. {config['name']}")
        print(f"   {config['description']}")
        
        output_dir = f"output/recommended_gradients/{config['method']}"
        
        try:
            plot_path = analyzer.create_gradient_plots(
                data, 
                output_dir,
                enhancement_method=config['method'],
                **config['params']
            )
            print(f"   ✓ 成功生成: {plot_path}")
            
        except Exception as e:
            print(f"   ✗ 生成失败: {e}")
    
    print(f"\n✅ 陡峭梯度图表生成完成！")
    print("📁 推荐图表保存在 output/recommended_gradients/ 目录下")
    
    print("\n🎯 各方法特点：")
    print("1. 一阶导数：最直观，直接显示变化率")
    print("2. 高通滤波：突出快速变化，滤除缓慢趋势")
    print("3. 锐化滤波：增强边缘，使变化更陡峭")
    print("4. Sobel梯度：检测变化边界，突出转折点")
    print("5. 差分增强：局部变化检测，敏感度高")

if __name__ == "__main__":
    generate_steep_gradient_plots()

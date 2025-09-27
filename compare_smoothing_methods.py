#!/usr/bin/env python3
"""
对比不同平滑滤波方法的效果
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

from analyze_spot_temporal import SpotTemporalAnalyzer

def compare_smoothing_methods():
    """对比不同平滑滤波方法"""
    
    print("=== 对比不同平滑滤波方法 ===")
    
    # 初始化分析器
    analyzer = SpotTemporalAnalyzer()
    
    # 读取已有数据
    import pandas as pd
    data_file = "output/temporal_analysis/spot_temporal_data.csv"
    
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请先运行 analyze_spot_temporal.py 生成数据")
        return
    
    # 读取数据
    df = pd.read_csv(data_file)
    data = df.to_dict('records')
    
    print(f"读取了 {len(data)} 个数据点")
    
    # 定义不同的平滑方法
    smoothing_configs = [
        {'method': 'gaussian', 'sigma': 5.0, 'window_size': 50, 'name': '高斯滤波 (σ=5)'},
        {'method': 'gaussian', 'sigma': 10.0, 'window_size': 50, 'name': '高斯滤波 (σ=10)'},
        {'method': 'gaussian', 'sigma': 15.0, 'window_size': 50, 'name': '高斯滤波 (σ=15)'},
        {'method': 'moving_avg', 'sigma': 10.0, 'window_size': 30, 'name': '移动平均 (窗口=30)'},
        {'method': 'moving_avg', 'sigma': 10.0, 'window_size': 50, 'name': '移动平均 (窗口=50)'},
        {'method': 'savgol', 'sigma': 10.0, 'window_size': 31, 'name': 'Savitzky-Golay (窗口=31)'},
        {'method': 'savgol', 'sigma': 10.0, 'window_size': 51, 'name': 'Savitzky-Golay (窗口=51)'},
        {'method': 'median', 'sigma': 10.0, 'window_size': 21, 'name': '中值滤波 (窗口=21)'},
    ]
    
    # 为每种方法生成图表
    for i, config in enumerate(smoothing_configs):
        print(f"\n生成方法 {i+1}/{len(smoothing_configs)}: {config['name']}")
        
        output_dir = f"output/smoothing_comparison/{config['method']}_{config['window_size']}_{config['sigma']}"
        
        try:
            plot_path = analyzer.create_temporal_plots(
                data, 
                output_dir,
                smoothing_method=config['method'],
                window_size=config['window_size'],
                sigma=config['sigma']
            )
            print(f"✓ 成功生成: {plot_path}")
            
        except Exception as e:
            print(f"✗ 生成失败: {e}")
    
    print(f"\n✅ 平滑方法对比完成！")
    print("📁 所有对比图表保存在 output/smoothing_comparison/ 目录下")

if __name__ == "__main__":
    compare_smoothing_methods()

#!/usr/bin/env python3
"""
在平滑滤波基础上增强梯度
用于识别换钢材时间段
"""

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
import platform
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

# 设置中文字体
def setup_chinese_font():
    """设置中文字体"""
    import matplotlib.font_manager as fm
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = []
        
        preferred_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
        
        for font in preferred_fonts:
            if font in available_fonts:
                chinese_fonts.append(font)
        
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"设置中文字体: {chinese_fonts[0]}")
            return True
    
    elif system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("设置Windows中文字体")
        return True
    
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("设置Linux中文字体")
        return True
    
    print("无法设置中文字体，将使用英文标签")
    return False

class EnhancedSmoothedGradientAnalyzer:
    """在平滑滤波基础上增强梯度的分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.data = []
    
    def apply_smoothing_then_gradient(self, data: List[Dict[str, Any]], 
                                    smoothing_sigma: float = 10.0,
                                    gradient_method: str = 'derivative',
                                    gradient_strength: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        先应用高斯平滑，再增强梯度
        
        Args:
            data: 斑块分析数据
            smoothing_sigma: 高斯平滑的sigma值
            gradient_method: 梯度增强方法
            gradient_strength: 梯度增强强度
            
        Returns:
            时间序列、原始平滑数据、增强梯度数据
        """
        time_seconds = np.array([d['time_seconds'] for d in data])
        spot_counts = np.array([d['spot_count'] for d in data])
        spot_densities = np.array([d['spot_density'] for d in data])
        
        # 第一步：高斯平滑
        smoothed_counts = gaussian_filter1d(spot_counts, sigma=smoothing_sigma)
        smoothed_densities = gaussian_filter1d(spot_densities, sigma=smoothing_sigma)
        
        # 第二步：在平滑数据基础上增强梯度
        if gradient_method == 'derivative':
            # 一阶导数
            enhanced_counts = np.gradient(smoothed_counts) * gradient_strength
            enhanced_densities = np.gradient(smoothed_densities) * gradient_strength
            
        elif gradient_method == 'high_pass':
            # 高通滤波
            cutoff = 0.1
            nyquist = 0.5
            normal_cutoff = cutoff / nyquist
            b, a = butter(4, normal_cutoff, btype='high', analog=False)
            
            enhanced_counts = filtfilt(b, a, smoothed_counts) * gradient_strength
            enhanced_densities = filtfilt(b, a, smoothed_densities) * gradient_strength
            
        elif gradient_method == 'sharpening':
            # 锐化滤波
            alpha = 0.5
            sharpened_counts = smoothed_counts + alpha * (smoothed_counts - gaussian_filter1d(smoothed_counts, sigma=1))
            sharpened_densities = smoothed_densities + alpha * (smoothed_densities - gaussian_filter1d(smoothed_densities, sigma=1))
            
            enhanced_counts = (sharpened_counts - smoothed_counts) * gradient_strength
            enhanced_densities = (sharpened_densities - smoothed_densities) * gradient_strength
            
        elif gradient_method == 'difference':
            # 差分增强
            window = 5
            enhanced_counts = np.zeros_like(smoothed_counts)
            enhanced_densities = np.zeros_like(smoothed_densities)
            
            for i in range(window, len(smoothed_counts) - window):
                front_avg_counts = np.mean(smoothed_counts[i:i+window])
                back_avg_counts = np.mean(smoothed_counts[i-window:i])
                enhanced_counts[i] = (front_avg_counts - back_avg_counts) * gradient_strength
                
                front_avg_densities = np.mean(smoothed_densities[i:i+window])
                back_avg_densities = np.mean(smoothed_densities[i-window:i])
                enhanced_densities[i] = (front_avg_densities - back_avg_densities) * gradient_strength
        
        else:
            # 默认使用导数
            enhanced_counts = np.gradient(smoothed_counts) * gradient_strength
            enhanced_densities = np.gradient(smoothed_densities) * gradient_strength
        
        return time_seconds, smoothed_counts, smoothed_densities, enhanced_counts, enhanced_densities
    
    def detect_change_periods(self, time_seconds: np.ndarray, enhanced_data: np.ndarray, 
                            threshold: float = 2.0, min_duration: float = 100.0) -> List[Dict[str, Any]]:
        """
        检测变化时间段（可能的换钢材时间段）
        
        Args:
            time_seconds: 时间序列
            enhanced_data: 增强后的梯度数据
            threshold: 变化阈值
            min_duration: 最小持续时间（秒）
            
        Returns:
            检测到的变化时间段列表
        """
        # 找到超过阈值的点
        above_threshold = np.abs(enhanced_data) > threshold
        
        # 找到连续的区域
        change_periods = []
        in_change = False
        start_time = None
        
        for i, is_change in enumerate(above_threshold):
            if is_change and not in_change:
                # 开始变化
                in_change = True
                start_time = time_seconds[i]
            elif not is_change and in_change:
                # 结束变化
                end_time = time_seconds[i-1]
                duration = end_time - start_time
                
                if duration >= min_duration:
                    change_periods.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'start_frame': i - int(duration/5),  # 假设每5秒一帧
                        'end_frame': i-1,
                        'max_change': np.max(np.abs(enhanced_data[i-int(duration/5):i-1]))
                    })
                
                in_change = False
        
        # 处理最后一个变化段
        if in_change:
            end_time = time_seconds[-1]
            duration = end_time - start_time
            if duration >= min_duration:
                change_periods.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'start_frame': len(time_seconds) - int(duration/5),
                    'end_frame': len(time_seconds)-1,
                    'max_change': np.max(np.abs(enhanced_data[len(time_seconds)-int(duration/5):]))
                })
        
        return change_periods
    
    def create_enhanced_gradient_plots(self, data: List[Dict[str, Any]], output_dir: str = "output",
                                     smoothing_sigma: float = 10.0, gradient_method: str = 'derivative',
                                     gradient_strength: float = 2.0, threshold: float = 2.0):
        """
        创建增强梯度图表，用于识别换钢材时间段
        
        Args:
            data: 斑块分析数据
            output_dir: 输出目录
            smoothing_sigma: 平滑sigma值
            gradient_method: 梯度增强方法
            gradient_strength: 梯度增强强度
            threshold: 变化检测阈值
        """
        if not data:
            print("没有数据可以绘制")
            return
        
        # 设置中文字体
        font_success = setup_chinese_font()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 应用平滑+梯度增强
        time_seconds, smoothed_counts, smoothed_densities, enhanced_counts, enhanced_densities = self.apply_smoothing_then_gradient(
            data, smoothing_sigma, gradient_method, gradient_strength)
        
        # 检测变化时间段
        change_periods_counts = self.detect_change_periods(time_seconds, enhanced_counts, threshold)
        change_periods_densities = self.detect_change_periods(time_seconds, enhanced_densities, threshold)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # 原始数据 vs 平滑数据
        ax1.plot(time_seconds, [d['spot_count'] for d in data], 'b-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax1.plot(time_seconds, smoothed_counts, 'b-', linewidth=2.5, alpha=0.9, label='高斯平滑')
        ax1.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax1.set_ylabel('斑块数量' if font_success else 'Spot Count')
        ax1.set_title('斑块数量 - 高斯平滑' if font_success else 'Spot Count - Gaussian Smoothed')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(time_seconds))
        ax1.legend()
        
        # 增强梯度 - 斑块数量
        ax2.plot(time_seconds, enhanced_counts, 'r-', linewidth=2, alpha=0.9)
        ax2.fill_between(time_seconds, enhanced_counts, alpha=0.3, color='red')
        
        # 标记变化时间段
        for period in change_periods_counts:
            ax2.axvspan(period['start_time'], period['end_time'], alpha=0.2, color='yellow', 
                       label='疑似换钢材时间段' if period == change_periods_counts[0] else "")
        
        ax2.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, label=f'阈值: ±{threshold}')
        ax2.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
        ax2.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax2.set_ylabel('梯度增强斑块数量' if font_success else 'Enhanced Gradient Spot Count')
        ax2.set_title('斑块数量梯度增强 - 换钢材检测' if font_success else 'Spot Count Gradient Enhancement - Steel Change Detection')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(time_seconds))
        ax2.legend()
        
        # 原始数据 vs 平滑数据 - 密度
        ax3.plot(time_seconds, [d['spot_density'] for d in data], 'g-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax3.plot(time_seconds, smoothed_densities, 'g-', linewidth=2.5, alpha=0.9, label='高斯平滑')
        ax3.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax3.set_ylabel('斑块密度' if font_success else 'Spot Density')
        ax3.set_title('斑块密度 - 高斯平滑' if font_success else 'Spot Density - Gaussian Smoothed')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max(time_seconds))
        ax3.legend()
        
        # 增强梯度 - 斑块密度
        ax4.plot(time_seconds, enhanced_densities, 'm-', linewidth=2, alpha=0.9)
        ax4.fill_between(time_seconds, enhanced_densities, alpha=0.3, color='magenta')
        
        # 标记变化时间段
        for period in change_periods_densities:
            ax4.axvspan(period['start_time'], period['end_time'], alpha=0.2, color='yellow',
                       label='疑似换钢材时间段' if period == change_periods_densities[0] else "")
        
        ax4.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, label=f'阈值: ±{threshold}')
        ax4.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
        ax4.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax4.set_ylabel('梯度增强斑块密度' if font_success else 'Enhanced Gradient Spot Density')
        ax4.set_title('斑块密度梯度增强 - 换钢材检测' if font_success else 'Spot Density Gradient Enhancement - Steel Change Detection')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, max(time_seconds))
        ax4.legend()
        
        # 添加方法说明
        method_names = {
            'derivative': '一阶导数',
            'high_pass': '高通滤波',
            'sharpening': '锐化滤波',
            'difference': '差分增强'
        }
        method_name = method_names.get(gradient_method, gradient_method)
        
        fig.suptitle(f'换钢材时间段检测 (σ={smoothing_sigma}, {method_name}, 强度={gradient_strength}, 阈值={threshold})', 
                    fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 保存图表
        plot_path = os.path.join(output_dir, f"steel_change_detection_{gradient_method}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"换钢材检测图表已保存: {plot_path}")
        
        # 输出检测结果
        print(f"\n=== 换钢材时间段检测结果 ===")
        print(f"基于斑块数量的检测结果:")
        for i, period in enumerate(change_periods_counts):
            print(f"  时间段 {i+1}: {period['start_time']:.1f}s - {period['end_time']:.1f}s "
                  f"(持续 {period['duration']:.1f}s, 最大变化: {period['max_change']:.2f})")
        
        print(f"\n基于斑块密度的检测结果:")
        for i, period in enumerate(change_periods_densities):
            print(f"  时间段 {i+1}: {period['start_time']:.1f}s - {period['end_time']:.1f}s "
                  f"(持续 {period['duration']:.1f}s, 最大变化: {period['max_change']:.4f})")
        
        # 保存检测结果
        results = {
            'detection_parameters': {
                'smoothing_sigma': smoothing_sigma,
                'gradient_method': gradient_method,
                'gradient_strength': gradient_strength,
                'threshold': threshold
            },
            'change_periods_counts': change_periods_counts,
            'change_periods_densities': change_periods_densities
        }
        
        results_path = os.path.join(output_dir, f"steel_change_detection_{gradient_method}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"检测结果已保存: {results_path}")
        
        return plot_path, results

def main():
    """主函数"""
    print("=== 换钢材时间段检测分析 ===")
    
    # 读取数据
    data_file = "output/temporal_analysis/spot_temporal_data.csv"
    
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请先运行 analyze_spot_temporal.py 生成数据")
        return
    
    df = pd.read_csv(data_file)
    data = df.to_dict('records')
    
    print(f"读取了 {len(data)} 个数据点")
    
    # 初始化分析器
    analyzer = EnhancedSmoothedGradientAnalyzer()
    
    # 测试不同的梯度增强方法
    methods = [
        {'method': 'derivative', 'strength': 3.0, 'threshold': 2.0, 'name': '一阶导数'},
        {'method': 'derivative', 'strength': 5.0, 'threshold': 3.0, 'name': '一阶导数(强)'},
        {'method': 'high_pass', 'strength': 2.0, 'threshold': 1.5, 'name': '高通滤波'},
        {'method': 'difference', 'strength': 2.0, 'threshold': 2.0, 'name': '差分增强'}
    ]
    
    for config in methods:
        print(f"\n=== 测试方法: {config['name']} ===")
        
        output_dir = f"output/steel_change_detection/{config['method']}_{config['strength']}"
        
        try:
            plot_path, results = analyzer.create_enhanced_gradient_plots(
                data,
                output_dir=output_dir,
                smoothing_sigma=10.0,
                gradient_method=config['method'],
                gradient_strength=config['strength'],
                threshold=config['threshold']
            )
            
            print(f"✓ 成功生成: {plot_path}")
            
        except Exception as e:
            print(f"✗ 生成失败: {e}")
    
    print(f"\n✅ 换钢材时间段检测完成！")
    print("📁 检测图表保存在 output/steel_change_detection/ 目录下")
    print("\n🎯 使用方法：")
    print("1. 查看生成的图表，黄色区域为疑似换钢材时间段")
    print("2. 调整阈值参数来优化检测精度")
    print("3. 结合斑块数量和密度两种指标进行综合判断")

if __name__ == "__main__":
    main()

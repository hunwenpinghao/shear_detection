#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
两层滤波和波峰面积分析测试脚本
专门用于展示在红色滤波结果基础上再进行一层滤波，去除毛刺，并计算波峰面积变化
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import platform
from typing import List, Dict, Tuple, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_chinese_font():
    """设置中文字体"""
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
        print("设置中文字体: SimHei")
        return True
    
    print("无法设置中文字体，将使用英文标签")
    return False

class TwoLayerFilterAnalyzer:
    """两层滤波分析器"""
    
    def __init__(self, data_path: str):
        """
        初始化分析器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
            print(f"加载CSV数据: {len(self.data)} 条记录")
        else:
            raise ValueError("不支持的文件格式，请使用CSV文件")
    
    def apply_second_layer_filter(self, data: np.ndarray, method: str = 'savgol', window_length: int = 21) -> np.ndarray:
        """
        在已有滤波结果基础上再进行一层平滑滤波
        
        Args:
            data: 已有滤波的数据
            method: 第二层滤波方法 ('savgol', 'gaussian', 'moving_avg')
            window_length: 滤波窗口长度
            
        Returns:
            第二层滤波后的数据
        """
        if method == 'savgol':
            # Savitzky-Golay滤波
            actual_length = min(window_length, len(data))
            if actual_length % 2 == 0:
                actual_length -= 1
            if actual_length > 3:
                return signal.savgol_filter(data, actual_length, 3)
        
        elif method == 'gaussian':
            # 高斯滤波
            sigma = window_length / 4
            return gaussian_filter1d(data, sigma=sigma)
        
        elif method == 'moving_avg':
            # 移动平均滤波
            actual_window = min(window_length, len(data))
            if actual_window % 2 == 0:
                actual_window += 1
            return np.convolve(data, np.ones(actual_window)/actual_window, mode='same')
        
        # 默认返回原数据
        return data
    
    def detect_peaks(self, data: np.ndarray, height: float = None, distance: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测信号中的波峰
        
        Args:
            data: 信号数据
            height: 波峰最小高度
            distance: 相邻波峰最小距离
            
        Returns:
            波峰位置和高度
        """
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(data, height=height, distance=distance)
        
        # 获取波峰高度
        if 'peak_heights' in properties:
            heights = properties['peak_heights']
        else:
            # 如果没有peak_heights属性，直接从data中获取
            heights = data[peaks]
        
        return peaks, heights
    
    def detect_valleys(self, data: np.ndarray, height: float = None, distance: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测信号中的波谷（最低点）
        
        Args:
            data: 信号数据
            height: 波谷最大高度阈值
            distance: 相邻波谷最小距离
            
        Returns:
            波谷位置和高度
        """
        from scipy.signal import find_peaks
        
        # 通过找负信号的峰值来找到波谷
        valleys, properties = find_peaks(-data, height=height, distance=distance)
        
        # 获取波谷高度
        if 'peak_heights' in properties:
            heights = -properties['peak_heights']  # 转回正值
        else:
            # 如果没有peak_heights属性，直接从data中获取
            heights = data[valleys]
        
        return valleys, heights
    
    def segment_hills(self, data: np.ndarray, time_data: np.ndarray, 
                     valley_distance: int = 200, min_hill_width: int = 100) -> List[Dict]:
        """
        将信号分割成多个山丘
        
        Args:
            data: 信号数据
            time_data: 时间数据
            valley_distance: 波谷最小距离
            min_hill_width: 最小山丘宽度
            
        Returns:
            山丘信息列表
        """
        # 检测波谷
        valleys, valley_heights = self.detect_valleys(data, distance=valley_distance)
        
        # 如果没有检测到波谷，整个信号作为一个山丘
        if len(valleys) == 0:
            return [{
                'index': 1,
                'start_idx': 0,
                'end_idx': len(data) - 1,
                'start_time': time_data[0],
                'end_time': time_data[-1],
                'width': len(data),
                'max_height': np.max(data),
                'area': np.trapz(data)
            }]
        
        # 分割山丘
        hills = []
        start_idx = 0
        
        for i, valley_idx in enumerate(valleys):
            end_idx = valley_idx
            
            # 检查山丘宽度是否足够
            if end_idx - start_idx >= min_hill_width:
                hill_data = data[start_idx:end_idx]
                hill_time = time_data[start_idx:end_idx]
                
                hills.append({
                    'index': len(hills) + 1,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': hill_time[0],
                    'end_time': hill_time[-1],
                    'width': end_idx - start_idx,
                    'max_height': np.max(hill_data),
                    'area': np.trapz(hill_data)
                })
            
            start_idx = valley_idx
        
        # 处理最后一个山丘
        if start_idx < len(data) - min_hill_width:
            end_idx = len(data) - 1
            hill_data = data[start_idx:end_idx]
            hill_time = time_data[start_idx:end_idx]
            
            hills.append({
                'index': len(hills) + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': hill_time[0],
                'end_time': hill_time[-1],
                'width': end_idx - start_idx,
                'max_height': np.max(hill_data),
                'area': np.trapz(hill_data)
            })
        
        return hills
    
    def apply_strong_smoothing(self, data: np.ndarray, method: str = 'gaussian', 
                             sigma: float = 30, window_size: int = 100) -> np.ndarray:
        """
        应用强平滑滤波，将多个小山峰平滑成大的山丘
        
        Args:
            data: 信号数据
            method: 平滑方法 ('gaussian', 'moving_avg', 'savgol')
            sigma: 高斯滤波的sigma值
            window_size: 窗口大小
            
        Returns:
            强平滑后的数据
        """
        if method == 'gaussian':
            return gaussian_filter1d(data, sigma=sigma)
        elif method == 'moving_avg':
            actual_window = min(window_size, len(data))
            if actual_window % 2 == 0:
                actual_window += 1
            return np.convolve(data, np.ones(actual_window)/actual_window, mode='same')
        elif method == 'savgol':
            actual_length = min(window_size, len(data))
            if actual_length % 2 == 0:
                actual_length -= 1
            if actual_length > 3:
                return signal.savgol_filter(data, actual_length, 3)
        
        return data
    
    def calculate_peak_area(self, data: np.ndarray, peak_idx: int, window: int = 50) -> float:
        """
        计算单个波峰的面积
        
        Args:
            data: 信号数据
            peak_idx: 波峰位置索引
            window: 计算面积的窗口大小
            
        Returns:
            波峰面积
        """
        # 定义计算面积的范围
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(data), peak_idx + window + 1)
        
        # 获取波峰范围内的数据
        peak_data = data[start_idx:end_idx]
        
        # 使用梯形积分计算面积
        area = np.trapz(peak_data)
        
        return area
    
    def analyze_peak_areas_over_time(self, data: np.ndarray, time_data: np.ndarray,
                                   peak_height_threshold: float = None, peak_distance: int = 50,
                                   area_window: int = 50) -> Tuple[List[Dict], np.ndarray]:
        """
        分析所有波峰的面积变化
        
        Args:
            data: 信号数据
            time_data: 时间数据
            peak_height_threshold: 波峰高度阈值
            peak_distance: 相邻波峰最小距离
            area_window: 计算面积的窗口大小
            
        Returns:
            波峰信息列表和面积时间序列
        """
        # 检测波峰
        peaks, heights = self.detect_peaks(data, height=peak_height_threshold, distance=peak_distance)
        
        # 计算每个波峰的面积
        peak_info = []
        areas = []
        
        for i, (peak_idx, height) in enumerate(zip(peaks, heights)):
            area = self.calculate_peak_area(data, peak_idx, area_window)
            
            peak_info.append({
                'index': i + 1,
                'position': peak_idx,
                'time': time_data[peak_idx],
                'height': height,
                'area': area
            })
            
            areas.append(area)
        
        # 创建面积时间序列（只在波峰位置有值）
        area_time_series = np.zeros_like(data)
        for peak_idx, area in zip(peaks, areas):
            area_time_series[peak_idx] = area
        
        return peak_info, area_time_series
    
    def create_hill_analysis(self, column_name: str = 'num_burrs', output_dir: str = "test_output"):
        """
        创建山丘分割和面积分析
        
        Args:
            column_name: 要分析的列名
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取数据
        time_seconds = self.data['time_seconds'].values
        original_data = self.data[column_name].values
        
        # 第一层滤波：高斯滤波 (σ=10) - 对应之前的红色曲线
        first_layer = gaussian_filter1d(original_data, sigma=10)
        
        # 强平滑滤波：将多个小山峰平滑成大的山丘
        strong_smooth = self.apply_strong_smoothing(first_layer, method='gaussian', sigma=30)
        
        # 分割山丘
        hills = self.segment_hills(strong_smooth, time_seconds, valley_distance=200, min_hill_width=100)
        
        # 创建专门的可视化图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'山丘分割及面积分析 - {column_name}', fontsize=16)
        
        # 原始数据 vs 第一层滤波（红色曲线）
        axes[0, 0].plot(time_seconds, original_data, 'b-', alpha=0.5, linewidth=0.8, label='原始数据')
        axes[0, 0].plot(time_seconds, first_layer, 'r-', linewidth=2, label='第一层滤波 (σ=10)')
        axes[0, 0].set_title('第一层滤波效果（对应之前的红色曲线）')
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel(column_name)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 强平滑效果
        axes[0, 1].plot(time_seconds, first_layer, 'r-', alpha=0.5, linewidth=1, label='第一层滤波')
        axes[0, 1].plot(time_seconds, strong_smooth, 'g-', linewidth=2, label='强平滑滤波 (σ=30)')
        axes[0, 1].set_title('强平滑效果（小山峰合并成大山丘）')
        axes[0, 1].set_xlabel('时间 (秒)')
        axes[0, 1].set_ylabel(column_name)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 山丘分割结果
        axes[1, 0].plot(time_seconds, strong_smooth, 'g-', linewidth=2, label='强平滑数据')
        
        # 用不同颜色标记不同的山丘
        colors = plt.cm.Set3(np.linspace(0, 1, len(hills)))
        for i, hill in enumerate(hills):
            start_idx = hill['start_idx']
            end_idx = hill['end_idx']
            hill_data = strong_smooth[start_idx:end_idx]
            hill_time = time_seconds[start_idx:end_idx]
            
            axes[1, 0].fill_between(hill_time, hill_data, alpha=0.3, color=colors[i], 
                                   label=f'山丘 {hill["index"]}')
        
        # 标记波谷位置
        valleys, _ = self.detect_valleys(strong_smooth, distance=200)
        if len(valleys) > 0:
            axes[1, 0].scatter(time_seconds[valleys], strong_smooth[valleys], 
                             color='red', s=50, zorder=5, label='分割点（波谷）')
        
        axes[1, 0].set_title('山丘分割结果')
        axes[1, 0].set_xlabel('时间 (秒)')
        axes[1, 0].set_ylabel(column_name)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 山丘面积变化曲线
        hill_areas = [hill['area'] for hill in hills]
        hill_centers = [(hill['start_time'] + hill['end_time']) / 2 for hill in hills]
        
        axes[1, 1].bar(range(len(hills)), hill_areas, color=colors, alpha=0.7)
        axes[1, 1].set_title('山丘面积分布')
        axes[1, 1].set_xlabel('山丘编号')
        axes[1, 1].set_ylabel('山丘面积')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for i, area in enumerate(hill_areas):
            axes[1, 1].text(i, area + max(hill_areas) * 0.01, f'{area:.0f}', 
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'hill_segmentation_analysis_{column_name}.png'),
                   dpi=300, bbox_inches='tight')
        
        # 打印详细分析结果
        print(f"\n{'='*80}")
        print(f"山丘分割及面积分析结果 - {column_name}")
        print(f"{'='*80}")
        
        print(f"\n滤波参数:")
        print(f"  第一层滤波: 高斯滤波 (σ=10)")
        print(f"  强平滑滤波: 高斯滤波 (σ=30)")
        print(f"  波谷最小距离: 200个数据点")
        print(f"  最小山丘宽度: 100个数据点")
        
        print(f"\n山丘分割结果:")
        print(f"  检测到 {len(hills)} 个山丘")
        
        print(f"\n山丘详细信息:")
        for hill in hills:
            print(f"  山丘 {hill['index']:2d}: "
                  f"时间={hill['start_time']:6.1f}-{hill['end_time']:6.1f}秒, "
                  f"宽度={hill['width']:4d}点, "
                  f"最大高度={hill['max_height']:6.2f}, "
                  f"面积={hill['area']:8.2f}")
        
        # 计算面积统计信息
        if hills:
            hill_areas = [hill['area'] for hill in hills]
            hill_heights = [hill['max_height'] for hill in hills]
            hill_widths = [hill['width'] for hill in hills]
            
            print(f"\n山丘面积统计:")
            print(f"  平均面积: {np.mean(hill_areas):.2f}")
            print(f"  最大面积: {np.max(hill_areas):.2f}")
            print(f"  最小面积: {np.min(hill_areas):.2f}")
            print(f"  面积标准差: {np.std(hill_areas):.2f}")
            print(f"  面积变异系数: {np.std(hill_areas)/np.mean(hill_areas)*100:.1f}%")
            
            print(f"\n山丘高度统计:")
            print(f"  平均高度: {np.mean(hill_heights):.2f}")
            print(f"  最大高度: {np.max(hill_heights):.2f}")
            print(f"  最小高度: {np.min(hill_heights):.2f}")
            print(f"  高度标准差: {np.std(hill_heights):.2f}")
            
            print(f"\n山丘宽度统计:")
            print(f"  平均宽度: {np.mean(hill_widths):.2f}点")
            print(f"  最大宽度: {np.max(hill_widths):.0f}点")
            print(f"  最小宽度: {np.min(hill_widths):.0f}点")
            print(f"  宽度标准差: {np.std(hill_widths):.2f}点")
            
            # 计算高度与面积的相关性
            correlation = np.corrcoef(hill_heights, hill_areas)[0, 1]
            print(f"\n高度与面积相关性: {correlation:.3f}")
        
        return hills, strong_smooth

def main():
    """主函数"""
    # 设置中文字体
    setup_chinese_font()
    
    # 数据文件路径
    data_path = "../data_burr_density_curve/burr_density_analysis.csv"
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return
    
    # 创建分析器
    analyzer = TwoLayerFilterAnalyzer(data_path)
    
    # 重点分析num_burrs列（对应之前看到的红色曲线）
    column_to_analyze = 'num_burrs'
    
    if column_to_analyze in analyzer.data.columns:
        print(f"\n{'='*80}")
        print(f"开始分析列: {column_to_analyze}")
        print(f"{'='*80}")
        
        # 创建山丘分割和面积分析
        hills, strong_smooth = analyzer.create_hill_analysis(column_to_analyze)
        
        print(f"\n分析完成！结果已保存到 test_output/hill_segmentation_analysis_{column_to_analyze}.png")
        print(f"成功将多个小山峰平滑成 {len(hills)} 个大山丘，并通过波谷位置进行了分割。")
    else:
        print(f"警告: 列 '{column_to_analyze}' 不存在于数据中")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
滤波强度测试脚本
测试不同强度的滤波效果，分析曲线的能量变化
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
import json

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

class FilterIntensityTester:
    """滤波强度测试器"""
    
    def __init__(self, data_path: str):
        """
        初始化测试器
        
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
        elif self.data_path.endswith('.json'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.data = pd.DataFrame(self.data)
            print(f"加载JSON数据: {len(self.data)} 条记录")
        else:
            raise ValueError("不支持的文件格式，请使用CSV或JSON文件")
    
    def apply_gaussian_filter(self, data: np.ndarray, sigma_values: List[float]) -> Dict[float, np.ndarray]:
        """
        应用不同强度的高斯滤波
        
        Args:
            data: 原始数据
            sigma_values: 不同的sigma值列表
            
        Returns:
            不同sigma值对应的滤波结果
        """
        filtered_results = {}
        
        for sigma in sigma_values:
            filtered_data = gaussian_filter1d(data, sigma=sigma)
            filtered_results[sigma] = filtered_data
        
        return filtered_results
    
    def apply_moving_average_filter(self, data: np.ndarray, window_sizes: List[int]) -> Dict[int, np.ndarray]:
        """
        应用不同窗口大小的移动平均滤波
        
        Args:
            data: 原始数据
            window_sizes: 不同的窗口大小列表
            
        Returns:
            不同窗口大小对应的滤波结果
        """
        filtered_results = {}
        
        for window_size in window_sizes:
            # 确保窗口大小不超过数据长度
            actual_window = min(window_size, len(data))
            if actual_window % 2 == 0:
                actual_window += 1  # 确保为奇数
            
            filtered_data = np.convolve(data, np.ones(actual_window)/actual_window, mode='same')
            filtered_results[window_size] = filtered_data
        
        return filtered_results
    
    def apply_savgol_filter(self, data: np.ndarray, window_lengths: List[int], polyorder: int = 3) -> Dict[int, np.ndarray]:
        """
        应用不同窗口长度的Savitzky-Golay滤波
        
        Args:
            data: 原始数据
            window_lengths: 不同的窗口长度列表
            polyorder: 多项式阶数
            
        Returns:
            不同窗口长度对应的滤波结果
        """
        filtered_results = {}
        
        for window_length in window_lengths:
            # 确保窗口长度不超过数据长度且为奇数
            actual_length = min(window_length, len(data))
            if actual_length % 2 == 0:
                actual_length -= 1
            
            if actual_length > polyorder:
                filtered_data = signal.savgol_filter(data, actual_length, polyorder)
                filtered_results[window_length] = filtered_data
            else:
                print(f"警告: 窗口长度 {window_length} 太小，跳过")
        
        return filtered_results
    
    def calculate_energy(self, data: np.ndarray) -> float:
        """
        计算信号能量
        
        Args:
            data: 信号数据
            
        Returns:
            信号能量
        """
        return np.sum(np.square(data))
    
    def calculate_energy_reduction(self, original_data: np.ndarray, filtered_data: np.ndarray) -> float:
        """
        计算滤波后的能量减少比例
        
        Args:
            original_data: 原始数据
            filtered_data: 滤波后数据
            
        Returns:
            能量减少比例 (0-1)
        """
        original_energy = self.calculate_energy(original_data)
        filtered_energy = self.calculate_energy(filtered_data)
        
        if original_energy == 0:
            return 0
        
        return (original_energy - filtered_energy) / original_energy
    
    def calculate_smoothness(self, data: np.ndarray) -> float:
        """
        计算信号平滑度（基于二阶差分的方差）

        Args:
            data: 信号数据

        Returns:
            平滑度指标（越小越平滑）
        """
        second_diff = np.diff(data, n=2)
        return np.var(second_diff)

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
            sigma = window_length / 4  # 根据窗口长度计算sigma
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

    def create_two_layer_filter_analysis(self, column_name: str, output_dir: str = "test_output"):
        """
        创建两层滤波分析

        Args:
            column_name: 要分析的列名
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 获取数据
        time_seconds = self.data['time_seconds'].values
        original_data = self.data[column_name].values

        # 第一层滤波：高斯滤波 (σ=10)
        first_layer = gaussian_filter1d(original_data, sigma=10)

        # 第二层滤波：Savitzky-Golay滤波
        second_layer = self.apply_second_layer_filter(first_layer, method='savgol', window_length=21)

        # 检测波峰并计算面积
        peak_info, area_time_series = self.analyze_peak_areas_over_time(
            second_layer, time_seconds, peak_distance=100, area_window=50
        )

        # 创建可视化图
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle(f'两层滤波及波峰分析 - {column_name}', fontsize=16)

        # 原始数据
        axes[0, 0].plot(time_seconds, original_data, 'b-', alpha=0.7, linewidth=1, label='原始数据')
        axes[0, 0].set_title('原始数据')
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel(column_name)
        axes[0, 0].grid(True, alpha=0.3)

        # 第一层滤波结果
        axes[0, 1].plot(time_seconds, original_data, 'b-', alpha=0.5, linewidth=0.8, label='原始')
        axes[0, 1].plot(time_seconds, first_layer, 'r-', linewidth=2, label='第一层滤波 (σ=10)')
        axes[0, 1].set_title('第一层高斯滤波')
        axes[0, 1].set_xlabel('时间 (秒)')
        axes[0, 1].set_ylabel(column_name)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 第二层滤波结果
        axes[1, 0].plot(time_seconds, original_data, 'b-', alpha=0.5, linewidth=0.8, label='原始')
        axes[1, 0].plot(time_seconds, first_layer, 'r-', alpha=0.5, linewidth=1, label='第一层')
        axes[1, 0].plot(time_seconds, second_layer, 'g-', linewidth=2, label='第二层滤波')
        axes[1, 0].set_title('两层滤波对比')
        axes[1, 0].set_xlabel('时间 (秒)')
        axes[1, 0].set_ylabel(column_name)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 波峰检测结果
        peaks, heights = self.detect_peaks(second_layer, distance=100)
        axes[1, 1].plot(time_seconds, second_layer, 'g-', linewidth=2, label='第二层滤波')
        axes[1, 1].scatter(time_seconds[peaks], heights, color='red', s=50, zorder=5, label='检测波峰')
        axes[1, 1].set_title('波峰检测结果')
        axes[1, 1].set_xlabel('时间 (秒)')
        axes[1, 1].set_ylabel(column_name)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 波峰面积时间序列
        axes[2, 0].plot(time_seconds, area_time_series, 'purple', linewidth=2, marker='o', markersize=4)
        axes[2, 0].set_title('波峰面积变化曲线')
        axes[2, 0].set_xlabel('时间 (秒)')
        axes[2, 0].set_ylabel('波峰面积')
        axes[2, 0].grid(True, alpha=0.3)

        # 波峰高度与面积的关系
        peak_times = [info['time'] for info in peak_info]
        peak_heights = [info['height'] for info in peak_info]
        peak_areas = [info['area'] for info in peak_info]

        axes[2, 1].scatter(peak_heights, peak_areas, alpha=0.7, s=60)
        axes[2, 1].set_title('波峰高度 vs 面积')
        axes[2, 1].set_xlabel('波峰高度')
        axes[2, 1].set_ylabel('波峰面积')
        axes[2, 1].grid(True, alpha=0.3)

        # 添加趋势线
        if len(peak_heights) > 2:
            z = np.polyfit(peak_heights, peak_areas, 1)
            p = np.poly1d(z)
            axes[2, 1].plot(peak_heights, p(peak_heights), "r--", alpha=0.8, label=f'趋势线: y={z[0]:.2f}x+{z[1]:.2f}')
            axes[2, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'two_layer_peak_analysis_{column_name}.png'),
                   dpi=300, bbox_inches='tight')

        # 打印波峰分析结果
        print(f"\n=== 波峰分析结果 - {column_name} ===")
        print(f"检测到 {len(peak_info)} 个波峰")

        for i, info in enumerate(peak_info[:10]):  # 只显示前10个
            print(f"波峰 {i+1}: 时间={info['time']:.1f}秒, 高度={info['height']:.2f}, 面积={info['area']:.2f}")

        if len(peak_info) > 10:
            print(f"... 还有 {len(peak_info) - 10} 个波峰")

        # 计算面积统计信息
        if peak_areas:
            print("\n面积统计:")
            print(f"  平均面积: {np.mean(peak_areas):.2f}")
            print(f"  最大面积: {np.max(peak_areas):.2f}")
            print(f"  最小面积: {np.min(peak_areas):.2f}")
            print(f"  面积标准差: {np.std(peak_areas):.2f}")

        return peak_info, area_time_series
    
    def test_gaussian_filter_intensity(self, column_name: str = 'burr_density'):
        """
        测试高斯滤波的不同强度
        
        Args:
            column_name: 要测试的列名
        """
        print(f"\n=== 测试高斯滤波强度 - {column_name} ===")
        
        # 获取数据
        original_data = self.data[column_name].values
        
        # 定义不同的sigma值
        sigma_values = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
        
        # 应用滤波
        filtered_results = self.apply_gaussian_filter(original_data, sigma_values)
        
        # 计算能量变化
        energy_analysis = {}
        smoothness_analysis = {}
        
        for sigma, filtered_data in filtered_results.items():
            energy_reduction = self.calculate_energy_reduction(original_data, filtered_data)
            smoothness = self.calculate_smoothness(filtered_data)
            
            energy_analysis[sigma] = {
                'energy_reduction': energy_reduction,
                'original_energy': self.calculate_energy(original_data),
                'filtered_energy': self.calculate_energy(filtered_data)
            }
            
            smoothness_analysis[sigma] = smoothness
            
            print(f"Sigma = {sigma:4.1f}: 能量减少 = {energy_reduction:.3f}, 平滑度 = {smoothness:.3f}")
        
        return filtered_results, energy_analysis, smoothness_analysis
    
    def test_moving_average_filter_intensity(self, column_name: str = 'burr_density'):
        """
        测试移动平均滤波的不同强度
        
        Args:
            column_name: 要测试的列名
        """
        print(f"\n=== 测试移动平均滤波强度 - {column_name} ===")
        
        # 获取数据
        original_data = self.data[column_name].values
        
        # 定义不同的窗口大小
        window_sizes = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
        
        # 应用滤波
        filtered_results = self.apply_moving_average_filter(original_data, window_sizes)
        
        # 计算能量变化
        energy_analysis = {}
        smoothness_analysis = {}
        
        for window_size, filtered_data in filtered_results.items():
            energy_reduction = self.calculate_energy_reduction(original_data, filtered_data)
            smoothness = self.calculate_smoothness(filtered_data)
            
            energy_analysis[window_size] = {
                'energy_reduction': energy_reduction,
                'original_energy': self.calculate_energy(original_data),
                'filtered_energy': self.calculate_energy(filtered_data)
            }
            
            smoothness_analysis[window_size] = smoothness
            
            print(f"窗口大小 = {window_size:3d}: 能量减少 = {energy_reduction:.3f}, 平滑度 = {smoothness:.3f}")
        
        return filtered_results, energy_analysis, smoothness_analysis
    
    def test_savgol_filter_intensity(self, column_name: str = 'burr_density'):
        """
        测试Savitzky-Golay滤波的不同强度
        
        Args:
            column_name: 要测试的列名
        """
        print(f"\n=== 测试Savitzky-Golay滤波强度 - {column_name} ===")
        
        # 获取数据
        original_data = self.data[column_name].values
        
        # 定义不同的窗口长度
        window_lengths = [5, 9, 13, 17, 21, 25, 31, 41, 51, 61, 81, 101]
        
        # 应用滤波
        filtered_results = self.apply_savgol_filter(original_data, window_lengths)
        
        # 计算能量变化
        energy_analysis = {}
        smoothness_analysis = {}
        
        for window_length, filtered_data in filtered_results.items():
            energy_reduction = self.calculate_energy_reduction(original_data, filtered_data)
            smoothness = self.calculate_smoothness(filtered_data)
            
            energy_analysis[window_length] = {
                'energy_reduction': energy_reduction,
                'original_energy': self.calculate_energy(original_data),
                'filtered_energy': self.calculate_energy(filtered_data)
            }
            
            smoothness_analysis[window_length] = smoothness
            
            print(f"窗口长度 = {window_length:3d}: 能量减少 = {energy_reduction:.3f}, 平滑度 = {smoothness:.3f}")
        
        return filtered_results, energy_analysis, smoothness_analysis
    
    def create_gaussian_comparison_plot(self, column_name: str, output_dir: str = "test_output"):
        """创建高斯滤波对比图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取时间轴数据和原始数据
        time_seconds = self.data['time_seconds'].values
        original_data = self.data[column_name].values
        
        # 测试高斯滤波
        gaussian_results, gaussian_energy, gaussian_smoothness = self.test_gaussian_filter_intensity(column_name)
        
        # 创建高斯滤波对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'高斯滤波强度测试 - {column_name}', fontsize=16)
        
        # 原始数据
        axes[0, 0].plot(time_seconds, original_data, 'b-', alpha=0.7, linewidth=1, label='原始数据')
        axes[0, 0].set_title('原始数据')
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel(column_name)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 不同sigma值的滤波结果
        colors = plt.cm.viridis(np.linspace(0, 1, len(gaussian_results)))
        for i, (sigma, filtered_data) in enumerate(gaussian_results.items()):
            axes[0, 1].plot(time_seconds, filtered_data, color=colors[i], 
                           linewidth=1.5, alpha=0.8, label=f'σ={sigma}')
        axes[0, 1].set_title('不同强度的滤波结果')
        axes[0, 1].set_xlabel('时间 (秒)')
        axes[0, 1].set_ylabel(column_name)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 能量减少分析
        sigmas = list(gaussian_energy.keys())
        energy_reductions = [gaussian_energy[s]['energy_reduction'] for s in sigmas]
        axes[1, 0].plot(sigmas, energy_reductions, 'ro-', linewidth=2, markersize=6)
        axes[1, 0].set_title('能量减少 vs 滤波强度')
        axes[1, 0].set_xlabel('Sigma值')
        axes[1, 0].set_ylabel('能量减少比例')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 平滑度分析
        smoothness_values = [gaussian_smoothness[s] for s in sigmas]
        axes[1, 1].plot(sigmas, smoothness_values, 'go-', linewidth=2, markersize=6)
        axes[1, 1].set_title('平滑度 vs 滤波强度')
        axes[1, 1].set_xlabel('Sigma值')
        axes[1, 1].set_ylabel('平滑度指标')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gaussian_filter_comparison_{column_name}.png'), 
                   dpi=300, bbox_inches='tight')
        
        return gaussian_results, gaussian_energy, gaussian_smoothness
    
    def create_comprehensive_comparison(self, column_name: str, output_dir: str = "test_output"):
        """创建综合对比图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取时间轴数据和原始数据
        time_seconds = self.data['time_seconds'].values
        original_data = self.data[column_name].values
        
        # 测试三种滤波方法
        gaussian_results, _, _ = self.test_gaussian_filter_intensity(column_name)
        ma_results, _, _ = self.test_moving_average_filter_intensity(column_name)
        sg_results, _, _ = self.test_savgol_filter_intensity(column_name)
        
        # 创建综合对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'滤波方法综合对比 - {column_name}', fontsize=16)
        
        # 原始数据
        axes[0, 0].plot(time_seconds, original_data, 'b-', alpha=0.7, linewidth=1, label='原始数据')
        axes[0, 0].set_title('原始数据')
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel(column_name)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 高斯滤波 - 中等强度
        gaussian_medium = gaussian_results[10.0]  # sigma=10
        axes[0, 1].plot(time_seconds, original_data, 'b-', alpha=0.5, linewidth=0.8, label='原始')
        axes[0, 1].plot(time_seconds, gaussian_medium, 'r-', linewidth=2, label='高斯滤波 (σ=10)')
        axes[0, 1].set_title('高斯滤波效果')
        axes[0, 1].set_xlabel('时间 (秒)')
        axes[0, 1].set_ylabel(column_name)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 移动平均滤波 - 中等窗口
        ma_medium = ma_results[25]  # 窗口=25
        axes[1, 0].plot(time_seconds, original_data, 'b-', alpha=0.5, linewidth=0.8, label='原始')
        axes[1, 0].plot(time_seconds, ma_medium, 'g-', linewidth=2, label='移动平均 (窗口=25)')
        axes[1, 0].set_title('移动平均滤波效果')
        axes[1, 0].set_xlabel('时间 (秒)')
        axes[1, 0].set_ylabel(column_name)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Savitzky-Golay滤波 - 中等窗口
        sg_medium = sg_results[21]  # 窗口=21
        axes[1, 1].plot(time_seconds, original_data, 'b-', alpha=0.5, linewidth=0.8, label='原始')
        axes[1, 1].plot(time_seconds, sg_medium, 'm-', linewidth=2, label='Savitzky-Golay (窗口=21)')
        axes[1, 1].set_title('Savitzky-Golay滤波效果')
        axes[1, 1].set_xlabel('时间 (秒)')
        axes[1, 1].set_ylabel(column_name)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comprehensive_filter_comparison_{column_name}.png'), 
                   dpi=300, bbox_inches='tight')
        
        # 打印推荐参数
        print(f"\n=== 滤波参数推荐 - {column_name} ===")
        print("高斯滤波推荐参数:")
        print("  - 轻度平滑: σ=2.0-5.0")
        print("  - 中度平滑: σ=10.0-15.0")
        print("  - 强度平滑: σ=20.0-30.0")
        
        print("\n移动平均滤波推荐参数:")
        print("  - 轻度平滑: 窗口=5-15")
        print("  - 中度平滑: 窗口=20-30")
        print("  - 强度平滑: 窗口=40-60")
        
        print("\nSavitzky-Golay滤波推荐参数:")
        print("  - 轻度平滑: 窗口=5-13")
        print("  - 中度平滑: 窗口=17-25")
        print("  - 强度平滑: 窗口=31-51")

def main():
    """主函数"""
    # 设置中文字体
    setup_chinese_font()
    
    # 数据文件路径
    data_path = "../data_burr_density_curve/burr_density_analysis.csv"
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return
    
    # 创建测试器
    tester = FilterIntensityTester(data_path)
    
    # 测试的列名
    columns_to_test = ['burr_density', 'num_burrs', 'tear_region_density']
    
    # 对每个列进行测试
    for column in columns_to_test:
        if column in tester.data.columns:
            print(f"\n{'='*60}")
            print(f"开始测试列: {column}")
            print(f"{'='*60}")
            
            # 创建两层滤波及波峰分析
            peak_info, area_time_series = tester.create_two_layer_filter_analysis(column)
            
            # 可选：创建高斯滤波对比图
            # tester.create_gaussian_comparison_plot(column)
            
            # 可选：创建综合对比图
            # tester.create_comprehensive_comparison(column)
        else:
            print(f"警告: 列 '{column}' 不存在于数据中")

if __name__ == "__main__":
    main()

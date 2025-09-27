#!/usr/bin/env python3
"""
梯度增强分析器
主要功能：
1. 使用多种方法增强时间序列的变化梯度
2. 突出显示快速变化的区域
3. 生成梯度增强的对比图表
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
from scipy.ndimage import gaussian_filter1d, laplace
from scipy.signal import butter, filtfilt, savgol_filter

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

class GradientEnhancementAnalyzer:
    """梯度增强分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.data = []
    
    def apply_gradient_enhancement(self, data: List[Dict[str, Any]], 
                                 enhancement_method: str = 'high_pass',
                                 **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对时间序列数据应用梯度增强
        
        Args:
            data: 斑块分析数据
            enhancement_method: 增强方法
            **kwargs: 方法特定参数
            
        Returns:
            时间序列、增强后的斑块数量、增强后的斑块密度
        """
        time_seconds = np.array([d['time_seconds'] for d in data])
        spot_counts = np.array([d['spot_count'] for d in data])
        spot_densities = np.array([d['spot_density'] for d in data])
        
        if enhancement_method == 'high_pass':
            # 高通滤波器 - 突出快速变化
            cutoff = kwargs.get('cutoff', 0.1)  # 归一化截止频率
            order = kwargs.get('order', 4)
            enhanced_counts = self._high_pass_filter(spot_counts, cutoff, order)
            enhanced_densities = self._high_pass_filter(spot_densities, cutoff, order)
            
        elif enhancement_method == 'derivative':
            # 一阶导数 - 直接计算变化率
            enhanced_counts = np.gradient(spot_counts)
            enhanced_densities = np.gradient(spot_densities)
            
        elif enhancement_method == 'second_derivative':
            # 二阶导数 - 突出加速度变化
            enhanced_counts = np.gradient(np.gradient(spot_counts))
            enhanced_densities = np.gradient(np.gradient(spot_densities))
            
        elif enhancement_method == 'sharpening':
            # 锐化滤波器
            alpha = kwargs.get('alpha', 0.3)  # 锐化强度
            enhanced_counts = self._sharpening_filter(spot_counts, alpha)
            enhanced_densities = self._sharpening_filter(spot_densities, alpha)
            
        elif enhancement_method == 'laplacian':
            # 拉普拉斯算子 - 突出边缘和变化
            enhanced_counts = laplace(spot_counts)
            enhanced_densities = laplace(spot_densities)
            
        elif enhancement_method == 'sobel':
            # Sobel算子 - 梯度检测
            enhanced_counts = self._sobel_gradient(spot_counts)
            enhanced_densities = self._sobel_gradient(spot_densities)
            
        elif enhancement_method == 'difference':
            # 差分增强
            window = kwargs.get('window', 5)
            enhanced_counts = self._difference_enhancement(spot_counts, window)
            enhanced_densities = self._difference_enhancement(spot_densities, window)
            
        elif enhancement_method == 'wavelet':
            # 小波变换增强高频
            level = kwargs.get('level', 2)
            enhanced_counts = self._wavelet_enhancement(spot_counts, level)
            enhanced_densities = self._wavelet_enhancement(spot_densities, level)
            
        else:
            # 默认使用高通滤波
            enhanced_counts = self._high_pass_filter(spot_counts, 0.1, 4)
            enhanced_densities = self._high_pass_filter(spot_densities, 0.1, 4)
        
        return time_seconds, enhanced_counts, enhanced_densities
    
    def _high_pass_filter(self, signal_data: np.ndarray, cutoff: float, order: int) -> np.ndarray:
        """高通滤波器"""
        # 归一化截止频率
        nyquist = 0.5
        normal_cutoff = cutoff / nyquist
        
        # 设计巴特沃斯高通滤波器
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        
        # 应用滤波器
        filtered_data = filtfilt(b, a, signal_data)
        
        # 增强幅度
        filtered_data = filtered_data * 2  # 增强因子
        
        return filtered_data
    
    def _sharpening_filter(self, signal_data: np.ndarray, alpha: float) -> np.ndarray:
        """锐化滤波器"""
        # 高斯平滑
        smoothed = gaussian_filter1d(signal_data, sigma=2)
        
        # 锐化：原信号 + alpha * (原信号 - 平滑信号)
        sharpened = signal_data + alpha * (signal_data - smoothed)
        
        return sharpened
    
    def _sobel_gradient(self, signal_data: np.ndarray) -> np.ndarray:
        """Sobel梯度算子"""
        # 1D Sobel算子 [1, 0, -1]
        sobel_kernel = np.array([1, 0, -1])
        
        # 应用卷积
        gradient = np.convolve(signal_data, sobel_kernel, mode='same')
        
        # 取绝对值并增强
        gradient = np.abs(gradient) * 2
        
        return gradient
    
    def _difference_enhancement(self, signal_data: np.ndarray, window: int) -> np.ndarray:
        """差分增强"""
        enhanced = np.zeros_like(signal_data)
        
        for i in range(window, len(signal_data) - window):
            # 计算前后窗口的差异
            front_avg = np.mean(signal_data[i:i+window])
            back_avg = np.mean(signal_data[i-window:i])
            enhanced[i] = (front_avg - back_avg) * 2
        
        return enhanced
    
    def _wavelet_enhancement(self, signal_data: np.ndarray, level: int) -> np.ndarray:
        """小波变换增强"""
        try:
            import pywt
            
            # 进行小波分解
            coeffs = pywt.wavedec(signal_data, 'db4', level=level)
            
            # 增强高频系数
            coeffs_enhanced = []
            for i, coeff in enumerate(coeffs):
                if i == 0:  # 低频系数保持
                    coeffs_enhanced.append(coeff)
                else:  # 高频系数增强
                    coeffs_enhanced.append(coeff * 2)
            
            # 重构信号
            enhanced = pywt.waverec(coeffs_enhanced, 'db4')
            
            return enhanced
            
        except ImportError:
            print("警告：pywt未安装，使用差分增强替代")
            return self._difference_enhancement(signal_data, 5)
    
    def create_gradient_plots(self, data: List[Dict[str, Any]], output_dir: str = "output",
                            enhancement_method: str = 'high_pass', **kwargs):
        """
        创建梯度增强图表
        
        Args:
            data: 斑块分析数据
            output_dir: 输出目录
            enhancement_method: 增强方法
            **kwargs: 方法特定参数
        """
        if not data:
            print("没有数据可以绘制")
            return
        
        # 设置中文字体
        font_success = setup_chinese_font()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取原始数据
        time_seconds = np.array([d['time_seconds'] for d in data])
        spot_counts = np.array([d['spot_count'] for d in data])
        spot_densities = np.array([d['spot_density'] for d in data])
        
        # 应用梯度增强
        _, enhanced_counts, enhanced_densities = self.apply_gradient_enhancement(
            data, enhancement_method, **kwargs)
        
        # 创建图表
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16))
        
        # 原始斑块数量
        ax1.plot(time_seconds, spot_counts, 'b-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax1.set_ylabel('斑块数量' if font_success else 'Spot Count')
        ax1.set_title('原始斑块数量' if font_success else 'Original Spot Count')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(time_seconds))
        
        # 增强后的斑块数量
        ax2.plot(time_seconds, enhanced_counts, 'r-', linewidth=2, alpha=0.9)
        ax2.fill_between(time_seconds, enhanced_counts, alpha=0.3, color='red')
        ax2.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax2.set_ylabel('增强斑块数量' if font_success else 'Enhanced Spot Count')
        ax2.set_title('梯度增强斑块数量' if font_success else 'Gradient Enhanced Spot Count')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(time_seconds))
        
        # 原始斑块密度
        ax3.plot(time_seconds, spot_densities, 'g-', linewidth=2, alpha=0.8)
        ax3.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax3.set_ylabel('斑块密度' if font_success else 'Spot Density')
        ax3.set_title('原始斑块密度' if font_success else 'Original Spot Density')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max(time_seconds))
        
        # 增强后的斑块密度
        ax4.plot(time_seconds, enhanced_densities, 'm-', linewidth=2, alpha=0.9)
        ax4.fill_between(time_seconds, enhanced_densities, alpha=0.3, color='magenta')
        ax4.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax4.set_ylabel('增强斑块密度' if font_success else 'Enhanced Spot Density')
        ax4.set_title('梯度增强斑块密度' if font_success else 'Gradient Enhanced Spot Density')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, max(time_seconds))
        
        # 添加方法说明
        method_names = {
            'high_pass': '高通滤波',
            'derivative': '一阶导数',
            'second_derivative': '二阶导数',
            'sharpening': '锐化滤波',
            'laplacian': '拉普拉斯算子',
            'sobel': 'Sobel梯度',
            'difference': '差分增强',
            'wavelet': '小波变换'
        }
        method_name = method_names.get(enhancement_method, enhancement_method)
        
        # 添加参数说明
        param_str = ""
        if 'cutoff' in kwargs:
            param_str += f" 截止频率={kwargs['cutoff']}"
        if 'alpha' in kwargs:
            param_str += f" 锐化强度={kwargs['alpha']}"
        if 'window' in kwargs:
            param_str += f" 窗口={kwargs['window']}"
        
        fig.suptitle(f'梯度增强方法: {method_name}{param_str}', fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.96)
        
        # 保存图表
        plot_path = os.path.join(output_dir, f"gradient_enhanced_{enhancement_method}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"梯度增强图表已保存: {plot_path}")
        
        return plot_path
    
    def run_comparison_analysis(self, data_file: str = "output/temporal_analysis/spot_temporal_data.csv"):
        """运行梯度增强对比分析"""
        
        print("=== 梯度增强对比分析 ===")
        
        # 读取数据
        if not os.path.exists(data_file):
            print(f"数据文件不存在: {data_file}")
            return
        
        df = pd.read_csv(data_file)
        data = df.to_dict('records')
        
        print(f"读取了 {len(data)} 个数据点")
        
        # 定义不同的梯度增强方法
        enhancement_configs = [
            {'method': 'high_pass', 'params': {'cutoff': 0.05, 'order': 4}, 'name': '高通滤波 (截止=0.05)'},
            {'method': 'high_pass', 'params': {'cutoff': 0.1, 'order': 4}, 'name': '高通滤波 (截止=0.1)'},
            {'method': 'high_pass', 'params': {'cutoff': 0.2, 'order': 4}, 'name': '高通滤波 (截止=0.2)'},
            {'method': 'derivative', 'params': {}, 'name': '一阶导数'},
            {'method': 'second_derivative', 'params': {}, 'name': '二阶导数'},
            {'method': 'sharpening', 'params': {'alpha': 0.2}, 'name': '锐化滤波 (α=0.2)'},
            {'method': 'sharpening', 'params': {'alpha': 0.5}, 'name': '锐化滤波 (α=0.5)'},
            {'method': 'laplacian', 'params': {}, 'name': '拉普拉斯算子'},
            {'method': 'sobel', 'params': {}, 'name': 'Sobel梯度'},
            {'method': 'difference', 'params': {'window': 3}, 'name': '差分增强 (窗口=3)'},
            {'method': 'difference', 'params': {'window': 5}, 'name': '差分增强 (窗口=5)'},
        ]
        
        # 为每种方法生成图表
        for i, config in enumerate(enhancement_configs):
            print(f"\n生成方法 {i+1}/{len(enhancement_configs)}: {config['name']}")
            
            output_dir = f"output/gradient_enhancement/{config['method']}_{i+1:02d}"
            
            try:
                plot_path = self.create_gradient_plots(
                    data, 
                    output_dir,
                    enhancement_method=config['method'],
                    **config['params']
                )
                print(f"✓ 成功生成: {plot_path}")
                
            except Exception as e:
                print(f"✗ 生成失败: {e}")
        
        print(f"\n✅ 梯度增强对比分析完成！")
        print("📁 所有对比图表保存在 output/gradient_enhancement/ 目录下")
        print("\n🎯 推荐方法：")
        print("  - 高通滤波 (截止=0.1-0.2)：突出快速变化")
        print("  - 一阶导数：直接显示变化率")
        print("  - 锐化滤波 (α=0.5)：增强边缘和突变")
        print("  - Sobel梯度：检测变化边界")


def main():
    """主函数"""
    analyzer = GradientEnhancementAnalyzer()
    analyzer.run_comparison_analysis()


if __name__ == "__main__":
    main()

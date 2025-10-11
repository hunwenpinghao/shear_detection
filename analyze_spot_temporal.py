#!/usr/bin/env python3
"""
撕裂面斑块数量和密度时间序列分析
主要功能：
1. 从斑块检测结果中提取斑块数量和密度数据
2. 绘制斑块数量和密度随时间变化的曲线图
3. 生成时间序列分析报告
4. 支持命令行参数配置

用法:
    python analyze_spot_temporal.py --roi_dir data/roi_imgs --output_dir output
    
示例:
    # 基本用法
    python analyze_spot_temporal.py --roi_dir data/roi_imgs --output_dir output/spot_analysis
    
    # 自定义平滑参数
    python analyze_spot_temporal.py --roi_dir data/roi_imgs --output_dir output \
        --smoothing_method savgol --window_size 100 --sigma 15.0
    
    # 保存每一帧的可视化（每100帧）
    python analyze_spot_temporal.py --roi_dir data/roi_imgs --output_dir output \
        --viz_interval 100
"""

import cv2
import numpy as np
import os
import glob
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
import platform
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))

from feature_extractor import FeatureExtractor
from config import PREPROCESS_CONFIG

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

class SpotTemporalAnalyzer:
    """斑块时间序列分析器"""
    
    def __init__(self, viz_interval: int = None):
        """
        初始化分析器
        
        Args:
            viz_interval: 可视化采样间隔，每隔多少帧保存一次标注图（默认None=不保存）
        """
        self.feature_extractor = FeatureExtractor(PREPROCESS_CONFIG)
        self.data = []
        self.viz_interval = viz_interval
        
    def extract_frame_info(self, filename: str) -> int:
        """从文件名提取帧号"""
        try:
            basename = os.path.basename(filename)
            # 提取frame_XXXXXX中的数字
            frame_num = int(basename.split('_')[1])
            return frame_num
        except (IndexError, ValueError):
            return -1
    
    def analyze_roi_spots(self, roi_dir: str, output_dir: str = "output") -> List[Dict[str, Any]]:
        """
        分析ROI图像的斑块特征
        
        Args:
            roi_dir: ROI图像目录路径
            output_dir: 输出目录（用于保存可视化）
            
        Returns:
            斑块分析结果列表
        """
        print("开始分析ROI图像的斑块特征...")
        if self.viz_interval:
            print(f"可视化采样间隔: 每 {self.viz_interval} 帧")
        
        # 获取所有ROI图像文件
        roi_pattern = os.path.join(roi_dir, "*_roi.png")
        roi_files = sorted(glob.glob(roi_pattern), key=self.extract_frame_info)
        
        if not roi_files:
            print(f"在目录 {roi_dir} 中未找到ROI图像文件")
            return []
        
        print(f"找到 {len(roi_files)} 个ROI图像文件")
        
        # 创建可视化目录
        viz_count = 0
        if self.viz_interval:
            viz_dir = os.path.join(output_dir, 'spot_visualizations')
            os.makedirs(viz_dir, exist_ok=True)
        
        results = []
        for i, roi_file in enumerate(tqdm(roi_files, desc="分析斑块")):
            frame_num = self.extract_frame_info(roi_file)
            if frame_num == -1:
                continue
                
            try:
                # 读取ROI图像
                roi_image = cv2.imread(roi_file, cv2.IMREAD_GRAYSCALE)
                if roi_image is None:
                    continue
                
                # 检测斑块
                spot_result = self.feature_extractor.detect_all_white_spots(roi_image)
                
                # 提取关键信息
                result = {
                    'frame_num': frame_num,
                    'time_seconds': frame_num * 5,  # 假设每5秒一帧
                    'spot_count': spot_result.get('all_spot_count', 0),
                    'spot_density': spot_result.get('all_spot_density', 0.0),
                    'image_shape': roi_image.shape,
                    'roi_file': roi_file
                }
                
                results.append(result)
                
                # 保存可视化（按采样间隔）
                if self.viz_interval and i % self.viz_interval == 0:
                    self._save_spot_visualization(
                        roi_image, spot_result, frame_num, viz_dir
                    )
                    viz_count += 1
                    
            except Exception as e:
                print(f"\n分析ROI图像 {roi_file} 时出错: {e}")
                continue
        
        print(f"\n斑块分析完成，成功分析 {len(results)} 个ROI图像")
        if self.viz_interval:
            print(f"已保存可视化: {viz_count} 张（采样间隔: {self.viz_interval}）")
        
        return results
    
    def _save_spot_visualization(self, roi_image: np.ndarray, spot_result: dict, 
                                frame_num: int, viz_dir: str):
        """
        保存斑块检测可视化结果
        
        Args:
            roi_image: ROI图像
            spot_result: 斑块检测结果
            frame_num: 帧号
            viz_dir: 可视化输出目录
        """
        try:
            # 创建彩色图像用于标注
            vis_image = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2RGB)
            
            # 获取所有斑块的轮廓
            if 'all_spot_contours' in spot_result and spot_result['all_spot_contours']:
                # 绘制斑块轮廓
                cv2.drawContours(vis_image, spot_result['all_spot_contours'], -1, 
                               (0, 255, 0), 2)  # 绿色轮廓
                
                # 标注斑块中心和编号
                for idx, contour in enumerate(spot_result['all_spot_contours'], 1):
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # 绘制中心点
                        cv2.circle(vis_image, (cx, cy), 3, (255, 0, 0), -1)  # 蓝色中心点
                        # 标注编号（只标注前50个，避免太密集）
                        if idx <= 50:
                            cv2.putText(vis_image, str(idx), (cx+5, cy-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 添加统计信息
            spot_count = spot_result.get('all_spot_count', 0)
            spot_density = spot_result.get('all_spot_density', 0.0)
            
            info_text = [
                f"Frame: {frame_num}",
                f"Spot Count: {spot_count}",
                f"Density: {spot_density:.4f}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(vis_image, text, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 30
            
            # 保存图像
            viz_path = os.path.join(viz_dir, f"frame_{frame_num:06d}_spots.png")
            cv2.imwrite(viz_path, vis_image)
            
        except Exception as e:
            print(f"\n保存可视化失败 frame {frame_num}: {e}")
    
    def apply_smoothing_filters(self, data: List[Dict[str, Any]], 
                               smoothing_method: str = 'gaussian',
                               window_size: int = 50,
                               sigma: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对时间序列数据应用平滑滤波
        
        Args:
            data: 斑块分析数据
            smoothing_method: 平滑方法 ('gaussian', 'moving_avg', 'savgol', 'median')
            window_size: 滤波窗口大小
            sigma: 高斯滤波的标准差
            
        Returns:
            时间序列、平滑后的斑块数量、平滑后的斑块密度
        """
        time_seconds = np.array([d['time_seconds'] for d in data])
        spot_counts = np.array([d['spot_count'] for d in data])
        spot_densities = np.array([d['spot_density'] for d in data])
        
        if smoothing_method == 'gaussian':
            # 高斯滤波
            smoothed_counts = gaussian_filter1d(spot_counts, sigma=sigma)
            smoothed_densities = gaussian_filter1d(spot_densities, sigma=sigma)
            
        elif smoothing_method == 'moving_avg':
            # 移动平均滤波
            smoothed_counts = np.convolve(spot_counts, np.ones(window_size)/window_size, mode='same')
            smoothed_densities = np.convolve(spot_densities, np.ones(window_size)/window_size, mode='same')
            
        elif smoothing_method == 'savgol':
            # Savitzky-Golay滤波
            window_length = min(window_size, len(spot_counts))
            if window_length % 2 == 0:
                window_length -= 1
            smoothed_counts = signal.savgol_filter(spot_counts, window_length, 3)
            smoothed_densities = signal.savgol_filter(spot_densities, window_length, 3)
            
        elif smoothing_method == 'median':
            # 中值滤波
            smoothed_counts = signal.medfilt(spot_counts, kernel_size=window_size)
            smoothed_densities = signal.medfilt(spot_densities, kernel_size=window_size)
            
        else:
            # 默认使用高斯滤波
            smoothed_counts = gaussian_filter1d(spot_counts, sigma=sigma)
            smoothed_densities = gaussian_filter1d(spot_densities, sigma=sigma)
        
        return time_seconds, smoothed_counts, smoothed_densities
    
    def create_temporal_plots(self, data: List[Dict[str, Any]], output_dir: str = "output",
                            smoothing_method: str = 'gaussian', window_size: int = 50, sigma: float = 10.0):
        """
        创建时间序列图表（带平滑滤波）
        
        Args:
            data: 斑块分析数据
            output_dir: 输出目录
            smoothing_method: 平滑方法
            window_size: 滤波窗口大小
            sigma: 高斯滤波标准差
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
        
        # 应用平滑滤波
        _, smoothed_counts, smoothed_densities = self.apply_smoothing_filters(
            data, smoothing_method, window_size, sigma)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 斑块数量随时间变化（原始数据+平滑曲线）
        ax1.plot(time_seconds, spot_counts, 'b-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax1.plot(time_seconds, smoothed_counts, 'b-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax1.fill_between(time_seconds, smoothed_counts, alpha=0.3, color='blue')
        ax1.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax1.set_ylabel('斑块数量' if font_success else 'Spot Count')
        ax1.set_title('斑块数量随时间变化 (平滑滤波)' if font_success else 'Spot Count Over Time (Smoothed)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_count = np.mean(spot_counts)
        max_count = np.max(smoothed_counts)
        ax1.axhline(y=mean_count, color='red', linestyle='--', alpha=0.7, 
                   label=f'平均值: {mean_count:.1f}')
        ax1.legend()
        
        # 斑块密度随时间变化（原始数据+平滑曲线）
        ax2.plot(time_seconds, spot_densities, 'r-', linewidth=0.8, alpha=0.3, label='原始数据')
        ax2.plot(time_seconds, smoothed_densities, 'r-', linewidth=2.5, alpha=0.9, label='平滑曲线')
        ax2.fill_between(time_seconds, smoothed_densities, alpha=0.3, color='red')
        ax2.set_xlabel('时间 (秒)' if font_success else 'Time (seconds)')
        ax2.set_ylabel('斑块密度' if font_success else 'Spot Density')
        ax2.set_title('斑块密度随时间变化 (平滑滤波)' if font_success else 'Spot Density Over Time (Smoothed)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_density = np.mean(spot_densities)
        max_density = np.max(smoothed_densities)
        ax2.axhline(y=mean_density, color='blue', linestyle='--', alpha=0.7,
                   label=f'平均值: {mean_density:.4f}')
        ax2.legend()
        
        # 添加滤波方法说明
        method_names = {
            'gaussian': '高斯滤波',
            'moving_avg': '移动平均',
            'savgol': 'Savitzky-Golay滤波',
            'median': '中值滤波'
        }
        method_name = method_names.get(smoothing_method, smoothing_method)
        fig.suptitle(f'平滑方法: {method_name} (σ={sigma}, 窗口={window_size})', fontsize=12, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 保存图表
        plot_path = os.path.join(output_dir, "spot_temporal_analysis_smoothed.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"平滑时间序列图表已保存: {plot_path}")
        
        # 创建统计摘要图
        self.create_statistics_summary(data, output_dir)
        
        return plot_path
    
    def create_statistics_summary(self, data: List[Dict[str, Any]], output_dir: str):
        """
        创建统计摘要图表
        
        Args:
            data: 斑块分析数据
            output_dir: 输出目录
        """
        if not data:
            return
        
        # 设置中文字体
        font_success = setup_chinese_font()
        
        # 提取数据
        spot_counts = [d['spot_count'] for d in data]
        spot_densities = [d['spot_density'] for d in data]
        
        # 创建统计图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 斑块数量分布直方图
        ax1.hist(spot_counts, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('斑块数量' if font_success else 'Spot Count')
        ax1.set_ylabel('频次' if font_success else 'Frequency')
        ax1.set_title('斑块数量分布' if font_success else 'Spot Count Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 斑块密度分布直方图
        ax2.hist(spot_densities, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax2.set_xlabel('斑块密度' if font_success else 'Spot Density')
        ax2.set_ylabel('频次' if font_success else 'Frequency')
        ax2.set_title('斑块密度分布' if font_success else 'Spot Density Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 斑块数量与密度的散点图
        ax3.scatter(spot_counts, spot_densities, alpha=0.6, color='green')
        ax3.set_xlabel('斑块数量' if font_success else 'Spot Count')
        ax3.set_ylabel('斑块密度' if font_success else 'Spot Density')
        ax3.set_title('斑块数量与密度关系' if font_success else 'Spot Count vs Density')
        ax3.grid(True, alpha=0.3)
        
        # 计算相关系数
        correlation = np.corrcoef(spot_counts, spot_densities)[0, 1]
        ax3.text(0.05, 0.95, f'相关系数: {correlation:.3f}', 
                transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 统计信息表格
        ax4.axis('off')
        stats_text = f"""
统计摘要:

斑块数量:
  平均值: {np.mean(spot_counts):.2f}
  标准差: {np.std(spot_counts):.2f}
  最小值: {np.min(spot_counts)}
  最大值: {np.max(spot_counts)}
  中位数: {np.median(spot_counts):.2f}

斑块密度:
  平均值: {np.mean(spot_densities):.6f}
  标准差: {np.std(spot_densities):.6f}
  最小值: {np.min(spot_densities):.6f}
  最大值: {np.max(spot_densities):.6f}
  中位数: {np.median(spot_densities):.6f}

数据点总数: {len(data)}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存统计图表
        stats_path = os.path.join(output_dir, "spot_statistics_summary.png")
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"统计摘要图表已保存: {stats_path}")
    
    def save_data_to_csv(self, data: List[Dict[str, Any]], output_dir: str = "output"):
        """
        保存数据到CSV文件
        
        Args:
            data: 斑块分析数据
            output_dir: 输出目录
        """
        if not data:
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 保存CSV文件
        csv_path = os.path.join(output_dir, "spot_temporal_data.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"数据已保存到CSV文件: {csv_path}")
        
        # 保存JSON格式
        json_path = os.path.join(output_dir, "spot_temporal_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"数据已保存到JSON文件: {json_path}")
    
    def run_analysis(self, roi_dir: str = "data/roi_imgs", output_dir: str = "output",
                    smoothing_method: str = 'gaussian', window_size: int = 50, sigma: float = 10.0):
        """
        运行完整的斑块时间序列分析（带平滑滤波）
        
        Args:
            roi_dir: ROI图像目录路径
            output_dir: 输出目录
            smoothing_method: 平滑方法 ('gaussian', 'moving_avg', 'savgol', 'median')
            window_size: 滤波窗口大小
            sigma: 高斯滤波标准差
        """
        print("\n" + "="*80)
        print("ROI斑块时间序列分析 (整个ROI区域，不区分撕裂面和剪切面)")
        print("="*80)
        print(f"ROI目录: {roi_dir}")
        print(f"输出目录: {output_dir}")
        print(f"平滑方法: {smoothing_method}")
        if self.viz_interval:
            print(f"可视化间隔: 每 {self.viz_interval} 帧")
        
        # 分析ROI图像的斑块特征
        data = self.analyze_roi_spots(roi_dir, output_dir)
        
        if not data:
            print("没有可分析的数据")
            return
        
        # 创建时间序列图表（带平滑滤波）
        plot_path = self.create_temporal_plots(data, output_dir, smoothing_method, window_size, sigma)
        
        # 保存数据
        self.save_data_to_csv(data, output_dir)
        
        # 输出统计摘要
        self.print_statistics_summary(data)
        
        print(f"\n{'='*80}")
        print("✅ 斑块时间序列分析完成！")
        print(f"{'='*80}")
        print(f"📊 平滑图表: {plot_path}")
        print(f"📁 数据文件: {output_dir}")
        print(f"🔧 平滑方法: {smoothing_method}, 窗口大小: {window_size}, σ: {sigma}")
        if self.viz_interval:
            print(f"🖼️  可视化: {output_dir}/spot_visualizations/")
        
        return data
    
    def print_statistics_summary(self, data: List[Dict[str, Any]]):
        """打印统计摘要"""
        if not data:
            return
        
        spot_counts = [d['spot_count'] for d in data]
        spot_densities = [d['spot_density'] for d in data]
        
        print("\n=== 斑块时间序列统计摘要 ===")
        print(f"数据点总数: {len(data)}")
        print(f"时间跨度: {data[0]['time_seconds']:.1f} - {data[-1]['time_seconds']:.1f} 秒")
        print(f"帧数范围: {data[0]['frame_num']} - {data[-1]['frame_num']}")
        
        print("\n斑块数量统计:")
        print(f"  平均值: {np.mean(spot_counts):.2f}")
        print(f"  标准差: {np.std(spot_counts):.2f}")
        print(f"  最小值: {np.min(spot_counts)}")
        print(f"  最大值: {np.max(spot_counts)}")
        print(f"  中位数: {np.median(spot_counts):.2f}")
        
        print("\n斑块密度统计:")
        print(f"  平均值: {np.mean(spot_densities):.6f}")
        print(f"  标准差: {np.std(spot_densities):.6f}")
        print(f"  最小值: {np.min(spot_densities):.6f}")
        print(f"  最大值: {np.max(spot_densities):.6f}")
        print(f"  中位数: {np.median(spot_densities):.6f}")
        
        # 计算相关系数
        correlation = np.corrcoef(spot_counts, spot_densities)[0, 1]
        print(f"\n斑块数量与密度相关系数: {correlation:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='ROI斑块时间序列分析工具 - 统计整个ROI区域的斑块数量和密度变化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python analyze_spot_temporal.py --roi_dir data/roi_imgs --output_dir output/spot_analysis
  
  # 自定义平滑参数（Savitzky-Golay滤波）
  python analyze_spot_temporal.py --roi_dir data/roi_imgs --output_dir output \
    --smoothing_method savgol --window_size 100
  
  # 保存每一帧的可视化（每50帧）
  python analyze_spot_temporal.py --roi_dir data/roi_imgs --output_dir output \
    --viz_interval 50
  
  # 使用高斯滤波+自定义σ值
  python analyze_spot_temporal.py --roi_dir data/roi_imgs --output_dir output \
    --smoothing_method gaussian --sigma 15.0
  
  # 批量处理多个视频
  python analyze_spot_temporal.py --roi_dir video1/roi_imgs --output_dir video1/spot_analysis
  python analyze_spot_temporal.py --roi_dir video2/roi_imgs --output_dir video2/spot_analysis --viz_interval 100
        """
    )
    
    # 必需参数
    parser.add_argument('--roi_dir', type=str, default='data/roi_imgs',
                       help='ROI图像目录路径 (默认: data/roi_imgs)')
    parser.add_argument('--output_dir', type=str, default='output/spot_temporal',
                       help='输出目录路径 (默认: output/spot_temporal)')
    
    # 可选参数
    parser.add_argument('--smoothing_method', type=str, default='gaussian',
                       choices=['gaussian', 'moving_avg', 'savgol', 'median'],
                       help='平滑方法: gaussian=高斯滤波, moving_avg=移动平均, '
                            'savgol=Savitzky-Golay滤波, median=中值滤波 (默认: gaussian)')
    parser.add_argument('--window_size', type=int, default=50,
                       help='滤波窗口大小 (默认: 50)')
    parser.add_argument('--sigma', type=float, default=10.0,
                       help='高斯滤波的标准差σ (默认: 10.0)')
    parser.add_argument('--viz_interval', type=int, default=None,
                       help='可视化采样间隔，每隔多少帧保存一次斑块标注图 (默认: None=不保存)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.roi_dir):
        print(f"错误: ROI目录不存在: {args.roi_dir}")
        return 1
    
    # 初始化分析器
    analyzer = SpotTemporalAnalyzer(viz_interval=args.viz_interval)
    
    # 运行分析
    data = analyzer.run_analysis(
        roi_dir=args.roi_dir,
        output_dir=args.output_dir,
        smoothing_method=args.smoothing_method,
        window_size=args.window_size,
        sigma=args.sigma
    )
    
    if data:
        print(f"\n🎯 分析完成！共分析了 {len(data)} 个时间点的斑块数据")
        print("📈 生成了平滑时间序列曲线图和统计摘要")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

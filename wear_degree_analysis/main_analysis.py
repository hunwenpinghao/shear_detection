#!/usr/bin/env python3
"""
剪刀磨损程度综合分析系统 - 通用版
整合所有分析功能的主脚本

模型信息：
- 模型名称: Claude Sonnet 4.5
- 模型版本: claude-sonnet-4-20250514
- 更新日期: 2025年5月14日

使用方法：
  python main_analysis.py [选项]
  
示例：
  # 使用默认数据目录
  python main_analysis.py
  
  # 指定数据目录和输出目录
  python main_analysis.py --data_dir /path/to/data --output_dir /path/to/output
  
  # 只运行特定的分析模块
  python main_analysis.py --modules basic enhanced coil
  
  # 跳过特征提取，直接使用已有特征文件
  python main_analysis.py --skip_extraction --features_csv results/features/wear_features.csv
  
  # 指定钢卷参数
  python main_analysis.py --n_coils 12 --coil_start_id 1
"""

import os
import sys
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from tqdm import tqdm
from math import pi

# 科学计算库
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d, maximum_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.stats import spearmanr

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessor import ImagePreprocessor
from src.geometry_features import GeometryFeatureExtractor
from src.visualizer import WearVisualizer
from src.utils import ensure_dir, save_json, compute_trend_slope, calculate_statistics


class IntegratedWearAnalyzer:
    """整合的剪刀磨损分析器 - 包含所有分析功能"""
    
    def __init__(self, data_dir: str, output_dir: str, max_frames: int = None,
                 n_coils: int = 9, coil_start_id: int = 4):
        """
        初始化分析器
        
        Args:
            data_dir: 数据目录（包含frame_*_roi.png图像）
            output_dir: 输出目录
            max_frames: 最大处理帧数（None表示处理所有帧）
            n_coils: 钢卷数量
            coil_start_id: 起始钢卷编号
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_coils = n_coils
        self.coil_start_id = coil_start_id
        
        # 自动检测图片总数
        if max_frames is None:
            import glob
            image_files = glob.glob(os.path.join(data_dir, 'frame_*_roi.png'))
            self.max_frames = len(image_files)
            if self.max_frames > 0:
                print(f"✓ 自动检测到 {self.max_frames} 帧图像")
            else:
                print(f"⚠ 警告: 数据目录未找到图像文件")
        else:
            self.max_frames = max_frames
        
        # 创建输出目录
        self.features_dir = os.path.join(output_dir, 'results', 'features')
        self.viz_dir = os.path.join(output_dir, 'results', 'visualizations')
        self.diagnosis_dir = os.path.join(self.viz_dir, 'frame_diagnosis')
        
        ensure_dir(self.features_dir)
        ensure_dir(self.viz_dir)
        ensure_dir(self.diagnosis_dir)
        
        # 初始化各模块（只在需要时初始化）
        self.preprocessor = None
        self.feature_extractor = None
        self.visualizer = None
        
        # 核心特征定义
        self.core_features = {
            'avg_rms_roughness': '平均RMS粗糙度',
            'avg_gradient_energy': '平均梯度能量（锐度）',
            'max_notch_depth': '最大缺口深度',
            'left_peak_density': '左侧峰密度',
            'right_peak_density': '右侧峰密度',
            'tear_shear_area_ratio': '撕裂/剪切面积比'
        }
        
        print(f"✓ 分析器初始化完成")
        print(f"  - 数据目录: {data_dir}")
        print(f"  - 输出目录: {output_dir}")
        print(f"  - 钢卷信息: 第{coil_start_id}-{coil_start_id+n_coils-1}卷（共{n_coils}卷）")
    
    def _init_processors(self):
        """延迟初始化处理器"""
        if self.preprocessor is None:
            self.preprocessor = ImagePreprocessor()
            self.feature_extractor = GeometryFeatureExtractor()
            self.visualizer = WearVisualizer(self.viz_dir)
    
    # ============================================================================
    # 特征提取模块
    # ============================================================================
    
    def extract_features(self) -> pd.DataFrame:
        """提取所有帧的特征"""
        self._init_processors()
        
        print(f"\n{'='*80}")
        print(f"步骤 1/7: 特征提取")
        print(f"{'='*80}")
        print(f"开始处理 {self.max_frames} 帧图像...")
        
        all_features = []
        
        for frame_id in tqdm(range(self.max_frames), desc="提取特征"):
            save_diagnosis = (frame_id < 10) or (frame_id % 100 == 0)
            
            try:
                filename = f"frame_{frame_id:06d}_roi.png"
                filepath = os.path.join(self.data_dir, filename)
                
                if not os.path.exists(filepath):
                    continue
                
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                preprocessed = self.preprocessor.process(image)
                if not preprocessed['success']:
                    continue
                
                features = self.feature_extractor.extract_features(preprocessed)
                features['frame_id'] = frame_id
                all_features.append(features)
                
                if save_diagnosis:
                    diagnosis_path = os.path.join(
                        self.diagnosis_dir, f"frame_{frame_id:06d}_diagnosis.png"
                    )
                    self.visualizer.visualize_single_frame_diagnosis(
                        image, preprocessed, features, frame_id, diagnosis_path
                    )
                    
            except Exception as e:
                print(f"\n⚠ 警告: 处理帧 {frame_id} 时出错: {str(e)}")
        
        if len(all_features) == 0:
            raise RuntimeError("❌ 错误: 没有成功处理任何帧")
        
        df = pd.DataFrame(all_features)
        print(f"\n✓ 成功处理 {len(df)} / {self.max_frames} 帧")
        
        # 保存特征
        csv_path = os.path.join(self.features_dir, 'wear_features.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 特征数据已保存: {csv_path}")
        
        return df
    
    # ============================================================================
    # 基础可视化模块
    # ============================================================================
    
    def generate_basic_visualizations(self, df: pd.DataFrame):
        """生成基础可视化"""
        self._init_processors()
        
        print(f"\n{'='*80}")
        print(f"步骤 2/7: 基础可视化")
        print(f"{'='*80}")
        
        self.visualizer.plot_temporal_trends(
            df, os.path.join(self.viz_dir, 'temporal_trends.png')
        )
        print("✓ 时序趋势曲线")
        
        self.visualizer.plot_feature_correlations(
            df, os.path.join(self.viz_dir, 'feature_correlations.png')
        )
        print("✓ 特征相关性分析")
        
        self.visualizer.plot_wear_progression(
            df, os.path.join(self.viz_dir, 'wear_progression.png')
        )
        print("✓ 磨损递进综合图")
    
    # ============================================================================
    # 增强可视化模块（集成 enhanced_visualization.py 的功能）
    # ============================================================================
    
    def generate_enhanced_visualizations(self, df: pd.DataFrame):
        """生成增强可视化"""
        print(f"\n{'='*80}")
        print(f"步骤 3/7: 增强可视化")
        print(f"{'='*80}")
        
        # 调用增强可视化脚本
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), 'enhanced_visualization.py')
        if os.path.exists(script_path):
            subprocess.run([sys.executable, script_path], cwd=self.output_dir)
            print("✓ 增强可视化已生成（峰值连线、周期对比、累积磨损、首尾对比）")
        else:
            print("⚠ 警告: 未找到 enhanced_visualization.py，跳过此模块")
    
    # ============================================================================
    # 深度趋势分析模块（集成 deep_trend_analysis.py 的功能）
    # ============================================================================
    
    def generate_deep_trend_analysis(self, df: pd.DataFrame):
        """深度趋势分析"""
        print(f"\n{'='*80}")
        print(f"步骤 4/7: 深度趋势分析")
        print(f"{'='*80}")
        
        # 调用深度分析脚本
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), 'deep_trend_analysis.py')
        if os.path.exists(script_path):
            subprocess.run([sys.executable, script_path], cwd=self.output_dir)
            print("✓ 深度趋势分析已完成（峰值包络、分段趋势、低通滤波）")
        else:
            print("⚠ 警告: 未找到 deep_trend_analysis.py，跳过此模块")
    
    # ============================================================================
    # 按卷分析模块（集成 coil_by_coil_analysis.py 的功能）
    # ============================================================================
    
    def generate_coil_analysis(self, df: pd.DataFrame):
        """按卷分析"""
        print(f"\n{'='*80}")
        print(f"步骤 5/7: 按卷分析")
        print(f"{'='*80}")
        
        # 调用按卷分析脚本
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), 'coil_by_coil_analysis.py')
        if os.path.exists(script_path):
            subprocess.run([sys.executable, script_path], cwd=self.output_dir)
            print("✓ 按卷分析已完成（箱线图、统计图、热力图、雷达图）")
        else:
            print("⚠ 警告: 未找到 coil_by_coil_analysis.py，跳过此模块")
    
    # ============================================================================
    # 最佳指标评估模块（集成 best_indicator_analysis.py 的功能）
    # ============================================================================
    
    def generate_best_indicator_analysis(self, df: pd.DataFrame):
        """最佳指标评估"""
        print(f"\n{'='*80}")
        print(f"步骤 6/7: 最佳指标评估")
        print(f"{'='*80}")
        
        # 调用指标评估脚本
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), 'best_indicator_analysis.py')
        if os.path.exists(script_path):
            subprocess.run([sys.executable, script_path], cwd=self.output_dir)
            print("✓ 指标评估已完成")
        else:
            print("⚠ 警告: 未找到 best_indicator_analysis.py，跳过此模块")
    
    # ============================================================================
    # 平滑长期趋势模块（集成 smooth_longterm_trend.py 的功能）
    # ============================================================================
    
    def generate_smooth_longterm_trend(self, df: pd.DataFrame):
        """平滑长期趋势分析"""
        print(f"\n{'='*80}")
        print(f"步骤 7/7: 平滑长期趋势分析")
        print(f"{'='*80}")
        
        # 调用平滑趋势脚本
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), 'smooth_longterm_trend.py')
        if os.path.exists(script_path):
            subprocess.run([sys.executable, script_path], cwd=self.output_dir)
            print("✓ 平滑趋势分析已完成（包络线、峰值拟合、全局平滑）")
        else:
            print("⚠ 警告: 未找到 smooth_longterm_trend.py，跳过此模块")
    
    # ============================================================================
    # 报告生成
    # ============================================================================
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """生成综合分析报告"""
        report = []
        report.append("# 剪刀磨损程度综合分析报告\n\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report.append(f"**分析帧数**: {len(df)}\n\n")
        report.append("---\n\n")
        
        # 数据概览
        report.append("## 1. 数据概览\n\n")
        report.append(f"- 成功处理帧数: {len(df)} / {self.max_frames}\n")
        report.append(f"- 帧编号范围: {df['frame_id'].min()} - {df['frame_id'].max()}\n")
        report.append(f"- 钢卷信息: 第{self.coil_start_id}-{self.coil_start_id+self.n_coils-1}卷（共{self.n_coils}卷）\n\n")
        
        # 核心特征统计
        report.append("## 2. 核心特征统计\n\n")
        for feature_name, feature_label in self.core_features.items():
            if feature_name in df.columns:
                values = df[feature_name].values
                stats = calculate_statistics(values)
                trend_slope = compute_trend_slope(values)
                
                report.append(f"### {feature_label}\n\n")
                report.append(f"| 指标 | 值 |\n")
                report.append(f"|------|----|\n")
                report.append(f"| 均值 | {stats['mean']:.6f} |\n")
                report.append(f"| 标准差 | {stats['std']:.6f} |\n")
                report.append(f"| 最小值 | {stats['min']:.6f} |\n")
                report.append(f"| 最大值 | {stats['max']:.6f} |\n")
                report.append(f"| 中位数 | {stats['median']:.6f} |\n")
                report.append(f"| 趋势斜率 | {trend_slope:.8f} |\n\n")
                
                if trend_slope > 1e-6:
                    report.append(f"✅ **趋势**: 递增（符合磨损预期）\n\n")
                elif trend_slope < -1e-6:
                    report.append(f"⚠️ **趋势**: 递减\n\n")
                else:
                    report.append(f"➡️ **趋势**: 平稳\n\n")
        
        # 生成的可视化文件
        report.append("## 3. 生成的可视化文件\n\n")
        report.append("### 基础可视化\n")
        report.append("- `temporal_trends.png` - 时序趋势曲线\n")
        report.append("- `feature_correlations.png` - 特征相关性分析\n")
        report.append("- `wear_progression.png` - 磨损递进综合图\n\n")
        
        report.append("### 增强可视化\n")
        report.append("- `peaks_trend.png` - 峰值连线图\n")
        report.append("- `cycle_comparison.png` - 周期起终点对比\n")
        report.append("- `cumulative_wear.png` - 累积磨损指数\n")
        report.append("- `first_last_comparison.png` - 首尾对比图\n\n")
        
        report.append("### 深度分析\n")
        report.append("- `envelope_analysis.png` - 峰值包络线分析\n")
        report.append("- `segment_analysis.png` - 分段趋势分析\n")
        report.append("- `longterm_trend.png` - 低通滤波长期趋势\n\n")
        
        report.append("### 按卷分析\n")
        report.append("- `coil_by_coil_boxplot.png` - 箱线图对比\n")
        report.append("- `coil_by_coil_bars.png` - 统计柱状图\n")
        report.append("- `coil_heatmap.png` - 特征热力图\n")
        report.append("- `coil_progression_detailed.png` - 逐卷递进趋势\n")
        report.append("- `coil_radar_comparison.png` - 雷达图对比\n\n")
        
        report.append("### 指标评估\n")
        report.append("- `best_indicators_comparison.png` - 最佳指标对比\n")
        report.append("- `recommended_indicators.png` - 推荐指标卡片\n\n")
        
        report.append("### 平滑趋势\n")
        report.append("- `smooth_method1_envelope.png` - 移动最大值包络\n")
        report.append("- `smooth_method2_peaks.png` - 周期峰值拟合\n")
        report.append("- `smooth_method3_global.png` - 全局二次平滑\n")
        report.append("- `smooth_comparison_final.png` - 方法对比\n\n")
        
        report.append("---\n\n")
        report.append("*本报告由剪刀磨损综合分析系统自动生成*\n")
        
        return ''.join(report)
    
    def save_report(self, report_content: str):
        """保存报告"""
        report_path = os.path.join(self.output_dir, 'results', 'analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\n✓ 分析报告已保存: {report_path}")
    
    # ============================================================================
    # 主流程
    # ============================================================================
    
    def run(self, skip_extraction=False, features_csv=None, modules=None):
        """
        运行完整分析流程
        
        Args:
            skip_extraction: 是否跳过特征提取
            features_csv: 已有特征文件路径
            modules: 要运行的模块列表
        """
        print("\n" + "="*80)
        print("剪刀磨损程度综合分析系统")
        print("="*80)
        
        if modules is None:
            modules = ['all']
        
        if 'all' in modules:
            modules = ['basic', 'enhanced', 'deep', 'coil', 'indicator', 'smooth']
        
        try:
            # 1. 特征提取或加载
            if skip_extraction and features_csv:
                print(f"\n跳过特征提取，加载特征文件: {features_csv}")
                df = pd.read_csv(features_csv)
                print(f"✓ 加载了 {len(df)} 帧的特征数据")
            else:
                df = self.extract_features()
            
            # 2. 运行各个分析模块
            if 'basic' in modules:
                self.generate_basic_visualizations(df)
            
            if 'enhanced' in modules:
                self.generate_enhanced_visualizations(df)
            
            if 'deep' in modules:
                self.generate_deep_trend_analysis(df)
            
            if 'coil' in modules:
                self.generate_coil_analysis(df)
            
            if 'indicator' in modules:
                self.generate_best_indicator_analysis(df)
            
            if 'smooth' in modules:
                self.generate_smooth_longterm_trend(df)
            
            # 3. 生成报告
            report = self.generate_report(df)
            self.save_report(report)
            
            print("\n" + "="*80)
            print("✅ 分析完成！")
            print("="*80)
            print(f"\n📁 结果保存目录: {os.path.join(self.output_dir, 'results')}")
            print(f"  - 特征数据: {os.path.join(self.output_dir, 'results/features')}")
            print(f"  - 可视化图表: {os.path.join(self.output_dir, 'results/visualizations')}")
            print(f"  - 分析报告: {os.path.join(self.output_dir, 'results/analysis_report.md')}")
            
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='剪刀磨损程度综合分析系统 - 通用版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认设置处理数据
  python main_analysis.py
  
  # 指定数据目录和输出目录
  python main_analysis.py --data_dir /path/to/roi_imgs --output_dir /path/to/output
  
  # 只运行特定的分析模块
  python main_analysis.py --modules basic enhanced coil
  
  # 跳过特征提取，使用已有特征文件
  python main_analysis.py --skip_extraction --features_csv results/features/wear_features.csv
  
  # 自定义钢卷参数
  python main_analysis.py --n_coils 12 --coil_start_id 1
        """
    )
    
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录（包含frame_*_roi.png图像），默认: ../data/roi_imgs')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录，默认: 当前目录')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='最大处理帧数（默认处理所有帧）')
    parser.add_argument('--n_coils', type=int, default=9,
                       help='钢卷数量（默认9）')
    parser.add_argument('--coil_start_id', type=int, default=4,
                       help='起始钢卷编号（默认4）')
    parser.add_argument('--skip_extraction', action='store_true',
                       help='跳过特征提取，使用已有特征文件')
    parser.add_argument('--features_csv', type=str, default=None,
                       help='已有特征CSV文件路径（配合--skip_extraction使用）')
    parser.add_argument('--modules', nargs='+',
                       choices=['basic', 'enhanced', 'deep', 'coil', 'indicator', 'smooth', 'all'],
                       default=['all'],
                       help='要运行的分析模块（默认全部）')
    
    args = parser.parse_args()
    
    # 配置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.data_dir is None:
        data_dir = os.path.join(os.path.dirname(base_dir), 'data', 'roi_imgs')
    else:
        data_dir = args.data_dir
    
    if args.output_dir is None:
        output_dir = base_dir
    else:
        output_dir = args.output_dir
    
    # 检查数据目录
    if not args.skip_extraction and not os.path.exists(data_dir):
        print(f"❌ 错误: 数据目录不存在: {data_dir}")
        print(f"\n提示: 请使用 --data_dir 参数指定正确的数据目录")
        return
    
    # 创建分析器并运行
    analyzer = IntegratedWearAnalyzer(
        data_dir=data_dir,
        output_dir=output_dir,
        max_frames=args.max_frames,
        n_coils=args.n_coils,
        coil_start_id=args.coil_start_id
    )
    
    analyzer.run(
        skip_extraction=args.skip_extraction,
        features_csv=args.features_csv,
        modules=args.modules
    )


if __name__ == '__main__':
    main()


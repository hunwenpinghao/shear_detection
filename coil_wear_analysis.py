"""
通用的剪刀磨损按卷分析脚本
自动检测钢卷边界，支持任意视频数据的特征提取和按卷分析

用法:
    python coil_wear_analysis.py --roi_dir <ROI图像目录> --output_dir <输出目录> [选项]

示例:
    python coil_wear_analysis.py --roi_dir data/roi_imgs --output_dir data/analysis --name "第一周期"
    
特点:
    - 基于统计变化点检测，自动识别钢卷切换边界
    - 无需手动指定钢卷数量
    - 多特征融合提高检测准确性
"""
import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
from tqdm import tqdm
from math import pi
from sklearn.preprocessing import StandardScaler
import ruptures as rp  # 用于变化点检测

# 添加主项目的模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'wear_degree_analysis', 'src'))

from preprocessor import ImagePreprocessor
from geometry_features import GeometryFeatureExtractor
from visualizer import WearVisualizer
from utils import ensure_dir
import seaborn as sns

# 设置中文字体 - 多个备选方案确保兼容性
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Songti SC', 'STSong', 'SimHei', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11
# 强制使用TrueType字体，避免字符丢失
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class UniversalWearAnalyzer:
    """通用的磨损分析器"""
    
    def __init__(self, roi_dir: str, output_dir: str, analysis_name: str = "视频分析", 
                 min_coils: int = 5, max_coils: int = 15):
        """
        初始化分析器
        
        Args:
            roi_dir: ROI图像目录
            output_dir: 输出目录
            analysis_name: 分析名称
            min_coils: 最小钢卷数
            max_coils: 最大钢卷数
        """
        self.roi_dir = os.path.abspath(roi_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.analysis_name = analysis_name
        self.min_coils = min_coils
        self.max_coils = max_coils
        
        # 创建输出目录
        ensure_dir(output_dir)
        
        # 统计帧数
        import glob
        self.image_files = sorted(glob.glob(os.path.join(roi_dir, 'frame_*_roi.png')))
        self.total_frames = len(self.image_files)
        
        print(f"\n{'='*80}")
        print(f"{analysis_name} - 分析初始化")
        print(f"{'='*80}")
        print(f"ROI目录: {roi_dir}")
        print(f"输出目录: {output_dir}")
        print(f"检测到帧数: {self.total_frames}")
        
        # 初始化特征提取器
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = GeometryFeatureExtractor()
        self.visualizer = WearVisualizer(output_dir)
    
    def extract_features(self, save_diagnosis: bool = True) -> pd.DataFrame:
        """
        提取所有帧的特征
        
        Args:
            save_diagnosis: 是否保存帧诊断图（前10帧和每100帧）
        """
        print(f"\n开始提取{self.analysis_name}的特征...")
        
        # 创建诊断图目录
        if save_diagnosis:
            diagnosis_dir = os.path.join(self.output_dir, 'visualizations', 'frame_diagnosis')
            ensure_dir(diagnosis_dir)
        
        all_features = []
        
        for frame_id in tqdm(range(self.total_frames), desc=f"提取特征"):
            try:
                # 构造文件路径
                filename = f"frame_{frame_id:06d}_roi.png"
                filepath = os.path.join(self.roi_dir, filename)
                
                if not os.path.exists(filepath):
                    continue
                
                # 读取图像
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                # 预处理
                preprocessed = self.preprocessor.process(image)
                
                if not preprocessed['success']:
                    continue
                
                # 提取特征
                features = self.feature_extractor.extract_features(preprocessed)
                features['frame_id'] = frame_id
                all_features.append(features)
                
                # 保存诊断图（前10帧和每100帧）
                if save_diagnosis and (frame_id < 10 or frame_id % 100 == 0):
                    diagnosis_path = os.path.join(diagnosis_dir, f"frame_{frame_id:06d}_diagnosis.png")
                    self.visualizer.visualize_single_frame_diagnosis(
                        image, preprocessed, features, frame_id, diagnosis_path
                    )
                
            except Exception as e:
                print(f"警告: 处理帧{frame_id}时出错: {e}")
                continue
        
        if len(all_features) == 0:
            raise RuntimeError("没有成功提取任何特征")
        
        df = pd.DataFrame(all_features)
        print(f"成功提取 {len(df)} / {self.total_frames} 帧的特征")
        
        # 保存特征
        features_dir = os.path.join(self.output_dir, 'features')
        ensure_dir(features_dir)
        
        csv_path = os.path.join(features_dir, 'wear_features.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存特征文件: {csv_path}")
        
        return df
    
    def detect_coil_boundaries(self, df: pd.DataFrame) -> list:
        """
        自动检测钢卷边界
        
        Args:
            df: 特征数据
            
        Returns:
            钢卷边界索引列表
        """
        print(f"\n自动检测钢卷边界...")
        print(f"钢卷数范围: {self.min_coils}-{self.max_coils}个")
        
        # 选择多个关键特征进行分析
        key_features = ['avg_gradient_energy', 'max_notch_depth', 'avg_rms_roughness']
        
        # 标准化特征
        scaler = StandardScaler()
        features_for_detection = []
        
        for feature in key_features:
            if feature in df.columns:
                # 平滑处理减少噪声
                smoothed = savgol_filter(df[feature].values, 
                                        window_length=min(51, len(df)//10*2+1), 
                                        polyorder=3)
                features_for_detection.append(smoothed)
        
        # 组合多个特征
        if len(features_for_detection) == 0:
            print("警告: 没有足够的特征用于检测，使用默认分割")
            return None
        
        combined_signal = np.column_stack(features_for_detection)
        combined_signal = scaler.fit_transform(combined_signal)
        
        # 使用Pelt算法检测变化点
        try:
            model = "l2"  # L2损失函数
            algo = rp.Pelt(model=model, min_size=len(df)//self.max_coils, jump=5)
            algo.fit(combined_signal)
            
            # 尝试不同的penalty参数，找到合适的钢卷数
            best_boundaries = None
            best_n_coils = 0
            
            for penalty in np.linspace(1, 10, 20):
                boundaries = algo.predict(pen=penalty)
                n_segments = len(boundaries)
                
                if self.min_coils <= n_segments <= self.max_coils:
                    best_boundaries = boundaries
                    best_n_coils = n_segments
                    break
            
            if best_boundaries is None:
                # 如果没找到合适的，用中间值
                print(f"未找到最优分割，使用默认分割")
                return None
            
            # 去掉最后的边界点（总是等于数据长度）
            boundaries = [0] + best_boundaries[:-1]
            
            print(f"✓ 检测到 {len(boundaries)} 个钢卷")
            print(f"边界位置: {boundaries}")
            
            return boundaries
            
        except Exception as e:
            print(f"变化点检测失败: {e}")
            print("使用默认均匀分割")
            return None
    
    def analyze_by_coil(self, df: pd.DataFrame):
        """
        按卷分析（自动检测钢卷边界）
        
        Args:
            df: 特征数据
        """
        print(f"\n{'='*80}")
        
        # 自动检测钢卷边界
        boundaries = self.detect_coil_boundaries(df)
        
        if boundaries is not None:
            # 使用检测到的边界分配卷号
            df['coil_id'] = 0
            for i, boundary in enumerate(boundaries):
                if i < len(boundaries) - 1:
                    df.loc[boundary:boundaries[i+1]-1, 'coil_id'] = i + 1
                else:
                    df.loc[boundary:, 'coil_id'] = i + 1
            
            n_coils = len(boundaries)
        else:
            # 检测失败，使用默认均匀分割
            print("⚠️ 自动检测失败，使用默认均匀分割（9个钢卷）")
            n_coils = 9
            coil_size = len(df) // n_coils
            df['coil_id'] = df.index // coil_size + 1
            df.loc[df['coil_id'] > n_coils, 'coil_id'] = n_coils
        
        print(f"{self.analysis_name} - 按卷分析（共{n_coils}个钢卷）")
        print(f"{'='*80}")
        
        print("\n每卷帧数分布:")
        coil_counts = df['coil_id'].value_counts().sort_index()
        for coil_id, count in coil_counts.items():
            print(f"  第{int(coil_id)}卷: {count}帧")
        
        # 保存带卷号的特征文件
        features_dir = os.path.join(self.output_dir, 'features')
        csv_with_coils = os.path.join(features_dir, 'wear_features_with_coils.csv')
        df.to_csv(csv_with_coils, index=False, encoding='utf-8-sig')
        
        # 创建可视化目录
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        ensure_dir(viz_dir)
        
        # 核心特征
        key_features = {
            'avg_rms_roughness': 'RMS粗糙度',
            'max_notch_depth': '最大缺口深度',
            'right_peak_density': '右侧峰密度（剪切面）',
            'avg_gradient_energy': '梯度能量（锐度）',
            'tear_shear_area_ratio': '撕裂/剪切面积比'
        }
        
        # 生成可视化
        self._plot_boxplot(df, key_features, viz_dir)
        self._plot_bars(df, key_features, viz_dir)
        self._plot_heatmap(df, key_features, viz_dir)
        self._plot_radar(df, key_features, viz_dir, n_coils)
        self._plot_progression(df, key_features, viz_dir)
        
        # 生成额外的分析图
        print("\n生成额外分析图...")
        self._plot_temporal_trends(df, os.path.join(viz_dir, 'temporal_trends.png'))
        self._plot_feature_correlations(df, os.path.join(viz_dir, 'feature_correlations.png'))
        self._plot_wear_progression(df, os.path.join(viz_dir, 'wear_progression.png'))
        self._plot_longterm_trend(df, os.path.join(viz_dir, 'longterm_trend.png'))
        self._plot_recommended_indicators(df, os.path.join(viz_dir, 'recommended_indicators.png'))
        print("✓ 额外分析图生成完成")
        
        # 生成分析报告
        self._generate_report(df, key_features, n_coils)
    
    def _plot_boxplot(self, df, key_features, viz_dir):
        """绘制箱线图"""
        print("\n生成可视化: 箱线图...")
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, (feature, label) in enumerate(list(key_features.items())[:5]):
            ax = axes[idx]
            
            coil_data = []
            coil_labels = []
            
            for coil_id in sorted(df['coil_id'].unique()):
                coil_df = df[df['coil_id'] == coil_id]
                coil_data.append(coil_df[feature].values)
                coil_labels.append(f'卷{int(coil_id)}')
            
            bp = ax.boxplot(coil_data, labels=coil_labels, patch_artist=True,
                           widths=0.6, boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(linewidth=2, color='red'))
            
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(coil_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            means = [np.mean(data) for data in coil_data]
            ax.plot(range(1, len(means)+1), means, 'bo-', linewidth=3,
                   markersize=8, label='均值趋势', zorder=10)
            
            x = np.arange(len(means))
            z = np.polyfit(x, means, 1)
            trend = np.poly1d(z)
            ax.plot(range(1, len(means)+1), trend(x), 'g--', linewidth=3,
                   label=f'线性趋势(斜率={z[0]:.4f})', alpha=0.8)
            
            change_pct = ((means[-1] - means[0]) / (means[0] + 1e-8)) * 100
            
            if change_pct > 5:
                trend_text = f'✓ 显著递增 +{change_pct:.1f}%'
                box_color = 'lightgreen'
            elif change_pct > 0:
                trend_text = f'轻微递增 +{change_pct:.1f}%'
                box_color = 'lightyellow'
            elif change_pct > -5:
                trend_text = f'基本平稳 {change_pct:.1f}%'
                box_color = 'lightgray'
            else:
                trend_text = f'递减 {change_pct:.1f}%'
                box_color = 'lightcoral'
            
            ax.text(0.5, 0.98, trend_text, transform=ax.transAxes,
                   fontsize=14, fontweight='bold', ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=1', facecolor=box_color,
                            alpha=0.8, edgecolor='black', linewidth=2))
            
            ax.set_xlabel('钢卷编号', fontweight='bold', fontsize=13)
            ax.set_ylabel(label, fontweight='bold', fontsize=13)
            ax.set_title(f'{label}\n按卷演变趋势', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
        
        axes[-1].axis('off')
        
        plt.suptitle(f'{self.analysis_name} - 剪刀磨损按卷分析（箱线图）',
                    fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'coil_by_coil_boxplot.png'), dpi=300, bbox_inches='tight')
        print(f"已保存: {viz_dir}/coil_by_coil_boxplot.png")
    
    def _plot_bars(self, df, key_features, viz_dir):
        """绘制柱状图"""
        print("\n生成可视化: 柱状图...")
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, (feature, label) in enumerate(list(key_features.items())[:5]):
            ax = axes[idx]
            
            coil_ids = []
            coil_means = []
            coil_maxes = []
            
            for coil_id in sorted(df['coil_id'].unique()):
                coil_df = df[df['coil_id'] == coil_id]
                coil_ids.append(int(coil_id))
                coil_means.append(coil_df[feature].mean())
                coil_maxes.append(coil_df[feature].max())
            
            x = np.arange(len(coil_ids))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, coil_means, width, label='均值',
                          color='steelblue', edgecolor='navy', linewidth=2, alpha=0.8)
            bars2 = ax.bar(x + width/2, coil_maxes, width, label='最大值',
                          color='coral', edgecolor='darkred', linewidth=2, alpha=0.8)
            
            for bar1, bar2 in zip(bars1, bars2):
                height1 = bar1.get_height()
                height2 = bar2.get_height()
                ax.text(bar1.get_x() + bar1.get_width()/2, height1,
                       f'{height1:.2f}', ha='center', va='bottom', fontsize=9)
                ax.text(bar2.get_x() + bar2.get_width()/2, height2,
                       f'{height2:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax.plot(x, coil_means, 'b--', linewidth=2, alpha=0.6)
            ax.plot(x, coil_maxes, 'r--', linewidth=2, alpha=0.6)
            
            z_mean = np.polyfit(x, coil_means, 1)
            change_pct = ((coil_means[-1] - coil_means[0]) / (coil_means[0] + 1e-8)) * 100
            
            trend_text = f'均值变化: {change_pct:+.1f}%\n斜率: {z_mean[0]:.4f}'
            box_color = 'lightgreen' if change_pct > 0 else 'lightcoral'
            
            ax.text(0.02, 0.98, trend_text, transform=ax.transAxes,
                   fontsize=12, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor=box_color, alpha=0.7))
            
            ax.set_xlabel('钢卷编号', fontweight='bold', fontsize=13)
            ax.set_ylabel(label, fontweight='bold', fontsize=13)
            ax.set_title(f'{label}\n各卷统计对比', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'卷{cid}' for cid in coil_ids])
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
        
        axes[-1].axis('off')
        
        plt.suptitle(f'{self.analysis_name} - 剪刀磨损按卷统计分析',
                    fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'coil_by_coil_bars.png'), dpi=300, bbox_inches='tight')
        print(f"已保存: {viz_dir}/coil_by_coil_bars.png")
    
    def _plot_heatmap(self, df, key_features, viz_dir):
        """绘制热力图"""
        print("\n生成可视化: 热力图...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        feature_names = list(key_features.values())
        matrix_data = []
        
        for feature in key_features.keys():
            row = []
            for coil_id in sorted(df['coil_id'].unique()):
                coil_df = df[df['coil_id'] == coil_id]
                row.append(coil_df[feature].mean())
            matrix_data.append(row)
        
        matrix = np.array(matrix_data)
        
        # 归一化
        matrix_norm = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            row = matrix[i, :]
            matrix_norm[i, :] = (row - row.min()) / (row.max() - row.min() + 1e-8)
        
        im = ax.imshow(matrix_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        coil_ids = sorted(df['coil_id'].unique())
        ax.set_xticks(np.arange(len(coil_ids)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels([f'第{int(cid)}卷' for cid in coil_ids], fontsize=12)
        ax.set_yticklabels(feature_names, fontsize=12)
        
        for i in range(len(feature_names)):
            for j in range(len(coil_ids)):
                text = ax.text(j, i, f'{matrix_norm[i, j]:.2f}',
                             ha="center", va="center", color="black",
                             fontsize=10, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('归一化特征值 (0=最小, 1=最大)', fontsize=12, fontweight='bold')
        
        ax.set_title(f'{self.analysis_name} - 各卷磨损特征热力图\n（颜色越红=该特征在该卷的值越大）',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('钢卷编号', fontsize=13, fontweight='bold')
        ax.set_ylabel('磨损特征', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'coil_heatmap.png'), dpi=300, bbox_inches='tight')
        print(f"已保存: {viz_dir}/coil_heatmap.png")
    
    def _plot_radar(self, df, key_features, viz_dir, n_coils):
        """绘制雷达图"""
        print("\n生成可视化: 雷达图...")
        
        # 选择开始、中期、结束三个卷
        coil_ids = sorted(df['coil_id'].unique())
        if len(coil_ids) >= 3:
            representative_coils = [coil_ids[0], coil_ids[len(coil_ids)//2], coil_ids[-1]]
            coil_labels = [f'第{int(representative_coils[0])}卷(开始)',
                          f'第{int(representative_coils[1])}卷(中期)',
                          f'第{int(representative_coils[2])}卷(结束)']
        else:
            representative_coils = coil_ids
            coil_labels = [f'第{int(cid)}卷' for cid in coil_ids]
        
        colors = ['blue', 'orange', 'red']
        
        fig, axes = plt.subplots(1, len(representative_coils), figsize=(20, 7),
                                subplot_kw=dict(projection='polar'))
        
        if len(representative_coils) == 1:
            axes = [axes]
        
        for plot_idx, (coil_id, coil_label, color) in enumerate(zip(representative_coils, coil_labels, colors)):
            ax = axes[plot_idx]
            
            coil_df = df[df['coil_id'] == coil_id]
            
            if len(coil_df) == 0:
                ax.text(0.5, 0.5, f'{coil_label}\n无数据',
                       transform=ax.transAxes, ha='center', va='center')
                continue
            
            categories = list(key_features.values())
            values = []
            
            for feature in key_features.keys():
                values.append(coil_df[feature].mean())
            
            # 归一化
            global_max = []
            global_min = []
            for feature in key_features.keys():
                global_max.append(df[feature].max())
                global_min.append(df[feature].min())
            
            values_norm = [(v - vmin) / (vmax - vmin + 1e-8)
                          for v, vmin, vmax in zip(values, global_min, global_max)]
            
            values_norm += values_norm[:1]
            
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]
            
            ax.plot(angles, values_norm, 'o-', linewidth=3, color=color,
                   label=coil_label, markersize=8)
            ax.fill(angles, values_norm, alpha=0.25, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=9)
            ax.grid(True, alpha=0.3)
            
            ax.set_title(coil_label, fontsize=15, fontweight='bold', pad=20)
        
        plt.suptitle(f'{self.analysis_name} - 雷达图对比：开始、中期、结束卷的磨损特征',
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'coil_radar_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"已保存: {viz_dir}/coil_radar_comparison.png")
    
    def _plot_progression(self, df, key_features, viz_dir):
        """绘制逐卷递进趋势图"""
        print("\n生成可视化: 逐卷递进趋势图...")
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 1, hspace=0.3)
        
        focus_features = {
            'right_peak_density': '右侧峰密度（剪切面微缺口）',
            'avg_gradient_energy': '梯度能量（刀口锐度）',
            'max_notch_depth': '最大缺口深度'
        }
        
        for idx, (feature, label) in enumerate(focus_features.items()):
            ax = fig.add_subplot(gs[idx])
            
            coil_ids = []
            coil_means = []
            coil_maxes = []
            coil_mins = []
            coil_q25 = []
            coil_q75 = []
            
            for coil_id in sorted(df['coil_id'].unique()):
                coil_df = df[df['coil_id'] == coil_id]
                values = coil_df[feature].values
                
                coil_ids.append(int(coil_id))
                coil_means.append(np.mean(values))
                coil_maxes.append(np.max(values))
                coil_mins.append(np.min(values))
                coil_q25.append(np.percentile(values, 25))
                coil_q75.append(np.percentile(values, 75))
            
            coil_ids = np.array(coil_ids)
            coil_means = np.array(coil_means)
            coil_maxes = np.array(coil_maxes)
            coil_mins = np.array(coil_mins)
            coil_q25 = np.array(coil_q25)
            coil_q75 = np.array(coil_q75)
            
            ax.fill_between(coil_ids, coil_mins, coil_maxes,
                          alpha=0.2, color='gray', label='最小-最大范围')
            ax.fill_between(coil_ids, coil_q25, coil_q75,
                          alpha=0.3, color='lightblue', label='25%-75%分位数')
            
            ax.plot(coil_ids, coil_means, 'o-', linewidth=4, markersize=12,
                   color='darkblue', label='均值', markeredgewidth=2,
                   markeredgecolor='white', zorder=10)
            
            ax.plot(coil_ids, coil_maxes, 's-', linewidth=3, markersize=10,
                   color='darkred', label='最大值', alpha=0.7, zorder=9)
            
            z = np.polyfit(coil_ids, coil_means, 1)
            trend = np.poly1d(z)
            ax.plot(coil_ids, trend(coil_ids), '--', linewidth=3,
                   color='green', label=f'均值趋势线', alpha=0.8)
            
            change_pct = ((coil_means[-1] - coil_means[0]) / (coil_means[0] + 1e-8)) * 100
            
            if feature == 'avg_gradient_energy':
                is_wear_increasing = (change_pct < 0)
                trend_desc = f'锐度下降{abs(change_pct):.1f}% → 磨损加重' if change_pct < 0 else f'锐度上升{change_pct:.1f}%'
            else:
                is_wear_increasing = (change_pct > 0)
                trend_desc = f'递增{change_pct:.1f}% → 磨损加重' if change_pct > 0 else f'递减{abs(change_pct):.1f}%'
            
            if is_wear_increasing:
                conclusion_text = f'✓ {trend_desc}'
                box_color = 'lightgreen'
            else:
                conclusion_text = f'{trend_desc}'
                box_color = 'lightyellow'
            
            ax.text(0.98, 0.98, conclusion_text, transform=ax.transAxes,
                   fontsize=14, fontweight='bold', ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=1', facecolor=box_color,
                            alpha=0.8, edgecolor='black', linewidth=2))
            
            ax.annotate(f'起始\n{coil_means[0]:.2f}',
                       xy=(coil_ids[0], coil_means[0]),
                       xytext=(coil_ids[0]-0.5, coil_means[0]*1.1),
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', lw=2))
            
            ax.annotate(f'结束\n{coil_means[-1]:.2f}',
                       xy=(coil_ids[-1], coil_means[-1]),
                       xytext=(coil_ids[-1]+0.5, coil_means[-1]*1.1),
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', lw=2))
            
            ax.set_xlabel('钢卷编号', fontweight='bold', fontsize=13)
            ax.set_ylabel(label, fontweight='bold', fontsize=13)
            ax.set_title(f'{label} - 逐卷演变', fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(coil_ids)
        
        plt.suptitle(f'{self.analysis_name} - 剪刀磨损逐卷演变分析',
                    fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'coil_progression_detailed.png'), dpi=300, bbox_inches='tight')
        print(f"已保存: {viz_dir}/coil_progression_detailed.png")
    
    def _plot_temporal_trends(self, df: pd.DataFrame, save_path: str):
        """绘制时序趋势图"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        features = [
            ('avg_rms_roughness', '平均RMS粗糙度 (像素)', 'blue'),
            ('max_notch_depth', '最大缺口深度 (像素)', 'red'),
            ('right_peak_density', '剪切面峰密度 (个/单位)', 'green'),
            ('avg_gradient_energy', '平均梯度能量', 'purple'),
            ('tear_shear_area_ratio', '撕裂/剪切面积比', 'orange'),
        ]
        
        for idx, (feat, label, color) in enumerate(features):
            ax = axes[idx // 2, idx % 2]
            if feat in df.columns:
                ax.plot(df['frame_id'], df[feat], color=color, alpha=0.5, linewidth=0.5, label='原始数据')
                
                # 平滑曲线
                window = min(101, len(df)//10*2+1)
                if window >= 5:
                    smoothed = savgol_filter(df[feat].values, window_length=window, polyorder=3)
                    ax.plot(df['frame_id'], smoothed, color=color, linewidth=2, label='平滑曲线')
                
                ax.set_xlabel('帧编号', fontsize=12, fontweight='bold')
                ax.set_ylabel(label, fontsize=12, fontweight='bold')
                ax.set_title(f'{label}时序变化', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # 隐藏多余的子图
        axes[-1, -1].axis('off')
        
        plt.suptitle(f'{self.analysis_name} - 特征时序趋势分析', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _plot_feature_correlations(self, df: pd.DataFrame, save_path: str):
        """绘制特征相关性矩阵"""
        features = ['avg_rms_roughness', 'max_notch_depth', 'right_peak_density',
                   'avg_gradient_energy', 'tear_shear_area_ratio']
        
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 2:
            print("警告: 特征数量不足，跳过相关性分析")
            return
        
        corr_matrix = df[available_features].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True,
                   cbar_kws={'label': '相关系数'}, ax=ax)
        
        # 设置特征标签（中文）
        feature_labels = {
            'avg_rms_roughness': '平均RMS粗糙度',
            'max_notch_depth': '最大缺口深度',
            'right_peak_density': '剪切面峰密度',
            'avg_gradient_energy': '平均梯度能量',
            'tear_shear_area_ratio': '撕裂/剪切面积比'
        }
        labels = [feature_labels.get(f, f) for f in available_features]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        ax.set_title(f'{self.analysis_name} - 特征相关性矩阵\n(1=完全正相关, -1=完全负相关)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _plot_wear_progression(self, df: pd.DataFrame, save_path: str):
        """绘制磨损递进图（滑动窗口平均）"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        features = [
            ('avg_rms_roughness', '平均RMS粗糙度'),
            ('max_notch_depth', '最大缺口深度'),
            ('right_peak_density', '剪切面峰密度'),
            ('avg_gradient_energy', '平均梯度能量'),
            ('tear_shear_area_ratio', '撕裂/剪切面积比'),
        ]
        
        window_size = max(10, len(df) // 20)  # 至少10帧
        
        for idx, (feat, label) in enumerate(features):
            ax = axes[idx // 3, idx % 3]
            if feat in df.columns:
                # 滑动窗口平均
                rolling_mean = df[feat].rolling(window=window_size, center=True).mean()
                rolling_std = df[feat].rolling(window=window_size, center=True).std()
                
                ax.plot(df['frame_id'], rolling_mean, color='darkblue', linewidth=2, label='滑动平均')
                ax.fill_between(df['frame_id'], 
                               rolling_mean - rolling_std,
                               rolling_mean + rolling_std,
                               alpha=0.3, color='lightblue', label='±1标准差')
                
                ax.set_xlabel('帧编号', fontsize=11, fontweight='bold')
                ax.set_ylabel(label, fontsize=11, fontweight='bold')
                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        axes[-1, -1].axis('off')
        
        plt.suptitle(f'{self.analysis_name} - 磨损递进分析（滑动窗口={window_size}帧）', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _plot_longterm_trend(self, df: pd.DataFrame, save_path: str):
        """绘制长期趋势图（线性拟合）"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        features = [
            ('avg_rms_roughness', '平均RMS粗糙度', 'blue'),
            ('max_notch_depth', '最大缺口深度', 'red'),
            ('right_peak_density', '剪切面峰密度', 'green'),
            ('avg_gradient_energy', '平均梯度能量', 'purple'),
            ('tear_shear_area_ratio', '撕裂/剪切面积比', 'orange'),
        ]
        
        for idx, (feat, label, color) in enumerate(features):
            ax = axes[idx // 3, idx % 3]
            if feat in df.columns:
                # 散点图
                ax.scatter(df['frame_id'], df[feat], alpha=0.3, s=10, color=color)
                
                # 线性拟合
                z = np.polyfit(df['frame_id'], df[feat], 1)
                p = np.poly1d(z)
                ax.plot(df['frame_id'], p(df['frame_id']), 
                       color='darkred', linewidth=3, linestyle='--', 
                       label=f'线性趋势: y={z[0]:.6f}x+{z[1]:.2f}')
                
                # 计算趋势方向
                trend = "增加" if z[0] > 0 else "减少"
                ax.text(0.05, 0.95, f'趋势: {trend}', 
                       transform=ax.transAxes, fontsize=11, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xlabel('帧编号', fontsize=11, fontweight='bold')
                ax.set_ylabel(label, fontsize=11, fontweight='bold')
                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        axes[-1, -1].axis('off')
        
        plt.suptitle(f'{self.analysis_name} - 长期磨损趋势分析（线性拟合）', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _plot_recommended_indicators(self, df: pd.DataFrame, save_path: str):
        """绘制推荐指标图（综合评分）"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 归一化特征
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        key_features = ['avg_rms_roughness', 'max_notch_depth', 
                       'avg_gradient_energy', 'tear_shear_area_ratio']
        available = [f for f in key_features if f in df.columns]
        
        if len(available) == 0:
            print("警告: 没有足够的特征生成推荐指标")
            return
        
        df_norm = pd.DataFrame(
            scaler.fit_transform(df[available]),
            columns=available,
            index=df.index
        )
        
        # 计算综合磨损指数
        weights = {'avg_rms_roughness': 0.3, 'max_notch_depth': 0.3,
                  'avg_gradient_energy': 0.2, 'tear_shear_area_ratio': 0.2}
        
        wear_index = np.zeros(len(df))
        for feat in available:
            weight = weights.get(feat, 0.25)
            wear_index += df_norm[feat].values * weight
        
        # 1. 综合磨损指数
        ax1 = axes[0, 0]
        ax1.plot(df['frame_id'], wear_index, color='darkred', linewidth=2)
        ax1.fill_between(df['frame_id'], 0, wear_index, alpha=0.3, color='red')
        ax1.set_xlabel('帧编号', fontsize=12, fontweight='bold')
        ax1.set_ylabel('综合磨损指数 (0-1)', fontsize=12, fontweight='bold')
        ax1.set_title('综合磨损指数（加权平均）', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.7, color='red', linestyle='--', label='警戒线')
        ax1.legend()
        
        # 2. 各特征贡献度
        ax2 = axes[0, 1]
        feature_labels = {
            'avg_rms_roughness': 'RMS粗糙度',
            'max_notch_depth': '缺口深度',
            'avg_gradient_energy': '梯度能量',
            'tear_shear_area_ratio': '面积比'
        }
        for feat in available:
            weight = weights.get(feat, 0.25)
            contribution = df_norm[feat].values * weight
            ax2.plot(df['frame_id'], contribution, 
                    label=feature_labels.get(feat, feat), 
                    linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('帧编号', fontsize=12, fontweight='bold')
        ax2.set_ylabel('贡献度', fontsize=12, fontweight='bold')
        ax2.set_title('各特征对磨损指数的贡献', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. 磨损阶段判断
        ax3 = axes[1, 0]
        stages = []
        for wi in wear_index:
            if wi < 0.3:
                stages.append(0)
            elif wi < 0.6:
                stages.append(1)
            else:
                stages.append(2)
        
        colors = ['green' if s == 0 else 'orange' if s == 1 else 'red' for s in stages]
        ax3.scatter(df['frame_id'], wear_index, c=colors, s=20, alpha=0.6)
        ax3.set_xlabel('帧编号', fontsize=12, fontweight='bold')
        ax3.set_ylabel('综合磨损指数', fontsize=12, fontweight='bold')
        ax3.set_title('磨损阶段分布', fontsize=14, fontweight='bold')
        ax3.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='轻度阈值')
        ax3.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='中度阈值')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 磨损统计
        ax4 = axes[1, 1]
        stage_counts = [stages.count(0), stages.count(1), stages.count(2)]
        colors_bar = ['green', 'orange', 'red']
        bars = ax4.bar(['轻度磨损', '中度磨损', '严重磨损'], stage_counts, color=colors_bar, alpha=0.7)
        ax4.set_ylabel('帧数', fontsize=12, fontweight='bold')
        ax4.set_title('磨损阶段统计', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}\n({height/len(df)*100:.1f}%)',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.suptitle(f'{self.analysis_name} - 推荐磨损指标分析', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _generate_report(self, df, key_features, n_coils):
        """生成分析报告"""
        print(f"\n{'='*80}")
        print(f"{self.analysis_name} - 按卷分析结论")
        print(f"{'='*80}")
        
        focus_features = {
            'right_peak_density': '右侧峰密度（剪切面微缺口）',
            'avg_gradient_energy': '梯度能量（刀口锐度）',
            'max_notch_depth': '最大缺口深度'
        }
        
        report_lines = []
        report_lines.append(f"# {self.analysis_name} - 按卷分析报告\n")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"总帧数: {len(df)}\n")
        report_lines.append(f"钢卷数: {n_coils}\n\n")
        
        for feature, label in focus_features.items():
            coil_means = []
            coil_ids_list = []
            for coil_id in sorted(df['coil_id'].unique()):
                coil_df = df[df['coil_id'] == coil_id]
                coil_means.append(coil_df[feature].mean())
                coil_ids_list.append(int(coil_id))
            
            change_pct = ((coil_means[-1] - coil_means[0]) / (coil_means[0] + 1e-8)) * 100
            
            print(f"\n【{label}】")
            print(f"  第{coil_ids_list[0]}卷均值: {coil_means[0]:.4f}")
            print(f"  第{coil_ids_list[-1]}卷均值: {coil_means[-1]:.4f}")
            print(f"  变化率: {change_pct:+.1f}%")
            
            report_lines.append(f"## {label}\n")
            report_lines.append(f"- 第{coil_ids_list[0]}卷均值: {coil_means[0]:.4f}\n")
            report_lines.append(f"- 第{coil_ids_list[-1]}卷均值: {coil_means[-1]:.4f}\n")
            report_lines.append(f"- 变化率: {change_pct:+.1f}%\n")
            
            increases = sum(1 for i in range(len(coil_means)-1)
                          if coil_means[i+1] > coil_means[i])
            total = len(coil_means) - 1
            
            print(f"  逐卷递增次数: {increases}/{total} = {increases/total*100:.0f}%")
            report_lines.append(f"- 逐卷递增次数: {increases}/{total} = {increases/total*100:.0f}%\n")
            
            if feature == 'avg_gradient_energy':
                if change_pct < 0:
                    conclusion = "✓ 锐度下降 → 刀口磨钝，符合磨损预期"
                    print(f"  {conclusion}")
                    report_lines.append(f"- {conclusion}\n")
            else:
                if change_pct > 0:
                    conclusion = "✓ 数值递增 → 磨损加重，符合预期"
                    print(f"  {conclusion}")
                    report_lines.append(f"- {conclusion}\n")
            
            report_lines.append("\n")
        
        print(f"\n{'='*80}")
        print("分析完成！")
        print(f"{'='*80}")
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        print(f"\n已保存分析报告: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='通用的剪刀磨损按卷分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（自动检测钢卷边界）
  python coil_wear_analysis.py --roi_dir data/roi_imgs --output_dir data/analysis
  
  # 指定分析名称
  python coil_wear_analysis.py --roi_dir data/roi_imgs --output_dir data/analysis --name "第一周期"
  
  # 处理多个视频
  python coil_wear_analysis.py --roi_dir video1/roi_imgs --output_dir video1/analysis --name "视频1"
  python coil_wear_analysis.py --roi_dir video2/roi_imgs --output_dir video2/analysis --name "视频2"
        """
    )
    
    parser.add_argument('--roi_dir', required=True, help='ROI图像目录路径')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    parser.add_argument('--name', default='视频分析', help='分析名称 (默认: 视频分析)')
    parser.add_argument('--min_coils', type=int, default=5, help='最小钢卷数 (默认: 5)')
    parser.add_argument('--max_coils', type=int, default=15, help='最大钢卷数 (默认: 15)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.roi_dir):
        print(f"错误: ROI目录不存在: {args.roi_dir}")
        return 1
    
    # 创建分析器
    analyzer = UniversalWearAnalyzer(
        roi_dir=args.roi_dir,
        output_dir=args.output_dir,
        analysis_name=args.name,
        min_coils=args.min_coils,
        max_coils=args.max_coils
    )
    
    # 提取特征
    df = analyzer.extract_features()
    
    # 按卷分析（自动检测）
    analyzer.analyze_by_coil(df)
    
    print(f"\n{'='*80}")
    print(f"分析完成！结果已保存到: {args.output_dir}")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


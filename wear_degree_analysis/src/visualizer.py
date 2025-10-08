"""
可视化模块
实现单帧诊断图、时序曲线和统计分析图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional
import os

# 设置中文字体 - 多个备选方案确保兼容性
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Songti SC', 'STSong', 'SimHei', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
# 强制使用TrueType字体，避免字符丢失
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class WearVisualizer:
    """磨损分析可视化器"""
    
    def __init__(self, output_dir: str):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_single_frame_diagnosis(self, image: np.ndarray, 
                                        preprocessed: dict,
                                        features: dict,
                                        frame_id: int,
                                        save_path: Optional[str] = None) -> None:
        """
        生成单帧诊断图
        
        Args:
            image: 原始图像
            preprocessed: 预处理结果
            features: 提取的特征
            frame_id: 帧编号
            save_path: 保存路径
        """
        if not preprocessed.get('success', False):
            print(f"帧 {frame_id} 预处理失败，跳过可视化")
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 原图 + 中心线 + 法线网格
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image, cmap='gray')
        
        # 绘制中心线
        xs = preprocessed['centerline_x']
        ys = preprocessed['centerline_y']
        ax1.plot(xs, ys, 'r-', linewidth=2, label='中心线')
        
        # 绘制法线（每5条显示一条）
        normals = preprocessed['normals']
        for i, normal in enumerate(normals):
            if i % 5 == 0:  # 每5条显示一条
                y = normal['y']
                x_start = normal['x_start']
                x_end = normal['x_end']
                ax1.plot([x_start, x_end], [y, y], 'g-', alpha=0.3, linewidth=0.5)
        
        ax1.set_title(f'帧 {frame_id}: 原图 + 中心线 + 法线', fontsize=12)
        ax1.legend()
        ax1.axis('off')
        
        # 2. 左右边界检测结果
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(image, cmap='gray')
        
        # 绘制中心线
        ax2.plot(xs, ys, 'r-', linewidth=2, alpha=0.5)
        
        # 绘制左右边界
        left_edges = preprocessed['left_edges']
        right_edges = preprocessed['right_edges']
        
        for i, normal in enumerate(normals):
            y = normal['y']
            x_center = normal['x_center']
            if i < len(left_edges):
                x_left = x_center + left_edges[i]
                ax2.plot(x_left, y, 'b.', markersize=2)
            if i < len(right_edges):
                x_right = x_center + right_edges[i]
                ax2.plot(x_right, y, 'g.', markersize=2)
        
        ax2.set_title('边界检测结果\n蓝色=左边界(撕裂) 绿色=右边界(剪切)', fontsize=12)
        ax2.axis('off')
        
        # 3. 边界位置序列曲线（左侧）
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(left_edges, 'b-', linewidth=1, label='左边界位置')
        ax3.axhline(y=np.mean(left_edges), color='r', linestyle='--', 
                   label=f'均值={np.mean(left_edges):.2f}')
        ax3.fill_between(range(len(left_edges)), 
                        np.mean(left_edges) - features['left_rms_roughness'],
                        np.mean(left_edges) + features['left_rms_roughness'],
                        alpha=0.3, color='orange', label=f'RMS={features["left_rms_roughness"]:.2f}')
        ax3.set_xlabel('法线索引')
        ax3.set_ylabel('相对位置')
        ax3.set_title(f'左边界位置序列\n峰数={features["left_peak_count"]}', fontsize=12)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. 边界位置序列曲线（右侧）
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(right_edges, 'g-', linewidth=1, label='右边界位置')
        ax4.axhline(y=np.mean(right_edges), color='r', linestyle='--',
                   label=f'均值={np.mean(right_edges):.2f}')
        ax4.fill_between(range(len(right_edges)),
                        np.mean(right_edges) - features['right_rms_roughness'],
                        np.mean(right_edges) + features['right_rms_roughness'],
                        alpha=0.3, color='orange', label=f'RMS={features["right_rms_roughness"]:.2f}')
        ax4.set_xlabel('法线索引')
        ax4.set_ylabel('相对位置')
        ax4.set_title(f'右边界位置序列\n峰数={features["right_peak_count"]}', fontsize=12)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. 梯度能量热图
        ax5 = plt.subplot(2, 3, 5)
        denoised = preprocessed['denoised']
        grad_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        im = ax5.imshow(grad_mag, cmap='hot')
        ax5.plot(xs, ys, 'c-', linewidth=1, alpha=0.5)
        plt.colorbar(im, ax=ax5)
        ax5.set_title('梯度能量热图', fontsize=12)
        ax5.axis('off')
        
        # 6. 关键特征摘要
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
关键特征摘要

【粗糙度】
左侧RMS: {features['left_rms_roughness']:.3f}
右侧RMS: {features['right_rms_roughness']:.3f}
平均RMS: {features['avg_rms_roughness']:.3f}

【梯度能量（锐度）】
左侧: {features['left_gradient_energy']:.2f}
右侧: {features['right_gradient_energy']:.2f}
平均: {features['avg_gradient_energy']:.2f}

【缺口深度】
左侧最大: {features['left_max_notch']:.3f}
右侧最大: {features['right_max_notch']:.3f}
整体最大: {features['max_notch_depth']:.3f}

【峰统计】
左侧峰数: {features['left_peak_count']}
右侧峰数: {features['right_peak_count']}
左侧峰密度: {features['left_peak_density']:.4f}
右侧峰密度: {features['right_peak_density']:.4f}

【面积比】
撕裂/剪切: {features['tear_shear_area_ratio']:.3f}
        """
        
        ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存诊断图: {save_path}")
        
        plt.close()
    
    def plot_temporal_trends(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        绘制时序趋势曲线
        
        Args:
            df: 特征数据DataFrame
            save_path: 保存路径
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        
        # 定义要绘制的核心特征
        features_to_plot = [
            ('avg_rms_roughness', '平均RMS粗糙度', 'blue'),
            ('avg_gradient_energy', '平均梯度能量（锐度）', 'green'),
            ('max_notch_depth', '最大缺口深度', 'red'),
            ('left_peak_density', '左侧峰密度', 'purple'),
            ('right_peak_density', '右侧峰密度', 'orange'),
            ('tear_shear_area_ratio', '撕裂/剪切面积比', 'brown')
        ]
        
        for idx, (feature_name, feature_label, color) in enumerate(features_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            if feature_name not in df.columns:
                continue
            
            values = df[feature_name].values
            frames = df['frame_id'].values
            
            # 原始曲线
            ax.plot(frames, values, 'o-', color=color, alpha=0.3, 
                   linewidth=1, markersize=3, label='原始数据')
            
            # 滑动平均平滑
            if len(values) >= 5:
                from scipy.ndimage import uniform_filter1d
                smoothed = uniform_filter1d(values, size=5)
                ax.plot(frames, smoothed, '-', color=color, 
                       linewidth=2, label='滑动平均(窗口=5)')
            
            # 线性拟合趋势线
            if len(values) >= 2:
                z = np.polyfit(frames, values, 1)
                p = np.poly1d(z)
                ax.plot(frames, p(frames), '--', color='red', 
                       linewidth=2, alpha=0.7, label=f'趋势(斜率={z[0]:.6f})')
            
            ax.set_xlabel('帧编号', fontsize=12)
            ax.set_ylabel(feature_label, fontsize=12)
            ax.set_title(f'{feature_label}随时间变化', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axhline(y=mean_val, color='gray', linestyle=':', alpha=0.5)
            ax.text(0.02, 0.98, f'均值±标准差: {mean_val:.3f}±{std_val:.3f}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'剪刀磨损指标时序趋势分析 (第4-12卷钢卷, 共{len(df)}帧)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"已保存时序趋势图: {save_path}")
        
        plt.close()
    
    def plot_feature_correlations(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        绘制特征相关性分析
        
        Args:
            df: 特征数据DataFrame
            save_path: 保存路径
        """
        # 选择数值型特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'frame_id' in numeric_cols:
            numeric_cols.remove('frame_id')
        
        if len(numeric_cols) < 2:
            print("特征数量不足，跳过相关性分析")
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 相关性矩阵热图
        ax1 = plt.subplot(2, 2, 1)
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('特征相关性矩阵', fontsize=14, fontweight='bold')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)
        
        # 2. 核心特征分布箱线图
        ax2 = plt.subplot(2, 2, 2)
        core_features = ['avg_rms_roughness', 'avg_gradient_energy', 
                        'max_notch_depth', 'tear_shear_area_ratio']
        core_features = [f for f in core_features if f in numeric_cols]
        
        if core_features:
            # 归一化以便在同一图中显示
            df_normalized = df[core_features].copy()
            for col in core_features:
                df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
            
            df_normalized.boxplot(ax=ax2)
            ax2.set_title('核心特征分布（归一化）', fontsize=14, fontweight='bold')
            ax2.set_ylabel('归一化值')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax2.grid(True, alpha=0.3)
        
        # 3. 特征统计摘要
        ax3 = plt.subplot(2, 2, 3)
        ax3.axis('off')
        
        stats_text = "特征统计摘要\n" + "="*50 + "\n\n"
        for col in core_features[:6]:  # 显示前6个特征
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                stats_text += f"{col}:\n"
                stats_text += f"  均值={mean_val:.4f}, 标准差={std_val:.4f}\n"
                stats_text += f"  范围=[{min_val:.4f}, {max_val:.4f}]\n\n"
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 4. 帧间变化率分析
        ax4 = plt.subplot(2, 2, 4)
        
        # 计算帧间变化率（针对avg_rms_roughness）
        if 'avg_rms_roughness' in df.columns and len(df) > 1:
            values = df['avg_rms_roughness'].values
            change_rate = np.diff(values) / (values[:-1] + 1e-8) * 100
            frames = df['frame_id'].values[1:]
            
            ax4.plot(frames, change_rate, 'o-', color='purple', alpha=0.6, markersize=3)
            ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax4.set_xlabel('帧编号', fontsize=12)
            ax4.set_ylabel('变化率 (%)', fontsize=12)
            ax4.set_title('平均RMS粗糙度帧间变化率', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_change = np.mean(change_rate)
            ax4.text(0.02, 0.98, f'平均变化率: {mean_change:.2f}%',
                    transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.suptitle('特征相关性与统计分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"已保存相关性分析图: {save_path}")
        
        plt.close()
    
    def plot_wear_progression(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        绘制磨损递进综合图（验证磨损递增趋势）
        
        Args:
            df: 特征数据DataFrame
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        frames = df['frame_id'].values
        
        # 选择最能代表磨损的特征并归一化
        wear_indicators = {
            'avg_rms_roughness': ('平均粗糙度↑', 'red'),
            'max_notch_depth': ('最大缺口深度↑', 'orange'),
            'left_peak_density': ('左侧峰密度↑', 'blue'),
            'right_peak_density': ('右侧峰密度↑', 'green')
        }
        
        for feature_name, (label, color) in wear_indicators.items():
            if feature_name in df.columns:
                values = df[feature_name].values
                
                # 归一化到0-1
                values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
                
                # 平滑
                if len(values_norm) >= 7:
                    from scipy.ndimage import uniform_filter1d
                    values_smooth = uniform_filter1d(values_norm, size=7)
                else:
                    values_smooth = values_norm
                
                ax.plot(frames, values_smooth, '-', color=color, 
                       linewidth=2.5, label=label, alpha=0.8)
        
        # 计算综合磨损指数（多个指标的平均）
        if all(f in df.columns for f in ['avg_rms_roughness', 'max_notch_depth']):
            rms_norm = (df['avg_rms_roughness'] - df['avg_rms_roughness'].min()) / \
                      (df['avg_rms_roughness'].max() - df['avg_rms_roughness'].min() + 1e-8)
            notch_norm = (df['max_notch_depth'] - df['max_notch_depth'].min()) / \
                        (df['max_notch_depth'].max() - df['max_notch_depth'].min() + 1e-8)
            
            composite_index = (rms_norm.values + notch_norm.values) / 2
            
            if len(composite_index) >= 7:
                from scipy.ndimage import uniform_filter1d
                composite_smooth = uniform_filter1d(composite_index, size=7)
            else:
                composite_smooth = composite_index
            
            ax.plot(frames, composite_smooth, '-', color='black', 
                   linewidth=3, label='综合磨损指数', alpha=0.9, linestyle='--')
            
            # 拟合趋势
            z = np.polyfit(frames, composite_smooth, 1)
            p = np.poly1d(z)
            ax.plot(frames, p(frames), ':', color='purple', 
                   linewidth=2, label=f'趋势线(斜率={z[0]:.6f})')
        
        ax.set_xlabel('帧编号 (第4-12卷钢卷过程)', fontsize=13, fontweight='bold')
        ax.set_ylabel('归一化磨损指标', fontsize=13, fontweight='bold')
        ax.set_title('剪刀磨损递进趋势分析', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加背景色区分阶段
        total_frames = len(df)
        stage1 = int(total_frames * 0.2)  # 前20%
        stage3 = int(total_frames * 0.8)  # 后20%
        
        ax.axvspan(0, stage1, alpha=0.1, color='green', label='早期(卷4-5)')
        ax.axvspan(stage1, stage3, alpha=0.1, color='yellow')
        ax.axvspan(stage3, total_frames, alpha=0.1, color='red', label='后期(卷11-12)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"已保存磨损递进图: {save_path}")
        
        plt.close()


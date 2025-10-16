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
from typing import Dict, List, Optional, Tuple
import os
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.interpolate import UnivariateSpline

# 设置中文字体 - 多个备选方案确保兼容性
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Songti SC', 'STSong', 'SimHei', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
# 强制使用TrueType字体，避免字符丢失
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def compute_envelope(signal: np.ndarray, window: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算信号的上下包络线
    
    Args:
        signal: 输入信号
        window: 滑动窗口大小
        
    Returns:
        upper_envelope: 上包络线
        lower_envelope: 下包络线
    """
    if len(signal) < window:
        return signal.copy(), signal.copy()
    
    # 使用最大/最小滤波器计算包络
    upper_envelope = maximum_filter1d(signal, size=window, mode='nearest')
    lower_envelope = minimum_filter1d(signal, size=window, mode='nearest')
    
    return upper_envelope, lower_envelope


def robust_curve_fit(signal: np.ndarray, percentile_range: Tuple[float, float] = (5, 95),
                     smoothing: float = None, use_local_filter: bool = True, 
                     local_window: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    鲁棒曲线拟合：去除离群点后用样条曲线拟合
    
    优化策略（2025-10-14更新）：
    - 自适应平滑参数（根据数据变异系数调整）
    - 对稀疏峰值数据使用更小的平滑参数
    - **滑动窗口局部离群点检测**（避免将局部凹陷区域的所有点标记为离群点）
    
    Args:
        signal: 输入信号
        percentile_range: 保留数据的百分位范围（用于全局粗筛选）
        smoothing: 样条平滑参数（None表示自动）
        use_local_filter: 是否使用局部滑动窗口过滤（推荐True）
        local_window: 局部窗口大小（None表示自动）
        
    Returns:
        fitted_curve: 拟合曲线
        inlier_mask: 内点掩码（True表示非离群点）
        density_score: 点密度得分（用于可视化）
    """
    if len(signal) < 10:
        return signal.copy(), np.ones(len(signal), dtype=bool), np.ones(len(signal))
    
    # === 第1阶段：全局粗筛选（去除极端离群点） ===
    lower_bound = np.percentile(signal, percentile_range[0])
    upper_bound = np.percentile(signal, percentile_range[1])
    global_inlier_mask = (signal >= lower_bound) & (signal <= upper_bound)
    
    # === 第2阶段：局部滑动窗口精细过滤 ===
    if use_local_filter:
        # 自动确定窗口大小
        if local_window is None:
            local_window = max(min(len(signal) // 15, 101), 21)
            if local_window % 2 == 0:
                local_window += 1
        
        # 初始化局部内点掩码
        local_inlier_mask = np.ones(len(signal), dtype=bool)
        
        # 滑动窗口检测
        half_window = local_window // 2
        for i in range(len(signal)):
            start = max(0, i - half_window)
            end = min(len(signal), i + half_window + 1)
            
            window_mask = global_inlier_mask[start:end]
            window_data = signal[start:end][window_mask]
            
            if len(window_data) < 3:
                continue
            
            local_mean = np.mean(window_data)
            local_std = np.std(window_data)
            
            # 3-sigma规则
            if local_std > 0:
                z_score = abs(signal[i] - local_mean) / local_std
                if z_score > 3.0:
                    local_inlier_mask[i] = False
        
        # 综合全局和局部掩码
        inlier_mask = global_inlier_mask & local_inlier_mask
    else:
        inlier_mask = global_inlier_mask
    
    # === 计算密度得分（可视化用） ===
    density_window = min(21, len(signal) // 5)
    if density_window % 2 == 0:
        density_window += 1
    
    density_score = np.zeros(len(signal))
    for i in range(len(signal)):
        start = max(0, i - density_window // 2)
        end = min(len(signal), i + density_window // 2 + 1)
        local_vals = signal[start:end]
        local_var = np.var(local_vals)
        density_score[i] = 1.0 / (local_var + 1e-6)
    
    density_score = (density_score - density_score.min()) / (density_score.max() - density_score.min() + 1e-6)
    
    # === 第3阶段：样条曲线拟合 ===
    x_inliers = np.where(inlier_mask)[0]
    y_inliers = signal[inlier_mask]
    
    if len(x_inliers) < 4:
        fitted_curve = np.full(len(signal), np.mean(signal))
        return fitted_curve, inlier_mask, density_score
    
    # 自动计算平滑参数（自适应策略）
    if smoothing is None:
        mean_val = np.abs(np.mean(y_inliers)) + 1e-10
        std_val = np.std(y_inliers)
        cv = std_val / mean_val
        
        if cv > 0.5:
            smoothing = len(x_inliers) * 0.05
        elif cv > 0.2:
            smoothing = len(x_inliers) * 0.15
        else:
            smoothing = len(x_inliers) * 0.3
    
    try:
        spline = UnivariateSpline(x_inliers, y_inliers, s=smoothing, k=3)
        x_full = np.arange(len(signal))
        fitted_curve = spline(x_full)
    except:
        degree = min(3, len(x_inliers) - 1)
        coeffs = np.polyfit(x_inliers, y_inliers, degree)
        poly = np.poly1d(coeffs)
        x_full = np.arange(len(signal))
        fitted_curve = poly(x_full)
    
    return fitted_curve, inlier_mask, density_score


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
        生成单帧诊断图（3×3增强版）
        
        基于原始中心线检测缺口和峰，用红色mask标注
        
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
        
        from scipy.signal import find_peaks
        
        # 获取数据
        centerline_x_smooth = np.array(preprocessed.get('centerline_x', []))
        centerline_x = np.array(preprocessed.get('centerline_x_raw', centerline_x_smooth))  # 使用原始中心线
        
        # 创建3×3布局
        fig = plt.figure(figsize=(24, 18))
        
        # 1. 原图 + 中心线 + 法线网格
        ax1 = plt.subplot(3, 3, 1)
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
        ax2 = plt.subplot(3, 3, 2)
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
        
        # 3. 中心线位置序列（核心粗糙度指标）+ 包络线 + 鲁棒拟合
        ax3 = plt.subplot(3, 3, 3)
        if len(centerline_x) > 0:
            # 原始中心线（半透明）
            ax3.plot(centerline_x, 'b-', linewidth=0.8, label='原始中心线', alpha=0.4)
            
            # 计算包络线
            upper_env, lower_env = compute_envelope(centerline_x, window=15)
            ax3.plot(upper_env, 'r:', linewidth=1.5, label='上包络', alpha=0.7)
            ax3.plot(lower_env, 'g:', linewidth=1.5, label='下包络', alpha=0.7)
            ax3.fill_between(range(len(centerline_x)), lower_env, upper_env, 
                           alpha=0.1, color='gray', label='包络范围')
            
            # 鲁棒拟合曲线（去除离群点）
            fitted_curve, inlier_mask, density_score = robust_curve_fit(centerline_x)
            ax3.plot(fitted_curve, 'purple', linewidth=2, label='鲁棒拟合', alpha=0.8)
            
            # 标注离群点
            outlier_indices = np.where(~inlier_mask)[0]
            if len(outlier_indices) > 0:
                ax3.scatter(outlier_indices, centerline_x[outlier_indices], 
                          c='orange', s=20, marker='x', alpha=0.6, label=f'离群点({len(outlier_indices)}个)')
            
            # 均值线
            mean_centerline = np.mean(centerline_x)
            ax3.axhline(y=mean_centerline, color='gray', linestyle='--',
                       linewidth=1, alpha=0.5)
            
            # 标注RMS粗糙度
            rms = features.get('centerline_rms_roughness', 0)
            ax3.text(0.02, 0.98, f'RMS粗糙度: {rms:.3f}\n内点率: {inlier_mask.sum()/len(inlier_mask)*100:.1f}%',
                    transform=ax3.transAxes, fontsize=9, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax3.set_xlabel('法线索引', fontsize=10)
        ax3.set_ylabel('横向位置 (像素)', fontsize=10)
        ax3.set_title(f'中心线位置序列（核心粗糙度指标）\nRMS={features.get("centerline_rms_roughness", 0):.3f}',
                     fontsize=11, fontweight='bold')
        ax3.legend(fontsize=7, loc='best', ncol=2)
        ax3.grid(True, alpha=0.3)
        
        # 4. 中心线位置序列 + 峰检测（基于原始中心线）+ 包络线
        ax4 = plt.subplot(3, 3, 4)
        
        peak_count = features.get('centerline_peak_count', 0)
        peak_density = features.get('centerline_peak_density', 0.0)
        
        if len(centerline_x) > 0:
            # 原始中心线（半透明）
            ax4.plot(centerline_x, 'g-', linewidth=0.8, label='原始中心线', alpha=0.4)
            
            # 计算包络线
            upper_env, lower_env = compute_envelope(centerline_x, window=15)
            ax4.plot(upper_env, 'r:', linewidth=1.5, label='上包络', alpha=0.7)
            ax4.plot(lower_env, 'b:', linewidth=1.5, label='下包络', alpha=0.7)
            ax4.fill_between(range(len(centerline_x)), lower_env, upper_env, 
                           alpha=0.1, color='green', label='包络范围')
            
            # 鲁棒拟合曲线
            fitted_curve, inlier_mask, _ = robust_curve_fit(centerline_x)
            ax4.plot(fitted_curve, 'orange', linewidth=2, label='鲁棒拟合', alpha=0.8)
            
            # 重新检测峰以标注位置（基于原始数据）
            if len(centerline_x) > 3:
                x_fit = np.arange(len(centerline_x))
                z = np.polyfit(x_fit, centerline_x, deg=min(3, len(centerline_x)-1))
                p = np.poly1d(z)
                fitted_centerline = p(x_fit)
                residuals = centerline_x - fitted_centerline
                inverted_residuals = -residuals
                peaks, properties = find_peaks(inverted_residuals, prominence=1.0, distance=10)
                
                # 只保留负残差的峰
                valid_peaks = [pk for pk in peaks if residuals[pk] < 0]
                if len(valid_peaks) > 0:
                    peak_heights = [centerline_x[int(p)] for p in valid_peaks]
                    ax4.plot(valid_peaks, peak_heights, 'r*', markersize=12,
                            label=f'检测到{len(valid_peaks)}个峰', zorder=10,
                            markeredgecolor='darkred', markeredgewidth=1)
            
            # 标注峰密度
            ax4.text(0.02, 0.98, 
                    f'峰数量: {peak_count}\n峰密度: {peak_density:.4f}',
                    transform=ax4.transAxes, fontsize=9, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        ax4.set_xlabel('法线索引', fontsize=10)
        ax4.set_ylabel('横向位置 (像素)', fontsize=10)
        ax4.set_title('中心线位置序列 + 峰检测\n（剪切面微缺口）',
                     fontsize=11, fontweight='bold')
        ax4.legend(fontsize=7, loc='best', ncol=2)
        ax4.grid(True, alpha=0.3)
        
        # 5. 中心线位置序列 + 缺口检测（基于原始中心线）+ 包络线
        ax5 = plt.subplot(3, 3, 5)
        
        notch_depth = features.get('centerline_max_notch_depth', 0.0)
        notch_idx = int(features.get('centerline_notch_idx', 0))
        
        if len(centerline_x) > 3:
            # 原始中心线（半透明）
            ax5.plot(centerline_x, 'b-', linewidth=0.8, label='原始中心线', alpha=0.4)
            
            # 计算包络线
            upper_env, lower_env = compute_envelope(centerline_x, window=15)
            ax5.plot(upper_env, 'r:', linewidth=1.5, label='上包络', alpha=0.7)
            ax5.plot(lower_env, 'g:', linewidth=1.5, label='下包络', alpha=0.7)
            ax5.fill_between(range(len(centerline_x)), lower_env, upper_env, 
                           alpha=0.1, color='blue', label='包络范围')
            
            # 鲁棒拟合曲线（主趋势线）
            fitted_curve, inlier_mask, _ = robust_curve_fit(centerline_x)
            ax5.plot(fitted_curve, 'purple', linewidth=2, label='鲁棒拟合', alpha=0.8)
            
            # 多项式拟合（用于检测缺口）
            x_fit = np.arange(len(centerline_x))
            z = np.polyfit(x_fit, centerline_x, deg=min(3, len(centerline_x)-1))
            p = np.poly1d(z)
            fitted_line = p(x_fit)
            ax5.plot(x_fit, fitted_line, 'orange', linestyle='--', linewidth=1.5, 
                    label='多项式基准', alpha=0.6)
            
            # 标注最大缺口
            if notch_depth > 0 and notch_idx < len(centerline_x):
                ax5.plot(notch_idx, centerline_x[notch_idx], 'r*', markersize=15,
                        label=f'最大缺口', zorder=10,
                        markeredgecolor='darkred', markeredgewidth=1)
                # 简化标注，只显示数值，紧靠五角星
                ax5.text(notch_idx + 5, centerline_x[notch_idx] - 1, 
                        f'{notch_depth:.1f}',
                        fontsize=10, color='red', fontweight='bold',
                        va='top', ha='left')
        
        ax5.set_xlabel('法线索引', fontsize=10)
        ax5.set_ylabel('横向位置 (像素)', fontsize=10)
        ax5.set_title(f'中心线位置序列 + 缺口检测\n最大缺口深度={notch_depth:.3f}',
                     fontsize=11, fontweight='bold')
        ax5.legend(fontsize=7, loc='best', ncol=2)
        ax5.grid(True, alpha=0.3)
        
        # 6. 梯度能量热图
        ax6 = plt.subplot(3, 3, 6)
        denoised = preprocessed['denoised']
        grad_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 使用原始宽高比，不要auto拉伸
        im = ax6.imshow(grad_mag, cmap='hot')
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        ax6.set_title(f'梯度能量热图\n平均值={features.get("avg_gradient_energy", 0):.1f}',
                     fontsize=11, fontweight='bold')
        ax6.axis('off')
        
        # 7. 最大缺口位置可视化（基于中心线 + 红色mask标注）
        ax7 = plt.subplot(3, 3, 7)
        
        # 复制图像用于绘制
        if len(image.shape) == 2:
            display_img1 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_img1 = image.copy()
        
        # 绘制原始中心线（青色）
        for y_idx in range(len(centerline_x)):
            x_pos = int(centerline_x[y_idx])
            if 0 <= x_pos < display_img1.shape[1] and 0 <= y_idx < display_img1.shape[0]:
                display_img1[y_idx, x_pos] = [0, 255, 255]  # 青色中心线
        
        # 标注最大缺口（红色mask）
        if notch_depth > 0 and notch_idx < len(centerline_x):
            notch_y = notch_idx
            notch_x = int(centerline_x[notch_idx])
            
            # 涂红色区域（在缺口位置向左涂）
            radius = 8
            for dy in range(-radius, radius+1):
                y_pos = notch_y + dy
                if 0 <= y_pos < len(centerline_x) and 0 <= y_pos < display_img1.shape[0]:
                    x_pos = int(centerline_x[y_pos])
                    if 0 <= x_pos < display_img1.shape[1]:
                        # 向左涂6个像素
                        for dx in range(-5, 1):
                            if 0 <= x_pos + dx < display_img1.shape[1]:
                                display_img1[y_pos, x_pos + dx] = [255, 0, 0]  # 红色
        
        ax7.imshow(display_img1)
        ax7.set_title(f'最大缺口位置可视化\n深度={notch_depth:.2f}像素', 
                     fontsize=13, fontweight='bold')
        ax7.axis('off')
        
        # 8. 剪切面峰值位置可视化（基于中心线 + 红色mask标注）
        ax8 = plt.subplot(3, 3, 8)
        
        # 复制图像用于绘制
        if len(image.shape) == 2:
            display_img2 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_img2 = image.copy()
        
        # 绘制原始中心线（青色）
        for y_idx in range(len(centerline_x)):
            x_pos = int(centerline_x[y_idx])
            if 0 <= x_pos < display_img2.shape[1] and 0 <= y_idx < display_img2.shape[0]:
                display_img2[y_idx, x_pos] = [0, 255, 255]  # 青色中心线
        
        # 重新检测峰值位置（用于可视化）
        valid_peak_indices = []
        if len(centerline_x) > 3:
            x_fit = np.arange(len(centerline_x))
            z = np.polyfit(x_fit, centerline_x, deg=min(3, len(centerline_x)-1))
            p = np.poly1d(z)
            fitted_centerline = p(x_fit)
            residuals = centerline_x - fitted_centerline
            inverted_residuals = -residuals
            peaks, properties = find_peaks(inverted_residuals, prominence=1.0, distance=10)
            
            # 只保留负残差的峰
            valid_peak_indices = [pk for pk in peaks if residuals[pk] < 0]
        
        # 标注峰值（红色mask）
        for peak_idx in valid_peak_indices:
            peak_y = int(peak_idx)
            peak_x = int(centerline_x[peak_y])
            
            # 涂红色区域
            radius = 8
            for dy in range(-radius, radius+1):
                y_pos = peak_y + dy
                if 0 <= y_pos < len(centerline_x) and 0 <= y_pos < display_img2.shape[0]:
                    x_pos = int(centerline_x[y_pos])
                    if 0 <= x_pos < display_img2.shape[1]:
                        # 向左涂6个像素（峰是向左突出）
                        for dx in range(-5, 1):
                            if 0 <= x_pos + dx < display_img2.shape[1]:
                                display_img2[y_pos, x_pos + dx] = [255, 0, 0]  # 红色
        
        ax8.imshow(display_img2)
        ax8.set_title(f'剪切面峰值位置可视化\n检测到{len(valid_peak_indices)}个峰', 
                     fontsize=13, fontweight='bold')
        ax8.axis('off')
        
        # 9. 关键特征摘要
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_text = f"""
【关键特征摘要】

基于中心线检测指标：
1. 中心线RMS粗糙度: {features.get('centerline_rms_roughness', 0):.3f}
   (核心磨损指标，反映刃口形状变化)

2. 最大缺口深度: {features.get('centerline_max_notch_depth', 0):.3f} 像素
   (撕裂面凹陷，位于y={features.get('centerline_notch_idx', 0)})

3. 剪切面峰数量: {features.get('centerline_peak_count', 0)}
   (剪切面向左突起，反映微缺口)

4. 峰密度: {features.get('centerline_peak_density', 0):.4f}
   (反映缺口分布密集度)

5. 梯度能量: {features.get('avg_gradient_energy', 0):.1f}
   (反映表面粗糙度和剪切质量)

注: 以上缺口和峰均基于原始中心线检测
        """
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'帧 {frame_id} 诊断分析（3×3增强版）', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            # print(f"已保存诊断图: {save_path}")  # 减少打印信息
        
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


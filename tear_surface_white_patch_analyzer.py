"""
撕裂面白色斑块分析器
专门针对用户观察到的"撕裂面白色斑块随钢卷数量增加而变化"现象进行详细分析

用法:
    python tear_surface_white_patch_analyzer.py --roi_dir data/roi_imgs --output_dir data/white_patch_analysis
    
功能:
    - 4种检测方法对比
    - 4种量化指标分析  
    - 时序曲线和按卷统计
    - 方法推荐报告
"""
import os
import sys
import argparse
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Songti SC', 'STSong', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class WhitePatchAnalyzer:
    """白色斑块分析器"""
    
    def __init__(self, roi_dir: str, output_dir: str):
        """
        初始化分析器
        
        Args:
            roi_dir: ROI图像目录
            output_dir: 输出目录
        """
        self.roi_dir = os.path.abspath(roi_dir)
        self.output_dir = os.path.abspath(output_dir)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有ROI文件
        self.image_files = sorted(glob.glob(os.path.join(roi_dir, 'frame_*_roi.png')))
        print(f"找到 {len(self.image_files)} 个ROI图像")
    
    def extract_left_region_and_mask(self, image: np.ndarray):
        """
        提取左侧撕裂面区域及掩码
        
        Args:
            image: 输入图像
            
        Returns:
            (left_region, left_mask)
        """
        height, width = image.shape
        
        # 简单方法：找白色区域中最暗点作为分界线
        mask_white = image > 100
        centerline_x = []
        
        for y in range(height):
            row = image[y, :]
            white_indices = np.where(mask_white[y, :])[0]
            
            if len(white_indices) > 10:
                # 在白色区域内找最暗点
                search_start = white_indices[0] + 5
                search_end = white_indices[-1] - 5
                
                if search_end > search_start:
                    min_idx = search_start + np.argmin(row[search_start:search_end])
                    centerline_x.append(min_idx)
                else:
                    centerline_x.append((white_indices[0] + white_indices[-1]) // 2)
            else:
                centerline_x.append(width // 2)
        
        # 平滑中心线
        if len(centerline_x) > 51:
            centerline_x = savgol_filter(centerline_x, 51, 3)
        centerline_x = np.array(centerline_x, dtype=int)
        
        # 创建左侧掩码
        left_mask = np.zeros_like(image, dtype=np.uint8)
        for y in range(height):
            if y < len(centerline_x):
                left_mask[y, :centerline_x[y]] = 255
        
        # 提取左侧区域
        left_region = cv2.bitwise_and(image, image, mask=left_mask)
        
        return left_region, left_mask
    
    def detect_white_patches_method1(self, image: np.ndarray, mask: np.ndarray, threshold: int = 200):
        """方法1：固定阈值法"""
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_and(binary, binary, mask=mask)
        return binary
    
    def detect_white_patches_method2(self, image: np.ndarray, mask: np.ndarray, min_threshold: int = 170):
        """方法2：自适应阈值法（Otsu + 最小阈值约束）"""
        masked_pixels = image[mask > 0]
        if len(masked_pixels) == 0:
            return np.zeros_like(image)
        
        # 计算Otsu阈值
        otsu_threshold, _ = cv2.threshold(masked_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 使用Otsu和最小阈值的较大值，避免阈值过低
        # 当前min_threshold=170，在过度检测和漏检之间取得平衡
        threshold = max(otsu_threshold, min_threshold)
        
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_and(binary, binary, mask=mask)
        return binary
    
    def detect_white_patches_method3(self, image: np.ndarray, mask: np.ndarray, sigma_factor: float = 1.5):
        """方法3：相对亮度法（统计阈值）"""
        masked_pixels = image[mask > 0]
        if len(masked_pixels) == 0:
            return np.zeros_like(image)
        
        mean_val = np.mean(masked_pixels)
        std_val = np.std(masked_pixels)
        threshold = mean_val + sigma_factor * std_val
        
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_and(binary, binary, mask=mask)
        return binary
    
    def detect_white_patches_method4(self, image: np.ndarray, mask: np.ndarray, kernel_size: int = 15):
        """方法4：局部对比度法（形态学Top-Hat）"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        tophat_masked = tophat[mask > 0]
        if len(tophat_masked) == 0 or tophat_masked.max() == 0:
            return np.zeros_like(image)
        
        threshold, _ = cv2.threshold(tophat_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary = cv2.threshold(tophat, max(threshold, 1), 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_and(binary, binary, mask=mask)
        return binary
    
    def compute_metrics(self, binary: np.ndarray, original: np.ndarray, mask: np.ndarray):
        """计算8个量化指标"""
        total_area = np.sum(mask > 0)
        white_area = np.sum(binary > 0)
        area_ratio = (white_area / total_area * 100) if total_area > 0 else 0.0
        
        num_labels, _ = cv2.connectedComponents(binary)
        patch_count = num_labels - 1
        
        white_pixels = original[binary > 0]
        if len(white_pixels) > 0:
            avg_brightness = np.mean(white_pixels)
            brightness_std = np.std(white_pixels)
        else:
            avg_brightness = 0.0
            brightness_std = 0.0
        
        # 单个白斑平均面积
        avg_patch_area = (area_ratio / patch_count) if patch_count > 0 else 0.0
        
        # 综合指标：白斑数量+亮度标准差
        composite_index = patch_count + brightness_std
        
        # 亮度直方图熵
        brightness_entropy = self._compute_brightness_entropy(white_pixels)
        
        # 斑块面积分布熵
        patch_area_entropy = self._compute_patch_area_entropy(binary)
        
        return {
            'area_ratio': area_ratio,
            'patch_count': patch_count,
            'avg_brightness': avg_brightness,
            'brightness_std': brightness_std,
            'avg_patch_area': avg_patch_area,
            'composite_index': composite_index,
            'brightness_entropy': brightness_entropy,
            'patch_area_entropy': patch_area_entropy
        }
    
    def _compute_brightness_entropy(self, pixels: np.ndarray, bins: int = 32):
        """计算亮度直方图的香农熵"""
        if len(pixels) == 0:
            return 0.0
        
        try:
            hist, _ = np.histogram(pixels, bins=bins, range=(0, 256))
            hist_norm = hist / (np.sum(hist) + 1e-10)
            hist_norm = hist_norm[hist_norm > 0]
            entropy = -np.sum(hist_norm * np.log2(hist_norm))
        except:
            entropy = 0.0
        
        return float(entropy)
    
    def _compute_patch_area_entropy(self, binary: np.ndarray, bins: int = 20):
        """计算斑块面积分布的香农熵"""
        try:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            if num_labels <= 1:
                return 0.0
            
            areas = stats[1:, cv2.CC_STAT_AREA]
            
            if len(areas) == 0 or areas.max() == areas.min():
                return 0.0
            
            hist, _ = np.histogram(areas, bins=bins)
            hist_norm = hist / (np.sum(hist) + 1e-10)
            hist_norm = hist_norm[hist_norm > 0]
            entropy = -np.sum(hist_norm * np.log2(hist_norm))
        except:
            entropy = 0.0
        
        return float(entropy)
    
    def analyze_all_frames(self):
        """分析所有帧，提取4种方法×8种指标的特征"""
        print("\n开始分析所有帧...")
        
        results = []
        
        for filepath in tqdm(self.image_files, desc="处理图像"):
            try:
                # 读取图像
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                # 提取帧ID
                basename = os.path.basename(filepath)
                frame_id = int(basename.split('_')[1])
                
                # 提取左侧区域
                left_region, left_mask = self.extract_left_region_and_mask(image)
                
                # 4种方法检测
                binary1 = self.detect_white_patches_method1(left_region, left_mask)
                binary2 = self.detect_white_patches_method2(left_region, left_mask)
                binary3 = self.detect_white_patches_method3(left_region, left_mask)
                binary4 = self.detect_white_patches_method4(left_region, left_mask)
                
                # 计算指标
                metrics1 = self.compute_metrics(binary1, left_region, left_mask)
                metrics2 = self.compute_metrics(binary2, left_region, left_mask)
                metrics3 = self.compute_metrics(binary3, left_region, left_mask)
                metrics4 = self.compute_metrics(binary4, left_region, left_mask)
                
                # 保存结果
                row = {'frame_id': frame_id}
                for i, metrics in enumerate([metrics1, metrics2, metrics3, metrics4], 1):
                    row[f'area_ratio_m{i}'] = metrics['area_ratio']
                    row[f'patch_count_m{i}'] = metrics['patch_count']
                    row[f'avg_brightness_m{i}'] = metrics['avg_brightness']
                    row[f'brightness_std_m{i}'] = metrics['brightness_std']
                    row[f'avg_patch_area_m{i}'] = metrics['avg_patch_area']
                    row[f'composite_index_m{i}'] = metrics['composite_index']
                    row[f'brightness_entropy_m{i}'] = metrics['brightness_entropy']
                    row[f'patch_area_entropy_m{i}'] = metrics['patch_area_entropy']
                
                results.append(row)
                
            except Exception as e:
                print(f"\n处理 {filepath} 时出错: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        # 保存CSV
        csv_path = os.path.join(self.output_dir, 'white_patch_features.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n已保存特征文件: {csv_path}")
        
        return df
    
    def compare_methods(self, df: pd.DataFrame):
        """生成方法对比可视化（抽样6帧）"""
        print("\n生成方法对比可视化...")
        
        # 选择6帧：早/中/晚各2帧
        n_frames = len(df)
        sample_indices = [
            n_frames // 10,
            n_frames // 5,
            n_frames // 2 - n_frames // 10,
            n_frames // 2 + n_frames // 10,
            n_frames * 4 // 5,
            n_frames * 9 // 10
        ]
        
        fig, axes = plt.subplots(6, 6, figsize=(24, 24))
        
        for row_idx, sample_idx in enumerate(sample_indices):
            frame_id = int(df.iloc[sample_idx]['frame_id'])
            
            # 读取原图
            filepath = os.path.join(self.roi_dir, f'frame_{frame_id:06d}_roi.png')
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                continue
            
            left_region, left_mask = self.extract_left_region_and_mask(image)
            
            # 绘制原图
            axes[row_idx, 0].imshow(image, cmap='gray')
            axes[row_idx, 0].set_title(f'帧{frame_id}\n原图', fontsize=10)
            axes[row_idx, 0].axis('off')
            
            # 绘制左侧区域
            axes[row_idx, 1].imshow(left_region, cmap='gray')
            axes[row_idx, 1].set_title('左侧撕裂面', fontsize=10)
            axes[row_idx, 1].axis('off')
            
            # 4种方法
            binary1 = self.detect_white_patches_method1(left_region, left_mask)
            binary2 = self.detect_white_patches_method2(left_region, left_mask)
            binary3 = self.detect_white_patches_method3(left_region, left_mask)
            binary4 = self.detect_white_patches_method4(left_region, left_mask)
            
            methods = [binary1, binary2, binary3, binary4]
            method_names = ['固定阈值', 'Otsu自适应', '相对亮度', '形态学Top-Hat']
            
            for col_idx, (binary, name) in enumerate(zip(methods, method_names), 2):
                # 叠加显示
                overlay = cv2.cvtColor(left_region, cv2.COLOR_GRAY2RGB)
                overlay[binary > 0] = [255, 0, 0]  # 红色标记白斑
                
                axes[row_idx, col_idx].imshow(overlay)
                axes[row_idx, col_idx].set_title(f'方法{col_idx-1}:{name}', fontsize=10)
                axes[row_idx, col_idx].axis('off')
        
        plt.suptitle('撕裂面白斑检测方法对比（红色=检测到的白斑）', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'method_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {self.output_dir}/method_comparison.png")
    
    def plot_temporal_curves(self, df: pd.DataFrame):
        """绘制时序曲线（4方法×8指标=32条曲线）"""
        print("\n生成时序曲线...")
        
        fig, axes = plt.subplots(8, 4, figsize=(20, 32))
        
        methods = ['m1', 'm2', 'm3', 'm4']
        method_names = ['固定阈值', 'Otsu自适应', '相对亮度', '形态学Top-Hat']
        metrics = ['area_ratio', 'patch_count', 'avg_brightness', 'brightness_std', 'avg_patch_area', 'composite_index', 'brightness_entropy', 'patch_area_entropy']
        metric_names = ['白斑面积占比(%)', '白斑数量(个)', '平均亮度', '亮度标准差', '单个白斑平均面积(%)', '综合指标(数量+亮度std)', '亮度直方图熵', '斑块面积分布熵']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for row_idx, metric in enumerate(metrics):
            for col_idx, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
                ax = axes[row_idx, col_idx]
                
                col_name = f'{metric}_{method}'
                values = df[col_name].values
                frames = df['frame_id'].values
                
                # 原始数据
                ax.plot(frames, values, '-', alpha=0.3, color=color, linewidth=0.5)
                
                # 平滑曲线
                window = min(51, len(values)//10*2+1)
                if window >= 5:
                    smoothed = savgol_filter(values, window_length=window, polyorder=3)
                    ax.plot(frames, smoothed, '-', color=color, linewidth=2.5, label='平滑曲线')
                
                # 线性趋势
                z = np.polyfit(frames, values, 1)
                trend = np.poly1d(z)
                ax.plot(frames, trend(frames), '--', color='red', linewidth=2, alpha=0.7, 
                       label=f'趋势(斜率={z[0]:.2e})')
                
                ax.set_xlabel('帧编号', fontsize=10)
                ax.set_ylabel(metric_names[row_idx], fontsize=10)
                ax.set_title(f'{method_name} - {metric_names[row_idx]}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('撕裂面白斑特征时序演变（4方法×8指标）', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'temporal_curves_4x8.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {self.output_dir}/temporal_curves_4x8.png")
    
    def plot_coil_statistics(self, df: pd.DataFrame, coil_csv: str = None):
        """绘制按卷统计（如果有卷号信息）"""
        print("\n生成按卷统计...")
        
        # 尝试加载卷号信息
        if coil_csv and os.path.exists(coil_csv):
            coil_df = pd.read_csv(coil_csv, encoding='utf-8-sig')
            if 'frame_id' in coil_df.columns and 'coil_id' in coil_df.columns:
                df = df.merge(coil_df[['frame_id', 'coil_id']], on='frame_id', how='left')
        
        if 'coil_id' not in df.columns or df['coil_id'].isna().all():
            print("  警告: 无卷号信息，跳过按卷统计")
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        methods = ['m1', 'm2', 'm3', 'm4']
        method_names = ['固定阈值', 'Otsu自适应', '相对亮度', '形态学Top-Hat']
        
        for idx, (method, method_name) in enumerate(zip(methods, method_names)):
            ax1 = axes[idx]
            ax2 = axes[idx + 4]
            
            # 面积占比
            coil_ids = sorted(df['coil_id'].unique())
            area_means = [df[df['coil_id']==cid][f'area_ratio_{method}'].mean() for cid in coil_ids]
            
            ax1.bar(range(len(coil_ids)), area_means, color='steelblue', alpha=0.7)
            ax1.plot(range(len(coil_ids)), area_means, 'ro-', linewidth=2, markersize=6)
            ax1.set_xlabel('钢卷编号', fontsize=10)
            ax1.set_ylabel('白斑面积占比(%)', fontsize=10)
            ax1.set_title(f'{method_name} - 面积占比按卷变化', fontsize=11, fontweight='bold')
            ax1.set_xticks(range(len(coil_ids)))
            ax1.set_xticklabels([f'卷{int(c)}' for c in coil_ids], rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 斑块数量
            count_means = [df[df['coil_id']==cid][f'patch_count_{method}'].mean() for cid in coil_ids]
            
            ax2.bar(range(len(coil_ids)), count_means, color='coral', alpha=0.7)
            ax2.plot(range(len(coil_ids)), count_means, 'go-', linewidth=2, markersize=6)
            ax2.set_xlabel('钢卷编号', fontsize=10)
            ax2.set_ylabel('白斑数量(个)', fontsize=10)
            ax2.set_title(f'{method_name} - 白斑数量按卷变化', fontsize=11, fontweight='bold')
            ax2.set_xticks(range(len(coil_ids)))
            ax2.set_xticklabels([f'卷{int(c)}' for c in coil_ids], rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('撕裂面白斑按卷统计分析', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'coil_statistics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {self.output_dir}/coil_statistics.png")
    
    def visualize_white_patches_with_markers(self, sample_interval: int = 100):
        """
        可视化白斑检测结果，用小圆圈标注每个白斑
        
        Args:
            sample_interval: 采样间隔（每隔多少帧保存一次）
        """
        print(f"\n生成白斑标注可视化（每隔{sample_interval}帧）...")
        
        # 创建输出目录
        markers_dir = os.path.join(self.output_dir, 'white_patch_markers')
        os.makedirs(markers_dir, exist_ok=True)
        
        # 选择要可视化的帧
        sampled_indices = list(range(0, len(self.image_files), sample_interval))
        
        methods = [
            ('method1', self.detect_white_patches_method1, '固定阈值'),
            ('method2', self.detect_white_patches_method2, 'Otsu自适应'),
            ('method3', self.detect_white_patches_method3, '相对亮度'),
            ('method4', self.detect_white_patches_method4, '形态学Top-Hat')
        ]
        
        for idx in tqdm(sampled_indices, desc="生成标注图"):
            try:
                filepath = self.image_files[idx]
                
                # 读取图像
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                # 提取帧ID
                basename = os.path.basename(filepath)
                frame_id = int(basename.split('_')[1])
                
                # 提取左侧区域
                left_region, left_mask = self.extract_left_region_and_mask(image)
                
                # 创建3x2子图布局（4个标注图 + 2个直方图）
                fig = plt.figure(figsize=(18, 24))
                gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
                
                # 前4个子图：4种方法的标注
                all_binaries = []  # 保存二值图用于直方图绘制
                all_areas = []  # 保存所有方法检测到的白斑面积
                
                for method_idx, (method_name, detect_func, display_name) in enumerate(methods):
                    row = method_idx // 2
                    col = method_idx % 2
                    ax = fig.add_subplot(gs[row, col])
                    
                    # 检测白斑
                    binary = detect_func(left_region, left_mask)
                    all_binaries.append(binary)
                    
                    # 转换为彩色图便于标注
                    display_img = cv2.cvtColor(left_region, cv2.COLOR_GRAY2RGB)
                    
                    # 查找白斑的连通域
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    
                    # 统计白斑数量和面积（排除背景）
                    valid_patches = 0
                    patch_areas = []
                    
                    # 在每个白斑质心画圆圈
                    for i in range(1, num_labels):  # 从1开始，跳过背景
                        area = stats[i, cv2.CC_STAT_AREA]
                        
                        # 过滤太小的斑块（面积小于5像素）
                        if area < 5:
                            continue
                        
                        valid_patches += 1
                        patch_areas.append(area)
                        
                        # 获取质心坐标
                        cx, cy = int(centroids[i][0]), int(centroids[i][1])
                        
                        # 根据面积决定圆圈大小
                        radius = max(3, min(int(np.sqrt(area) * 0.5), 10))
                        
                        # 画红色圆圈
                        cv2.circle(display_img, (cx, cy), radius, (255, 0, 0), 2)
                        
                        # 可选：画一个小点标记质心
                        cv2.circle(display_img, (cx, cy), 1, (0, 255, 0), -1)
                    
                    all_areas.append(patch_areas)
                    
                    # 显示图像
                    ax.imshow(display_img)
                    ax.set_title(f'{display_name}\n帧{frame_id} - 检测到{valid_patches}个白斑', 
                               fontsize=14, fontweight='bold')
                    ax.axis('off')
                    
                    # 添加比例尺信息
                    h, w = left_region.shape
                    ax.text(0.02, 0.98, f'白斑数: {valid_patches}', 
                           transform=ax.transAxes, fontsize=12, 
                           verticalalignment='top', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                # 第5个子图：亮度直方图对比（左下）
                ax_brightness = fig.add_subplot(gs[2, 0])
                
                # 提取撕裂面整体亮度
                tear_pixels = left_region[left_mask > 0]
                if len(tear_pixels) > 0:
                    # 撕裂面整体亮度直方图（灰色）
                    ax_brightness.hist(tear_pixels, bins=50, color='gray', alpha=0.5, 
                                      label='撕裂面整体亮度', edgecolor='black', linewidth=0.5)
                    
                    # 4种方法检测到的白斑亮度直方图
                    colors_hist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    for idx, (binary, method_name) in enumerate(zip(all_binaries, ['方法1', '方法2', '方法3', '方法4'])):
                        white_pixels = left_region[binary > 0]
                        if len(white_pixels) > 0:
                            ax_brightness.hist(white_pixels, bins=50, color=colors_hist[idx], 
                                             alpha=0.3, label=f'{method_name}白斑', 
                                             edgecolor=colors_hist[idx], linewidth=1)
                
                ax_brightness.set_xlabel('亮度值', fontsize=12, fontweight='bold')
                ax_brightness.set_ylabel('像素数量', fontsize=12, fontweight='bold')
                ax_brightness.set_title(f'撕裂面亮度分布对比\n帧{frame_id}', fontsize=14, fontweight='bold')
                ax_brightness.legend(fontsize=10, loc='best')
                ax_brightness.grid(True, alpha=0.3, axis='y')
                
                # 第6个子图：斑块面积直方图对比（右下）
                ax_area = fig.add_subplot(gs[2, 1])
                
                colors_hist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                for idx, (areas, method_name) in enumerate(zip(all_areas, ['方法1', '方法2', '方法3', '方法4'])):
                    if len(areas) > 0:
                        ax_area.hist(areas, bins=20, color=colors_hist[idx], alpha=0.5,
                                   label=f'{method_name} ({len(areas)}个)',
                                   edgecolor=colors_hist[idx], linewidth=1)
                
                ax_area.set_xlabel('斑块面积 (像素数)', fontsize=12, fontweight='bold')
                ax_area.set_ylabel('斑块数量', fontsize=12, fontweight='bold')
                ax_area.set_title(f'白斑面积分布对比\n帧{frame_id}', fontsize=14, fontweight='bold')
                ax_area.legend(fontsize=10, loc='best')
                ax_area.grid(True, alpha=0.3, axis='y')
                
                plt.suptitle(f'撕裂面白斑综合分析 - 帧{frame_id}\n（上：标注图，下：直方图对比）', 
                           fontsize=18, fontweight='bold')
                plt.tight_layout()
                
                # 保存图像
                save_path = os.path.join(markers_dir, f'frame_{frame_id:06d}_markers.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"\n处理帧{idx}时出错: {e}")
                continue
        
        print(f"✓ 已保存标注图到: {markers_dir}")
        print(f"  共生成 {len(sampled_indices)} 张标注图")
    
    def generate_recommendation_report(self, df: pd.DataFrame):
        """生成方法推荐报告"""
        print("\n生成方法推荐报告...")
        
        report_lines = []
        report_lines.append("# 撕裂面白斑检测方法推荐报告\n\n")
        
        methods = ['m1', 'm2', 'm3', 'm4']
        method_names = ['方法1:固定阈值法', '方法2:Otsu自适应法', '方法3:相对亮度法', '方法4:形态学Top-Hat法']
        metrics = ['area_ratio', 'patch_count']
        
        report_lines.append("## 方法评估\n\n")
        report_lines.append("评估维度：\n")
        report_lines.append("1. **单调性**：与帧序号的Spearman相关系数（反映是否随磨损递增）\n")
        report_lines.append("2. **稳定性**：变异系数CV（标准差/均值，越小越稳定）\n")
        report_lines.append("3. **灵敏度**：数值变化范围\n\n")
        
        # 评估每种方法
        evaluation_results = []
        
        for method, method_name in zip(methods, method_names):
            report_lines.append(f"### {method_name}\n\n")
            
            for metric in metrics:
                col_name = f'{metric}_{method}'
                values = df[col_name].values
                frames = df['frame_id'].values
                
                # 单调性
                corr, pval = spearmanr(frames, values)
                
                # 稳定性
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val > 0 else 0
                
                # 灵敏度
                value_range = np.max(values) - np.min(values)
                
                metric_cn = '面积占比' if metric == 'area_ratio' else '斑块数量'
                
                report_lines.append(f"**指标: {metric_cn}**\n")
                report_lines.append(f"- 单调性（Spearman相关系数）: {corr:.4f} (p-value={pval:.4e})\n")
                report_lines.append(f"- 稳定性（变异系数CV）: {cv:.4f}\n")
                report_lines.append(f"- 灵敏度（数值范围）: {value_range:.2f}\n")
                report_lines.append(f"- 均值: {mean_val:.2f}, 标准差: {std_val:.2f}\n\n")
                
                evaluation_results.append({
                    'method': method_name,
                    'metric': metric_cn,
                    'monotonicity': abs(corr),
                    'stability': 1/(cv+0.01),  # 转换为稳定性得分
                    'sensitivity': value_range
                })
        
        # 综合推荐
        report_lines.append("## 综合推荐\n\n")
        
        # 找最佳方法
        eval_df = pd.DataFrame(evaluation_results)
        eval_df['综合得分'] = eval_df['monotonicity'] * 0.5 + eval_df['stability'] * 0.01 + eval_df['sensitivity'] * 0.001
        
        best_method = eval_df.loc[eval_df['综合得分'].idxmax()]
        
        report_lines.append(f"**推荐方法**: {best_method['method']}\n")
        report_lines.append(f"**推荐指标**: {best_method['metric']}\n")
        report_lines.append(f"**综合得分**: {best_method['综合得分']:.4f}\n\n")
        
        report_lines.append("**说明**:\n")
        report_lines.append("- 该方法在单调性、稳定性和灵敏度方面取得了最佳平衡\n")
        report_lines.append("- 建议在后续分析中使用该方法作为主要指标\n")
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'method_recommendation.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        print(f"已保存: {report_path}")
        print(f"\n推荐方法: {best_method['method']} - {best_method['metric']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='撕裂面白色斑块分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tear_surface_white_patch_analyzer.py --roi_dir data/roi_imgs --output_dir data/white_patch_analysis
  
  # 如果有卷号信息，可以指定CSV文件
  python tear_surface_white_patch_analyzer.py --roi_dir data/roi_imgs --output_dir data/white_patch_analysis --coil_csv data/analysis/features/wear_features_with_coils.csv
        """
    )
    
    parser.add_argument('--roi_dir', required=True, help='ROI图像目录路径')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    parser.add_argument('--coil_csv', default=None, help='包含卷号信息的CSV文件路径（可选）')
    parser.add_argument('--marker_interval', type=int, default=100, help='白斑标注图采样间隔（默认每100帧）')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.roi_dir):
        print(f"错误: ROI目录不存在: {args.roi_dir}")
        return 1
    
    # 创建分析器
    analyzer = WhitePatchAnalyzer(
        roi_dir=args.roi_dir,
        output_dir=args.output_dir
    )
    
    # 分析所有帧
    df = analyzer.analyze_all_frames()
    
    if len(df) == 0:
        print("错误: 没有成功分析任何帧")
        return 1
    
    # 生成可视化
    analyzer.compare_methods(df)
    analyzer.plot_temporal_curves(df)
    analyzer.plot_coil_statistics(df, coil_csv=args.coil_csv)
    analyzer.generate_recommendation_report(df)
    
    # 生成白斑标注图
    analyzer.visualize_white_patches_with_markers(sample_interval=args.marker_interval)
    
    print(f"\n{'='*80}")
    print(f"分析完成！所有结果已保存到: {args.output_dir}")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


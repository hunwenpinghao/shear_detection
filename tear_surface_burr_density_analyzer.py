"""
撕裂面毛刺密度分析器
专门针对撕裂面毛刺的数量和密度进行详细分析

用法:
    python tear_surface_burr_density_analyzer.py --roi_dir data/roi_imgs --output_dir data/burr_patch_analysis
    
功能:
    - 4种检测参数对比
    - 8种量化指标分析  
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

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_process'))
from feature_extractor import FeatureExtractor
from config import PREPROCESS_CONFIG

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Songti SC', 'STSong', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class TearSurfaceBurrAnalyzer:
    """撕裂面毛刺分析器"""
    
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
        
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(PREPROCESS_CONFIG)
    
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
    
    def detect_burrs_method1(self, image: np.ndarray, mask: np.ndarray):
        """方法1：默认参数的毛刺检测"""
        burr_result = self.feature_extractor.detect_burs(image, mask=mask)
        if 'burs_binary_mask' in burr_result and burr_result['burs_binary_mask'] is not None:
            return burr_result['burs_binary_mask']
        return np.zeros_like(image)
    
    def detect_burrs_method2(self, image: np.ndarray, mask: np.ndarray):
        """方法2：更敏感的毛刺检测（降低阈值）"""
        # 备份原配置
        original_config = PREPROCESS_CONFIG.copy()
        
        # 修改配置使检测更敏感
        temp_config = PREPROCESS_CONFIG.copy()
        if 'burr_detection' in temp_config:
            temp_config['burr_detection']['threshold_factor'] = 0.8  # 降低阈值因子
        
        # 创建临时特征提取器
        temp_extractor = FeatureExtractor(temp_config)
        burr_result = temp_extractor.detect_burs(image, mask=mask)
        
        if 'burs_binary_mask' in burr_result and burr_result['burs_binary_mask'] is not None:
            return burr_result['burs_binary_mask']
        return np.zeros_like(image)
    
    def detect_burrs_method3(self, image: np.ndarray, mask: np.ndarray):
        """方法3：基于形态学的毛刺检测"""
        # 使用形态学方法检测突出部分
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # 边缘检测
        edges = cv2.Canny(masked_image, 50, 150)
        
        # 膨胀边缘以连接近邻的毛刺点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 应用mask
        burr_binary = cv2.bitwise_and(dilated, dilated, mask=mask)
        
        return burr_binary
    
    def detect_burrs_method4(self, image: np.ndarray, mask: np.ndarray):
        """方法4：基于梯度的毛刺检测"""
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # 计算梯度
        grad_x = cv2.Sobel(masked_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(masked_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 归一化到0-255
        gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 阈值处理
        masked_pixels = gradient_mag[mask > 0]
        if len(masked_pixels) > 0:
            threshold = np.percentile(masked_pixels, 85)  # 使用85%分位数作为阈值
            _, burr_binary = cv2.threshold(gradient_mag, threshold, 255, cv2.THRESH_BINARY)
            burr_binary = cv2.bitwise_and(burr_binary, burr_binary, mask=mask)
            return burr_binary
        
        return np.zeros_like(image)
    
    def compute_metrics(self, binary: np.ndarray, original: np.ndarray, mask: np.ndarray):
        """计算8个量化指标"""
        total_area = np.sum(mask > 0)
        burr_area = np.sum(binary > 0)
        area_ratio = (burr_area / total_area * 100) if total_area > 0 else 0.0
        
        # 毛刺数量（连通域分析）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        burr_count = num_labels - 1  # 减去背景
        
        # 毛刺密度：数量除以毛刺总面积
        burr_density = (burr_count / burr_area) if burr_area > 0 else 0.0
        
        # 单个毛刺平均面积
        if burr_count > 0:
            burr_areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
            avg_burr_area = np.mean(burr_areas)
            burr_area_std = np.std(burr_areas)
        else:
            avg_burr_area = 0.0
            burr_area_std = 0.0
        
        # 毛刺区域平均亮度
        burr_pixels = original[binary > 0]
        if len(burr_pixels) > 0:
            avg_brightness = np.mean(burr_pixels)
            brightness_std = np.std(burr_pixels)
        else:
            avg_brightness = 0.0
            brightness_std = 0.0
        
        # 综合指标：毛刺数量加权面积标准差
        composite_index = burr_count + burr_area_std * 0.01
        
        # 毛刺面积分布熵
        burr_area_entropy = self._compute_burr_area_entropy(binary)
        
        return {
            'area_ratio': area_ratio,                # 毛刺面积占比(%)
            'burr_count': burr_count,                # 毛刺数量(个)
            'burr_density': burr_density,            # 毛刺密度(个/像素)
            'avg_burr_area': avg_burr_area,          # 单个毛刺平均面积(像素)
            'burr_area_std': burr_area_std,          # 毛刺面积标准差
            'avg_brightness': avg_brightness,         # 毛刺区域平均亮度
            'brightness_std': brightness_std,         # 毛刺区域亮度标准差
            'composite_index': composite_index,       # 综合指标
            'burr_area_entropy': burr_area_entropy   # 毛刺面积分布熵
        }
    
    def _compute_burr_area_entropy(self, binary: np.ndarray, bins: int = 20):
        """计算毛刺面积分布的香农熵"""
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
                
                # 提取左侧撕裂面区域
                left_region, left_mask = self.extract_left_region_and_mask(image)
                
                # 4种方法检测毛刺
                binary1 = self.detect_burrs_method1(left_region, left_mask)
                binary2 = self.detect_burrs_method2(left_region, left_mask)
                binary3 = self.detect_burrs_method3(left_region, left_mask)
                binary4 = self.detect_burrs_method4(left_region, left_mask)
                
                # 计算指标
                metrics1 = self.compute_metrics(binary1, left_region, left_mask)
                metrics2 = self.compute_metrics(binary2, left_region, left_mask)
                metrics3 = self.compute_metrics(binary3, left_region, left_mask)
                metrics4 = self.compute_metrics(binary4, left_region, left_mask)
                
                # 保存结果
                row = {'frame_id': frame_id}
                for i, metrics in enumerate([metrics1, metrics2, metrics3, metrics4], 1):
                    row[f'area_ratio_m{i}'] = metrics['area_ratio']
                    row[f'burr_count_m{i}'] = metrics['burr_count']
                    row[f'burr_density_m{i}'] = metrics['burr_density']
                    row[f'avg_burr_area_m{i}'] = metrics['avg_burr_area']
                    row[f'burr_area_std_m{i}'] = metrics['burr_area_std']
                    row[f'avg_brightness_m{i}'] = metrics['avg_brightness']
                    row[f'brightness_std_m{i}'] = metrics['brightness_std']
                    row[f'composite_index_m{i}'] = metrics['composite_index']
                    row[f'burr_area_entropy_m{i}'] = metrics['burr_area_entropy']
                
                results.append(row)
                
            except Exception as e:
                print(f"\n处理 {filepath} 时出错: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        # 保存CSV
        csv_path = os.path.join(self.output_dir, 'burr_features.csv')
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
            
            # 绘制左侧撕裂面区域
            axes[row_idx, 1].imshow(left_region, cmap='gray')
            axes[row_idx, 1].set_title('左侧撕裂面', fontsize=10)
            axes[row_idx, 1].axis('off')
            
            # 4种方法
            binary1 = self.detect_burrs_method1(left_region, left_mask)
            binary2 = self.detect_burrs_method2(left_region, left_mask)
            binary3 = self.detect_burrs_method3(left_region, left_mask)
            binary4 = self.detect_burrs_method4(left_region, left_mask)
            
            methods = [binary1, binary2, binary3, binary4]
            method_names = ['默认参数', '敏感检测', '形态学', '梯度法']
            
            for col_idx, (binary, name) in enumerate(zip(methods, method_names), 2):
                # 叠加显示
                overlay = cv2.cvtColor(left_region, cv2.COLOR_GRAY2RGB)
                overlay[binary > 0] = [255, 165, 0]  # 橙色标记毛刺
                
                axes[row_idx, col_idx].imshow(overlay)
                axes[row_idx, col_idx].set_title(f'方法{col_idx-1}:{name}', fontsize=10)
                axes[row_idx, col_idx].axis('off')
        
        plt.suptitle('撕裂面毛刺检测方法对比（橙色=检测到的毛刺）', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'method_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {self.output_dir}/method_comparison.png")
    
    def plot_temporal_curves(self, df: pd.DataFrame):
        """绘制时序曲线（4方法×8指标=32条曲线）"""
        print("\n生成时序曲线...")
        
        fig, axes = plt.subplots(9, 4, figsize=(20, 36))
        
        methods = ['m1', 'm2', 'm3', 'm4']
        method_names = ['默认参数', '敏感检测', '形态学', '梯度法']
        metrics = ['area_ratio', 'burr_count', 'burr_density', 'avg_burr_area', 
                   'burr_area_std', 'avg_brightness', 'brightness_std', 
                   'composite_index', 'burr_area_entropy']
        metric_names = ['毛刺面积占比(%)', '毛刺数量(个)', '毛刺密度(个/像素)', 
                       '单个毛刺平均面积', '毛刺面积标准差', '毛刺区域平均亮度', 
                       '毛刺区域亮度标准差', '综合指标', '毛刺面积分布熵']
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
                if window >= 5 and len(values) >= window:
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
        
        plt.suptitle('撕裂面毛刺特征时序演变（4方法×9指标）', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'temporal_curves_4x9.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {self.output_dir}/temporal_curves_4x9.png")
    
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
        method_names = ['默认参数', '敏感检测', '形态学', '梯度法']
        
        for idx, (method, method_name) in enumerate(zip(methods, method_names)):
            ax1 = axes[idx]
            ax2 = axes[idx + 4]
            
            # 面积占比
            coil_ids = sorted(df['coil_id'].unique())
            area_means = [df[df['coil_id']==cid][f'area_ratio_{method}'].mean() for cid in coil_ids]
            
            ax1.bar(range(len(coil_ids)), area_means, color='steelblue', alpha=0.7)
            ax1.plot(range(len(coil_ids)), area_means, 'ro-', linewidth=2, markersize=6)
            ax1.set_xlabel('钢卷编号', fontsize=10)
            ax1.set_ylabel('毛刺面积占比(%)', fontsize=10)
            ax1.set_title(f'{method_name} - 面积占比按卷变化', fontsize=11, fontweight='bold')
            ax1.set_xticks(range(len(coil_ids)))
            ax1.set_xticklabels([f'卷{int(c)}' for c in coil_ids], rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 毛刺数量
            count_means = [df[df['coil_id']==cid][f'burr_count_{method}'].mean() for cid in coil_ids]
            
            ax2.bar(range(len(coil_ids)), count_means, color='coral', alpha=0.7)
            ax2.plot(range(len(coil_ids)), count_means, 'go-', linewidth=2, markersize=6)
            ax2.set_xlabel('钢卷编号', fontsize=10)
            ax2.set_ylabel('毛刺数量(个)', fontsize=10)
            ax2.set_title(f'{method_name} - 毛刺数量按卷变化', fontsize=11, fontweight='bold')
            ax2.set_xticks(range(len(coil_ids)))
            ax2.set_xticklabels([f'卷{int(c)}' for c in coil_ids], rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('撕裂面毛刺按卷统计分析', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'coil_statistics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {self.output_dir}/coil_statistics.png")
    
    def visualize_burrs_with_markers(self, sample_interval: int = 100):
        """
        可视化毛刺检测结果，用小圆圈标注每个毛刺
        
        Args:
            sample_interval: 采样间隔（每隔多少帧保存一次）
        """
        print(f"\n生成毛刺标注可视化（每隔{sample_interval}帧）...")
        
        # 创建输出目录
        markers_dir = os.path.join(self.output_dir, 'burr_markers')
        os.makedirs(markers_dir, exist_ok=True)
        
        # 选择要可视化的帧
        sampled_indices = list(range(0, len(self.image_files), sample_interval))
        
        methods = [
            ('method1', self.detect_burrs_method1, '默认参数'),
            ('method2', self.detect_burrs_method2, '敏感检测'),
            ('method3', self.detect_burrs_method3, '形态学'),
            ('method4', self.detect_burrs_method4, '梯度法')
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
                
                # 提取左侧撕裂面区域
                left_region, left_mask = self.extract_left_region_and_mask(image)
                
                # 创建3x2子图布局
                fig = plt.figure(figsize=(18, 24))
                gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
                
                # 前4个子图：4种方法的标注
                all_binaries = []
                all_areas = []
                
                for method_idx, (method_name, detect_func, display_name) in enumerate(methods):
                    row = method_idx // 2
                    col = method_idx % 2
                    ax = fig.add_subplot(gs[row, col])
                    
                    # 检测毛刺
                    binary = detect_func(left_region, left_mask)
                    all_binaries.append(binary)
                    
                    # 转换为彩色图便于标注
                    display_img = cv2.cvtColor(left_region, cv2.COLOR_GRAY2RGB)
                    
                    # 查找毛刺的连通域
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    
                    # 统计毛刺数量和面积
                    valid_burrs = 0
                    burr_areas = []
                    
                    # 在每个毛刺质心画圆圈
                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        
                        # 过滤太小的毛刺（面积小于3像素）
                        if area < 3:
                            continue
                        
                        valid_burrs += 1
                        burr_areas.append(area)
                        
                        # 获取质心坐标
                        cx, cy = int(centroids[i][0]), int(centroids[i][1])
                        
                        # 根据面积决定圆圈大小
                        radius = max(2, min(int(np.sqrt(area) * 0.3), 8))
                        
                        # 画橙色圆圈
                        cv2.circle(display_img, (cx, cy), radius, (255, 165, 0), 2)
                        
                        # 画一个小点标记质心
                        cv2.circle(display_img, (cx, cy), 1, (0, 255, 0), -1)
                    
                    all_areas.append(burr_areas)
                    
                    # 显示图像
                    ax.imshow(display_img)
                    ax.set_title(f'{display_name}\n帧{frame_id} - 检测到{valid_burrs}个毛刺', 
                               fontsize=14, fontweight='bold')
                    ax.axis('off')
                    
                    # 添加信息
                    h, w = left_region.shape
                    ax.text(0.02, 0.98, f'毛刺数: {valid_burrs}', 
                           transform=ax.transAxes, fontsize=12, 
                           verticalalignment='top', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
                
                # 第5个子图：毛刺亮度分布对比（左下）
                ax_brightness = fig.add_subplot(gs[2, 0])
                
                colors_hist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                for idx, (binary, method_name) in enumerate(zip(all_binaries, ['方法1', '方法2', '方法3', '方法4'])):
                    burr_pixels = left_region[binary > 0]
                    if len(burr_pixels) > 0:
                        ax_brightness.hist(burr_pixels, bins=50, color=colors_hist[idx], 
                                         alpha=0.4, label=f'{method_name}', 
                                         edgecolor=colors_hist[idx], linewidth=1)
                
                ax_brightness.set_xlabel('亮度值', fontsize=12, fontweight='bold')
                ax_brightness.set_ylabel('像素数量', fontsize=12, fontweight='bold')
                ax_brightness.set_title(f'毛刺区域亮度分布对比\n帧{frame_id}', fontsize=14, fontweight='bold')
                ax_brightness.legend(fontsize=10, loc='best')
                ax_brightness.grid(True, alpha=0.3, axis='y')
                
                # 第6个子图：毛刺面积分布对比（右下）
                ax_area = fig.add_subplot(gs[2, 1])
                
                for idx, (areas, method_name) in enumerate(zip(all_areas, ['方法1', '方法2', '方法3', '方法4'])):
                    if len(areas) > 0:
                        ax_area.hist(areas, bins=20, color=colors_hist[idx], alpha=0.5,
                                   label=f'{method_name} ({len(areas)}个)',
                                   edgecolor=colors_hist[idx], linewidth=1)
                
                ax_area.set_xlabel('毛刺面积 (像素数)', fontsize=12, fontweight='bold')
                ax_area.set_ylabel('毛刺数量', fontsize=12, fontweight='bold')
                ax_area.set_title(f'毛刺面积分布对比\n帧{frame_id}', fontsize=14, fontweight='bold')
                ax_area.legend(fontsize=10, loc='best')
                ax_area.grid(True, alpha=0.3, axis='y')
                
                plt.suptitle(f'撕裂面毛刺综合分析 - 帧{frame_id}\n（上：标注图，下：直方图对比）', 
                           fontsize=18, fontweight='bold')
                plt.tight_layout()
                
                # 保存图像
                save_path = os.path.join(markers_dir, f'frame_{frame_id:06d}_burr_markers.png')
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
        report_lines.append("# 撕裂面毛刺检测方法推荐报告\n\n")
        
        methods = ['m1', 'm2', 'm3', 'm4']
        method_names = ['方法1:默认参数', '方法2:敏感检测', '方法3:形态学', '方法4:梯度法']
        metrics = ['area_ratio', 'burr_count', 'burr_density']
        
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
                
                metric_cn = '面积占比' if metric == 'area_ratio' else ('毛刺数量' if metric == 'burr_count' else '毛刺密度')
                
                report_lines.append(f"**指标: {metric_cn}**\n")
                report_lines.append(f"- 单调性（Spearman相关系数）: {corr:.4f} (p-value={pval:.4e})\n")
                report_lines.append(f"- 稳定性（变异系数CV）: {cv:.4f}\n")
                report_lines.append(f"- 灵敏度（数值范围）: {value_range:.4f}\n")
                report_lines.append(f"- 均值: {mean_val:.4f}, 标准差: {std_val:.4f}\n\n")
                
                evaluation_results.append({
                    'method': method_name,
                    'metric': metric_cn,
                    'monotonicity': abs(corr),
                    'stability': 1/(cv+0.01),
                    'sensitivity': value_range
                })
        
        # 综合推荐
        report_lines.append("## 综合推荐\n\n")
        
        # 找最佳方法
        eval_df = pd.DataFrame(evaluation_results)
        eval_df['综合得分'] = eval_df['monotonicity'] * 0.5 + eval_df['stability'] * 0.01 + eval_df['sensitivity'] * 0.01
        
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
        description='撕裂面毛刺密度分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tear_surface_burr_density_analyzer.py --roi_dir data/roi_imgs --output_dir data/burr_patch_analysis
  
  # 如果有卷号信息，可以指定CSV文件
  python tear_surface_burr_density_analyzer.py --roi_dir data/roi_imgs --output_dir data/burr_patch_analysis --coil_csv data/analysis/features/wear_features_with_coils.csv
        """
    )
    
    parser.add_argument('--roi_dir', required=True, help='ROI图像目录路径')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    parser.add_argument('--coil_csv', default=None, help='包含卷号信息的CSV文件路径（可选）')
    parser.add_argument('--marker_interval', type=int, default=100, help='毛刺标注图采样间隔（默认每100帧）')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.roi_dir):
        print(f"错误: ROI目录不存在: {args.roi_dir}")
        return 1
    
    # 创建分析器
    analyzer = TearSurfaceBurrAnalyzer(
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
    
    # 生成毛刺标注图
    analyzer.visualize_burrs_with_markers(sample_interval=args.marker_interval)
    
    print(f"\n{'='*80}")
    print(f"分析完成！所有结果已保存到: {args.output_dir}")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


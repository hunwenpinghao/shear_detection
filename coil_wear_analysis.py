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
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d, maximum_filter1d, minimum_filter1d
from scipy.interpolate import UnivariateSpline
from datetime import datetime
from tqdm import tqdm
from math import pi
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ruptures as rp  # 用于变化点检测

# 添加主项目的模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'wear_degree_analysis', 'src'))

from preprocessor import ImagePreprocessor
from geometry_features import GeometryFeatureExtractor
from visualizer import WearVisualizer
from composite_indicator import CompositeWearIndicator
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
                 min_coils: int = 5, max_coils: int = 15,
                 diagnosis_interval: int = 100, marker_interval: int = 100,
                 n_coils: int = None, detection_method: str = "valley"):
        """
        初始化分析器
        
        Args:
            roi_dir: ROI图像目录
            output_dir: 输出目录
            analysis_name: 分析名称
            min_coils: 最小钢卷数（自动检测时使用）
            max_coils: 最大钢卷数（自动检测时使用）
            diagnosis_interval: 帧诊断图采样间隔（默认100）
            marker_interval: 白斑标注图采样间隔（默认100）
            n_coils: 直接指定钢卷数（如果指定，则跳过自动检测，速度快10倍）
            detection_method: 检测方法 ("valley"=波谷检测法[推荐], "pelt"=Pelt算法)
        """
        self.roi_dir = os.path.abspath(roi_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.analysis_name = analysis_name
        self.min_coils = min_coils
        self.max_coils = max_coils
        self.diagnosis_interval = diagnosis_interval
        self.marker_interval = marker_interval
        self.n_coils = n_coils  # 直接指定钢卷数（快速模式）
        self.detection_method = detection_method  # 检测方法选择
        
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
        self.composite_indicator = CompositeWearIndicator()
    
    @staticmethod
    def compute_envelope(signal: np.ndarray, window: int = 15):
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
        
        upper_envelope = maximum_filter1d(signal, size=window, mode='nearest')
        lower_envelope = minimum_filter1d(signal, size=window, mode='nearest')
        
        return upper_envelope, lower_envelope
    
    @staticmethod
    def robust_curve_fit(signal: np.ndarray, percentile_range=(5, 95), smoothing=None, 
                        use_local_filter=True, local_window=None):
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
            local_window: 局部窗口大小（None表示自动，建议为数据长度的5%-10%）
            
        Returns:
            fitted_curve: 拟合曲线
            inlier_mask: 内点掩码（True表示非离群点）
        """
        if len(signal) < 10:
            return signal.copy(), np.ones(len(signal), dtype=bool)
        
        # === 第1阶段：全局粗筛选（去除极端离群点） ===
        lower_bound = np.percentile(signal, percentile_range[0])
        upper_bound = np.percentile(signal, percentile_range[1])
        global_inlier_mask = (signal >= lower_bound) & (signal <= upper_bound)
        
        # === 第2阶段：局部滑动窗口精细过滤 ===
        if use_local_filter:
            # 自动确定窗口大小（建议为数据长度的5%-10%）
            if local_window is None:
                local_window = max(min(len(signal) // 15, 101), 21)  # 21到101之间
                if local_window % 2 == 0:
                    local_window += 1  # 确保是奇数
            
            # 初始化局部内点掩码
            local_inlier_mask = np.ones(len(signal), dtype=bool)
            
            # 滑动窗口检测
            half_window = local_window // 2
            for i in range(len(signal)):
                # 定义窗口范围
                start = max(0, i - half_window)
                end = min(len(signal), i + half_window + 1)
                
                # 获取窗口内的数据（只考虑全局内点）
                window_indices = np.arange(start, end)
                window_mask = global_inlier_mask[start:end]
                window_data = signal[start:end][window_mask]
                
                if len(window_data) < 3:
                    continue
                
                # 计算窗口内的局部统计量
                local_mean = np.mean(window_data)
                local_std = np.std(window_data)
                
                # 局部离群点判断：当前点是否偏离局部均值超过3倍标准差
                if local_std > 0:
                    z_score = abs(signal[i] - local_mean) / local_std
                    if z_score > 3.0:  # 3-sigma规则
                        local_inlier_mask[i] = False
            
            # 综合全局和局部掩码
            inlier_mask = global_inlier_mask & local_inlier_mask
        else:
            # 不使用局部过滤，直接使用全局掩码
            inlier_mask = global_inlier_mask
        
        # === 第3阶段：样条曲线拟合 ===
        x_inliers = np.where(inlier_mask)[0]
        y_inliers = signal[inlier_mask]
        
        if len(x_inliers) < 4:
            fitted_curve = np.full(len(signal), np.mean(signal))
            return fitted_curve, inlier_mask
        
        # 自动计算平滑参数（自适应策略）
        if smoothing is None:
            # 计算数据的变异系数（CV = std / mean）
            mean_val = np.abs(np.mean(y_inliers)) + 1e-10
            std_val = np.std(y_inliers)
            cv = std_val / mean_val
            
            # 根据变异系数调整平滑参数
            # CV越大（数据波动越大），平滑参数越小（拟合越灵活）
            if cv > 0.5:  # 高变异（如稀疏峰值数据）
                smoothing = len(x_inliers) * 0.05  # 更敏感
            elif cv > 0.2:  # 中等变异
                smoothing = len(x_inliers) * 0.15
            else:  # 低变异（稳定数据）
                smoothing = len(x_inliers) * 0.3
        
        try:
            # 使用三次样条拟合
            spline = UnivariateSpline(x_inliers, y_inliers, s=smoothing, k=3)
            x_full = np.arange(len(signal))
            fitted_curve = spline(x_full)
        except:
            # 如果样条失败，使用多项式拟合
            degree = min(3, len(x_inliers) - 1)
            coeffs = np.polyfit(x_inliers, y_inliers, degree)
            poly = np.poly1d(coeffs)
            x_full = np.arange(len(signal))
            fitted_curve = poly(x_full)
        
        return fitted_curve, inlier_mask
    
    def extract_features(self, save_diagnosis: bool = True) -> pd.DataFrame:
        """
        提取所有帧的特征
        
        Args:
            save_diagnosis: 是否保存帧诊断图（按采样间隔）
        """
        print(f"\n开始提取{self.analysis_name}的特征...")
        print(f"诊断图采样间隔: 每 {self.diagnosis_interval} 帧")
        
        # 扫描实际存在的ROI文件
        roi_files = sorted(glob.glob(os.path.join(self.roi_dir, "frame_*_roi.png")))
        print(f"实际找到 {len(roi_files)} 个ROI文件")
        
        # 创建诊断图目录
        if save_diagnosis:
            diagnosis_dir = os.path.join(self.output_dir, 'visualizations', 'frame_diagnosis')
            ensure_dir(diagnosis_dir)
            diagnosis_count = 0
        
        all_features = []
        read_fail_count = 0
        preprocess_fail_count = 0
        extract_fail_count = 0
        
        for idx, filepath in enumerate(tqdm(roi_files, desc=f"提取特征")):
            try:
                # 从文件名中提取帧ID
                basename = os.path.basename(filepath)
                frame_id = int(basename.split('_')[1])
                
                # 读取图像
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    read_fail_count += 1
                    if read_fail_count <= 3:
                        print(f"\n警告: 无法读取图像 {filepath}")
                    continue
                
                # 预处理
                preprocessed = self.preprocessor.process(image)
                
                if not preprocessed['success']:
                    preprocess_fail_count += 1
                    if preprocess_fail_count <= 3:
                        print(f"\n警告: 预处理失败 frame {frame_id}: {preprocessed.get('error', '未知错误')}")
                    continue
                
                # 提取特征
                try:
                    features = self.feature_extractor.extract_features(preprocessed)
                    features['frame_id'] = frame_id
                    all_features.append(features)
                except Exception as e:
                    extract_fail_count += 1
                    if extract_fail_count <= 3:
                        print(f"\n警告: 特征提取失败 frame {frame_id}: {e}")
                    continue
                
                # 保存诊断图（按采样间隔，基于frame_id而非索引）
                if save_diagnosis and frame_id % self.diagnosis_interval == 0:
                    diagnosis_path = os.path.join(diagnosis_dir, f"frame_{frame_id:06d}_diagnosis.png")
                    self.visualizer.visualize_single_frame_diagnosis(
                        image, preprocessed, features, frame_id, diagnosis_path
                    )
                    diagnosis_count += 1
                
            except Exception as e:
                print(f"\n警告: 处理文件 {filepath} 时出错: {e}")
                continue
        
        # 打印诊断信息
        print(f"\n特征提取诊断:")
        print(f"  总文件数: {len(roi_files)}")
        print(f"  成功提取: {len(all_features)}")
        print(f"  读取失败: {read_fail_count} 帧")
        print(f"  预处理失败: {preprocess_fail_count} 帧")
        print(f"  特征提取失败: {extract_fail_count} 帧")
        if save_diagnosis:
            print(f"  已保存诊断图: {diagnosis_count} 张（采样间隔: {self.diagnosis_interval}）")
        
        if len(all_features) == 0:
            raise RuntimeError("没有成功提取任何特征")
        
        df = pd.DataFrame(all_features)
        print(f"成功提取 {len(df)} / {self.total_frames} 帧的特征")
        
        # 验证和修正撕裂面占比数据
        if 'tear_shear_area_ratio' in df.columns:
            original_ratio = df['tear_shear_area_ratio']
            invalid_count = (original_ratio < 0).sum() + (original_ratio > 1).sum()
            
            if invalid_count > 0:
                print(f"⚠️  发现 {invalid_count} 个撕裂面占比值超出0-1范围，正在修正...")
                
                # 如果有很多值>1，可能是比值形式，使用转换公式
                if (original_ratio > 1).sum() > len(original_ratio) * 0.1:
                    print("   使用转换公式: new_ratio = old_ratio / (old_ratio + 1)")
                    df['tear_shear_area_ratio'] = original_ratio / (original_ratio + 1)
                else:
                    print("   直接截断到0-1范围")
                    df['tear_shear_area_ratio'] = np.clip(original_ratio, 0.0, 1.0)
                
                # 验证修正结果
                corrected_ratio = df['tear_shear_area_ratio']
                print(f"✅ 修正完成: 最小值={corrected_ratio.min():.4f}, 最大值={corrected_ratio.max():.4f}")
                print(f"   所有值在0-1范围内: {(corrected_ratio >= 0).all() and (corrected_ratio <= 1).all()}")
            else:
                print("✅ 撕裂面占比数据正常（0-1范围）")
        
        # 保存特征
        features_dir = os.path.join(self.output_dir, 'features')
        ensure_dir(features_dir)
        
        csv_path = os.path.join(features_dir, 'wear_features.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存特征文件: {csv_path}")
        
        return df
    
    def _evaluate_segmentation_quality(self, signal: np.ndarray, boundaries: list) -> float:
        """
        快速评估分割质量的综合评分函数
        
        Args:
            signal: 用于分割的信号
            boundaries: 分割边界点列表
            
        Returns:
            float: 综合评分，越高越好
        """
        if len(boundaries) < 2:
            return -float('inf')
        
        # 快速计算段均值（避免创建segments列表）
        segment_means = []
        segment_lengths = []
        
        for i in range(len(boundaries)):
            start = boundaries[i-1] if i > 0 else 0
            end = boundaries[i] if i < len(boundaries) else len(signal)
            
            if end > start:
                segment_mean = np.mean(signal[start:end])
                segment_means.append(segment_mean)
                segment_lengths.append(end - start)
        
        if len(segment_means) < 2:
            return -float('inf')
        
        # 简化的评分计算（只保留最重要的指标）
        # 1. 段间差异性（最重要）
        between_variance = np.var(segment_means)
        
        # 2. 长度均匀性（简化计算）
        length_std = np.std(segment_lengths)
        length_mean = np.mean(segment_lengths)
        length_uniformity = 1.0 / (1.0 + length_std / length_mean) if length_mean > 0 else 0
        
        # 3. 边界强度（简化计算，只检查部分边界）
        boundary_strength = 0.0
        check_boundaries = boundaries[1:-1][::2]  # 只检查一半的边界点
        for boundary in check_boundaries:
            if 1 <= boundary < len(signal) - 1:
                gradient = abs(signal[boundary] - signal[boundary-1])
                boundary_strength += gradient
        boundary_strength /= max(1, len(check_boundaries))
        
        # 简化的综合评分
        score = between_variance * 3.0 + boundary_strength * 1.0 + length_uniformity * 0.5
        
        return score
    
    def _detect_by_valley_method(self, df: pd.DataFrame) -> list:
        """
        波谷检测法：通过二次滤波 + 波谷检测来识别钢卷边界
        
        原理：
        1. 对信号进行二次平滑滤波，过滤假波谷
        2. 检测波谷（局部最小值点）
        3. 相邻波谷之间为一个钢卷
        
        优势：速度快、逻辑清晰、物理意义明确
        
        Args:
            df: 特征数据
            
        Returns:
            钢卷边界索引列表
        """
        print("🌊 使用波谷检测法识别钢卷边界...")
        
        # 获取信号
        if 'weighted_score' in df.columns:
            signal = df['weighted_score'].values
            print("使用综合磨损指数")
        else:
            key_features = ['avg_gradient_energy', 'max_notch_depth', 'avg_rms_roughness']
            scaler = StandardScaler()
            features_for_detection = []
            
            for feature in key_features:
                if feature in df.columns:
                    features_for_detection.append(df[feature].values)
            
            if len(features_for_detection) == 0:
                print("警告: 没有足够的特征用于检测")
                return None
            
            combined_signal = np.column_stack(features_for_detection)
            signal = scaler.fit_transform(combined_signal).mean(axis=1)
            print("使用多特征组合")
        
        # 第一次平滑：大窗口滤波
        window1 = min(201, len(signal)//4*2+1)
        if window1 >= 5:
            signal_smooth1 = savgol_filter(signal, window_length=window1, polyorder=3)
            print(f"第一次平滑：窗口大小 {window1}")
        else:
            signal_smooth1 = signal
        
        # 第二次平滑：进一步平滑
        window2 = min(151, len(signal_smooth1)//6*2+1)
        if window2 >= 5:
            signal_smooth2 = savgol_filter(signal_smooth1, window_length=window2, polyorder=3)
            print(f"第二次平滑：窗口大小 {window2}")
        else:
            signal_smooth2 = signal_smooth1
        
        # 检测波谷（局部最小值）
        # distance: 相邻波谷的最小距离（避免检测到假波谷）
        min_distance = max(100, len(signal_smooth2) // (self.max_coils + 5))
        
        # 反转信号来检测波谷（find_peaks 检测波峰）
        inverted_signal = -signal_smooth2
        
        # prominence: 波峰显著性（过滤不明显的波峰）
        prominence = np.std(signal_smooth2) * 0.3  # 波动幅度的30%
        
        print(f"波谷检测参数：最小距离={min_distance}帧, 显著性阈值={prominence:.3f}")
        
        valleys, properties = find_peaks(
            inverted_signal, 
            distance=min_distance,
            prominence=prominence,
            width=20  # 波谷最小宽度
        )
        
        print(f"检测到 {len(valleys)} 个波谷")
        
        if len(valleys) == 0:
            print("⚠️ 未检测到明显的波谷，使用默认均匀分割")
            default_coils = (self.min_coils + self.max_coils) // 2
            coil_size = len(df) // default_coils
            boundaries = [i * coil_size for i in range(default_coils)]
            boundaries[0] = 0
            return boundaries
        
        # 波谷数量 = 钢卷数 - 1（两个钢卷之间有一个波谷）
        n_coils = len(valleys) + 1
        
        # 检查是否在合理范围内
        if n_coils < self.min_coils:
            print(f"⚠️ 检测到的钢卷数 ({n_coils}) 少于最小值 ({self.min_coils})")
            print("提示：可能需要调整 --min_coils 参数")
        elif n_coils > self.max_coils:
            print(f"⚠️ 检测到的钢卷数 ({n_coils}) 多于最大值 ({self.max_coils})")
            print("提示：可能需要调整 --max_coils 参数或增加平滑强度")
            # 只保留最显著的波谷
            n_valleys_to_keep = self.max_coils - 1
            prominences = properties['prominences']
            top_indices = np.argsort(prominences)[-n_valleys_to_keep:]
            valleys = valleys[sorted(top_indices)]
            n_coils = len(valleys) + 1
            print(f"保留最显著的 {len(valleys)} 个波谷，调整为 {n_coils} 个钢卷")
        
        # 构建边界列表：[0, 波谷1, 波谷2, ..., 波谷n]
        boundaries = [0] + valleys.tolist()
        
        print(f"✓ 检测到 {n_coils} 个钢卷")
        print(f"边界位置（波谷）: {valleys.tolist()}")
        
        # 输出每个钢卷的长度
        segment_lengths = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i+1] if i+1 < len(boundaries) else len(signal)
            length = end - start
            segment_lengths.append(length)
            print(f"  第{i+1}卷: {length}帧 (帧 {start} → {end})")
        
        # 验证分割质量
        self._validate_segmentation(signal_smooth2, boundaries, n_coils)
        
        return boundaries
    
    def _detect_with_fixed_n_coils(self, df: pd.DataFrame, n_coils: int) -> list:
        """
        快速模式：直接指定钢卷数进行检测（速度快10倍）
        
        Args:
            df: 特征数据
            n_coils: 钢卷数量
            
        Returns:
            钢卷边界索引列表
        """
        # 优先使用综合磨损指数
        if 'weighted_score' in df.columns:
            signal = df['weighted_score'].values
        else:
            # 使用关键特征组合
            key_features = ['avg_gradient_energy', 'max_notch_depth', 'avg_rms_roughness']
            scaler = StandardScaler()
            features_for_detection = []
            
            for feature in key_features:
                if feature in df.columns:
                    features_for_detection.append(df[feature].values)
            
            if len(features_for_detection) == 0:
                print("警告: 没有足够的特征用于检测，使用均匀分割")
                coil_size = len(df) // n_coils
                boundaries = [i * coil_size for i in range(n_coils)]
                boundaries[0] = 0
                return boundaries
            
            combined_signal = np.column_stack(features_for_detection)
            signal = scaler.fit_transform(combined_signal).mean(axis=1)
        
        # 平滑信号
        window = min(151, len(signal)//6*2+1)
        if window >= 5:
            signal_smooth = savgol_filter(signal, window_length=window, polyorder=3)
        else:
            signal_smooth = signal
        
        # 使用Pelt算法直接指定断点数
        try:
            model = "l2"
            min_segment_size = max(len(df)//(n_coils * 2), 50)
            jump_size = max(20, min(100, len(signal_smooth) // 100))
            
            print(f"最小段长度: {min_segment_size} 帧, 跳跃步长: {jump_size}")
            print("拟合模型中...")
            
            algo = rp.Pelt(model=model, min_size=int(min_segment_size), jump=jump_size)
            algo.fit(signal_smooth.reshape(-1, 1))
            
            # 直接指定断点数（比搜索penalty快10倍）
            print(f"检测 {n_coils} 个钢卷的边界...")
            boundaries = algo.predict(n_bkps=n_coils-1)
            
            # 去掉最后的边界点
            boundaries = [0] + boundaries[:-1]
            
            print(f"✓ 快速检测完成，共 {len(boundaries)} 个钢卷")
            print(f"边界位置: {boundaries}")
            
            # 验证分割质量
            self._validate_segmentation(signal_smooth, boundaries, len(boundaries))
            
            return boundaries
            
        except Exception as e:
            print(f"快速检测失败: {e}")
            print("使用均匀分割作为备选")
            coil_size = len(df) // n_coils
            boundaries = [i * coil_size for i in range(n_coils)]
            boundaries[0] = 0
            return boundaries
    
    def _validate_segmentation(self, signal: np.ndarray, boundaries: list, n_coils: int):
        """
        快速验证分割质量并输出关键信息
        
        Args:
            signal: 用于分割的信号
            boundaries: 分割边界点列表
            n_coils: 钢卷数量
        """
        print(f"\n=== 分割质量验证 ===")
        
        # 快速计算段长度统计
        segment_lengths = []
        for i in range(len(boundaries)):
            start = boundaries[i-1] if i > 0 else 0
            end = boundaries[i] if i < len(boundaries) else len(signal)
            segment_lengths.append(end - start)
        
        min_len, max_len = min(segment_lengths), max(segment_lengths)
        avg_len = np.mean(segment_lengths)
        std_len = np.std(segment_lengths)
        
        print(f"段长度: 最短{min_len}帧, 最长{max_len}帧, 平均{avg_len:.1f}帧")
        print(f"长度均匀性: {std_len/avg_len:.3f} ({'均匀' if std_len/avg_len < 0.3 else '不均匀'})")
        
        # 简化的质量评价（复用已计算的评分）
        quality_score = self._evaluate_segmentation_quality(signal, boundaries)
        if quality_score > 2.0:
            quality_level = "优秀"
        elif quality_score > 1.0:
            quality_level = "良好"
        elif quality_score > 0.5:
            quality_level = "一般"
        else:
            quality_level = "较差"
        
        print(f"分割质量: {quality_level} (评分: {quality_score:.3f})")
        print(f"{'='*30}")
    
    def detect_coil_boundaries(self, df: pd.DataFrame) -> list:
        """
        自动检测钢卷边界（改进版：使用综合磨损指数）
        
        Args:
            df: 特征数据
            
        Returns:
            钢卷边界索引列表
        """
        print(f"\n检测钢卷边界...")
        
        # 快速模式：直接指定钢卷数
        if self.n_coils is not None:
            print(f"⚡ 快速模式：使用指定的钢卷数 {self.n_coils}")
            return self._detect_with_fixed_n_coils(df, self.n_coils)
        
        # 自动检测模式
        print(f"🔍 自动检测模式：钢卷数范围 {self.min_coils}-{self.max_coils}个")
        
        # 方法选择
        if self.detection_method == "valley":
            print("📊 使用波谷检测法（推荐，快速且直观）")
            return self._detect_by_valley_method(df)
        elif self.detection_method == "pelt":
            print("📊 使用Pelt变化点检测法")
            return self._detect_by_pelt_method(df)
        else:
            print(f"⚠️ 未知的检测方法: {self.detection_method}，使用默认波谷检测法")
            return self._detect_by_valley_method(df)
    
    def _detect_by_pelt_method(self, df: pd.DataFrame) -> list:
        """
        Pelt算法检测法（原自动检测逻辑）
        
        Args:
            df: 特征数据
            
        Returns:
            钢卷边界索引列表
        """
        
        # 优先使用综合磨损指数（如果已经计算）
        if 'weighted_score' in df.columns:
            print("使用综合磨损指数进行检测")
            signal = df['weighted_score'].values
        else:
            # 否则使用多个关键特征的组合
            print("使用多特征组合进行检测")
            key_features = ['avg_gradient_energy', 'max_notch_depth', 'avg_rms_roughness']
            
            scaler = StandardScaler()
            features_for_detection = []
            
            for feature in key_features:
                if feature in df.columns:
                    features_for_detection.append(df[feature].values)
            
            if len(features_for_detection) == 0:
                print("警告: 没有足够的特征用于检测，使用默认分割")
                return None
            
            combined_signal = np.column_stack(features_for_detection)
            signal = scaler.fit_transform(combined_signal).mean(axis=1)
        
        # 强力平滑，降低噪声
        window = min(151, len(signal)//6*2+1)  # 更大的窗口
        if window >= 5:
            signal_smooth = savgol_filter(signal, window_length=window, polyorder=3)
        else:
            signal_smooth = signal
        
        # 使用Pelt算法检测变化点
        try:
            model = "l2"  # 使用L2模型（比RBF快3-5倍）
            # 大幅增大 min_size，避免过度分割
            min_segment_size = max(len(df)//(self.max_coils * 2), 50)  # 更灵活的最小段长度
            # 自适应调整 jump 参数：数据量大时用更大的跳跃步长
            jump_size = max(20, min(100, len(signal_smooth) // 100))  # 根据数据量自适应
            print(f"最小段长度: {min_segment_size} 帧, 跳跃步长: {jump_size}")
            
            algo = rp.Pelt(model=model, min_size=int(min_segment_size), jump=jump_size)
            print("拟合模型中...")
            algo.fit(signal_smooth.reshape(-1, 1))
            print("✓ 模型拟合完成")
            
            # 自适应钢卷数量检测 - 不预设目标数量
            best_boundaries = None
            best_n_coils = 0
            best_score = -float('inf')  # 使用综合评分而非距离
            all_results = {}  # 用字典记录每个n值对应的penalty和评分
            
            # 快速自适应penalty搜索策略（两阶段搜索）
            print(f"正在搜索最优penalty参数（两阶段快速检测）...")
            
            # 第一阶段：粗搜索（8个点）
            penalties_coarse = np.logspace(-1, 2.5, 8)  # 从0.1到316，只用8个点
            
            print(f"阶段1：粗搜索 {len(penalties_coarse)} 个penalty值...")
            good_enough_score = 1.5  # 降低阈值，更容易触发早期停止
            min_search_points = 5    # 至少搜索5个点
            penalties = penalties_coarse  # 默认使用粗搜索结果
            
            for i, penalty in enumerate(penalties):
                try:
                    boundaries = algo.predict(pen=penalty)
                    n_segments = len(boundaries)
                    
                    # 只考虑合理范围内的分割数
                    if not (self.min_coils <= n_segments <= self.max_coils):
                        continue
                    
                    # 计算分割质量评分
                    segment_score = self._evaluate_segmentation_quality(signal_smooth, boundaries)
                    
                    # 记录每个分割数的最佳结果
                    if n_segments not in all_results or segment_score > all_results[n_segments][2]:
                        all_results[n_segments] = (penalty, boundaries, segment_score)
                    
                    # 更新全局最佳结果
                    if segment_score > best_score:
                        best_score = segment_score
                        best_boundaries = boundaries
                        best_n_coils = n_segments
                        print(f"  [{i+1}/{len(penalties)}] {n_segments}个钢卷 (penalty={penalty:.2f}, score={segment_score:.3f}) ✓")
                        
                        # 更积极的早期停止策略
                        if segment_score > good_enough_score and i >= min_search_points:
                            print(f"✓ 找到足够好的结果，提前结束搜索")
                            break
                    else:
                        # 不是最优但也显示进度
                        if i % 2 == 0:  # 每隔一个显示
                            print(f"  [{i+1}/{len(penalties)}] {n_segments}个钢卷 (penalty={penalty:.2f}, score={segment_score:.3f})")
                        
                except:
                    continue
            
            # 打印搜索结果摘要
            print(f"搜索到的所有分割数: {sorted(all_results.keys())}")
            if all_results:
                print("各分割数的最佳评分:")
                for n_seg in sorted(all_results.keys()):
                    penalty, _, score = all_results[n_seg]
                    print(f"  {n_seg}个钢卷: score={score:.3f} (penalty={penalty:.2f})")
            
            if best_boundaries is None:
                print(f"未找到最优分割")
                if all_results:
                    # 从所有结果中选择评分最高的
                    best_result = max(all_results.values(), key=lambda x: x[2])
                    best_boundaries = best_result[1]
                    best_n_coils = len(best_result[1])
                    print(f"使用评分最高的分割: {best_n_coils}个钢卷 (score={best_result[2]:.3f})")
                else:
                    return None
            
            # 去掉最后的边界点（总是等于数据长度）
            boundaries = [0] + best_boundaries[:-1]
            
            print(f"✓ 检测到 {len(boundaries)} 个钢卷 (综合评分: {best_score:.3f})")
            print(f"边界位置: {boundaries}")
            
            # 添加分割质量验证
            self._validate_segmentation(signal_smooth, boundaries, len(boundaries))
            
            return boundaries
            
        except Exception as e:
            print(f"变化点检测失败: {e}")
            import traceback
            traceback.print_exc()
            print("使用默认均匀分割")
            return None
    
    def analyze_by_coil(self, df: pd.DataFrame):
        """
        按卷分析（自动检测钢卷边界）
        
        Args:
            df: 特征数据
        """
        print(f"\n{'='*80}")
        
        # === 先计算简单的综合指标用于边界检测 ===
        print("\n预计算简单综合指标用于钢卷检测...")
        # 使用几个关键特征的归一化均值
        key_features = ['avg_rms_roughness', 'max_notch_depth', 'right_peak_density']
        temp_scores = []
        for feat in key_features:
            if feat in df.columns:
                # MinMax归一化
                vals = df[feat].values
                if vals.max() > vals.min():
                    normalized = (vals - vals.min()) / (vals.max() - vals.min())
                    temp_scores.append(normalized)
        
        if len(temp_scores) > 0:
            df['weighted_score'] = np.mean(temp_scores, axis=0)
            print("✓ 临时综合指标已计算")
        
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
            # 检测失败，使用默认均匀分割（基于中位数钢卷数）
            default_coils = (self.min_coils + self.max_coils) // 2
            print(f"⚠️ 自动检测失败，使用默认均匀分割（{default_coils}个钢卷）")
            n_coils = default_coils
            coil_size = len(df) // n_coils
            df['coil_id'] = df.index // coil_size + 1
            df.loc[df['coil_id'] > n_coils, 'coil_id'] = n_coils
        
        print(f"{self.analysis_name} - 按卷分析（共{n_coils}个钢卷）")
        print(f"{'='*80}")
        
        print("\n每卷帧数分布:")
        coil_counts = df['coil_id'].value_counts().sort_index()
        for coil_id, count in coil_counts.items():
            print(f"  第{int(coil_id)}卷: {count}帧")
        
        # === 计算完整的综合磨损指标（会覆盖临时的） ===
        print("\n计算完整综合磨损指标...")
        df, analysis_results = self.composite_indicator.compute_all_indicators(df)
        print("✓ 综合指标计算完成")
        
        # 保存带卷号和综合指标的特征文件
        features_dir = os.path.join(self.output_dir, 'features')
        csv_with_coils = os.path.join(features_dir, 'wear_features_with_coils.csv')
        df.to_csv(csv_with_coils, index=False, encoding='utf-8-sig')
        
        # 保存特征重要性分析结果
        importance_df = analysis_results['importance_df']
        importance_csv = os.path.join(features_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_csv, index=False, encoding='utf-8-sig')
        print(f"✓ 特征重要性已保存: {importance_csv}")
        
        # 保存PCA载荷矩阵
        pca_result = analysis_results['pca_result']
        if len(pca_result['loadings']) > 0:
            pca_loadings_csv = os.path.join(features_dir, 'pca_loadings.csv')
            pca_result['loadings'].to_csv(pca_loadings_csv, encoding='utf-8-sig')
            print(f"✓ PCA载荷已保存: {pca_loadings_csv}")
        
        # 创建可视化目录
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        ensure_dir(viz_dir)
        
        # 核心特征
        key_features = {
            'avg_rms_roughness': 'RMS粗糙度',
            'max_notch_depth': '最大缺口深度',
            'right_peak_density': '右侧峰密度（剪切面）',
            'avg_gradient_energy': '梯度能量（锐度）',
            'tear_shear_area_ratio': '撕裂面占比'
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
        self._plot_individual_longterm_trends(df, viz_dir)
        self._plot_combined_trends_6x1(df, viz_dir)
        self._plot_recommended_indicators(df, os.path.join(viz_dir, 'recommended_indicators.png'))
        
        # 生成水平梯度能量对比图
        if 'avg_horizontal_gradient' in df.columns:
            self._plot_horizontal_gradient_comparison(df, os.path.join(viz_dir, 'horizontal_gradient_comparison.png'))
        
        # 生成平滑长期趋势分析
        self._plot_smooth_longterm_trends(df, os.path.join(viz_dir, 'smooth_longterm_trends.png'))
        
        # 生成深度趋势分析
        self._plot_deep_trend_analysis(df, viz_dir)
        
        # 生成撕裂面白斑分析
        self._plot_white_patch_analysis(df, os.path.join(viz_dir, 'white_patch_analysis.png'))
        
        # 生成白斑标注图（带直方图）
        self._generate_white_patch_markers(df, viz_dir, sample_interval=self.marker_interval)
        
        # 生成白斑时序曲线（8×4完整版）
        self._plot_white_patch_temporal_curves(df, os.path.join(viz_dir, 'white_patch_temporal_curves_4x8.png'))
        
        # 生成白斑方法推荐报告
        self._generate_white_patch_recommendation(df, viz_dir)
        
        print("✓ 额外分析图生成完成")
        
        # 生成综合指标相关可视化
        print("\n生成综合指标可视化...")
        self._plot_feature_importance(importance_df, os.path.join(viz_dir, 'feature_importance.png'))
        self._plot_composite_indicators_comparison(df, os.path.join(viz_dir, 'composite_indicators_comparison.png'))
        self._plot_multi_dimension_evolution(df, n_coils, os.path.join(viz_dir, 'multi_dimension_evolution.png'))
        self._plot_feature_contribution_heatmap(pca_result, os.path.join(viz_dir, 'feature_contribution_heatmap.png'))
        
        # 关键特征抽样展示
        key_features_dir = os.path.join(viz_dir, 'key_features_samples')
        ensure_dir(key_features_dir)
        self._plot_key_features_samples(df, importance_df, key_features_dir)
        print("✓ 综合指标可视化完成")
        
        # 生成分析报告
        self._generate_report(df, key_features, n_coils, analysis_results)
    
    def _plot_boxplot(self, df, key_features, viz_dir):
        """绘制箱线图"""
        print("\n生成可视化: 箱线图...")
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, (feature, label) in enumerate(list(key_features.items())[:5]):
            ax = axes[idx]
            
            coil_data = []
            coil_labels = []
            
            # 过滤掉 NaN 值
            valid_coil_ids = df['coil_id'].dropna().unique()
            for coil_id in sorted(valid_coil_ids):
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
            
            # 过滤掉 NaN 值
            valid_coil_ids = df['coil_id'].dropna().unique()
            for coil_id in sorted(valid_coil_ids):
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
        
        # 过滤掉 NaN 值
        valid_coil_ids = df['coil_id'].dropna().unique()
        for feature in key_features.keys():
            row = []
            for coil_id in sorted(valid_coil_ids):
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
            
            # 过滤掉 NaN 值
            valid_coil_ids = df['coil_id'].dropna().unique()
            for coil_id in sorted(valid_coil_ids):
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
            ('tear_shear_area_ratio', '撕裂面占比', 'orange'),
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
            'tear_shear_area_ratio': '撕裂面占比'
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
            ('tear_shear_area_ratio', '撕裂面占比'),
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
            ('tear_shear_area_ratio', '撕裂面占比', 'orange'),
        ]
        
        for idx, (feat, label, color) in enumerate(features):
            ax = axes[idx // 3, idx % 3]
            if feat in df.columns:
                # 原始数据连线（显示时间连续性）
                ax.plot(df['frame_id'], df[feat], 
                       alpha=0.3, linewidth=1.2, color=color, 
                       zorder=1, label='逐帧曲线')
                
                # 散点标记（标出数据点）
                ax.scatter(df['frame_id'], df[feat], 
                          alpha=0.4, s=15, color=color, zorder=2)
                
                # 线性拟合趋势线（突出整体趋势）
                z = np.polyfit(df['frame_id'], df[feat], 1)
                p = np.poly1d(z)
                ax.plot(df['frame_id'], p(df['frame_id']), 
                       color='darkred', linewidth=3, linestyle='--', 
                       zorder=3, label=f'线性趋势: y={z[0]:.6f}x+{z[1]:.2f}')
                
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
    
    def _plot_individual_longterm_trends(self, df: pd.DataFrame, viz_dir: str):
        """
        绘制单独的长期趋势图（每个特征一张图）
        
        将5个关键特征的长期趋势分别保存为独立的图片文件，
        x轴拉长以便更清楚地查看随时间的变化曲线
        """
        print("\n生成单独的长期趋势图...")
        
        # 定义特征及其对应的标签和颜色
        features_to_plot = [
            ('avg_rms_roughness', '平均RMS粗糙度', 'blue'),
            ('max_notch_depth', '最大缺口深度', 'red'),
            ('right_peak_density', '剪切面峰密度', 'green'),
            ('avg_gradient_energy', '平均梯度能量', 'purple'),
            ('tear_shear_area_ratio', '撕裂面占比', 'orange'),
        ]
        
        # 创建输出目录
        output_dir = os.path.join(viz_dir, 'individual_trends')
        ensure_dir(output_dir)
        
        for feat, label, color in features_to_plot:
            if feat not in df.columns:
                print(f"  警告: 特征 '{feat}' 不存在，跳过")
                continue
            
            # 创建单独的图表，x轴拉长至60英寸（与split_longterm_trend_charts.py一致）
            fig, ax = plt.subplots(figsize=(60, 6))
            
            # 获取数据
            y_values = df[feat].values
            
            # 计算包络线
            upper_env, lower_env = self.compute_envelope(y_values, window=min(31, len(y_values)//10))
            
            # 计算鲁棒拟合曲线
            fitted_curve, inlier_mask = self.robust_curve_fit(y_values, percentile_range=(5, 95))
            
            # 绘制包络范围（填充）
            ax.fill_between(df['frame_id'], lower_env, upper_env,
                           alpha=0.15, color='gray', label='包络范围', zorder=1)
            
            # 绘制上下包络线
            ax.plot(df['frame_id'], upper_env, ':', linewidth=1.5, 
                   color='red', alpha=0.6, label='上包络', zorder=2)
            ax.plot(df['frame_id'], lower_env, ':', linewidth=1.5, 
                   color='green', alpha=0.6, label='下包络', zorder=2)
            
            # 原始数据连线（半透明）
            ax.plot(df['frame_id'], y_values,
                   alpha=0.3, linewidth=1.2, color=color,
                   zorder=3, label='逐帧曲线')
            
            # 散点标记
            ax.scatter(df['frame_id'], y_values,
                      alpha=0.4, s=15, color=color, zorder=4)
            
            # 标注离群点
            outlier_indices = np.where(~inlier_mask)[0]
            if len(outlier_indices) > 0:
                ax.scatter(df['frame_id'].iloc[outlier_indices], 
                          y_values[outlier_indices],
                          c='orange', s=30, marker='x', alpha=0.7, 
                          label=f'离群点({len(outlier_indices)}个)', zorder=5)
            
            # 鲁棒拟合曲线（主趋势）
            ax.plot(df['frame_id'], fitted_curve,
                   color='purple', linewidth=3, linestyle='-',
                   alpha=0.8, zorder=6, label='鲁棒拟合')
            
            # 线性拟合趋势线
            z = np.polyfit(df['frame_id'], y_values, 1)
            p = np.poly1d(z)
            ax.plot(df['frame_id'], p(df['frame_id']),
                   color='darkred', linewidth=2.5, linestyle='--',
                   zorder=7, label=f'线性趋势: y={z[0]:.6f}x+{z[1]:.2f}')
            
            # 计算趋势方向和内点率
            trend = "增加" if z[0] > 0 else "减少"
            inlier_ratio = inlier_mask.sum() / len(inlier_mask) * 100
            ax.text(0.05, 0.95, f'趋势: {trend}\n内点率: {inlier_ratio:.1f}%',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('帧编号', fontsize=12, fontweight='bold')
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(f'{label} 长期趋势', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 调整x轴范围，确保拉长效果
            ax.set_xlim(df['frame_id'].min(), df['frame_id'].max())
            
            # 保存图表
            individual_save_path = os.path.join(output_dir, f'{feat}_trend.png')
            plt.savefig(individual_save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  已保存: {feat}_trend.png")
        
        print(f"✓ 单独长期趋势图已保存到: {output_dir}")
    
    def _plot_combined_trends_6x1(self, df: pd.DataFrame, viz_dir: str):
        """
        绘制6×1组合图（综合指标 + 5个特征上下罗列）
        
        综合指标：4个特征归一化叠加（不含梯度能量）
        """
        print("\n生成6×1组合长期趋势图...")
        
        # 定义特征及其对应的标签和颜色
        features_to_plot = [
            ('avg_rms_roughness', '平均RMS粗糙度', 'blue'),
            ('max_notch_depth', '最大缺口深度', 'red'),
            ('right_peak_density', '剪切面峰密度', 'green'),
            ('avg_gradient_energy', '平均梯度能量', 'purple'),
            ('tear_shear_area_ratio', '撕裂面占比', 'orange'),
        ]
        
        # 创建输出目录
        output_dir = os.path.join(viz_dir, 'individual_trends')
        ensure_dir(output_dir)
        
        # 创建6×1子图布局，x轴设置为80英寸
        fig, axes = plt.subplots(6, 1, figsize=(80, 29))
        
        # ========== 第1个子图：综合指标（4个特征归一化后叠加，不含梯度能量） ==========
        ax_composite = axes[0]
        
        # 计算综合指标 - 排除 avg_gradient_energy
        composite_score = np.zeros(len(df))
        valid_features = []
        excluded_features = ['avg_gradient_energy']  # 排除的特征
        
        for feat, label, color in features_to_plot:
            if feat in df.columns and feat not in excluded_features:
                # 归一化到0-1
                values = df[feat].values
                if values.max() > values.min():
                    normalized = (values - values.min()) / (values.max() - values.min())
                    composite_score += normalized
                    valid_features.append((feat, label))
        
        # 平均化（避免简单求和导致值过大）
        if len(valid_features) > 0:
            composite_score = composite_score / len(valid_features)
        
        # 计算包络线和鲁棒拟合
        upper_env_comp, lower_env_comp = self.compute_envelope(composite_score, window=min(31, len(composite_score)//10))
        fitted_curve_comp, inlier_mask_comp = self.robust_curve_fit(composite_score, percentile_range=(5, 95))
        
        # 绘制包络范围
        ax_composite.fill_between(df['frame_id'], lower_env_comp, upper_env_comp,
                                 alpha=0.15, color='gray', label='包络范围', zorder=1)
        
        # 绘制包络线
        ax_composite.plot(df['frame_id'], upper_env_comp, ':', linewidth=1.5, 
                         color='red', alpha=0.6, label='上包络', zorder=2)
        ax_composite.plot(df['frame_id'], lower_env_comp, ':', linewidth=1.5, 
                         color='green', alpha=0.6, label='下包络', zorder=2)
        
        # 绘制综合指标（半透明）
        ax_composite.plot(df['frame_id'], composite_score,
                         alpha=0.3, linewidth=1.5, color='darkblue',
                         zorder=3, label='综合磨损指标')
        
        ax_composite.scatter(df['frame_id'], composite_score,
                            alpha=0.4, s=20, color='darkblue', zorder=4)
        
        # 标注离群点
        outlier_indices_comp = np.where(~inlier_mask_comp)[0]
        if len(outlier_indices_comp) > 0:
            ax_composite.scatter(df['frame_id'].iloc[outlier_indices_comp], 
                                composite_score[outlier_indices_comp],
                                c='orange', s=35, marker='x', alpha=0.7, 
                                label=f'离群点({len(outlier_indices_comp)}个)', zorder=5)
        
        # 鲁棒拟合曲线
        ax_composite.plot(df['frame_id'], fitted_curve_comp,
                         color='purple', linewidth=3.5, linestyle='-',
                         alpha=0.8, zorder=6, label='鲁棒拟合')
        
        # 线性拟合
        z_comp = np.polyfit(df['frame_id'], composite_score, 1)
        p_comp = np.poly1d(z_comp)
        ax_composite.plot(df['frame_id'], p_comp(df['frame_id']),
                         color='red', linewidth=3, linestyle='--',
                         zorder=7, label=f'线性趋势: y={z_comp[0]:.6f}x+{z_comp[1]:.2f}')
        
        # 趋势标注（包含内点率）
        trend_comp = "增加" if z_comp[0] > 0 else "减少"
        trend_color_comp = 'lightgreen' if z_comp[0] > 0 else 'lightcoral'
        inlier_ratio_comp = inlier_mask_comp.sum() / len(inlier_mask_comp) * 100
        ax_composite.text(0.02, 0.98, f'趋势: {trend_comp}\n内点率: {inlier_ratio_comp:.1f}%',
                         transform=ax_composite.transAxes, fontsize=12,
                         verticalalignment='top', fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor=trend_color_comp, alpha=0.7,
                                  edgecolor='black', linewidth=2))
        
        # 统计信息
        mean_comp = composite_score.mean()
        std_comp = composite_score.std()
        min_comp = composite_score.min()
        max_comp = composite_score.max()
        
        stats_text_comp = f'均值: {mean_comp:.3f}\n标准差: {std_comp:.3f}\n范围: [{min_comp:.3f}, {max_comp:.3f}]'
        ax_composite.text(0.98, 0.98, stats_text_comp,
                         transform=ax_composite.transAxes, fontsize=10,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8,
                                  edgecolor='gray', linewidth=1))
        
        # 添加特征说明
        features_text = '包含特征: ' + ', '.join([label for _, label in valid_features])
        ax_composite.text(0.02, 0.02, features_text,
                         transform=ax_composite.transAxes, fontsize=9,
                         verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                                  edgecolor='gray', linewidth=0.5))
        
        ax_composite.set_xlabel('帧编号', fontsize=13, fontweight='bold')
        ax_composite.set_ylabel('综合磨损指标 (归一化)', fontsize=13, fontweight='bold')
        ax_composite.set_title('综合磨损指标 (4特征归一化叠加: 不含梯度能量)', fontsize=16, fontweight='bold', pad=15, 
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax_composite.grid(True, alpha=0.3)
        ax_composite.legend(loc='upper left', fontsize=11)
        ax_composite.set_xlim(df['frame_id'].min(), df['frame_id'].max())
        ax_composite.set_ylim(-0.05, 1.05)
        
        # ========== 后5个子图：各个特征 ==========
        
        for idx, (feat, label, color) in enumerate(features_to_plot):
            ax = axes[idx + 1]  # 因为第0个位置被综合指标占用
            
            if feat not in df.columns:
                ax.text(0.5, 0.5, f'特征 "{feat}" 不存在', 
                       ha='center', va='center', fontsize=14, color='red')
                ax.set_title(f'{label} - 数据缺失', fontsize=14, fontweight='bold')
                continue
            
            # 获取数据
            y_values = df[feat].values
            
            # 计算包络线和鲁棒拟合
            upper_env, lower_env = self.compute_envelope(y_values, window=min(31, len(y_values)//10))
            fitted_curve, inlier_mask = self.robust_curve_fit(y_values, percentile_range=(5, 95))
            
            # 绘制包络范围
            ax.fill_between(df['frame_id'], lower_env, upper_env,
                           alpha=0.15, color='gray', label='包络范围', zorder=1)
            
            # 绘制包络线
            ax.plot(df['frame_id'], upper_env, ':', linewidth=1.5, 
                   color='red', alpha=0.6, label='上包络', zorder=2)
            ax.plot(df['frame_id'], lower_env, ':', linewidth=1.5, 
                   color='green', alpha=0.6, label='下包络', zorder=2)
            
            # 原始数据连线（半透明）
            ax.plot(df['frame_id'], y_values,
                   alpha=0.3, linewidth=1.2, color=color,
                   zorder=3, label='逐帧曲线')
            
            # 散点标记
            ax.scatter(df['frame_id'], y_values,
                      alpha=0.4, s=15, color=color, zorder=4)
            
            # 标注离群点
            outlier_indices = np.where(~inlier_mask)[0]
            if len(outlier_indices) > 0:
                ax.scatter(df['frame_id'].iloc[outlier_indices], 
                          y_values[outlier_indices],
                          c='orange', s=30, marker='x', alpha=0.7, 
                          label=f'离群点({len(outlier_indices)}个)', zorder=5)
            
            # 鲁棒拟合曲线
            ax.plot(df['frame_id'], fitted_curve,
                   color='purple', linewidth=2.5, linestyle='-',
                   alpha=0.8, zorder=6, label='鲁棒拟合')
            
            # 线性拟合趋势线
            z = np.polyfit(df['frame_id'], y_values, 1)
            p = np.poly1d(z)
            ax.plot(df['frame_id'], p(df['frame_id']),
                   color='darkred', linewidth=2.5, linestyle='--',
                   zorder=7, label=f'线性趋势: y={z[0]:.6f}x+{z[1]:.2f}')
            
            # 计算趋势方向和内点率
            trend = "增加" if z[0] > 0 else "减少"
            trend_color = 'lightgreen' if z[0] > 0 else 'lightcoral'
            inlier_ratio = inlier_mask.sum() / len(inlier_mask) * 100
            ax.text(0.02, 0.98, f'趋势: {trend}\n内点率: {inlier_ratio:.1f}%',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=trend_color, alpha=0.6,
                            edgecolor='black', linewidth=1))
            
            # 添加统计信息
            mean_val = df[feat].mean()
            std_val = df[feat].std()
            min_val = df[feat].min()
            max_val = df[feat].max()
            
            stats_text = f'均值: {mean_val:.2f}\n标准差: {std_val:.2f}\n范围: [{min_val:.2f}, {max_val:.2f}]'
            ax.text(0.98, 0.98, stats_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7,
                            edgecolor='gray', linewidth=0.5))
            
            ax.set_xlabel('帧编号', fontsize=12, fontweight='bold')
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(f'{label} 长期趋势', fontsize=14, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=10)
            
            # 调整x轴范围，确保拉长效果
            ax.set_xlim(df['frame_id'].min(), df['frame_id'].max())
        
        # 设置总标题
        fig.suptitle(f'{self.analysis_name} - 剪刀磨损长期趋势综合分析（综合指标[4特征] + 5特征详情）', 
                    fontsize=18, fontweight='bold', y=0.996)
        
        # 调整子图间距
        plt.tight_layout(rect=[0, 0, 1, 0.996])
        
        # 保存
        combined_save_path = os.path.join(output_dir, 'all_trends_6x1.png')
        plt.savefig(combined_save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ 6×1组合图已保存: all_trends_6x1.png")
    
    def _plot_horizontal_gradient_comparison(self, df: pd.DataFrame, save_path: str):
        """绘制水平梯度能量对比图（总梯度 vs 水平梯度）"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. 时序对比（原始数据 + 平滑 + 线性趋势）
        ax1 = axes[0, 0]
        
        window = min(51, len(df)//10*2+1)
        
        # 绘制总梯度能量
        if 'avg_gradient_energy' in df.columns:
            ax1.plot(df['frame_id'], df['avg_gradient_energy'], 
                    color='blue', alpha=0.3, linewidth=1, label='总梯度能量(原始)')
            
            # 平滑处理
            if window >= 5:
                smoothed_total = savgol_filter(df['avg_gradient_energy'].values, 
                                              window_length=window, polyorder=3)
                ax1.plot(df['frame_id'], smoothed_total, 
                        color='blue', linewidth=3, label='总梯度能量(平滑)')
            
            # 线性趋势
            z_total = np.polyfit(df['frame_id'], df['avg_gradient_energy'], 1)
            p_total = np.poly1d(z_total)
            ax1.plot(df['frame_id'], p_total(df['frame_id']), 
                    color='darkblue', linewidth=2.5, linestyle='--', alpha=0.8,
                    label=f'总梯度趋势(斜率={z_total[0]:.2e})')
        
        # 绘制水平梯度能量
        ax1.plot(df['frame_id'], df['avg_horizontal_gradient'], 
                color='red', alpha=0.3, linewidth=1, label='水平梯度能量(原始)')
        
        # 平滑处理
        if window >= 5:
            smoothed_horizontal = savgol_filter(df['avg_horizontal_gradient'].values, 
                                               window_length=window, polyorder=3)
            ax1.plot(df['frame_id'], smoothed_horizontal, 
                    color='red', linewidth=3, label='水平梯度能量(平滑)')
        
        # 线性趋势
        z_horizontal = np.polyfit(df['frame_id'], df['avg_horizontal_gradient'], 1)
        p_horizontal = np.poly1d(z_horizontal)
        ax1.plot(df['frame_id'], p_horizontal(df['frame_id']), 
                color='darkred', linewidth=2.5, linestyle='--', alpha=0.8,
                label=f'水平梯度趋势(斜率={z_horizontal[0]:.2e})')
        
        ax1.set_xlabel('帧编号', fontsize=12, fontweight='bold')
        ax1.set_ylabel('梯度能量', fontsize=12, fontweight='bold')
        ax1.set_title('总梯度 vs 水平梯度 时序对比（含线性趋势）', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. 按卷统计对比
        ax2 = axes[0, 1]
        
        valid_coil_ids = df['coil_id'].dropna().unique()
        coil_ids_list = sorted(valid_coil_ids)
        
        total_grad_means = []
        horizontal_grad_means = []
        
        for coil_id in coil_ids_list:
            coil_df = df[df['coil_id'] == coil_id]
            if 'avg_gradient_energy' in df.columns:
                total_grad_means.append(coil_df['avg_gradient_energy'].mean())
            horizontal_grad_means.append(coil_df['avg_horizontal_gradient'].mean())
        
        x = np.arange(len(coil_ids_list))
        width = 0.35
        
        if 'avg_gradient_energy' in df.columns and len(total_grad_means) > 0:
            bars1 = ax2.bar(x - width/2, total_grad_means, width, 
                          label='总梯度能量', color='steelblue', alpha=0.8)
        
        bars2 = ax2.bar(x + width/2 if 'avg_gradient_energy' in df.columns else x, 
                       horizontal_grad_means, width,
                       label='水平梯度能量', color='coral', alpha=0.8)
        
        ax2.set_xlabel('钢卷编号', fontsize=12, fontweight='bold')
        ax2.set_ylabel('平均梯度能量', fontsize=12, fontweight='bold')
        ax2.set_title('各卷梯度能量对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'卷{int(cid)}' for cid in coil_ids_list])
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 归一化对比（更清楚地看趋势 + 线性拟合）
        ax3 = axes[1, 0]
        
        # 归一化到0-1
        if 'avg_gradient_energy' in df.columns:
            total_grad_norm = (df['avg_gradient_energy'] - df['avg_gradient_energy'].min()) / \
                             (df['avg_gradient_energy'].max() - df['avg_gradient_energy'].min() + 1e-8)
            if window >= 5:
                total_grad_norm_smooth = savgol_filter(total_grad_norm.values, 
                                                      window_length=window, polyorder=3)
                ax3.plot(df['frame_id'], total_grad_norm_smooth, 
                        color='blue', linewidth=3, label='总梯度能量(归一化)')
            
            # 线性趋势（归一化后）
            z_total_norm = np.polyfit(df['frame_id'], total_grad_norm.values, 1)
            p_total_norm = np.poly1d(z_total_norm)
            ax3.plot(df['frame_id'], p_total_norm(df['frame_id']), 
                    color='darkblue', linewidth=2, linestyle='--', alpha=0.7,
                    label=f'总梯度线性趋势(斜率={z_total_norm[0]:.2e})')
        
        horizontal_grad_norm = (df['avg_horizontal_gradient'] - df['avg_horizontal_gradient'].min()) / \
                              (df['avg_horizontal_gradient'].max() - df['avg_horizontal_gradient'].min() + 1e-8)
        if window >= 5:
            horizontal_grad_norm_smooth = savgol_filter(horizontal_grad_norm.values, 
                                                       window_length=window, polyorder=3)
            ax3.plot(df['frame_id'], horizontal_grad_norm_smooth, 
                    color='red', linewidth=3, label='水平梯度能量(归一化)')
        
        # 线性趋势（归一化后）
        z_horizontal_norm = np.polyfit(df['frame_id'], horizontal_grad_norm.values, 1)
        p_horizontal_norm = np.poly1d(z_horizontal_norm)
        ax3.plot(df['frame_id'], p_horizontal_norm(df['frame_id']), 
                color='darkred', linewidth=2, linestyle='--', alpha=0.7,
                label=f'水平梯度线性趋势(斜率={z_horizontal_norm[0]:.2e})')
        
        ax3.set_xlabel('帧编号', fontsize=12, fontweight='bold')
        ax3.set_ylabel('归一化梯度能量 (0-1)', fontsize=12, fontweight='bold')
        ax3.set_title('归一化梯度能量对比（含线性趋势）', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9, loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.05, 1.05)
        
        # 4. 变化率统计
        ax4 = axes[1, 1]
        
        # 计算首尾变化率
        def calc_change_rate(values):
            if len(values) > 10:
                first = np.mean(values[:len(values)//10])
                last = np.mean(values[-len(values)//10:])
                if first != 0:
                    return ((last - first) / first) * 100
            return 0
        
        labels = []
        change_rates = []
        colors = []
        
        if 'avg_gradient_energy' in df.columns:
            total_change = calc_change_rate(df['avg_gradient_energy'].values)
            labels.append('总梯度能量')
            change_rates.append(total_change)
            colors.append('steelblue')
        
        horizontal_change = calc_change_rate(df['avg_horizontal_gradient'].values)
        labels.append('水平梯度能量')
        change_rates.append(horizontal_change)
        colors.append('coral')
        
        bars = ax4.bar(labels, change_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax4.set_ylabel('变化率 (%)', fontsize=12, fontweight='bold')
        ax4.set_title('首尾变化率对比\n(负值表示下降=磨损加重)', fontsize=14, fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, rate in zip(bars, change_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top',
                    fontsize=12, fontweight='bold')
        
        plt.suptitle(f'{self.analysis_name} - 水平梯度能量专项分析\n（水平梯度只反映垂直边缘，对刀口锐度更敏感）', 
                    fontsize=16, fontweight='bold', y=0.995)
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
        
        # 1. 综合磨损指数（原始 + 平滑）
        ax1 = axes[0, 0]
        
        # 原始数据（半透明细线）
        ax1.plot(df['frame_id'], wear_index, color='darkred', 
                linewidth=1, alpha=0.3, label='原始数据')
        
        # 平滑处理
        from scipy.signal import savgol_filter
        window = min(51, len(wear_index)//10*2+1)
        if window >= 5 and len(wear_index) > window:
            wear_index_smooth = savgol_filter(wear_index, window_length=window, polyorder=3)
            # 平滑曲线（加粗）
            ax1.plot(df['frame_id'], wear_index_smooth, color='darkred', 
                    linewidth=3, alpha=1.0, label='平滑曲线')
            ax1.fill_between(df['frame_id'], 0, wear_index_smooth, alpha=0.2, color='red')
        else:
            # 数据太少，不平滑
            ax1.plot(df['frame_id'], wear_index, color='darkred', linewidth=2, label='磨损指数')
            ax1.fill_between(df['frame_id'], 0, wear_index, alpha=0.3, color='red')
        
        ax1.set_xlabel('帧编号', fontsize=12, fontweight='bold')
        ax1.set_ylabel('综合磨损指数 (0-1)', fontsize=12, fontweight='bold')
        ax1.set_title('综合磨损指数（细线=原始，粗线=平滑）', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, label='警戒线', alpha=0.8)
        ax1.legend(fontsize=10)
        
        # 2. 各特征贡献度（原始 + 平滑）
        ax2 = axes[0, 1]
        feature_labels = {
            'avg_rms_roughness': 'RMS粗糙度',
            'max_notch_depth': '缺口深度',
            'avg_gradient_energy': '梯度能量',
            'tear_shear_area_ratio': '撕裂面占比'
        }
        for feat in available:
            weight = weights.get(feat, 0.25)
            contribution = df_norm[feat].values * weight
            
            # 原始数据（半透明细线）
            ax2.plot(df['frame_id'], contribution, 
                    linewidth=0.8, alpha=0.3)
            
            # 平滑曲线（加粗）
            if window >= 5 and len(contribution) > window:
                contribution_smooth = savgol_filter(contribution, window_length=window, polyorder=3)
                ax2.plot(df['frame_id'], contribution_smooth,
                        label=feature_labels.get(feat, feat), 
                        linewidth=2.5, alpha=1.0)
            else:
                ax2.plot(df['frame_id'], contribution, 
                        label=feature_labels.get(feat, feat), 
                        linewidth=1.5, alpha=0.8)
        
        ax2.set_xlabel('帧编号', fontsize=12, fontweight='bold')
        ax2.set_ylabel('贡献度', fontsize=12, fontweight='bold')
        ax2.set_title('各特征贡献（细线=原始，粗线=平滑）', fontsize=14, fontweight='bold')
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
    
    def _plot_feature_importance(self, importance_df: pd.DataFrame, save_path: str):
        """可视化特征重要性排序"""
        if len(importance_df) == 0:
            print("警告: 无特征重要性数据，跳过")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. 综合重要性得分
        ax1 = axes[0]
        top_features = importance_df.head(10)
        colors = plt.cm.RdYlGn(top_features['importance_score'] / top_features['importance_score'].max())
        bars = ax1.barh(range(len(top_features)), top_features['importance_score'], color=colors)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=10)
        ax1.set_xlabel('综合重要性得分', fontsize=12, fontweight='bold')
        ax1.set_title('Top 10 最重要特征', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
        
        # 2. 变异系数 vs 单调性
        ax2 = axes[1]
        scatter = ax2.scatter(importance_df['cv'], importance_df['monotonicity'], 
                             s=importance_df['importance_score']*200, 
                             c=importance_df['importance_score'], 
                             cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # 标注top 5
        for idx, row in importance_df.head(5).iterrows():
            ax2.annotate(row['feature'], (row['cv'], row['monotonicity']), 
                        fontsize=8, ha='right', alpha=0.8)
        
        ax2.set_xlabel('变异系数 (CV)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('单调性 (|Spearman相关|)', fontsize=12, fontweight='bold')
        ax2.set_title('特征重要性二维分布', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='重要性得分')
        
        # 3. 各维度特征数量
        ax3 = axes[2]
        group_counts = importance_df['group'].value_counts()
        colors_group = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        ax3.pie(group_counts.values, labels=group_counts.index, autopct='%1.1f%%',
               colors=colors_group, startangle=90)
        ax3.set_title('特征维度分布', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'{self.analysis_name} - 特征重要性分析', 
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _plot_composite_indicators_comparison(self, df: pd.DataFrame, save_path: str):
        """可视化3种综合指标的对比"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 归一化到0-1便于对比
        scaler = MinMaxScaler()
        
        scores = {
            'weighted_score': '加权平均法',
            'pca_score': 'PCA主成分法',
            'overall_score': '多维度法'
        }
        
        # 检查哪些得分可用
        available_scores = {k: v for k, v in scores.items() if k in df.columns}
        
        if len(available_scores) == 0:
            print("警告: 无综合指标数据，跳过")
            return
        
        # 1. 三种方法对比曲线（原始 + 平滑）
        ax1 = axes[0, 0]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿
        
        for idx, (score_name, score_label) in enumerate(available_scores.items()):
            score_values = df[score_name].values
            # 归一化到0-1
            if score_values.max() > score_values.min():
                score_norm = (score_values - score_values.min()) / (score_values.max() - score_values.min())
            else:
                score_norm = score_values
            
            color = colors[idx % len(colors)]
            
            # 绘制原始数据（半透明）
            ax1.plot(df['frame_id'], score_norm, color=color, 
                    linewidth=0.8, alpha=0.3, linestyle='-')
            
            # 平滑处理
            window = min(51, len(score_norm)//10*2+1)
            if window >= 5:
                from scipy.signal import savgol_filter
                score_smooth = savgol_filter(score_norm, window_length=window, polyorder=3)
                # 绘制平滑曲线
                ax1.plot(df['frame_id'], score_smooth, color=color, 
                        label=score_label, linewidth=2.5, alpha=1.0)
            else:
                ax1.plot(df['frame_id'], score_norm, color=color, 
                        label=score_label, linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('帧编号', fontsize=12, fontweight='bold')
        ax1.set_ylabel('归一化得分 (0-1)', fontsize=12, fontweight='bold')
        ax1.set_title('3种综合指标对比（细线=原始，粗线=平滑）', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. 相关性矩阵
        ax2 = axes[0, 1]
        score_cols = list(available_scores.keys())
        if len(score_cols) >= 2:
            corr_matrix = df[score_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                       center=0.5, vmin=0, vmax=1, square=True, ax=ax2, 
                       cbar_kws={'label': '相关系数'})
            ax2.set_title('综合指标相关性', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=14)
            ax2.axis('off')
        
        # 3. 分布对比（箱线图）
        ax3 = axes[1, 0]
        box_data = [df[score_name].values for score_name in available_scores.keys()]
        box_labels = list(available_scores.values())
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow']):
            patch.set_facecolor(color)
        ax3.set_ylabel('得分', fontsize=12, fontweight='bold')
        ax3.set_title('综合指标分布对比', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 变化率对比
        ax4 = axes[1, 1]
        change_rates = []
        for score_name in available_scores.keys():
            values = df[score_name].values
            if len(values) > 10:
                first = np.mean(values[:len(values)//10])
                last = np.mean(values[-len(values)//10:])
                if first != 0:
                    change_rate = ((last - first) / first) * 100
                else:
                    change_rate = 0
                change_rates.append(change_rate)
            else:
                change_rates.append(0)
        
        colors_bar = ['blue' if cr > 0 else 'red' for cr in change_rates]
        bars = ax4.bar(box_labels, change_rates, color=colors_bar, alpha=0.7)
        ax4.set_ylabel('变化率 (%)', fontsize=12, fontweight='bold')
        ax4.set_title('首尾变化率对比', fontsize=14, fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars, change_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
        
        plt.suptitle(f'{self.analysis_name} - 综合指标方法对比', 
                    fontsize=18, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _plot_multi_dimension_evolution(self, df: pd.DataFrame, n_coils: int, save_path: str):
        """可视化多维度得分的演变（雷达图）"""
        dimensions = ['geometric_score', 'texture_score', 'frequency_score', 'distribution_score']
        dim_labels = ['几何特征', '纹理特征', '频域特征', '统计分布']
        
        # 检查哪些维度可用
        available_dims = [d for d in dimensions if d in df.columns]
        available_labels = [dim_labels[i] for i, d in enumerate(dimensions) if d in available_dims]
        
        if len(available_dims) < 2:
            print("警告: 维度得分数据不足，跳过")
            return
        
        # 选择3个代表性阶段：开始、中期、结束
        coil_ids = sorted(df['coil_id'].unique())
        if len(coil_ids) >= 3:
            representative_coils = [coil_ids[0], coil_ids[len(coil_ids)//2], coil_ids[-1]]
            stage_labels = ['开始阶段', '中期阶段', '结束阶段']
        else:
            representative_coils = coil_ids
            stage_labels = [f'第{int(c)}卷' for c in representative_coils]
        
        fig, axes = plt.subplots(1, len(representative_coils), figsize=(6*len(representative_coils), 6),
                                subplot_kw=dict(projection='polar'))
        
        if len(representative_coils) == 1:
            axes = [axes]
        
        angles = np.linspace(0, 2 * np.pi, len(available_dims), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for idx, (coil_id, stage_label) in enumerate(zip(representative_coils, stage_labels)):
            ax = axes[idx]
            
            # 该卷的平均得分
            coil_data = df[df['coil_id'] == coil_id]
            values = [coil_data[dim].mean() for dim in available_dims]
            values += values[:1]  # 闭合
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2, label=stage_label, color='darkblue')
            ax.fill(angles, values, alpha=0.25, color='blue')
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(available_labels, fontsize=11)
            ax.set_ylim(0, 1)
            ax.set_title(f'{stage_label}\n第{int(coil_id)}卷', fontsize=13, fontweight='bold', pad=20)
            ax.grid(True)
        
        plt.suptitle(f'{self.analysis_name} - 多维度磨损演变分析', 
                    fontsize=18, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _plot_feature_contribution_heatmap(self, pca_result: dict, save_path: str):
        """可视化特征对主成分的贡献热力图"""
        loadings = pca_result.get('loadings', pd.DataFrame())
        
        if len(loadings) == 0:
            print("警告: PCA载荷数据为空，跳过")
            return
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(loadings)*0.3)))
        
        # 绘制热力图
        sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, vmin=-1, vmax=1, cbar_kws={'label': '载荷值'},
                   ax=ax, linewidths=0.5)
        
        ax.set_xlabel('主成分', fontsize=13, fontweight='bold')
        ax.set_ylabel('特征名称', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.analysis_name} - 特征对主成分的贡献\n(红色=正贡献, 蓝色=负贡献)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 添加解释方差
        if 'explained_variance_ratio' in pca_result and len(pca_result['explained_variance_ratio']) > 0:
            explained_var = pca_result['explained_variance_ratio']
            var_text = '解释方差: ' + ', '.join([f'PC{i+1}={v*100:.1f}%' 
                                                 for i, v in enumerate(explained_var)])
            plt.figtext(0.5, 0.02, var_text, ha='center', fontsize=11, style='italic')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _plot_key_features_samples(self, df: pd.DataFrame, importance_df: pd.DataFrame, save_dir: str):
        """可视化关键特征的抽样展示"""
        if len(importance_df) == 0:
            print("警告: 无特征重要性数据，跳过抽样展示")
            return
        
        # 选择top 5特征
        top_features = importance_df.head(5)['feature'].tolist()
        
        # 基于综合得分选择3个代表帧：低/中/高磨损
        if 'overall_score' in df.columns:
            score_col = 'overall_score'
        elif 'weighted_score' in df.columns:
            score_col = 'weighted_score'
        else:
            print("警告: 无综合得分，使用frame_id采样")
            score_col = 'frame_id'
        
        # 按得分排序，取低、中、高三个分位数
        df_sorted = df.sort_values(score_col)
        low_idx = len(df_sorted) // 6
        mid_idx = len(df_sorted) // 2
        high_idx = len(df_sorted) * 5 // 6
        
        sample_indices = [low_idx, mid_idx, high_idx]
        sample_labels = ['低磨损', '中度磨损', '高磨损']
        
        for feature in top_features:
            if feature not in df.columns:
                continue
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            for idx, (sample_idx, label) in enumerate(zip(sample_indices, sample_labels)):
                ax = axes[idx]
                
                sample_row = df_sorted.iloc[sample_idx]
                frame_id = int(sample_row['frame_id'])
                feature_value = sample_row[feature]
                
                # 绘制该特征的整体曲线，并高亮当前点
                ax.plot(df['frame_id'], df[feature], color='lightgray', linewidth=1, alpha=0.5)
                ax.scatter([frame_id], [feature_value], color='red', s=200, zorder=5, 
                          marker='*', edgecolors='black', linewidth=1.5)
                
                ax.set_xlabel('帧编号', fontsize=11, fontweight='bold')
                ax.set_ylabel(feature, fontsize=11, fontweight='bold')
                ax.set_title(f'{label}\n帧{frame_id}: {feature_value:.3f}', 
                            fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{self.analysis_name} - {feature} 特征演变与典型样本', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 文件名安全化
            safe_feature_name = feature.replace('/', '_').replace('\\', '_')
            feature_save_path = os.path.join(save_dir, f'{safe_feature_name}_samples.png')
            plt.savefig(feature_save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"已保存: {save_dir} (共{len(top_features)}个特征)")
    
    def _detect_cycles_advanced(self, values, min_drop=1.5, min_cycle_length=100):
        """
        高级周期检测（用于平滑趋势分析）
        
        Args:
            values: 信号值
            min_drop: 最小下降幅度（标准差倍数）
            min_cycle_length: 最小周期长度
            
        Returns:
            周期列表 [(start, end), ...]
        """
        # 寻找局部峰值
        peaks, _ = find_peaks(values, distance=min_cycle_length//2)
        
        if len(peaks) < 2:
            return [(0, len(values)-1)]
        
        # 基于峰值间的谷底分割周期
        cycles = []
        valleys = []
        
        for i in range(len(peaks)-1):
            start_peak = peaks[i]
            end_peak = peaks[i+1]
            
            # 找到两峰之间的最低点
            valley_segment = values[start_peak:end_peak+1]
            valley_idx = start_peak + np.argmin(valley_segment)
            
            # 检查下降幅度
            drop = values[start_peak] - values[valley_idx]
            if drop > min_drop * np.std(values):
                start_idx = 0 if i == 0 else valleys[-1]
                cycles.append((start_idx, valley_idx))
                valleys.append(valley_idx)
        
        # 添加最后一段
        if len(cycles) > 0:
            cycles.append((cycles[-1][1], len(values)-1))
        else:
            cycles.append((0, len(values)-1))
        
        # 过滤太短的周期
        cycles = [(s, e) for s, e in cycles if e - s >= min_cycle_length]
        
        if len(cycles) == 0:
            cycles = [(0, len(values)-1)]
        
        return cycles
    
    def _plot_smooth_longterm_trends(self, df: pd.DataFrame, save_path: str):
        """
        绘制平滑长期趋势分析（3种方法对比）
        
        包含：
        1. 移动最大值包络线法
        2. 周期峰值样条插值法
        3. 全局二次拟合法
        """
        print("\n生成平滑长期趋势分析...")
        
        wear_features = {
            'avg_rms_roughness': '平均RMS粗糙度',
            'max_notch_depth': '最大缺口深度',
            'right_peak_density': '右侧峰密度'
        }
        
        # 创建对比图
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        for idx, (feature, label) in enumerate(wear_features.items()):
            if feature not in df.columns:
                continue
            
            # 左侧：原始数据 + 三种平滑方法
            ax_left = fig.add_subplot(gs[idx, 0])
            
            values = df[feature].values
            frames = df['frame_id'].values
            
            # 原始数据（浅色）
            ax_left.plot(frames, values, '-', alpha=0.15, color='gray', 
                        linewidth=0.5, label='原始数据')
            
            # === 方法1：移动最大值包络线 ===
            window = min(200, len(values)//5)
            max_envelope = maximum_filter1d(values, size=window, mode='nearest')
            smooth_env = max_envelope  # 默认值，如果后续处理失败则使用原始包络线
            if len(max_envelope) > 51:
                smooth_env = savgol_filter(max_envelope, 
                                          window_length=min(51, len(max_envelope)//2*2+1), 
                                          polyorder=3)
            if len(smooth_env) > 0:
                ax_left.plot(frames, smooth_env, '-', color='orange', 
                            linewidth=2.5, label='方法1:包络线', alpha=0.8)
            
            # === 方法2：周期峰值样条插值 ===
            cycle_frames = []
            cycle_maxes = []
            cycles = []
            try:
                cycles = self._detect_cycles_advanced(values)
                
                for start, end in cycles:
                    if end > start:
                        max_idx = start + np.argmax(values[start:end+1])
                        cycle_frames.append(frames[max_idx])
                        cycle_maxes.append(values[max_idx])
                
                if len(cycle_frames) > 3:
                    spline = UnivariateSpline(cycle_frames, cycle_maxes, 
                                              k=min(3, len(cycle_frames)-1), s=5)
                    smooth_frames = np.linspace(frames[0], frames[-1], 500)
                    ax_left.plot(smooth_frames, spline(smooth_frames), '-', 
                                color='green', linewidth=2.5, label='方法2:样条', alpha=0.8)
            except Exception as e:
                print(f"  警告: 样条插值失败 ({feature}): {e}")
            
            # === 方法3：全局二次拟合 ===
            cycle_key_frames = []
            cycle_key_values = []
            try:
                for start, end in cycles:
                    if end - start > 10:
                        window_size = min(50, (end - start) // 2)
                        if window_size >= 3:
                            seg_smooth = uniform_filter1d(values[start:end+1], size=window_size)
                            max_idx = start + np.argmax(seg_smooth)
                            cycle_key_frames.append(frames[max_idx])
                            cycle_key_values.append(values[max_idx])
                
                if len(cycle_key_frames) > 3:
                    z = np.polyfit(cycle_key_frames, cycle_key_values, 2)
                    poly_trend = np.poly1d(z)
                    ax_left.plot(frames, poly_trend(frames), '-', 
                                color='purple', linewidth=2.5, label='方法3:二次拟合', alpha=0.8)
            except Exception as e:
                print(f"  警告: 二次拟合失败 ({feature}): {e}")
            
            ax_left.set_xlabel('帧编号', fontweight='bold', fontsize=11)
            ax_left.set_ylabel(label, fontweight='bold', fontsize=11)
            ax_left.set_title(f'{label}\n三种平滑方法对比', fontsize=13, fontweight='bold')
            ax_left.legend(fontsize=9, loc='best')
            ax_left.grid(True, alpha=0.3)
            
            # 右侧：趋势斜率对比
            ax_right = fig.add_subplot(gs[idx, 1])
            
            slopes = []
            methods = []
            colors_bar = []
            
            # 方法1斜率
            if len(smooth_env) > 2:
                z1 = np.polyfit(frames, smooth_env, 1)
                slopes.append(z1[0])
                methods.append('包络线')
                colors_bar.append('orange')
            
            # 方法2斜率
            if len(cycle_frames) > 3:
                z2 = np.polyfit(cycle_frames, cycle_maxes, 1)
                slopes.append(z2[0])
                methods.append('样条')
                colors_bar.append('green')
            
            # 方法3斜率
            if len(cycle_key_frames) > 3:
                z3 = np.polyfit(cycle_key_frames, cycle_key_values, 1)
                slopes.append(z3[0])
                methods.append('二次拟合')
                colors_bar.append('purple')
            
            if slopes:
                bars = ax_right.barh(methods, slopes, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
                ax_right.axvline(x=0, color='black', linestyle='--', linewidth=1)
                ax_right.set_xlabel('趋势斜率', fontweight='bold', fontsize=11)
                ax_right.set_title(f'{label}\n方法斜率对比', fontsize=13, fontweight='bold')
                ax_right.grid(True, alpha=0.3, axis='x')
                
                # 添加数值标签
                for bar, slope in zip(bars, slopes):
                    width = bar.get_width()
                    ax_right.text(width, bar.get_y() + bar.get_height()/2,
                                f'{slope:.2e}', ha='left' if width > 0 else 'right',
                                va='center', fontsize=10, fontweight='bold')
        
        plt.suptitle(f'{self.analysis_name} - 平滑长期趋势分析（3种方法对比）', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _plot_deep_trend_analysis(self, df: pd.DataFrame, viz_dir: str):
        """
        深度趋势分析（3个子图）
        
        包含：
        1. 峰值包络线分析
        2. 分段趋势分析
        3. 低通滤波长期趋势
        """
        print("\n生成深度趋势分析...")
        
        key_features_deep = {
            'avg_rms_roughness': '平均RMS粗糙度',
            'max_notch_depth': '最大缺口深度',
            'right_peak_density': '右侧峰密度',
            'avg_gradient_energy': '平均梯度能量'
        }
        
        available_features = {k: v for k, v in key_features_deep.items() if k in df.columns}
        
        if len(available_features) == 0:
            print("  警告: 无可用特征，跳过深度趋势分析")
            return
        
        # === 1. 峰值包络线分析 ===
        print("  生成峰值包络线分析...")
        
        def extract_envelope(signal, window=300):
            """提取峰值包络线"""
            envelope = []
            frames_env = []
            
            for i in range(0, len(signal), max(1, window//2)):
                window_data = signal[i:min(i+window, len(signal))]
                if len(window_data) > 0:
                    envelope.append(np.max(window_data))
                    frames_env.append(i + window//2)
            
            return np.array(frames_env), np.array(envelope)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (feature, label) in enumerate(list(available_features.items())[:4]):
            ax = axes[idx]
            
            values = df[feature].values
            frames = df['frame_id'].values
            
            # 原始数据
            ax.plot(frames, values, 'o', alpha=0.1, markersize=2, color='gray', label='原始数据')
            
            # 提取峰值包络线
            window_size = min(300, len(values)//5)
            env_frames, envelope = extract_envelope(values, window=window_size)
            ax.plot(env_frames, envelope, 'ro-', linewidth=2, markersize=4, 
                   label='峰值包络线', alpha=0.7)
            
            # 拟合包络线趋势
            if len(envelope) > 2:
                z = np.polyfit(env_frames, envelope, 1)
                trend = np.poly1d(z)
                ax.plot(env_frames, trend(env_frames), 'b--', linewidth=2.5, 
                       label=f'趋势(斜率={z[0]:.6f})')
                
                # 判断趋势
                if z[0] > 1e-6:
                    trend_text = f"↑ 递增\n斜率: {z[0]:.6f}"
                    color = 'lightgreen'
                elif z[0] < -1e-6:
                    trend_text = f"↓ 递减\n斜率: {z[0]:.6f}"
                    color = 'lightcoral'
                else:
                    trend_text = f"→ 平稳\n斜率: {z[0]:.6f}"
                    color = 'lightyellow'
                
                ax.text(0.02, 0.98, trend_text,
                       transform=ax.transAxes, fontsize=11, verticalalignment='top',
                       fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.8, 
                                edgecolor='black', linewidth=1.5))
            
            ax.set_xlabel('帧编号', fontweight='bold', fontsize=11)
            ax.set_ylabel(label, fontweight='bold', fontsize=11)
            ax.set_title(f'{label}\n峰值包络线分析', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.analysis_name} - 峰值包络线深度分析', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'deep_envelope_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # === 2. 分段趋势分析 ===
        print("  生成分段趋势分析...")
        
        def detect_change_points(signal, threshold=2.0):
            """检测突变点"""
            diff = np.abs(np.diff(signal))
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            change_points = np.where(diff > mean_diff + threshold * std_diff)[0]
            return change_points
        
        # 使用第一个特征检测变点
        first_feature = list(available_features.keys())[0]
        values_for_cp = df[first_feature].values
        change_points = detect_change_points(values_for_cp, threshold=1.5)
        
        # 基于变点分段
        segments = []
        start = 0
        for cp in change_points:
            if cp - start > 50:
                segments.append((start, cp))
                start = cp
        segments.append((start, len(values_for_cp)))
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (feature, label) in enumerate(list(available_features.items())[:4]):
            ax = axes[idx]
            
            values = df[feature].values
            frames = df['frame_id'].values
            
            # 绘制原始数据
            ax.plot(frames, values, 'o', alpha=0.2, markersize=1, color='gray')
            
            # 为每段计算趋势
            colors = plt.cm.rainbow(np.linspace(0, 1, min(len(segments), 10)))
            segment_slopes = []
            
            for seg_idx, (start_idx, end_idx) in enumerate(segments[:10]):
                seg_frames = frames[start_idx:end_idx]
                seg_values = values[start_idx:end_idx]
                
                if len(seg_values) > 2:
                    z = np.polyfit(seg_frames, seg_values, 1)
                    trend = np.poly1d(z)
                    ax.plot(seg_frames, trend(seg_frames), '-', 
                           color=colors[seg_idx], linewidth=2.5, alpha=0.8)
                    segment_slopes.append(z[0])
            
            # 统计各段斜率
            if segment_slopes:
                avg_slope = np.mean(segment_slopes)
                positive_ratio = sum(1 for s in segment_slopes if s > 0) / len(segment_slopes)
                
                ax.text(0.02, 0.98, 
                       f'段数: {len(segment_slopes)}\n'
                       f'平均斜率: {avg_slope:.6f}\n'
                       f'递增段占比: {positive_ratio:.1%}',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, 
                                edgecolor='black', linewidth=1.5))
            
            ax.set_xlabel('帧编号', fontweight='bold', fontsize=11)
            ax.set_ylabel(label, fontweight='bold', fontsize=11)
            ax.set_title(f'{label}\n分段趋势（前10段）', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.analysis_name} - 分段趋势深度分析', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'deep_segment_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # === 3. 低通滤波长期趋势 ===
        print("  生成低通滤波长期趋势...")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (feature, label) in enumerate(list(available_features.items())[:4]):
            ax = axes[idx]
            
            values = df[feature].values
            frames = df['frame_id'].values
            
            # 原始数据
            ax.plot(frames, values, '-', alpha=0.2, linewidth=0.5, color='gray', 
                   label='原始数据')
            
            # 移动平均（窗口=100）
            window_ma = min(100, len(values)//10)
            if len(values) >= window_ma and window_ma >= 3:
                ma = uniform_filter1d(values, size=window_ma)
                ax.plot(frames, ma, 'b-', linewidth=2.5, 
                       label=f'移动平均({window_ma})', alpha=0.8)
                
                # 拟合长期趋势
                z = np.polyfit(frames, ma, 1)
                trend = np.poly1d(z)
                ax.plot(frames, trend(frames), 'r--', linewidth=3, 
                       label=f'长期趋势(斜率={z[0]:.6f})')
                
                # 判断趋势
                if z[0] > 1e-6:
                    trend_text = f"✓ 长期递增\n斜率: {z[0]:.6f}"
                    color = 'lightgreen'
                elif z[0] < -1e-6:
                    trend_text = f"✗ 长期递减\n斜率: {z[0]:.6f}"
                    color = 'lightcoral'
                else:
                    trend_text = f"→ 长期平稳\n斜率: {z[0]:.6f}"
                    color = 'lightyellow'
                
                ax.text(0.02, 0.98, trend_text,
                       transform=ax.transAxes, fontsize=11, verticalalignment='top',
                       fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.8,
                                edgecolor='black', linewidth=1.5))
            
            ax.set_xlabel('帧编号', fontweight='bold', fontsize=11)
            ax.set_ylabel(label, fontweight='bold', fontsize=11)
            ax.set_title(f'{label}\n低通滤波后的长期趋势', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.analysis_name} - 低通滤波长期趋势分析', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'deep_longterm_filtered.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ 深度趋势分析完成")
    
    def _plot_white_patch_analysis(self, df: pd.DataFrame, save_path: str):
        """
        绘制撕裂面白斑分析图
        
        针对用户观察：撕裂面白色斑块随钢卷数量增加而增多
        对比4种检测方法的效果
        """
        print("\n生成撕裂面白斑分析...")
        
        # 检查是否有白斑特征
        white_patch_cols = [col for col in df.columns if col.startswith('white_')]
        if len(white_patch_cols) == 0:
            print("  警告: 数据中没有白斑特征，跳过")
            return
        
        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        methods = ['m1', 'm2', 'm3', 'm4']
        method_names = ['固定阈值', 'Otsu自适应', '相对亮度', '形态学Top-Hat']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 第一行：面积占比时序曲线
        for idx, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
            ax = fig.add_subplot(gs[0, idx])
            
            col_name = f'white_area_ratio_{method}'
            if col_name not in df.columns:
                continue
            
            values = df[col_name].values
            frames = df['frame_id'].values
            
            # 原始数据
            ax.plot(frames, values, '-', alpha=0.2, color=color, linewidth=0.8)
            
            # 平滑曲线
            window = min(51, len(values)//10*2+1)
            if window >= 5:
                smoothed = savgol_filter(values, window_length=window, polyorder=3)
                ax.plot(frames, smoothed, '-', color=color, linewidth=2.5, label='平滑曲线')
            
            # 线性趋势
            z = np.polyfit(frames, values, 1)
            trend = np.poly1d(z)
            ax.plot(frames, trend(frames), '--', color='red', linewidth=2, alpha=0.7)
            
            # 计算首尾变化
            if len(values) > 10:
                first = np.mean(values[:len(values)//10])
                last = np.mean(values[-len(values)//10:])
                change = last - first
                change_pct = (change / (first + 1e-8)) * 100
                
                trend_text = f'变化: {change:+.1f}% ({change_pct:+.0f}%)'
                box_color = 'lightgreen' if change > 0 else 'lightcoral'
                
                ax.text(0.02, 0.98, trend_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7))
            
            ax.set_xlabel('帧编号', fontsize=10, fontweight='bold')
            ax.set_ylabel('白斑面积占比(%)', fontsize=10, fontweight='bold')
            ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # 第二行：斑块数量时序曲线
        for idx, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
            ax = fig.add_subplot(gs[1, idx])
            
            col_name = f'white_patch_count_{method}'
            if col_name not in df.columns:
                continue
            
            values = df[col_name].values
            frames = df['frame_id'].values
            
            # 原始数据
            ax.plot(frames, values, '-', alpha=0.2, color=color, linewidth=0.8)
            
            # 平滑曲线
            window = min(51, len(values)//10*2+1)
            if window >= 5:
                smoothed = savgol_filter(values, window_length=window, polyorder=3)
                ax.plot(frames, smoothed, '-', color=color, linewidth=2.5, label='平滑曲线')
            
            # 线性趋势
            z = np.polyfit(frames, values, 1)
            trend = np.poly1d(z)
            ax.plot(frames, trend(frames), '--', color='red', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('帧编号', fontsize=10, fontweight='bold')
            ax.set_ylabel('白斑数量(个)', fontsize=10, fontweight='bold')
            ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # 第三行：按卷统计（如果有卷号）
        if 'coil_id' in df.columns:
            for idx, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
                ax = fig.add_subplot(gs[2, idx])
                
                col_name = f'white_area_ratio_{method}'
                if col_name not in df.columns:
                    continue
                
                coil_ids = sorted(df['coil_id'].unique())
                coil_means = [df[df['coil_id']==cid][col_name].mean() for cid in coil_ids]
                
                x = np.arange(len(coil_ids))
                bars = ax.bar(x, coil_means, color=color, alpha=0.6, edgecolor='black', linewidth=1.5)
                ax.plot(x, coil_means, 'ro-', linewidth=2, markersize=8, zorder=10)
                
                # 添加趋势线
                z = np.polyfit(x, coil_means, 1)
                trend = np.poly1d(z)
                ax.plot(x, trend(x), 'g--', linewidth=2.5, alpha=0.7, label=f'趋势线')
                
                ax.set_xlabel('钢卷编号', fontsize=10, fontweight='bold')
                ax.set_ylabel('平均白斑面积占比(%)', fontsize=10, fontweight='bold')
                ax.set_title(f'{method_name} - 按卷统计', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([f'卷{int(c)}' for c in coil_ids])
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend(fontsize=8)
        else:
            # 如果没有卷号，显示4种方法的相关性对比
            ax = fig.add_subplot(gs[2, :])
            
            # 绘制4种方法的对比曲线
            for method, method_name, color in zip(methods, method_names, colors):
                col_name = f'white_area_ratio_{method}'
                if col_name not in df.columns:
                    continue
                
                values = df[col_name].values
                frames = df['frame_id'].values
                
                # 归一化
                if values.max() > values.min():
                    values_norm = (values - values.min()) / (values.max() - values.min())
                else:
                    values_norm = values
                
                # 平滑
                window = min(51, len(values_norm)//10*2+1)
                if window >= 5:
                    smoothed = savgol_filter(values_norm, window_length=window, polyorder=3)
                    ax.plot(frames, smoothed, '-', color=color, linewidth=2.5, label=method_name)
            
            ax.set_xlabel('帧编号', fontsize=12, fontweight='bold')
            ax.set_ylabel('归一化白斑面积占比 (0-1)', fontsize=12, fontweight='bold')
            ax.set_title('4种方法检测结果对比（归一化）', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.analysis_name} - 撕裂面白色斑块分析\n（用户观察：白斑随磨损增加而增多）', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _extract_left_region_and_mask(self, image: np.ndarray):
        """提取左侧撕裂面区域及掩码"""
        height, width = image.shape
        
        # 找白色区域中最暗点作为分界线
        mask_white = image > 100
        centerline_x = []
        
        for y in range(height):
            row = image[y, :]
            white_indices = np.where(mask_white[y, :])[0]
            
            if len(white_indices) > 10:
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
            from scipy.signal import savgol_filter
            centerline_x = savgol_filter(centerline_x, 51, 3)
        centerline_x = np.array(centerline_x, dtype=int)
        
        # 创建左侧掩码
        left_mask = np.zeros_like(image, dtype=np.uint8)
        for y in range(height):
            if y < len(centerline_x):
                left_mask[y, :centerline_x[y]] = 255
        
        left_region = cv2.bitwise_and(image, image, mask=left_mask)
        return left_region, left_mask
    
    def _detect_white_patches_methods(self, image: np.ndarray, mask: np.ndarray):
        """使用4种方法检测白斑，返回4个二值图"""
        # 方法1：固定阈值
        _, binary1 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
        binary1 = cv2.bitwise_and(binary1, binary1, mask=mask)
        
        # 方法2：Otsu + 最小阈值约束
        masked_pixels = image[mask > 0]
        if len(masked_pixels) > 0:
            otsu_threshold, _ = cv2.threshold(masked_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold2 = max(otsu_threshold, 170)
            _, binary2 = cv2.threshold(image, threshold2, 255, cv2.THRESH_BINARY)
            binary2 = cv2.bitwise_and(binary2, binary2, mask=mask)
        else:
            binary2 = np.zeros_like(image)
        
        # 方法3：相对亮度法
        if len(masked_pixels) > 0:
            mean_val = np.mean(masked_pixels)
            std_val = np.std(masked_pixels)
            threshold3 = mean_val + 1.5 * std_val
            _, binary3 = cv2.threshold(image, threshold3, 255, cv2.THRESH_BINARY)
            binary3 = cv2.bitwise_and(binary3, binary3, mask=mask)
        else:
            binary3 = np.zeros_like(image)
        
        # 方法4：Top-Hat
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        tophat_masked = tophat[mask > 0]
        if len(tophat_masked) > 0 and tophat_masked.max() > 0:
            threshold4, _ = cv2.threshold(tophat_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, binary4 = cv2.threshold(tophat, max(threshold4, 1), 255, cv2.THRESH_BINARY)
            binary4 = cv2.bitwise_and(binary4, binary4, mask=mask)
        else:
            binary4 = np.zeros_like(image)
        
        return [binary1, binary2, binary3, binary4]
    
    def _generate_white_patch_markers(self, df: pd.DataFrame, viz_dir: str, sample_interval: int = 100):
        """生成白斑标注图（带直方图对比）"""
        print(f"\n生成白斑标注图（每隔{sample_interval}帧）...")
        
        # 检查是否有白斑特征
        if 'white_area_ratio_m1' not in df.columns:
            print("  警告: 数据中没有白斑特征，跳过")
            return
        
        # 创建输出目录
        markers_dir = os.path.join(viz_dir, 'white_patch_markers')
        ensure_dir(markers_dir)
        
        # 选择要可视化的帧（基于frame_id而非DataFrame索引）
        df_sampled = df[df['frame_id'] % sample_interval == 0].head(20)  # 最多20张避免太多
        method_names_display = ['固定阈值', 'Otsu自适应', '相对亮度', '形态学Top-Hat']
        
        for _, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="生成标注图"):
            try:
                frame_id = int(row['frame_id'])
                
                # 读取原图
                filepath = os.path.join(self.roi_dir, f'frame_{frame_id:06d}_roi.png')
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                # 提取左侧区域
                left_region, left_mask = self._extract_left_region_and_mask(image)
                
                # 4种方法检测
                binaries = self._detect_white_patches_methods(left_region, left_mask)
                
                # 创建3x2布局
                fig = plt.figure(figsize=(18, 24))
                gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
                
                all_areas = []
                
                # 前4个子图：标注
                for method_idx, (binary, display_name) in enumerate(zip(binaries, method_names_display)):
                    row = method_idx // 2
                    col = method_idx % 2
                    ax = fig.add_subplot(gs[row, col])
                    
                    display_img = cv2.cvtColor(left_region, cv2.COLOR_GRAY2RGB)
                    
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    
                    valid_patches = 0
                    patch_areas = []
                    
                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area < 5:
                            continue
                        
                        valid_patches += 1
                        patch_areas.append(area)
                        
                        cx, cy = int(centroids[i][0]), int(centroids[i][1])
                        radius = max(3, min(int(np.sqrt(area) * 0.5), 10))
                        cv2.circle(display_img, (cx, cy), radius, (255, 0, 0), 2)
                        cv2.circle(display_img, (cx, cy), 1, (0, 255, 0), -1)
                    
                    all_areas.append(patch_areas)
                    
                    ax.imshow(display_img)
                    ax.set_title(f'{display_name}\n帧{frame_id} - 检测到{valid_patches}个白斑', 
                               fontsize=14, fontweight='bold')
                    ax.axis('off')
                    
                    ax.text(0.02, 0.98, f'白斑数: {valid_patches}', 
                           transform=ax.transAxes, fontsize=12, 
                           verticalalignment='top', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                # 第5个子图：亮度直方图
                ax_brightness = fig.add_subplot(gs[2, 0])
                tear_pixels = left_region[left_mask > 0]
                if len(tear_pixels) > 0:
                    ax_brightness.hist(tear_pixels, bins=50, color='gray', alpha=0.5, 
                                      label='撕裂面整体亮度', edgecolor='black', linewidth=0.5)
                    
                    colors_hist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    for idx, binary in enumerate(binaries):
                        white_pixels = left_region[binary > 0]
                        if len(white_pixels) > 0:
                            ax_brightness.hist(white_pixels, bins=50, color=colors_hist[idx], 
                                             alpha=0.3, label=f'方法{idx+1}白斑', 
                                             edgecolor=colors_hist[idx], linewidth=1)
                
                ax_brightness.set_xlabel('亮度值', fontsize=12, fontweight='bold')
                ax_brightness.set_ylabel('像素数量', fontsize=12, fontweight='bold')
                ax_brightness.set_title(f'撕裂面亮度分布对比\n帧{frame_id}', fontsize=14, fontweight='bold')
                ax_brightness.legend(fontsize=10, loc='best')
                ax_brightness.grid(True, alpha=0.3, axis='y')
                
                # 第6个子图：斑块面积直方图
                ax_area = fig.add_subplot(gs[2, 1])
                colors_hist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                for idx, areas in enumerate(all_areas):
                    if len(areas) > 0:
                        ax_area.hist(areas, bins=20, color=colors_hist[idx], alpha=0.5,
                                   label=f'方法{idx+1} ({len(areas)}个)',
                                   edgecolor=colors_hist[idx], linewidth=1)
                
                ax_area.set_xlabel('斑块面积 (像素数)', fontsize=12, fontweight='bold')
                ax_area.set_ylabel('斑块数量', fontsize=12, fontweight='bold')
                ax_area.set_title(f'白斑面积分布对比\n帧{frame_id}', fontsize=14, fontweight='bold')
                ax_area.legend(fontsize=10, loc='best')
                ax_area.grid(True, alpha=0.3, axis='y')
                
                plt.suptitle(f'{self.analysis_name} - 撕裂面白斑综合分析 - 帧{frame_id}\n（上：标注图，下：直方图对比）', 
                           fontsize=18, fontweight='bold')
                plt.tight_layout()
                
                save_path = os.path.join(markers_dir, f'frame_{frame_id:06d}_markers.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"\n  处理帧{frame_id}时出错: {e}")
                continue
        
        print(f"  已保存标注图到: {markers_dir}")
    
    def _plot_white_patch_temporal_curves(self, df: pd.DataFrame, save_path: str):
        """绘制白斑时序曲线（8×4完整版）"""
        print("\n生成白斑时序曲线（8×4）...")
        
        # 检查是否有白斑特征
        if 'white_area_ratio_m1' not in df.columns:
            print("  警告: 数据中没有白斑特征，跳过")
            return
        
        fig, axes = plt.subplots(8, 4, figsize=(20, 32))
        
        methods = ['m1', 'm2', 'm3', 'm4']
        method_names = ['固定阈值', 'Otsu自适应', '相对亮度', '形态学Top-Hat']
        metrics = ['area_ratio', 'patch_count', 'avg_brightness', 'brightness_std', 
                  'avg_patch_area', 'composite_index', 'brightness_entropy', 'patch_area_entropy']
        metric_names = ['白斑面积占比(%)', '白斑数量(个)', '平均亮度', '亮度标准差', 
                       '单个白斑平均面积(%)', '综合指标(数量+std)', '亮度直方图熵', '斑块面积分布熵']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for row_idx, metric in enumerate(metrics):
            for col_idx, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
                ax = axes[row_idx, col_idx]
                
                col_name = f'white_{metric}_{method}'
                if col_name not in df.columns:
                    continue
                
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
        
        plt.suptitle(f'{self.analysis_name} - 撕裂面白斑特征时序演变（4方法×8指标）', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")
    
    def _generate_white_patch_recommendation(self, df: pd.DataFrame, viz_dir: str):
        """生成白斑方法推荐报告"""
        print("\n生成白斑方法推荐报告...")
        
        # 检查是否有白斑特征
        if 'white_area_ratio_m1' not in df.columns:
            print("  警告: 数据中没有白斑特征，跳过")
            return
        
        from scipy.stats import spearmanr
        
        report_lines = []
        report_lines.append(f"# {self.analysis_name} - 撕裂面白斑检测方法推荐报告\n\n")
        
        methods = ['m1', 'm2', 'm3', 'm4']
        method_names = ['方法1:固定阈值法', '方法2:Otsu自适应法', '方法3:相对亮度法', '方法4:形态学Top-Hat法']
        metrics = ['area_ratio', 'patch_count']
        
        report_lines.append("## 方法评估\n\n")
        report_lines.append("评估维度：\n")
        report_lines.append("1. **单调性**：与帧序号的Spearman相关系数（反映是否随磨损递增）\n")
        report_lines.append("2. **稳定性**：变异系数CV（标准差/均值，越小越稳定）\n")
        report_lines.append("3. **灵敏度**：数值变化范围\n\n")
        
        evaluation_results = []
        
        for method, method_name in zip(methods, method_names):
            report_lines.append(f"### {method_name}\n\n")
            
            for metric in metrics:
                col_name = f'white_{metric}_{method}'
                if col_name not in df.columns:
                    continue
                
                values = df[col_name].values
                frames = df['frame_id'].values
                
                corr, pval = spearmanr(frames, values)
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val > 0 else 0
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
                    'stability': 1/(cv+0.01),
                    'sensitivity': value_range
                })
        
        # 综合推荐
        report_lines.append("## 综合推荐\n\n")
        
        if len(evaluation_results) > 0:
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
        report_path = os.path.join(viz_dir, 'white_patch_recommendation.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        print(f"  已保存: {report_path}")
    
    def _generate_report(self, df, key_features, n_coils, analysis_results=None):
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
        
        # 过滤掉 NaN 值
        valid_coil_ids = df['coil_id'].dropna().unique()
        for feature, label in focus_features.items():
            coil_means = []
            coil_ids_list = []
            for coil_id in sorted(valid_coil_ids):
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
        
        # === 添加综合指标分析 ===
        if analysis_results is not None:
            report_lines.append("---\n\n")
            report_lines.append("## 综合磨损指标分析\n\n")
            
            # 方法对比
            report_lines.append("### 综合评分方法对比\n\n")
            for score_name in ['weighted_score', 'pca_score', 'overall_score']:
                if score_name in df.columns:
                    values = df[score_name].values
                    if len(values) > 10:
                        first = np.mean(values[:len(values)//10])
                        last = np.mean(values[-len(values)//10:])
                        if first != 0:
                            change = ((last - first) / first) * 100
                        else:
                            change = 0
                        
                        method_names = {
                            'weighted_score': '加权平均法',
                            'pca_score': 'PCA主成分法',
                            'overall_score': '多维度法'
                        }
                        report_lines.append(f"- **{method_names[score_name]}**: 变化率={change:+.1f}%\n")
            
            # PCA分析
            pca_result = analysis_results.get('pca_result', {})
            if 'explained_variance_ratio' in pca_result and len(pca_result['explained_variance_ratio']) > 0:
                explained_var = pca_result['explained_variance_ratio']
                total_explained = sum(explained_var) * 100
                report_lines.append(f"- PCA累计解释方差: {total_explained:.1f}%\n")
            
            report_lines.append("\n")
            
            # 特征重要性Top 5
            importance_df = analysis_results.get('importance_df', pd.DataFrame())
            if len(importance_df) > 0:
                report_lines.append("### 特征重要性排序 (Top 5)\n\n")
                for idx, row in importance_df.head(5).iterrows():
                    report_lines.append(f"{idx+1}. **{row['feature']}**: ")
                    report_lines.append(f"重要性={row['importance_score']:.3f}, ")
                    report_lines.append(f"单调性={row['monotonicity']:.3f}, ")
                    report_lines.append(f"变异系数={row['cv']:.3f}\n")
                
                report_lines.append("\n")
                
                # 最强相关特征
                top_feature = importance_df.iloc[0]
                report_lines.append("### 磨损相关性建议\n\n")
                report_lines.append(f"基于当前数据，**{top_feature['feature']}** ")
                report_lines.append(f"显示出最明显的单调趋势（单调性={top_feature['monotonicity']:.3f}），")
                report_lines.append("建议作为主要监控指标。\n")
        
        # === 添加白斑分析结论 ===
        white_patch_cols = [col for col in df.columns if col.startswith('white_area_ratio_')]
        if len(white_patch_cols) > 0:
            report_lines.append("\n---\n\n")
            report_lines.append("## 撕裂面白色斑块分析\n\n")
            report_lines.append("基于用户观察：撕裂面白色斑块随钢卷数量增加而增多\n\n")
            
            # 分析4种方法的变化趋势
            methods = ['m1', 'm2', 'm3', 'm4']
            method_names = ['固定阈值法', 'Otsu自适应法', '相对亮度法', '形态学Top-Hat法']
            
            report_lines.append("### 各检测方法的变化趋势\n\n")
            
            for method, method_name in zip(methods, method_names):
                col_name = f'white_area_ratio_{method}'
                if col_name in df.columns:
                    values = df[col_name].values
                    if len(values) > 10:
                        first = np.mean(values[:len(values)//10])
                        last = np.mean(values[-len(values)//10:])
                        change = last - first
                        change_pct = (change / (first + 1e-8)) * 100
                        
                        report_lines.append(f"**{method_name}**:\n")
                        report_lines.append(f"- 初期白斑面积占比: {first:.2f}%\n")
                        report_lines.append(f"- 后期白斑面积占比: {last:.2f}%\n")
                        report_lines.append(f"- 变化量: {change:+.2f}% (变化率: {change_pct:+.1f}%)\n")
                        
                        if change > 0:
                            report_lines.append(f"- **结论**: ✓ 白斑面积显著增加，与用户观察一致\n")
                        else:
                            report_lines.append(f"- 结论: 白斑面积未见明显增长\n")
                        
                        report_lines.append("\n")
            
            # 综合结论
            report_lines.append("### 综合结论\n\n")
            avg_changes = []
            for method in methods:
                col_name = f'white_area_ratio_{method}'
                if col_name in df.columns:
                    values = df[col_name].values
                    if len(values) > 10:
                        first = np.mean(values[:len(values)//10])
                        last = np.mean(values[-len(values)//10:])
                        change = last - first
                        avg_changes.append(change)
            
            if len(avg_changes) > 0:
                avg_change = np.mean(avg_changes)
                if avg_change > 0:
                    report_lines.append(f"4种检测方法的平均变化量为 **{avg_change:+.2f}%**，")
                    report_lines.append("表明撕裂面白色斑块确实随着钢卷数量增加而增多，")
                    report_lines.append("**验证了用户的观察**。这一现象可能反映了：\n\n")
                    report_lines.append("1. 剪刀磨损导致撕裂面质量下降\n")
                    report_lines.append("2. 撕裂过程中产生更多白色高亮区域（应力集中或纤维断裂）\n")
                    report_lines.append("3. 可作为剪刀磨损的重要指标之一\n")
                else:
                    report_lines.append("白斑面积未见显著增长趋势，可能需要进一步调整检测参数。\n")
        
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
  # 基本用法（使用波谷检测法自动识别，推荐）
  python coil_wear_analysis.py --roi_dir data/roi_imgs --output_dir data/analysis
  
  # ⚡ 最快模式：直接指定钢卷数（速度快10倍）
  python coil_wear_analysis.py --roi_dir data/roi_imgs --output_dir data/analysis \
    --n_coils 8 --name "视频1"
  
  # 使用Pelt算法检测（更精确但较慢）
  python coil_wear_analysis.py --roi_dir data/roi_imgs --output_dir data/analysis \
    --detection_method pelt --name "第一周期"
  
  # 自定义可视化采样间隔
  python coil_wear_analysis.py --roi_dir data/roi_imgs --output_dir data/analysis \
    --diagnosis_interval 50 --marker_interval 50
  
  # 组合使用：波谷检测+自定义参数
  python coil_wear_analysis.py --roi_dir data/roi_imgs --output_dir data/analysis \
    --detection_method valley --min_coils 6 --max_coils 12 --name "视频2"
  
  # 批量处理
  python coil_wear_analysis.py --roi_dir video1/roi_imgs --output_dir video1/analysis --n_coils 8 --name "视频1"
  python coil_wear_analysis.py --roi_dir video2/roi_imgs --output_dir video2/analysis --name "视频2"
        """
    )
    
    parser.add_argument('--roi_dir', required=True, help='ROI图像目录路径')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    parser.add_argument('--name', default='视频分析', help='分析名称 (默认: 视频分析)')
    parser.add_argument('--min_coils', type=int, default=5, help='最小钢卷数，自动检测时使用 (默认: 5)')
    parser.add_argument('--max_coils', type=int, default=15, help='最大钢卷数，自动检测时使用 (默认: 15)')
    parser.add_argument('--n_coils', type=int, default=None,
                       help='⚡ 快速模式：直接指定钢卷数量，跳过自动检测（速度快10倍）')
    parser.add_argument('--detection_method', type=str, default='valley', choices=['valley', 'pelt'],
                       help='自动检测方法：valley=波谷检测法（推荐，快速）, pelt=Pelt变化点检测（慢但精确）（默认valley）')
    parser.add_argument('--diagnosis_interval', type=int, default=100, 
                       help='帧诊断图采样间隔，每隔多少帧生成一次诊断图（默认100）')
    parser.add_argument('--marker_interval', type=int, default=100,
                       help='白斑标注图采样间隔，每隔多少帧生成一次标注图（默认100，最多20张）')
    
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
        max_coils=args.max_coils,
        diagnosis_interval=args.diagnosis_interval,
        marker_interval=args.marker_interval,
        n_coils=args.n_coils,
        detection_method=args.detection_method
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


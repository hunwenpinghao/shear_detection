"""
特征提取模块
主要功能：
1. 计算撕裂面/剪切面比例
2. 检测和计数撕裂面中的白色斑块
3. 提取其他形状和纹理特征
4. 为时序分析准备特征数据
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import ndimage
from skimage import measure, morphology
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from config import PREPROCESS_CONFIG, VIS_CONFIG
from font_utils import setup_chinese_font


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化特征提取器
        
        Args:
            config: 配置参数
        """
        self.config = config if config is not None else PREPROCESS_CONFIG
    
    def calculate_surface_ratio(self, tear_mask: np.ndarray, 
                              shear_mask: np.ndarray) -> Dict[str, float]:
        """
        计算撕裂面和剪切面的比例
        
        Args:
            tear_mask: 撕裂面掩码
            shear_mask: 剪切面掩码
            
        Returns:
            包含各种比例指标的字典
        """
        tear_area = np.sum(tear_mask > 0)
        shear_area = np.sum(shear_mask > 0)
        total_area = tear_area + shear_area
        
        if total_area == 0:
            return {
                'tear_area': 0,
                'shear_area': 0,
                'total_area': 0,
                'tear_ratio': 0.0,
                'shear_ratio': 0.0,
                'tear_to_shear_ratio': 0.0
            }
        
        tear_ratio = tear_area / total_area
        shear_ratio = shear_area / total_area
        tear_to_shear_ratio = tear_area / shear_area if shear_area > 0 else float('inf')
        
        return {
            'tear_area': int(tear_area),
            'shear_area': int(shear_area),
            'total_area': int(total_area),
            'tear_ratio': float(tear_ratio),
            'shear_ratio': float(shear_ratio),
            'tear_to_shear_ratio': float(tear_to_shear_ratio)
        }
    
    def detect_burs(self, image: np.ndarray, mask: np.ndarray = None) -> Dict[str, Any]:
        """
        检测毛刺特征 - 使用多种方法增强毛刺可见性
        
        Args:
            image: 原始图像
            mask: 可选的掩码区域，如果不提供则检测整张图像
            
        Returns:
            毛刺检测结果
        """
        # 如果没有提供掩码，创建全图掩码
        if mask is None:
            mask = np.ones(image.shape, dtype=bool)
        
        # 只在掩码区域内检测毛刺
        if mask is not None:
            # 将布尔mask转换为uint8图像用于位运算
            mask_uint8 = (mask * 255).astype(np.uint8)
            work_image = cv2.bitwise_and(image, mask_uint8)
        else:
            work_image = image.copy()
        
        enhanced_image = work_image.copy()
        
        # 1. 形态学梯度检测 - 突出毛刺边缘
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_gradient = cv2.morphologyEx(work_image, cv2.MORPH_GRADIENT, kernel_morph)
        
        # 2. 拉普拉斯算子高频边缘检测 - 突出细小毛刺
        laplacian = cv2.Laplacian(work_image, cv2.CV_64F, ksize=3)
        laplacian = np.absolute(laplacian).astype(np.uint8)
        
        # 3. Sobel算子梯度幅度 - 毛刺通常有较强的梯度变化
        sobel_x = cv2.Sobel(work_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(work_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
        
        # 4. 多尺度高斯核差分 - 检测不同大小毛刺
        kernel_sizes = [3, 5, 7]
        multi_scale_result = np.zeros_like(work_image)
        
        for size in kernel_sizes:
            if size % 2 == 1:
                gaussian = cv2.GaussianBlur(work_image, (size, size), 0)
                multi_scale_result = np.maximum(multi_scale_result, 
                    cv2.Laplacian(gaussian, cv2.CV_64F, ksize=3))
        
        multi_scale_result = np.absolute(multi_scale_result).astype(np.uint8)
        
        # 5. 梯度阈值检测
        gradient_threshold = np.mean(sobel_magnitude) + np.std(sobel_magnitude)
        sobel_thresholded = (sobel_magnitude > gradient_threshold).astype(np.uint8) * 255
        
        # 综合毛刺检测 (多种方法加权融合)
        combined_detection = (
            0.3 * morph_gradient + 
            0.3 * laplacian + 
            0.2 * sobel_thresholded +
            0.2 * multi_scale_result
        ).astype(np.uint8)
        
        # 形态学后处理 - 连接断断续续的毛刺
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        burs_enhanced = cv2.morphologyEx(combined_detection, cv2.MORPH_CLOSE, kernel_connect)
        
        # 二值化处理
        burs_threshold = np.mean(burs_enhanced) + np.std(burs_enhanced)
        burs_binary = (burs_enhanced > burs_threshold).astype(np.uint8) * 255
        
        # 连通域分析
        labeled_burs = measure.label(burs_binary)
        burs_regions = measure.regionprops(labeled_burs)
        
        # 毛刺特征过滤 - 毛刺通常是细长、不规则的形状
        burs_features = []
        burs_areas = []
        burs_perimeters = []
        burs_elongations = []
        
        for region in burs_regions:
            area = region.area
            if area > 5:  # 最小面积过滤
                perimeter = region.perimeter
                # 计算伸长率 (毛刺通常是细长的)
                if area > 0:
                    elongation = perimeter**2 / (4 * np.pi * area)  # 等圆度度量的变形
                else:
                    elongation = 0
                
                # 只保留形状特征符合毛刺的区域
                if elongation > 5.0:  # 高伸长率表示毛刺特征
                    burs_features.append(region)
                    burs_areas.append(area)
                    burs_perimeters.append(perimeter)
                    burs_elongations.append(elongation)
        
        # 计算毛刺统计特征
        total_burs_area = sum(burs_areas)
        mask_area = np.sum(mask > 0) if mask.dtype == np.uint8 else np.sum(mask)
        burs_density = total_burs_area / mask_area if mask_area > 0 else 0
        
        # 毛刺分布均匀性
        region_centroids = [r.centroid for r in burs_features]
        if len(region_centroids) > 1:
            centroids_array = np.array(region_centroids)
            centroid_std = np.std(centroids_array, axis=0)
            distribution_uniformity = 1.0 / (1.0 + np.mean(centroid_std))
        else:
            distribution_uniformity = 0.0
        
        return {
            'burs_count': len(burs_features),
            'burs_total_area': total_burs_area,
            'burs_density': burs_density,
            'average_burs_area': np.mean(burs_areas) if burs_areas else 0,
            'burs_area_std': np.std(burs_areas) if burs_areas else 0,
            'average_elongation': np.mean(burs_elongations) if burs_elongations else 0,
            'elongation_std': np.std(burs_elongations) if burs_elongations else 0,
            'burs_distribution_uniformity': distribution_uniformity,
            'burs_enhanced_image': burs_enhanced,
            'burs_binary_mask': burs_binary,
            'morph_gradient': morph_gradient,
            'laplacian': laplacian,
            'sobel_magnitude': sobel_magnitude,
            'combined_detection': combined_detection
        }
    
    def texture_clustering_analysis(self, image: np.ndarray, method: str = 'horizontal_profiles') -> Dict[str, Any]:
        """
        基于图像纹理特征的聚类分析 - 将左右半边分离为独立簇
        
        Args:
            image: 输入图像 
            method: 聚类方法 ('horizontal_profiles', 'kmeans', 'hierarchical', 'dbscan')
            
        Returns:
            聚类检测结果
        """
        
        # 预处理：只保留白色条状区域
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 使用白色斑块检测作为预处理
        white_threshold = self.config.get('white_spot_threshold', 200)
        _, binary = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
        
        # 寻找连通域 - 整个条状区域
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'left_cluster': None, 'right_cluster': None, 'clustering_result': None}
        
        # 选择最大轮廓（条状区域）
        contour = max(contours, key=cv2.contourArea)
        
        # 定义掩码
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [contour], 255)
        
        # 计算水平投影中心的波动
        height, width = image.shape[:2]
        
        if method == 'horizontal_profiles':
            return self._horizontal_profile_clustering(image, mask)
        elif method == 'kmeans':
            return self._kmeans_spatial_clustering(image, mask)
        elif method == 'hierarchical':
            return self._hierarchical_clustering(image, mask)
        elif method == 'dbscan':
            return self._dbscan_clustering(image, mask)
        else:
            return self._horizontal_profile_clustering(image, mask)
    
    def _horizontal_profile_clustering(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        基于水平投影的聚类方法 - 找出垂直纹理两侧的分界线
        
        这个方法针对垂直条状纹理，分析每一行的加权像素中心，将波动分解为左、右两个簇
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 只在掩码区域内分析
        masked_img = cv2.bitwise_and(gray, mask)
        
        height, width = masked_img.shape
        
        # 计算每行的水平投影加权中心
        row_centers = []
        row_intensities = []
        
        for row in range(height):
            row_data = masked_img[row, :].astype(np.float32)
            row_data[row_data == 0] = 0.01  # 避免零点除
            
            # 计算加权中心（只考虑有效像素）
            weights = row_data
            if np.sum(weights) > 0:
                x_positions = np.arange(width)
                weighted_center = np.average(x_positions, weights=weights)
                row_centers.append(weighted_center)
                row_intensities.append(np.sum(weights))
            else:
                row_centers.append(width // 2)  # 默认中心
                row_intensities.append(0)
        
        row_centers = np.array(row_centers)
        row_intensities = np.array(row_intensities)
        
        # 检测中心偏移模式的周期性变化
        # 使用Savitzky-Golay滤波器平滑曲线
        from scipy.signal import savgol_filter
        
        window_length = min(31, height // 10) + 2 if height // 10 % 2 == 0 else min(31, height // 10)
        if window_length < 5:
            window_length = 5
        
        smoothed_centers = savgol_filter(row_centers, window_length, 3)
        
        # 分析波动特性 - 极值点检测
        from scipy.signal import find_peaks
        from scipy.signal import argrelmax, argrelmin
        
        # 一次和二次导数分析找拐点
        smoothed_diff1 = np.gradient(smoothed_centers)
        smoothed_diff2 = np.gradient(smoothed_diff1)
        
        # 检测二阶导数变号点作为分界点
        sign_changes = np.where(np.diff(np.sign(smoothed_diff2)))[0]
        
        # 聚类每行的归属 - 基于与行中心的偏差
        central_trend = np.median(smoothed_centers)
        left_labels = (row_centers < central_trend).astype(int)
        right_labels = 1 - left_labels
        
        # 计算聚类质量指标
        left_coherence = 1.0 / (1.0 + np.std(smoothed_centers[left_labels == 1])) if np.sum(left_labels) > 0 else 0
        right_coherence = 1.0 / (1.0 + np.std(smoothed_centers[right_labels == 1])) if np.sum(right_labels) > 0 else 0
        
        # 连接连续性检查
        left_transitions = np.sum(np.abs(np.diff(left_labels)))
        right_transitions = np.sum(np.abs(np.diff(right_labels)))
        
        clustering_stability = 1.0 / (1.0 + (left_transitions + right_transitions) / height)
        
        return {
            'cluster_separation_method': 'horizontal_profile_analysis',
            'left_cluster_coherence': left_coherence,
            'right_cluster_coherence': right_coherence,
            'clustering_stability': clustering_stability,
            'central_trend_line': central_trend,
            'separation_quality': (left_coherence + right_coherence) / 2,
            'raw_centers': row_centers.tolist(),
            'smoothed_centers': smoothed_centers.tolist(),
            'left_labels': left_labels.tolist(),
            'right_labels': right_labels.tolist(),
            'clustering_result': {
                'success': clustering_stability > 0.5,
                'separation_center': float(central_trend),
                'clusters': {
                    'left': {'labels': left_labels.tolist(), 'coherence': float(left_coherence)},
                    'right': {'labels': right_labels.tolist(), 'coherence': float(right_coherence)}
                }
            }
        }
    
    def _kmeans_spatial_clustering(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        基于空间特征的K-Means聚类
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 提取特征像素点（基于强度和梯度）
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 创建特征矩阵：位置 + 强度 + 梯度
        height, width = gray.shape
        coords = np.column_stack([np.repeat(np.arange(height), width), 
                                 np.tile(np.arange(width), height)])
        features_matrix = np.column_stack([
            coords,
            gray.flatten(),
            gradient_magnitude.flatten()
        ])
        
        # 二值化掩码展开
        mask_flat = (mask.astype(bool)).flatten()
        features_matrix = features_matrix[mask_flat]
        
        if len(features_matrix) < 2:
            return {'cluster_separation_method': 'kmeans', 'clustering_result': {'success': False}}
        
        # 标准化特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_matrix)
        
        # K-Means 聚类 (k=2)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        # 分析聚类结果
        center_0 = kmeans.cluster_centers_[0]
        center_1 = kmeans.cluster_centers_[1]
        
        # 基于x坐标区分左右聚类
        if center_0[1] < center_1[1]:  # x坐标较小者为左簇
            left_cluster = clusters == 0
            right_cluster = clusters == 1
        else:
            left_cluster = clusters == 1
            right_cluster = clusters == 0
        
        left_coherence = 1.0 / (1.0 + np.std(features_matrix[left_cluster, 1]))
        right_coherence = 1.0 / (1.0 + np.std(features_matrix[right_cluster, 1]))
        
        # 为图像像素生成行级标签以支持颜色可视化
        height, width = gray.shape
        left_labels = np.zeros(height, dtype=int)
        right_labels = np.zeros(height, dtype=int)
        
        # 重建聚类标签到图像坐标
        mask_flat_tmp = mask.flatten().astype(bool)
        original_indices = np.arange(len(mask_flat_tmp))[mask_flat_tmp]
        
        # 根据聚类结果重建每个行的标签
        for i, idx in enumerate(original_indices):
            if idx < height * width:
                row_idx = idx // width
                col_idx = idx % width
                if row_idx < height:
                    if left_cluster[i]:
                        left_labels[row_idx] = 1
                    if right_cluster[i]:
                        right_labels[row_idx] = 1
        
        return {
            'cluster_separation_method': 'kmeans',
            'left_cluster_coherence': left_coherence,
            'right_cluster_coherence': right_coherence,
            'clustering_stability': (left_coherence + right_coherence) / 2,
            'separation_quality': (left_coherence + right_coherence) / 2,
            'left_labels': left_labels.tolist(),
            'right_labels': right_labels.tolist(),
            'clustering_result': {
                'success': (left_coherence + right_coherence) / 2 > 0.3,
                'left_centroid': center_0.tolist(),
                'right_centroid': center_1.tolist(),
                'clusters': {
                    'left': {'labels': left_labels.tolist()},
                    'right': {'labels': right_labels.tolist()}
                }
            }
        }
    
    def _hierarchical_clustering(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        基于层次聚类的区域分离
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 提取显著特征点
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        
        if corners is None or len(corners) < 2:
            return {'cluster_separation_method': 'hierarchical', 'clustering_result': {'success': False}}
        
        features = corners.reshape(-1, 2)
        
        # 层次聚类
        clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
        cluster_labels = clustering.fit_predict(features)
        
        # 分析聚类分离度
        cluster_0_points = features[cluster_labels == 0]
        cluster_1_points = features[cluster_labels == 1]
        
        left_cluster_mask = cluster_labels == 0 if np.mean(cluster_0_points[:, 1]) < np.mean(cluster_1_points[:, 1]) else cluster_labels == 1
        right_cluster_mask = cluster_labels != left_cluster_mask
        
        left_coherence = 1.0 / (1.0 + np.std(features[left_cluster_mask][:, 1])) if np.sum(left_cluster_mask) > 0 else 0
        right_coherence = 1.0 / (1.0 + np.std(features[right_cluster_mask][:, 1])) if np.sum(right_cluster_mask) > 0 else 0
        
        # 为颜色可视化生成行级标签
        height, width = gray.shape
        left_labels = np.zeros(height, dtype=int)
        right_labels = np.zeros(height, dtype=int)
        
        # 映射特征点到图像行级标签
        for i, feat_point in enumerate(features):
            row_idx = int(feat_point[0])
            if 0 <= row_idx < height:
                if left_cluster_mask[i]:
                    left_labels[row_idx] = 1
                if right_cluster_mask[i]:
                    right_labels[row_idx] = 1
        
        return {
            'cluster_separation_method': 'hierarchical',
            'left_cluster_coherence': left_coherence,
            'right_cluster_coherence': right_coherence,
            'clustering_stability': (left_coherence + right_coherence) / 2,
            'separation_quality': (left_coherence + right_coherence) / 2,
            'left_labels': left_labels.tolist(),
            'right_labels': right_labels.tolist(),
            'clustering_result': {
                'success': (left_coherence + right_coherence) / 2 > 0.3
            }
        }
    
    def _dbscan_clustering(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        基于DBSCAN的密度聚类
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 提取纹理特征
        from skimage.feature import local_binary_pattern
        
        # 计算LBP特征
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        # 创建特征向量
        height, width = gray.shape
        features = np.column_stack([
            np.repeat(np.arange(height), width).astype(np.float32),
            np.tile(np.arange(width), height).astype(np.float32),
            gray.flatten().astype(np.float32),
            lbp.flatten().astype(np.float32)
        ])
        
        mask_flat = mask.astype(bool).flatten()
        filtered_features = features[mask_flat]
        
        if len(filtered_features) < 10:
            return {'cluster_separation_method': 'dbscan', 'clustering_result': {'success': False}}
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(filtered_features)
        
        # DBSCAN聚类
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        cluster_labels = dbscan.fit_predict(scaled_features)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        if n_clusters < 2:
            return {'cluster_separation_method': 'dbscan', 'clustering_result': {'success': False}}
        
        # 分析主类簇分离
        main_clusters = set(cluster_labels) - {-1}
        clusters_x_coords = {}
        for cluster_id in main_clusters:
            cluster_points = filtered_features[cluster_labels == cluster_id]
            clusters_x_coords[cluster_id] = np.mean(cluster_points[:, 1])
        
        success = True if n_clusters >= 2 else False
        
        # 生成聚类标签用于颜色可视化
        height, width = mask.shape
        left_labels = np.zeros(height, dtype=int)
        right_labels = np.zeros(height, dtype=int)
        
        if len(clusters_x_coords) >= 2:
            cluster_ids = list(main_clusters)
            if len(cluster_ids) >= 2:
                # 按x坐标排序选择左聚和右簇
                sorted_clusters = sorted(cluster_ids, key=lambda x: clusters_x_coords[x])
                left_cluster_id = sorted_clusters[0]
                right_cluster_id = sorted_clusters[1] if len(sorted_clusters) > 1 else sorted_clusters[0]
                
                # 重建标签地图到图像行坐标
                mask_flat = mask.flatten()
                for i, pixel_idx in enumerate(range(len(mask_flat))):
                    if mask_flat[i] and i < len(cluster_labels):
                        row_idx = i // width
                        if row_idx < height:
                            if cluster_labels[i] == left_cluster_id:
                                left_labels[row_idx] = 1
                            elif cluster_labels[i] == right_cluster_id:
                                right_labels[row_idx] = 1
        
        return {
            'cluster_separation_method': 'dbscan',
            'left_cluster_coherence': 1.0 if success else 0.0,
            'right_cluster_coherence': 1.0 if success else 0.0,
            'clustering_stability': 1.0 if success else 0.0,
            'separation_quality': 1.0 if success else 0.0,
            'left_labels': left_labels.tolist(),
            'right_labels': right_labels.tolist(),
            'clustering_result': {
                'success': success,
                'n_clusters': n_clusters
            }
        }
    
    def visualize_clustering_results(self, image: np.ndarray, clustering_result: Dict[str, Any], 
                                   save_path: Optional[str] = None, spot_image: np.ndarray = None) -> None:
        """
        可视化聚类结果
        增加斑块图像显示以支持正确的聚类分析
        
        Args:
            image: 原始图像
            clustering_result: 聚类结果
            save_path: 保存路径
            spot_image: 白色斑块增强图像（可选）
        """
        setup_chinese_font()
        
        # 确保中文字体正确显示
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        grid_size = (2, 3)
        
        # 显示斑块增强图像（如果提供）
        if spot_image is not None:
            axes[0, 0].imshow(image, cmap='gray', alpha=0.7)
            axes[0, 0].imshow(spot_image, cmap='Reds', alpha=0.8)
            axes[0, 0].set_title('斑块图像+聚类源')
            axes[0, 0].axis('off')
        else:
            # 如果没有斑块图像，生成白色斑块增强
            white_spots_result = self.detect_all_white_spots(image)
            spot_mask = white_spots_result.get('all_white_binary_mask', None)
            if spot_mask is not None:
                axes[0, 0].imshow(image, cmap='gray', alpha=0.7)
                axes[0, 0].imshow(spot_mask, cmap='Reds', alpha=0.8)
                axes[0, 0].set_title('检测斑块+聚类源')
                axes[0, 0].axis('off')
            else:
                axes[0, 0].imshow(image, cmap='gray')
                axes[0, 0].set_title('原始图像')
                axes[0, 0].axis('off')
        
        # 聚类概况
        method = clustering_result.get('cluster_separation_method', 'Unknown')
        left_coherence = clustering_result.get('left_cluster_coherence', 0)
        right_coherence = clustering_result.get('right_cluster_coherence', 0)
        stability = clustering_result.get('clustering_stability', 0)
        quality = clustering_result.get('separation_quality', 0)
        
        clustering_info = [
            f'聚类方法: {method}',
            f'左簇关联度: {left_coherence:.3f}',
            f'右簇关联度: {right_coherence:.3f}', 
            f'聚类稳定性: {stability:.3f}',
            f'分割质量: {quality:.3f}',
            '' if not clustering_result.get('clustering_result', {}).get('success', False) else '聚类检测成功'
        ]
        
        axes[0, 1].text(0.05, 0.92, '\n'.join(clustering_info), 
                       transform=axes[0, 1].transAxes, fontsize=10, 
                       verticalalignment='top')
        axes[0, 1].set_title('聚类算法统计')
        axes[0, 1].axis('off')
        
        # 聚类质量整体评估
        metrics = ['左簇关联度', '右簇关联度', '聚类稳定性', '分割质量']
        values = [left_coherence, right_coherence, stability, quality]
        
        x = np.arange(len(metrics))
        bars = axes[0, 2].bar(x, values, color=['red', 'blue', 'green', 'purple'], alpha=0.7)
        axes[0, 2].set_xlabel('评估指标')
        axes[0, 2].set_ylabel('数值')
        axes[0, 2].set_title('聚类质量评估')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(metrics, rotation=45)
        
        for bar, value in zip(bars, values):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
            # 数据可视化图表 - 分割边界演变
        if 'smoothed_centers' in clustering_result:
            row_centers = clustering_result['raw_centers']
            smoothed_centers = clustering_result['smoothed_centers']
            
            y = np.arange(len(row_centers))
            axes[1, 0].plot(row_centers, y, 'b-', alpha=0.7, label='原始中心点')
            axes[1, 0].plot(smoothed_centers, y, 'r-', linewidth=2, label='平滑边界')
            axes[1, 0].set_ylabel('水平行像素')
            axes[1, 0].set_xlabel('纵向中心坐标')
            axes[1, 0].set_title('分割边界演变')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # 显示斑块强度轮廓（如果有斑块信息）
            if spot_image is not None:
                spot_profile = np.mean(spot_image, axis=0)
                x = np.arange(len(spot_profile))
                axes[1, 0].plot(x, spot_profile, 'g-', linewidth=2, label='斑块强度线')
                axes[1, 0].set_ylabel('斑块强度')
                axes[1, 0].set_xlabel('纵向坐标') 
                axes[1, 0].set_title('斑块强度统计分析')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, '无边界轮廓数据', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('分割边界演变')
                axes[1, 0].axis('off')
        
        # 聚类一致性分析
        left_coherence_val = clustering_result.get('left_cluster_coherence', 0)
        right_coherence_val = clustering_result.get('right_cluster_coherence', 0)
        
        coherence_data = ['撕裂面', '剪切面']
        coherence_values = [left_coherence_val, right_coherence_val]
        
        axes[1, 1].barh(coherence_data, coherence_values, color=['red', 'blue'], alpha=0.7)
        axes[1, 1].set_xlabel('关键一致性指数')
        axes[1, 1].set_title('分离态势分布')
        axes[1, 1].set_xlim(0, 1)
        
        for i, v in enumerate(coherence_values):
            axes[1, 1].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        # 总体聚类效果
        overall_success = clustering_result.get('clustering_result', {}).get('success', False)
        overall_mark = '聚类检测成功' if overall_success else '聚类可能失败'
        
        axes[1, 2].text(0.1, 0.7, overall_mark, transform=axes[1, 2].transAxes,
                       fontsize=14, ha='center', va='center', fontweight='bold',
                       color='green' if overall_success else 'red')
        
        axes[1, 2].set_title('聚类结果总结')
        axes[1, 2].axis('off')
        
        # 新增：聚类簇颜色可视化 - 在底部左边子图显示聚类结果  
        self._render_cluster_visualization(axes[1, 0], image, spot_image, clustering_result)
        
        # 底部中间子图替换为分离斑块颜色标识
        # 替换原始'聚类质量评估'为分离斑块视图
        # self._render_separated_spots_colored(axes[1, 1], image, spot_image, clustering_result)
        # 
        # 如果你需要一个独立的额外子图，可能需要修改整体布局
        # axes[1, 1]: 原本是"分离态势分布"
        self._render_separated_spots_colored(axes[1, 1], image, spot_image, clustering_result)
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
            plt.close()
    
    def _render_cluster_visualization(self, ax, image: np.ndarray, spot_image: np.ndarray, clustering_result: Dict[str, Any]):
        """
        渲染聚类簇颜色可视化到指定轴
        
        Args:
            ax: matplotlib轴对象
            image: 原始图像
            spot_image: 斑块图像
            clustering_result: 聚类结果
        """
        if 'left_labels' in clustering_result and 'right_labels' in clustering_result:
            left_labels = clustering_result['left_labels']
            right_labels = clustering_result['right_labels']
            
            # 检查图像尺寸
            image_h, image_w = image.shape[:2]
            
            # 初始化彩色可视化图像
            cluster_vis = np.ones((image_h, image_w, 3), dtype=np.float32)
            
            if isinstance(left_labels, list):
                left_labels = np.array(left_labels)
            if isinstance(right_labels, list):
                right_labels = np.array(right_labels)
                
            if len(left_labels) > 0 and len(right_labels) > 0:
                # 确保标签数组大小匹配
                num_labels = min(len(left_labels), image_h)
                left_labels_trunc = left_labels[:num_labels]
                right_labels_trunc = right_labels[:num_labels]
                
                # 按照标签分配颜色，特别加强斑块区域的颜色标识
                for row in range(num_labels):
                    # 只对斑块区域内的像素着色
                    if row < image_h:
                        for col in range(image_w):
                            # 检查在当前斑块内且行被归类
                            if spot_image is not None and row < spot_image.shape[0] and col < spot_image.shape[1]:
                                pixel_in_spot = spot_image[row, col] > 0
                                # 对于斑块中的像素，使用高对比颜色
                                if pixel_in_spot:
                                    is_strong_spot = spot_image[row, col] > 200  # 强斑块区域
                                    if len(left_labels_trunc) > row and left_labels_trunc[row] == 1:
                                        # 左簇：斑块用亮红色系，普通用深红色
                                        cluster_vis[row, col] = [1.0, 0.3, 0.3] if is_strong_spot else [0.8, 0.2, 0.2]
                                    elif len(right_labels_trunc) > row and right_labels_trunc[row] == 1:
                                        # 右簇：斑块用亮蓝色系，普通用深蓝色
                                        cluster_vis[row, col] = [0.3, 0.3, 1.0] if is_strong_spot else [0.2, 0.2, 0.8]
                                else:
                                    # 非斑块区域的轻度着色
                                    if len(left_labels_trunc) > row and left_labels_trunc[row] == 1:
                                        cluster_vis[row, col] = [0.2, 0.1, 0.1]  # 暗红色区域
                                    elif len(right_labels_trunc) > row and right_labels_trunc[row] == 1:
                                        cluster_vis[row, col] = [0.1, 0.1, 0.2]  # 暗蓝色区域
                            else:
                                # 如果没有提供斑块图像，使用中性颜色
                                if len(left_labels_trunc) > row and left_labels_trunc[row] == 1:
                                    cluster_vis[row, col] = [0.9, 0.4, 0.4]  # 温和红色
                                elif len(right_labels_trunc) > row and right_labels_trunc[row] == 1:
                                    cluster_vis[row, col] = [0.4, 0.4, 0.9]  # 温和蓝色
            
            # 图像显示叠加重合
            ax.imshow(cluster_vis)
            ax.set_title('聚类簇颜色可视化')
            ax.axis('off')
            
            # 叠加边界线以增加可辨识度
            if 'smoothed_centers' in clustering_result and len(clustering_result['smoothed_centers']) > 0:
                smoothed_centers = np.array(clustering_result['smoothed_centers'])
                ycoords = np.arange(len(smoothed_centers))
                xcoords = smoothed_centers
                
                # 边界线位于簇分界处
                ax.plot(xcoords, ycoords, 'k-', linewidth=3, alpha=1.0, label='分界线')
            else:
                # 尝试基于聚类中心点绘制分界线
                center_x = image_w // 2
                ax.axvline(x=center_x, color='k', linewidth=2, alpha=0.8)
            
            # 添加图例
            if len(left_labels_trunc) > 0 and len(right_labels_trunc) > 0:
                has_left = np.sum(left_labels_trunc) > 0
                has_right = np.sum(right_labels_trunc) > 0
                if has_left or has_right:
                    ax.plot([], [], 'r', linewidth=5, alpha=0.8, label='左簇（撕裂区域）' if has_left else '左簇（无数据）')
                    ax.plot([], [], 'b', linewidth=5, alpha=0.8, label='右簇（剪切区域）' if has_right else '右簇（无数据）')
                    ax.legend(loc='upper right', fontsize=8)
        else:
            # 无聚类数据，显示用户名
            ax.text(0.5, 0.5, "未提供聚类标签数据", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('聚类簇颜色可视化')
            ax.axis('off')
    
    def _render_separated_spots_colored(self, ax, image: np.ndarray, spot_image: np.ndarray, clustering_result: Dict[str, Any]):
        """
        渲染分离的斑块颜色可视化，让检测到的每个斑块根据它的簇分配不同的颜色  
        
        Args:
            ax: matplotlib轴对象
            image: 原始图像
            spot_image: 斑块图像
            clustering_result: 聚类结果
        """
        if 'left_labels' in clustering_result and 'right_labels' in clustering_result:
            left_labels = clustering_result['left_labels']
            right_labels = clustering_result['right_labels']
            
            # 检查图像尺寸
            image_h, image_w = image.shape[:2]
            
            if isinstance(left_labels, list):
                left_labels = np.array(left_labels)
            if isinstance(right_labels, list):
                right_labels = np.array(right_labels)
                
            if len(left_labels) > 0 and len(right_labels) > 0 and spot_image is not None and spot_image.size > 0:
                num_labels = min(len(left_labels), image_h)
                left_labels_trunc = left_labels[:num_labels] if len(left_labels) >= num_labels else np.zeros(num_labels)
                right_labels_trunc = right_labels[:num_labels] if len(right_labels) >= num_labels else np.zeros(num_labels)
                
                # 处理斑块图像
                if spot_image.dtype != np.uint8:
                    spot_binary = (spot_image > 0).astype(np.uint8) * 255
                else:
                    spot_binary = spot_image
                
                # 连通域获取斑块
                num_labels_img, labels_img = cv2.connectedComponents(spot_binary)
                print(f"检测到 {num_labels_img - 1} 个斑块")
                
                # 创建更强对比度的彩色显示画布
                colored_canvas = np.zeros((image_h, image_w, 3), dtype=np.uint8)  # 黑色背景用来与斑块对比
                
                if num_labels_img > 1:  # 有斑块
                    total_colored_pixels = 0
                    left_count = 0
                    right_count = 0
                    other_count = 0
                    
                    # 使用中心线x进行左右分割
                    center_x = image_w // 2 
                    print(f"使用中心线x={center_x} 对{num_labels_img-1}个斑块进行左右两簇着色")
                    
                    for spot_id in range(1, num_labels_img): 
                        spot_mask = (labels_img == spot_id)
                        spot_coords = np.where(spot_mask)
                        
                        if len(spot_coords[0]) > 0:
                            # 计算斑块几何中心
                            spot_center_x = int(np.mean(spot_coords[1]))  # 列坐标
                            spot_center_y = int(np.mean(spot_coords[0]))   # 行坐标
                            
                            print(f"斑块{spot_id}: 中心位置({spot_center_x},{spot_center_y}) -> 计算左右簇...")
                            
                            # 判断斑块属于哪个簇
                            if spot_center_x < center_x:
                                # 左簇：使用鲜艳红色表示撕裂面
                                patch_color = [255, 100, 100]  # 明亮的红色
                                cluster_assignment = "左簇(撕裂面)"
                                left_count += 1
                                print(f"  -> {cluster_assignment} <- 亮红色")
                            else:
                                # 右簇：使用鲜艳蓝色表示剪切面  
                                patch_color = [100, 100, 255]  # 明亮的蓝色
                                cluster_assignment = "右簇(剪切面)"
                                right_count += 1
                                print(f"  -> {cluster_assignment} <- 亮蓝色")

                            # 对该斑块的所有像素进行着色
                            pixels_colored = 0
                            for idx in range(len(spot_coords[0])):
                                y, x = spot_coords[0][idx], spot_coords[1][idx]
                                if 0 <= y < image_h and 0 <= x < image_w:
                                    colored_canvas[y, x] = patch_color
                                    pixels_colored += 1
                            print(f"  着色完成: {pixels_colored}个像素 -> {cluster_assignment}")
                            total_colored_pixels += pixels_colored
                    
                    # 强制显示正确斑块图像
                    if total_colored_pixels > 0:
                        print(f"显示斑块图像 - 着色像素总计: {total_colored_pixels}")
                        # 确保颜色范围正确进行显示
                        final_display = colored_canvas.astype(np.uint8)  # 确保值类型正确
                        ax.imshow(final_display, interpolation='nearest')
                        
                        print(f"成功着色细节: 左簇斑块 {left_count} 个, 右簇斑块 {right_count} 个, 其他 {other_count} 个")
                        print(f"显示画布类型: {final_display.dtype}, 最小值: {np.min(final_display)}, 最大值: {np.max(final_display)}")
                    else:
                        print("无斑块着色应用于可视化，备用显示.")
                        ax.imshow(image, cmap='gray')
                        if spot_image is not None:
                            ax.imshow(spot_image, cmap='Reds', alpha=0.8)
                else:
                    # 无斑块时显示灰度斑块图 
                    ax.imshow(image, cmap='gray')
                    ax.imshow(spot_image, cmap='reds', alpha=0.7)
            else:
                # 无聚类或斑块图像
                ax.imshow(image, cmap='gray')

            ax.set_title('簇分配斑块着色')
            ax.axis('off')
            
            # 添加图例
            ax.plot([], [], 'r', linewidth=5, alpha=0.8, label='左簇斑块（撕裂面）') 
            ax.plot([], [], 'b', linewidth=5, alpha=0.8, label='右簇斑块（剪切面）') 
            ax.plot([], [], 'orange', linewidth=5, alpha=0.8, label='未分配斑块') 
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, "无聚类数据或斑块图像", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('簇分配斑块着色')
            ax.axis('off')
    
    def detect_all_white_spots(self, image: np.ndarray) -> Dict[str, Any]:
        """
        检测原始图像中的所有白色斑块（不受分割限制）
        
        Args:
            image: 原始图像
            
        Returns:
            全部白色斑块检测结果
        """
        # 白斑检测阈值
        white_threshold = self.config['white_spot_threshold']
        
        # 二值化检测白色区域（整个图像）
        _, white_binary = cv2.threshold(image, white_threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_binary = cv2.morphologyEx(white_binary, cv2.MORPH_OPEN, kernel)
        
        # 连通域分析
        labeled_spots = measure.label(white_binary)
        regions = measure.regionprops(labeled_spots)
        
        # 过滤斑块大小
        min_area = self.config['min_spot_area']
        max_area = self.config['max_spot_area']
        
        valid_spots = []
        spot_areas = []
        spot_centroids = []
        
        for region in regions:
            if min_area <= region.area <= max_area:
                valid_spots.append(region)
                spot_areas.append(region.area)
                spot_centroids.append(region.centroid)
        
        # 计算统计特征
        total_spot_area = sum(spot_areas)
        total_image_area = image.shape[0] * image.shape[1]
        spot_density = total_spot_area / total_image_area if total_image_area > 0 else 0
        
        # 计算斑块分布特征
        if len(spot_centroids) > 1:
            centroids_array = np.array(spot_centroids)
            centroid_std = np.std(centroids_array, axis=0)
            distribution_uniformity = 1.0 / (1.0 + np.mean(centroid_std))
        else:
            distribution_uniformity = 0.0
        
        return {
            'all_spot_count': len(valid_spots),
            'all_total_spot_area': total_spot_area,
            'all_spot_density': spot_density,
            'all_average_spot_size': np.mean(spot_areas) if spot_areas else 0,
            'all_spot_size_std': np.std(spot_areas) if spot_areas else 0,
            'all_distribution_uniformity': distribution_uniformity,
            'all_spot_centroids': spot_centroids,
            'all_white_binary_mask': white_binary
        }

    def detect_white_spots(self, image: np.ndarray, 
                          tear_mask: np.ndarray) -> Dict[str, Any]:
        """
        检测撕裂面中的白色斑块
        
        Args:
            image: 原始图像
            tear_mask: 撕裂面掩码
            
        Returns:
            白色斑块检测结果
        """
        # 只在撕裂面区域内检测白斑
        masked_image = cv2.bitwise_and(image, tear_mask)
        
        # 白斑检测阈值
        white_threshold = self.config['white_spot_threshold']
        
        # 二值化检测白色区域
        _, white_binary = cv2.threshold(masked_image, white_threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_binary = cv2.morphologyEx(white_binary, cv2.MORPH_OPEN, kernel)
        
        # 连通域分析
        labeled_spots = measure.label(white_binary)
        regions = measure.regionprops(labeled_spots)
        
        # 过滤斑块大小
        min_area = self.config['min_spot_area']
        max_area = self.config['max_spot_area']
        
        valid_spots = []
        spot_areas = []
        spot_centroids = []
        
        for region in regions:
            if min_area <= region.area <= max_area:
                valid_spots.append(region)
                spot_areas.append(region.area)
                spot_centroids.append(region.centroid)
        
        # 计算统计特征
        total_spot_area = sum(spot_areas)
        tear_area = np.sum(tear_mask > 0)
        spot_density = total_spot_area / tear_area if tear_area > 0 else 0
        
        # 计算斑块分布特征
        if len(spot_centroids) > 1:
            centroids_array = np.array(spot_centroids)
            centroid_std = np.std(centroids_array, axis=0)
            distribution_uniformity = 1.0 / (1.0 + np.mean(centroid_std))
        else:
            distribution_uniformity = 0.0
        
        return {
            'spot_count': len(valid_spots),
            'total_spot_area': total_spot_area,
            'spot_density': spot_density,
            'average_spot_size': np.mean(spot_areas) if spot_areas else 0,
            'spot_size_std': np.std(spot_areas) if spot_areas else 0,
            'distribution_uniformity': distribution_uniformity,
            'spot_centroids': spot_centroids,
            'white_binary_mask': white_binary
        }
    
    def calculate_edge_roughness(self, tear_mask: np.ndarray, 
                                shear_mask: np.ndarray) -> Dict[str, float]:
        """
        计算边界粗糙度特征
        
        Args:
            tear_mask: 撕裂面掩码
            shear_mask: 剪切面掩码
            
        Returns:
            边界粗糙度特征
        """
        def edge_roughness(mask):
            """计算单个掩码的边界粗糙度"""
            # 获取边界
            edges = cv2.Canny(mask, 50, 150)
            
            # 找到边界轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                return 0.0
            
            # 选择最大轮廓
            max_contour = max(contours, key=cv2.contourArea)
            
            # 计算轮廓的凸包
            hull = cv2.convexHull(max_contour)
            
            # 轮廓面积与凸包面积的比值（越小表示越粗糙）
            contour_area = cv2.contourArea(max_contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area == 0:
                return 0.0
            
            roughness = 1.0 - (contour_area / hull_area)
            
            # 计算周长比
            contour_perimeter = cv2.arcLength(max_contour, True)
            hull_perimeter = cv2.arcLength(hull, True)
            
            perimeter_ratio = contour_perimeter / hull_perimeter if hull_perimeter > 0 else 1.0
            
            return {
                'area_roughness': roughness,
                'perimeter_ratio': perimeter_ratio,
                'contour_area': contour_area,
                'contour_perimeter': contour_perimeter
            }
        
        tear_roughness = edge_roughness(tear_mask)
        shear_roughness = edge_roughness(shear_mask)
        
        return {
            'tear_area_roughness': tear_roughness.get('area_roughness', 0) if isinstance(tear_roughness, dict) else 0,
            'tear_perimeter_ratio': tear_roughness.get('perimeter_ratio', 0) if isinstance(tear_roughness, dict) else 0,
            'shear_area_roughness': shear_roughness.get('area_roughness', 0) if isinstance(shear_roughness, dict) else 0,
            'shear_perimeter_ratio': shear_roughness.get('perimeter_ratio', 0) if isinstance(shear_roughness, dict) else 0,
            'roughness_difference': abs(tear_roughness.get('area_roughness', 0) - shear_roughness.get('area_roughness', 0)) if isinstance(tear_roughness, dict) and isinstance(shear_roughness, dict) else 0
        }
    
    def calculate_texture_features(self, image: np.ndarray, 
                                 tear_mask: np.ndarray, 
                                 shear_mask: np.ndarray) -> Dict[str, float]:
        """
        计算纹理特征
        
        Args:
            image: 原始图像
            tear_mask: 撕裂面掩码
            shear_mask: 剪切面掩码
            
        Returns:
            纹理特征
        """
        def region_texture(img, mask):
            """计算区域内的纹理特征"""
            masked_img = cv2.bitwise_and(img, mask)
            
            if np.sum(mask) == 0:
                return {
                    'mean_intensity': 0,
                    'std_intensity': 0,
                    'entropy': 0,
                    'contrast': 0
                }
            
            # 提取有效像素
            valid_pixels = masked_img[mask > 0]
            
            # 基本统计特征
            mean_intensity = np.mean(valid_pixels)
            std_intensity = np.std(valid_pixels)
            
            # 计算熵
            hist, _ = np.histogram(valid_pixels, bins=256, range=(0, 256))
            hist = hist + 1e-10  # 避免log(0)
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob))
            
            # 计算对比度（基于灰度共生矩阵的简化版本）
            # 使用局部标准差作为对比度的近似
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(masked_img.astype(np.float32), -1, kernel)
            local_sq_mean = cv2.filter2D((masked_img.astype(np.float32))**2, -1, kernel)
            local_variance = local_sq_mean - local_mean**2
            contrast = np.mean(local_variance[mask > 0])
            
            return {
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'entropy': float(entropy),
                'contrast': float(contrast)
            }
        
        tear_texture = region_texture(image, tear_mask)
        shear_texture = region_texture(image, shear_mask)
        
        # 组合特征
        features = {}
        for key in tear_texture.keys():
            features[f'tear_{key}'] = tear_texture[key]
            features[f'shear_{key}'] = shear_texture[key]
            features[f'{key}_difference'] = abs(tear_texture[key] - shear_texture[key])
        
        return features
    
    def extract_all_features(self, image: np.ndarray, 
                           tear_mask: np.ndarray, 
                           shear_mask: np.ndarray) -> Dict[str, Any]:
        """
        提取所有特征
        
        Args:
            image: 原始图像
            tear_mask: 撕裂面掩码
            shear_mask: 剪切面掩码
            
        Returns:
            完整的特征字典
        """
        features = {}
        
        # 1. 表面比例特征
        ratio_features = self.calculate_surface_ratio(tear_mask, shear_mask)
        features.update(ratio_features)
        
        # 2. 整体白色斑块特征（新增）
        all_spot_features = self.detect_all_white_spots(image)
        # 只保留数值特征，不包含图像数据
        numeric_all_spot_features = {k: v for k, v in all_spot_features.items() 
                                   if k not in ['all_spot_centroids', 'all_white_binary_mask']}
        features.update(numeric_all_spot_features)
        
        # 3. 撕裂面白色斑块特征
        spot_features = self.detect_white_spots(image, tear_mask)
        # 只保留数值特征，不包含图像数据
        numeric_spot_features = {k: v for k, v in spot_features.items() 
                               if k not in ['spot_centroids', 'white_binary_mask']}
        features.update(numeric_spot_features)
        
        # 4. 毛刺检测特征（新增）
        burs_tear_features = self.detect_burs(image, tear_mask)
        # 只保留数值特征
        numeric_burs_tear_features = {k: v for k, v in burs_tear_features.items() 
                                   if k not in ['burs_enhanced_image', 'burs_binary_mask', 
                                              'morph_gradient', 'laplacian', 'sobel_magnitude', 'combined_detection']}
        # 为撕裂面毛刺特征添加前缀
        tear_burs_features = {f'tear_burs_{k}': v for k, v in numeric_burs_tear_features.items()}
        features.update(tear_burs_features)
        
        # 检测剪切面毛刺特征
        burs_shear_features = self.detect_burs(image, shear_mask)
        numeric_burs_shear_features = {k: v for k, v in burs_shear_features.items() 
                                     if k not in ['burs_enhanced_image', 'burs_binary_mask',
                                                'morph_gradient', 'laplacian', 'sobel_magnitude', 'combined_detection']}
        shear_burs_features = {f'shear_burs_{k}': v for k, v in numeric_burs_shear_features.items()}
        features.update(shear_burs_features)
        
        # 5. 纹理聚类分析（新增）
        clustering_result = self.texture_clustering_analysis(image, method='horizontal_profiles')
        # 只保留数值特征，不包含可视化数据
        numeric_clustering_features = {k: v for k, v in clustering_result.items() 
                                     if k not in ['raw_centers', 'smoothed_centers', 'left_labels', 'right_labels', 'clustering_result']}
        clustering_features = {f'clustering_{k}': v for k, v in numeric_clustering_features.items()}
        features.update(clustering_features)
        
        # 6. 边界粗糙度特征
        roughness_features = self.calculate_edge_roughness(tear_mask, shear_mask)
        features.update(roughness_features)
        
        # 7. 纹理特征
        texture_features = self.calculate_texture_features(image, tear_mask, shear_mask)
        features.update(texture_features)
        
        # 8. 添加时间戳（用于时序分析）
        import time
        features['timestamp'] = time.time()
        
        return features
    
    def visualize_features(self, image: np.ndarray, 
                          tear_mask: np.ndarray, 
                          shear_mask: np.ndarray,
                          features: Dict[str, Any],
                          save_path: Optional[str] = None):
        """
        可视化特征提取结果
        
        Args:
            image: 原始图像
            tear_mask: 撕裂面掩码
            shear_mask: 剪切面掩码
            features: 提取的特征
            save_path: 保存路径
        """
        # 设置中文字体
        setup_chinese_font()
        
        # 检测白斑用于可视化
        spot_info = self.detect_white_spots(image, tear_mask)
        white_spots = spot_info['white_binary_mask']
        
        # 检测所有白色斑块（未分割前）
        all_spot_info = self.detect_all_white_spots(image)
        all_white_spots = all_spot_info['all_white_binary_mask']
        
        # 检测毛刺特征（新增） - 修改为不区分撕裂面和剪切面
        # 创建整体掩码（撕裂面+剪切面）
        whole_mask = np.maximum(tear_mask, shear_mask)
        burs_whole_info = self.detect_burs(image, whole_mask)
        burs_whole_enhanced = burs_whole_info['burs_enhanced_image']
        burs_whole_binary = burs_whole_info['burs_binary_mask']
        
        # 保留原有的分别检测（向后兼容，但不用于可视化）
        burs_tear_info = self.detect_burs(image, tear_mask)
        burs_tear_enhanced = burs_tear_info['burs_enhanced_image']
        burs_tear_binary = burs_tear_info['burs_binary_mask']
        burs_shear_info = self.detect_burs(image, shear_mask)
        burs_shear_enhanced = burs_shear_info['burs_enhanced_image']
        burs_shear_binary = burs_shear_info['burs_binary_mask']
        
        # 改为3行4列布局，增加毛刺检测图像
        fig, axes = plt.subplots(3, 4, figsize=(32, 18))
        
        # 第一行图像
        # 原始图像
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        # 分割结果
        overlay = np.zeros((image.shape[0], image.shape[1], 3))
        overlay[:, :, 0] = tear_mask / 255.0
        overlay[:, :, 2] = shear_mask / 255.0
        
        axes[0, 1].imshow(image, cmap='gray', alpha=0.7)
        axes[0, 1].imshow(overlay, alpha=0.5)
        axes[0, 1].set_title(f'分割结果\n撕裂面比例: {features["tear_ratio"]:.3f}')
        axes[0, 1].axis('off')
        
        # 新增：分割前的白色斑块（整体）
        axes[0, 2].imshow(image, cmap='gray', alpha=0.7)
        axes[0, 2].imshow(all_white_spots, cmap='Reds', alpha=0.8)
        all_spot_count = all_spot_info['all_spot_count']
        axes[0, 2].set_title(f'分割前整体白色斑块\n数量: {all_spot_count}')
        axes[0, 2].axis('off')
        
        # 撕裂面白色斑块
        axes[0, 3].imshow(image, cmap='gray', alpha=0.7)
        axes[0, 3].imshow(white_spots, cmap='Reds', alpha=0.8)
        axes[0, 3].set_title(f'撕裂面白色斑块\n数量: {features["spot_count"]}')
        axes[0, 3].axis('off')
        
        # 第二行图像（统一毛刺检测 - 不区分撕裂面和剪切面）
        # 整体毛刺增强图像
        axes[1, 0].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 0].imshow(burs_whole_enhanced, cmap='hot', alpha=0.8)
        whole_burs_count = burs_whole_info['burs_count']
        axes[1, 0].set_title(f'整体毛刺增强\n数量: {whole_burs_count}')
        axes[1, 0].axis('off')
        
        # 整体毛刺二值图
        axes[1, 1].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(burs_whole_binary, cmap='Blues', alpha=0.8)
        axes[1, 1].set_title(f'整体毛刺检测\n密度: {burs_whole_info.get("burs_density", 0):.3f}')
        axes[1, 1].axis('off')
        
        # 整体毛刺层次化检测图（利用梯度细节显示）
        axes[1, 2].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 2].imshow(burs_whole_info.get('combined_detection', burs_whole_binary), 
                         cmap='hot', alpha=0.8)
        axes[1, 2].set_title(f'毛刺层次检测\n总体积: {burs_whole_info.get("burs_total_area", 0)}')
        axes[1, 2].axis('off')
        
        # 毛刺形态学研究图（梯度+拉普拉斯显示）
        morph_result = np.maximum(
            burs_whole_info.get('morph_gradient', np.zeros_like(image)),
            burs_whole_info.get('laplacian', np.zeros_like(image))
        )
        axes[1, 3].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 3].imshow(morph_result, cmap='viridis', alpha=0.8)
        axes[1, 3].set_title(f'毛刺形态学分析\n标准差: {burs_whole_info.get("elongation_std", 0):.3f}')
        axes[1, 3].axis('off')
        
        # 第三行特征统计图表
        # 比例特征
        ratios = [features['tear_ratio'], features['shear_ratio']]
        labels = ['撕裂面', '剪切面']
        colors = ['red', 'blue']
        
        axes[2, 0].pie(ratios, labels=labels, colors=colors, autopct='%1.2f%%')
        axes[2, 0].set_title('表面比例分布')
        
        # 纹理特征对比
        feature_names = ['平均亮度', '标准差', '熵', '对比度']
        tear_values = [
            features.get('tear_mean_intensity', 0),
            features.get('tear_std_intensity', 0),
            features.get('tear_entropy', 0),
            features.get('tear_contrast', 0)
        ]
        shear_values = [
            features.get('shear_mean_intensity', 0),
            features.get('shear_std_intensity', 0),
            features.get('shear_entropy', 0),
            features.get('shear_contrast', 0)
        ]
        
        x = np.arange(len(feature_names))
        width = 0.35
        
        axes[2, 1].bar(x - width/2, tear_values, width, label='撕裂面', color='red', alpha=0.7)
        axes[2, 1].bar(x + width/2, shear_values, width, label='剪切面', color='blue', alpha=0.7)
        axes[2, 1].set_xlabel('特征类型')
        axes[2, 1].set_ylabel('特征值')
        axes[2, 1].set_title('纹理特征对比')
        axes[2, 1].set_xticks(x)
        axes[2, 1].set_xticklabels(feature_names, rotation=45)
        axes[2, 1].legend()
        
        # 整体毛刺特征分析（不区分撕裂面和剪切面）
        burs_values = ['毛刺数量', '毛刺密度', '平均伸长率', '毛刺总面积', '分布均匀性']
        whole_burs_values = [
            burs_whole_info.get('burs_count', 0),
            burs_whole_info.get('burs_density', 0),
            burs_whole_info.get('average_elongation', 0),
            burs_whole_info.get('burs_total_area', 0),
            burs_whole_info.get('burs_distribution_uniformity', 0)
        ]
        
        x_burs = np.arange(len(burs_values))
        
        axes[2, 2].bar(x_burs, whole_burs_values, color='green', alpha=0.7)
        axes[2, 2].set_xlabel('毛刺特征')
        axes[2, 2].set_ylabel('数值')
        axes[2, 2].set_title('整体毛刺特征分析')
        axes[2, 2].set_xticks(x_burs)
        axes[2, 2].set_xticklabels(burs_values, rotation=45)
        
        # 添加特征值标注
        for i, v in enumerate(whole_burs_values):
            axes[2, 2].text(i, v + max(whole_burs_values) * 0.01, f'{v:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        # 关键指标文本显示
        key_metrics = [
            f"撕裂面/剪切面比值: {features.get('tear_to_shear_ratio', 0):.3f}",
            f"撕裂面白斑密度: {features.get('spot_density', 0):.3f}",
            f"整体白斑密度: {all_spot_info.get('all_spot_density', 0):.3f}",
            f"整体毛刺数量: {burs_whole_info.get('burs_count', 0)}个",
            f"整体毛刺密度: {burs_whole_info.get('burs_density', 0):.3f}",
            f"平均毛刺面积: {burs_whole_info.get('average_burs_area', 0):.1f}",
            f"毛刺分布均匀性: {burs_whole_info.get('burs_distribution_uniformity', 0):.3f}",
            f"平均毛刺伸长率: {burs_whole_info.get('average_elongation', 0):.2f}"
        ]
        
        axes[2, 3].text(0.1, 0.9, '\n'.join(key_metrics), 
                       transform=axes[2, 3].transAxes, fontsize=9,
                       verticalalignment='top')
        axes[2, 3].set_title('检测统计')
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
            plt.close()


def test_feature_extraction():
    """测试特征提取功能"""
    import os
    from preprocessor import ImagePreprocessor
    from segmentation import SurfaceSegmentator
    from config import DATA_DIR, OUTPUT_DIR
    
    # 初始化
    preprocessor = ImagePreprocessor()
    segmentator = SurfaceSegmentator()
    extractor = FeatureExtractor()
    
    # 测试图像路径
    test_image = os.path.join(DATA_DIR, 'Image_20250710125452500.bmp')
    
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
    try:
        # 完整流水线测试
        print("开始特征提取测试...")
        
        # 预处理
        roi_image, _ = preprocessor.preprocess_pipeline(test_image, target_size=(128, 512))
        print("预处理完成")
        
        # 分割
        tear_mask, shear_mask, seg_info = segmentator.segment_surface(roi_image, method='hybrid')
        print("分割完成")
        
        # 特征提取
        features = extractor.extract_all_features(roi_image, tear_mask, shear_mask)
        print("特征提取完成")
        
        # 打印主要特征
        print("\n=== 主要特征 ===")
        print(f"撕裂面比例: {features['tear_ratio']:.3f}")
        print(f"剪切面比例: {features['shear_ratio']:.3f}")
        print(f"撕裂面/剪切面比值: {features['tear_to_shear_ratio']:.3f}")
        print(f"白斑数量: {features['spot_count']}")
        print(f"白斑密度: {features['spot_density']:.3f}")
        print(f"平均白斑大小: {features['average_spot_size']:.1f}")
        print(f"撕裂面粗糙度: {features['tear_area_roughness']:.3f}")
        print(f"剪切面粗糙度: {features['shear_area_roughness']:.3f}")
        
        # 可视化
        vis_path = os.path.join(OUTPUT_DIR, 'feature_extraction_result.png')
        extractor.visualize_features(roi_image, tear_mask, shear_mask, features, vis_path)
        print(f"可视化结果已保存到: {vis_path}")
        
        # 保存特征到JSON文件
        import json
        features_path = os.path.join(OUTPUT_DIR, 'extracted_features.json')
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=2, ensure_ascii=False)
        print(f"特征数据已保存到: {features_path}")
        
    except Exception as e:
        print(f"特征提取测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_feature_extraction()

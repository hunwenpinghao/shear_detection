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
        
        # 4. 边界粗糙度特征
        roughness_features = self.calculate_edge_roughness(tear_mask, shear_mask)
        features.update(roughness_features)
        
        # 5. 纹理特征
        texture_features = self.calculate_texture_features(image, tear_mask, shear_mask)
        features.update(texture_features)
        
        # 6. 添加时间戳（用于时序分析）
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
        
        # 改为2行4列布局，增加整体白斑检测图像
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        
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
        
        # 第二行特征统计图表
        # 比例特征
        ratios = [features['tear_ratio'], features['shear_ratio']]
        labels = ['撕裂面', '剪切面']
        colors = ['red', 'blue']
        
        axes[1, 0].pie(ratios, labels=labels, colors=colors, autopct='%1.2f%%')
        axes[1, 0].set_title('表面比例分布')
        
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
        
        axes[1, 1].bar(x - width/2, tear_values, width, label='撕裂面', color='red', alpha=0.7)
        axes[1, 1].bar(x + width/2, shear_values, width, label='剪切面', color='blue', alpha=0.7)
        axes[1, 1].set_xlabel('特征类型')
        axes[1, 1].set_ylabel('特征值')
        axes[1, 1].set_title('纹理特征对比')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(feature_names, rotation=45)
        axes[1, 1].legend()
        
        # 关键指标文本显示
        key_metrics = [
            f"撕裂面/剪切面比值: {features.get('tear_to_shear_ratio', 0):.3f}",
            f"撕裂面白斑密度: {features.get('spot_density', 0):.3f}",
            f"整体白斑密度: {all_spot_info.get('all_spot_density', 0):.3f}",
            f"平均白斑大小: {features.get('average_spot_size', 0):.1f}",
            f"撕裂面粗糙度: {features.get('tear_area_roughness', 0):.3f}",
            f"剪切面粗糙度: {features.get('shear_area_roughness', 0):.3f}",
            f"粗糙度差异: {features.get('roughness_difference', 0):.3f}"
        ]
        
        axes[1, 2].text(0.1, 0.9, '\n'.join(key_metrics), 
                       transform=axes[1, 2].transAxes, fontsize=10,
                       verticalalignment='top')
        axes[1, 2].set_title('关键指标')
        axes[1, 2].axis('off')
        
        # 第三列位置暂时为空或放置辅助信息
        axes[1, 3].axis('off')
        axes[1, 3].text(0.5, 0.5, f'整体检测白斑\n数量: {all_spot_count}', 
                       transform=axes[1, 3].transAxes, fontsize=12,
                       ha='center', va='center')
        axes[1, 3].set_title('统计总结')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
            plt.close()
        else:
            plt.show()


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

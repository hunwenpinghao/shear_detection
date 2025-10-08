"""
几何特征提取模块
实现10个核心磨损指标的计算（原5个 + 新增5个）
"""
import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy import stats
from skimage.feature import graycomatrix, graycoprops
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class GeometryFeatureExtractor:
    """几何特征提取器"""
    
    def __init__(self):
        """初始化特征提取器"""
        pass
    
    def compute_rms_roughness(self, edge_positions: np.ndarray) -> float:
        """
        计算边界RMS粗糙度
        
        Args:
            edge_positions: 边界位置序列
            
        Returns:
            RMS粗糙度值
        """
        if len(edge_positions) == 0:
            return 0.0
        
        mean_pos = np.mean(edge_positions)
        rms = np.sqrt(np.mean((edge_positions - mean_pos) ** 2))
        
        return float(rms)
    
    def compute_gradient_energy(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        计算梯度能量（Tenengrad）- 衡量锐度
        
        Args:
            image: 输入灰度图像
            mask: ROI掩码
            
        Returns:
            梯度能量值
        """
        # 计算Sobel梯度
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 梯度幅值的平方
        grad_magnitude_sq = grad_x ** 2 + grad_y ** 2
        
        # 在掩码区域内计算平均能量
        mask_bool = mask > 0
        if not mask_bool.any():
            return 0.0
        
        energy = np.mean(grad_magnitude_sq[mask_bool])
        
        return float(energy)
    
    def compute_horizontal_gradient_energy(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        计算水平方向梯度能量 - 只使用x方向梯度
        
        水平梯度主要反映垂直边缘（如刀口边缘）的锐度，
        对剪刀磨损导致的边缘钝化更敏感
        
        Args:
            image: 输入灰度图像
            mask: ROI掩码
            
        Returns:
            水平梯度能量值
        """
        # 只计算水平方向（x方向）的Sobel梯度
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        
        # 梯度的平方
        grad_x_sq = grad_x ** 2
        
        # 在掩码区域内计算平均能量
        mask_bool = mask > 0
        if not mask_bool.any():
            return 0.0
        
        energy = np.mean(grad_x_sq[mask_bool])
        
        return float(energy)
    
    def compute_max_notch_depth(self, edge_positions: np.ndarray, poly_degree: int = 3) -> float:
        """
        计算最大缺口深度
        通过多项式拟合找到边界的整体趋势，计算实际边界与拟合线的最大负偏差
        
        Args:
            edge_positions: 边界位置序列
            poly_degree: 多项式拟合阶数
            
        Returns:
            最大缺口深度（负值的绝对值）
        """
        if len(edge_positions) < poly_degree + 1:
            return 0.0
        
        # 多项式拟合
        x = np.arange(len(edge_positions))
        
        try:
            coeffs = np.polyfit(x, edge_positions, poly_degree)
            fitted = np.polyval(coeffs, x)
            
            # 计算残差
            residuals = edge_positions - fitted
            
            # 找最大负偏差（凹陷）
            negative_residuals = residuals[residuals < 0]
            
            if len(negative_residuals) > 0:
                max_notch = abs(np.min(negative_residuals))
            else:
                max_notch = 0.0
                
        except np.linalg.LinAlgError:
            max_notch = 0.0
        
        return float(max_notch)
    
    def compute_peak_statistics(self, edge_positions: np.ndarray, 
                               prominence: float = 1.0) -> Dict[str, float]:
        """
        计算峰统计特征
        
        Args:
            edge_positions: 边界位置序列
            prominence: 峰的显著性阈值
            
        Returns:
            包含峰统计的字典: peak_count, avg_peak_height, avg_peak_distance
        """
        if len(edge_positions) < 3:
            return {
                'peak_count': 0,
                'avg_peak_height': 0.0,
                'avg_peak_distance': 0.0,
                'peak_density': 0.0
            }
        
        # 检测峰值
        peaks, properties = find_peaks(edge_positions, prominence=prominence)
        
        peak_count = len(peaks)
        
        if peak_count > 0:
            avg_peak_height = float(np.mean(properties['prominences']))
            
            # 计算峰间距
            if peak_count > 1:
                peak_distances = np.diff(peaks)
                avg_peak_distance = float(np.mean(peak_distances))
            else:
                avg_peak_distance = 0.0
            
            # 峰密度：每单位长度的峰数量
            peak_density = float(peak_count / len(edge_positions))
        else:
            avg_peak_height = 0.0
            avg_peak_distance = 0.0
            peak_density = 0.0
        
        return {
            'peak_count': peak_count,
            'avg_peak_height': avg_peak_height,
            'avg_peak_distance': avg_peak_distance,
            'peak_density': peak_density
        }
    
    def compute_area_ratio(self, left_mask: np.ndarray, right_mask: np.ndarray) -> float:
        """
        计算撕裂区与剪切区的面积比
        
        Args:
            left_mask: 左侧（撕裂面）掩码
            right_mask: 右侧（剪切面）掩码
            
        Returns:
            面积比（撕裂区面积 / 剪切区面积）
        """
        left_area = np.sum(left_mask > 0)
        right_area = np.sum(right_mask > 0)
        
        if right_area == 0:
            return 0.0
        
        ratio = float(left_area / right_area)
        
        return ratio
    
    def compute_waviness(self, edge_positions: np.ndarray, cutoff_freq: float = 0.1) -> Dict[str, float]:
        """
        计算边界波纹幅度（低频分量）
        
        通过低通滤波分离长周期波动，量化刀口的整体弯曲程度
        
        Args:
            edge_positions: 边界位置序列
            cutoff_freq: 截止频率（相对于Nyquist频率）
            
        Returns:
            波纹特征字典
        """
        if len(edge_positions) < 10:
            return {
                'waviness_amplitude': 0.0,
                'waviness_wavelength': 0.0
            }
        
        try:
            # 使用savgol_filter进行平滑（低通）
            window_length = min(51, len(edge_positions) // 4 * 2 + 1)  # 确保是奇数
            if window_length < 5:
                window_length = 5
            
            low_freq = savgol_filter(edge_positions, window_length=window_length, polyorder=3)
            
            # 波纹振幅 = 低频分量的标准差
            waviness_amplitude = float(np.std(low_freq))
            
            # 使用FFT找主频，计算波长
            fft_vals = np.fft.fft(low_freq - np.mean(low_freq))
            power = np.abs(fft_vals) ** 2
            freqs = np.fft.fftfreq(len(low_freq))
            
            # 只看正频率
            positive_freqs = freqs[:len(freqs)//2]
            positive_power = power[:len(power)//2]
            
            if len(positive_power) > 1 and np.max(positive_power[1:]) > 0:
                # 找到最大功率对应的频率（排除DC分量）
                dominant_freq_idx = np.argmax(positive_power[1:]) + 1
                dominant_freq = positive_freqs[dominant_freq_idx]
                
                if dominant_freq > 0:
                    waviness_wavelength = float(1.0 / dominant_freq)
                else:
                    waviness_wavelength = 0.0
            else:
                waviness_wavelength = 0.0
                
        except Exception as e:
            waviness_amplitude = 0.0
            waviness_wavelength = 0.0
        
        return {
            'waviness_amplitude': waviness_amplitude,
            'waviness_wavelength': waviness_wavelength
        }
    
    def compute_boundary_trend(self, edge_positions: np.ndarray) -> Dict[str, float]:
        """
        计算边界整体偏移趋势
        
        通过线性拟合获得一阶系数，反映刀口的整体磨损方向
        
        Args:
            edge_positions: 边界位置序列
            
        Returns:
            趋势特征字典
        """
        if len(edge_positions) < 3:
            return {
                'trend_slope': 0.0,
                'trend_intercept': 0.0,
                'trend_r2': 0.0
            }
        
        try:
            x = np.arange(len(edge_positions))
            
            # 线性拟合
            slope, intercept = np.polyfit(x, edge_positions, 1)
            
            # 计算R²
            y_pred = slope * x + intercept
            ss_res = np.sum((edge_positions - y_pred) ** 2)
            ss_tot = np.sum((edge_positions - np.mean(edge_positions)) ** 2)
            
            if ss_tot > 0:
                r2 = 1 - (ss_res / ss_tot)
            else:
                r2 = 0.0
                
        except Exception as e:
            slope = 0.0
            intercept = 0.0
            r2 = 0.0
        
        return {
            'trend_slope': float(slope),
            'trend_intercept': float(intercept),
            'trend_r2': float(r2)
        }
    
    def compute_texture_contrast(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        计算GLCM灰度共生矩阵纹理特征
        
        磨损增加时，表面纹理更不规则，contrast和entropy升高
        
        Args:
            image: 输入灰度图像
            mask: ROI掩码
            
        Returns:
            纹理特征字典
        """
        if mask is None or not mask.any():
            return {
                'glcm_contrast': 0.0,
                'glcm_homogeneity': 0.0,
                'glcm_energy': 0.0,
                'glcm_correlation': 0.0
            }
        
        try:
            # 提取掩码区域
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) < 10:
                return {
                    'glcm_contrast': 0.0,
                    'glcm_homogeneity': 0.0,
                    'glcm_energy': 0.0,
                    'glcm_correlation': 0.0
                }
            
            # 创建ROI的bounding box
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            roi = image[y_min:y_max+1, x_min:x_max+1].copy()
            roi_mask = mask[y_min:y_max+1, x_min:x_max+1]
            
            # 将非掩码区域设为0
            roi[roi_mask == 0] = 0
            
            # 归一化到0-255范围（GLCM需要）
            if roi.max() > roi.min():
                roi_norm = ((roi - roi.min()) / (roi.max() - roi.min()) * 255).astype(np.uint8)
            else:
                roi_norm = roi.astype(np.uint8)
            
            # 计算GLCM（4个方向：0°, 45°, 90°, 135°）
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            glcm = graycomatrix(roi_norm, distances=distances, angles=angles, 
                               levels=256, symmetric=True, normed=True)
            
            # 计算4个方向的平均特征
            contrast = float(np.mean(graycoprops(glcm, 'contrast')))
            homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
            energy = float(np.mean(graycoprops(glcm, 'energy')))
            correlation = float(np.mean(graycoprops(glcm, 'correlation')))
            
        except Exception as e:
            contrast = 0.0
            homogeneity = 0.0
            energy = 0.0
            correlation = 0.0
        
        return {
            'glcm_contrast': contrast,
            'glcm_homogeneity': homogeneity,
            'glcm_energy': energy,
            'glcm_correlation': correlation
        }
    
    def compute_peak_skewness(self, edge_positions: np.ndarray) -> Dict[str, float]:
        """
        计算峰高分布的偏度和峰度
        
        偏度反映不对称磨损，峰度反映极端缺口的存在
        
        Args:
            edge_positions: 边界位置序列
            
        Returns:
            统计分布特征字典
        """
        if len(edge_positions) < 3:
            return {
                'peak_skewness': 0.0,
                'peak_kurtosis': 0.0,
                'peak_std': 0.0
            }
        
        try:
            # 去趋势（去除线性成分）
            x = np.arange(len(edge_positions))
            slope, intercept = np.polyfit(x, edge_positions, 1)
            trend = slope * x + intercept
            detrended = edge_positions - trend
            
            # 计算统计量
            skewness = float(stats.skew(detrended))
            kurtosis = float(stats.kurtosis(detrended))
            std = float(np.std(detrended))
            
            # 处理可能的nan值
            if np.isnan(skewness):
                skewness = 0.0
            if np.isnan(kurtosis):
                kurtosis = 0.0
                
        except Exception as e:
            skewness = 0.0
            kurtosis = 0.0
            std = 0.0
        
        return {
            'peak_skewness': skewness,
            'peak_kurtosis': kurtosis,
            'peak_std': std
        }
    
    def compute_frequency_features(self, edge_positions: np.ndarray) -> Dict[str, float]:
        """
        计算频域特征
        
        高频能量反映细小波动，刀口钝化时高频下降
        
        Args:
            edge_positions: 边界位置序列
            
        Returns:
            频域特征字典
        """
        if len(edge_positions) < 10:
            return {
                'high_freq_ratio': 0.0,
                'dominant_freq': 0.0,
                'spectral_centroid': 0.0
            }
        
        try:
            # 去均值
            signal = edge_positions - np.mean(edge_positions)
            
            # FFT变换
            fft_vals = np.fft.fft(signal)
            power = np.abs(fft_vals) ** 2
            freqs = np.fft.fftfreq(len(signal))
            
            # 只看正频率
            positive_freqs = freqs[:len(freqs)//2]
            positive_power = power[:len(power)//2]
            
            if len(positive_power) > 1:
                # 总能量
                total_energy = np.sum(positive_power)
                
                if total_energy > 0:
                    # 高频能量占比（>0.3 Nyquist频率）
                    high_freq_threshold = 0.3
                    high_freq_mask = positive_freqs >= high_freq_threshold
                    high_freq_energy = np.sum(positive_power[high_freq_mask])
                    high_freq_ratio = float(high_freq_energy / total_energy)
                    
                    # 主频率（最大功率对应的频率，排除DC）
                    if len(positive_power) > 1:
                        dominant_freq_idx = np.argmax(positive_power[1:]) + 1
                        dominant_freq = float(positive_freqs[dominant_freq_idx])
                    else:
                        dominant_freq = 0.0
                    
                    # 谱质心
                    spectral_centroid = float(np.sum(positive_freqs * positive_power) / total_energy)
                else:
                    high_freq_ratio = 0.0
                    dominant_freq = 0.0
                    spectral_centroid = 0.0
            else:
                high_freq_ratio = 0.0
                dominant_freq = 0.0
                spectral_centroid = 0.0
                
        except Exception as e:
            high_freq_ratio = 0.0
            dominant_freq = 0.0
            spectral_centroid = 0.0
        
        return {
            'high_freq_ratio': high_freq_ratio,
            'dominant_freq': dominant_freq,
            'spectral_centroid': spectral_centroid
        }
    
    def extract_features(self, preprocessed_data: dict) -> Dict[str, float]:
        """
        提取所有几何特征
        
        Args:
            preprocessed_data: 预处理结果字典
            
        Returns:
            包含所有特征的字典
        """
        if not preprocessed_data.get('success', False):
            return self._get_empty_features()
        
        # 获取预处理数据
        image = preprocessed_data['denoised']
        left_mask = preprocessed_data['left_mask']
        right_mask = preprocessed_data['right_mask']
        left_edges = preprocessed_data['left_edges']
        right_edges = preprocessed_data['right_edges']
        
        # 1. 边界RMS粗糙度（左右分别计算）
        left_rms = self.compute_rms_roughness(left_edges)
        right_rms = self.compute_rms_roughness(right_edges)
        
        # 2. 梯度能量（左右分别计算）
        left_gradient_energy = self.compute_gradient_energy(image, left_mask)
        right_gradient_energy = self.compute_gradient_energy(image, right_mask)
        
        # 2.1 水平梯度能量（左右分别计算）
        left_horizontal_gradient = self.compute_horizontal_gradient_energy(image, left_mask)
        right_horizontal_gradient = self.compute_horizontal_gradient_energy(image, right_mask)
        
        # 3. 最大缺口深度（左右分别计算）
        left_notch = self.compute_max_notch_depth(left_edges)
        right_notch = self.compute_max_notch_depth(right_edges)
        
        # 4. 峰统计（左右分别计算）
        left_peak_stats = self.compute_peak_statistics(left_edges)
        right_peak_stats = self.compute_peak_statistics(right_edges)
        
        # 5. 面积比
        area_ratio = self.compute_area_ratio(left_mask, right_mask)
        
        # === 新增特征 ===
        # 6. 波纹幅度（左右分别计算）
        left_waviness = self.compute_waviness(left_edges)
        right_waviness = self.compute_waviness(right_edges)
        
        # 7. 边界偏移趋势（左右分别计算）
        left_trend = self.compute_boundary_trend(left_edges)
        right_trend = self.compute_boundary_trend(right_edges)
        
        # 8. GLCM纹理特征（左右分别计算）
        left_texture = self.compute_texture_contrast(image, left_mask)
        right_texture = self.compute_texture_contrast(image, right_mask)
        
        # 9. 峰高偏度（左右分别计算）
        left_skew = self.compute_peak_skewness(left_edges)
        right_skew = self.compute_peak_skewness(right_edges)
        
        # 10. 频域特征（左右分别计算）
        left_freq = self.compute_frequency_features(left_edges)
        right_freq = self.compute_frequency_features(right_edges)
        
        # 组合特征
        features = {
            # === 原有5个核心特征 ===
            # 粗糙度特征
            'left_rms_roughness': left_rms,
            'right_rms_roughness': right_rms,
            'avg_rms_roughness': (left_rms + right_rms) / 2,
            
            # 梯度能量（锐度）特征
            'left_gradient_energy': left_gradient_energy,
            'right_gradient_energy': right_gradient_energy,
            'avg_gradient_energy': (left_gradient_energy + right_gradient_energy) / 2,
            
            # 水平梯度能量（垂直边缘锐度）特征
            'left_horizontal_gradient': left_horizontal_gradient,
            'right_horizontal_gradient': right_horizontal_gradient,
            'avg_horizontal_gradient': (left_horizontal_gradient + right_horizontal_gradient) / 2,
            
            # 缺口深度特征
            'left_max_notch': left_notch,
            'right_max_notch': right_notch,
            'max_notch_depth': max(left_notch, right_notch),
            
            # 峰统计特征（撕裂面）
            'left_peak_count': left_peak_stats['peak_count'],
            'left_peak_density': left_peak_stats['peak_density'],
            'left_avg_peak_height': left_peak_stats['avg_peak_height'],
            
            # 峰统计特征（剪切面）
            'right_peak_count': right_peak_stats['peak_count'],
            'right_peak_density': right_peak_stats['peak_density'],
            'right_avg_peak_height': right_peak_stats['avg_peak_height'],
            
            # 面积比特征
            'tear_shear_area_ratio': area_ratio,
            
            # === 新增5个扩展特征 ===
            # 波纹特征
            'left_waviness_amplitude': left_waviness['waviness_amplitude'],
            'right_waviness_amplitude': right_waviness['waviness_amplitude'],
            'avg_waviness_amplitude': (left_waviness['waviness_amplitude'] + right_waviness['waviness_amplitude']) / 2,
            'left_waviness_wavelength': left_waviness['waviness_wavelength'],
            'right_waviness_wavelength': right_waviness['waviness_wavelength'],
            
            # 趋势特征
            'left_trend_slope': left_trend['trend_slope'],
            'right_trend_slope': right_trend['trend_slope'],
            'avg_trend_slope': (left_trend['trend_slope'] + right_trend['trend_slope']) / 2,
            'left_trend_r2': left_trend['trend_r2'],
            'right_trend_r2': right_trend['trend_r2'],
            
            # 纹理特征
            'left_glcm_contrast': left_texture['glcm_contrast'],
            'right_glcm_contrast': right_texture['glcm_contrast'],
            'avg_glcm_contrast': (left_texture['glcm_contrast'] + right_texture['glcm_contrast']) / 2,
            'left_glcm_homogeneity': left_texture['glcm_homogeneity'],
            'right_glcm_homogeneity': right_texture['glcm_homogeneity'],
            'left_glcm_energy': left_texture['glcm_energy'],
            'right_glcm_energy': right_texture['glcm_energy'],
            'left_glcm_correlation': left_texture['glcm_correlation'],
            'right_glcm_correlation': right_texture['glcm_correlation'],
            
            # 统计分布特征
            'left_peak_skewness': left_skew['peak_skewness'],
            'right_peak_skewness': right_skew['peak_skewness'],
            'avg_peak_skewness': (left_skew['peak_skewness'] + right_skew['peak_skewness']) / 2,
            'left_peak_kurtosis': left_skew['peak_kurtosis'],
            'right_peak_kurtosis': right_skew['peak_kurtosis'],
            'left_peak_std': left_skew['peak_std'],
            'right_peak_std': right_skew['peak_std'],
            
            # 频域特征
            'left_high_freq_ratio': left_freq['high_freq_ratio'],
            'right_high_freq_ratio': right_freq['high_freq_ratio'],
            'avg_high_freq_ratio': (left_freq['high_freq_ratio'] + right_freq['high_freq_ratio']) / 2,
            'left_dominant_freq': left_freq['dominant_freq'],
            'right_dominant_freq': right_freq['dominant_freq'],
            'left_spectral_centroid': left_freq['spectral_centroid'],
            'right_spectral_centroid': right_freq['spectral_centroid'],
            'avg_spectral_centroid': (left_freq['spectral_centroid'] + right_freq['spectral_centroid']) / 2,
        }
        
        return features
    
    def _get_empty_features(self) -> Dict[str, float]:
        """返回空特征字典（所有值为0）"""
        return {
            # 原有特征
            'left_rms_roughness': 0.0,
            'right_rms_roughness': 0.0,
            'avg_rms_roughness': 0.0,
            'left_gradient_energy': 0.0,
            'right_gradient_energy': 0.0,
            'avg_gradient_energy': 0.0,
            'left_horizontal_gradient': 0.0,
            'right_horizontal_gradient': 0.0,
            'avg_horizontal_gradient': 0.0,
            'left_max_notch': 0.0,
            'right_max_notch': 0.0,
            'max_notch_depth': 0.0,
            'left_peak_count': 0,
            'left_peak_density': 0.0,
            'left_avg_peak_height': 0.0,
            'right_peak_count': 0,
            'right_peak_density': 0.0,
            'right_avg_peak_height': 0.0,
            'tear_shear_area_ratio': 0.0,
            # 新增特征
            'left_waviness_amplitude': 0.0,
            'right_waviness_amplitude': 0.0,
            'avg_waviness_amplitude': 0.0,
            'left_waviness_wavelength': 0.0,
            'right_waviness_wavelength': 0.0,
            'left_trend_slope': 0.0,
            'right_trend_slope': 0.0,
            'avg_trend_slope': 0.0,
            'left_trend_r2': 0.0,
            'right_trend_r2': 0.0,
            'left_glcm_contrast': 0.0,
            'right_glcm_contrast': 0.0,
            'avg_glcm_contrast': 0.0,
            'left_glcm_homogeneity': 0.0,
            'right_glcm_homogeneity': 0.0,
            'left_glcm_energy': 0.0,
            'right_glcm_energy': 0.0,
            'left_glcm_correlation': 0.0,
            'right_glcm_correlation': 0.0,
            'left_peak_skewness': 0.0,
            'right_peak_skewness': 0.0,
            'avg_peak_skewness': 0.0,
            'left_peak_kurtosis': 0.0,
            'right_peak_kurtosis': 0.0,
            'left_peak_std': 0.0,
            'right_peak_std': 0.0,
            'left_high_freq_ratio': 0.0,
            'right_high_freq_ratio': 0.0,
            'avg_high_freq_ratio': 0.0,
            'left_dominant_freq': 0.0,
            'right_dominant_freq': 0.0,
            'left_spectral_centroid': 0.0,
            'right_spectral_centroid': 0.0,
            'avg_spectral_centroid': 0.0,
        }


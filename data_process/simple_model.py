"""
简单模型验证模块
主要功能：
1. 基于规则的剪刀磨损检测模型
2. 时序特征分析和趋势检测
3. 变化点检测算法
4. 剪刀更换时机预测
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import json
import os
from datetime import datetime
from config import MODEL_CONFIG, VIS_CONFIG
from font_utils import setup_chinese_font


class SimpleShearDetectionModel:
    """简单的剪刀磨损检测模型"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化模型
        
        Args:
            config: 模型配置参数
        """
        self.config = config if config is not None else MODEL_CONFIG
        self.feature_history = []
        self.time_history = []
        self.predictions = []
        
    def add_observation(self, features: Dict[str, Any], timestamp: Optional[float] = None):
        """
        添加新的观测数据
        
        Args:
            features: 特征字典
            timestamp: 时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = features.get('timestamp', datetime.now().timestamp())
        
        self.feature_history.append(features)
        self.time_history.append(timestamp)
    
    def calculate_ema(self, values: List[float], alpha: Optional[float] = None) -> List[float]:
        """
        计算指数移动平均
        
        Args:
            values: 数值列表
            alpha: 平滑参数
            
        Returns:
            EMA值列表
        """
        if not values:
            return []
        
        if alpha is None:
            alpha = self.config['ema_alpha']
        
        ema = [values[0]]
        for i in range(1, len(values)):
            ema_value = alpha * values[i] + (1 - alpha) * ema[-1]
            ema.append(ema_value)
        
        return ema
    
    def detect_trend(self, values: List[float], window_size: int = 10) -> Dict[str, Any]:
        """
        检测趋势
        
        Args:
            values: 数值列表
            window_size: 窗口大小
            
        Returns:
            趋势分析结果
        """
        if len(values) < window_size:
            return {
                'trend': 'insufficient_data',
                'slope': 0,
                'r_squared': 0,
                'p_value': 1.0
            }
        
        # 使用最近的窗口数据
        recent_values = values[-window_size:]
        x = np.arange(len(recent_values))
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
        
        # 判断趋势
        if p_value < 0.05:  # 显著性检验
            if slope > 0.001:
                trend = 'increasing'
            elif slope < -0.001:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'no_significant_trend'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err
        }
    
    def detect_change_points(self, values: List[float]) -> List[int]:
        """
        使用简单的变化点检测算法
        
        Args:
            values: 数值列表
            
        Returns:
            变化点索引列表
        """
        if len(values) < 10:
            return []
        
        change_points = []
        sensitivity = self.config['change_detection_sensitivity']
        
        # 计算移动平均和标准差
        window = 5
        for i in range(window, len(values) - window):
            # 前窗口和后窗口的统计量
            before_window = values[i-window:i]
            after_window = values[i:i+window]
            
            before_mean = np.mean(before_window)
            after_mean = np.mean(after_window)
            before_std = np.std(before_window)
            after_std = np.std(after_window)
            
            # 检测均值变化
            if before_std > 0 and after_std > 0:
                # t检验
                pooled_std = np.sqrt((before_std**2 + after_std**2) / 2)
                t_stat = abs(before_mean - after_mean) / (pooled_std * np.sqrt(2/window))
                
                # 简单的阈值检测
                if t_stat > 2.0 and abs(before_mean - after_mean) > sensitivity:
                    change_points.append(i)
        
        # 去除相邻的变化点
        filtered_points = []
        for point in change_points:
            if not filtered_points or point - filtered_points[-1] > 10:
                filtered_points.append(point)
        
        return filtered_points
    
    def evaluate_blade_condition(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估当前剪刀状态
        
        Args:
            features: 当前特征
            
        Returns:
            剪刀状态评估结果
        """
        # 关键指标
        tear_ratio = features.get('tear_ratio', 0)
        spot_count = features.get('spot_count', 0)
        spot_density = features.get('spot_density', 0)
        roughness_diff = features.get('roughness_difference', 0)
        
        # 阈值判断
        tear_threshold = self.config['tear_ratio_threshold']
        spot_threshold = self.config['spot_count_threshold']
        
        # 计算风险分数（0-1，越高越需要更换）
        risk_score = 0
        
        # 撕裂面比例风险
        if tear_ratio > tear_threshold:
            risk_score += 0.4 * (tear_ratio - tear_threshold) / (1 - tear_threshold)
        
        # 白斑数量风险
        if spot_count > spot_threshold:
            risk_score += 0.3 * min(1.0, (spot_count - spot_threshold) / spot_threshold)
        
        # 白斑密度风险
        if spot_density > 0.1:
            risk_score += 0.2 * min(1.0, spot_density / 0.1)
        
        # 粗糙度差异风险
        if roughness_diff > 0.2:
            risk_score += 0.1 * min(1.0, roughness_diff / 0.2)
        
        risk_score = min(1.0, risk_score)
        
        # 状态判断
        if risk_score < 0.3:
            condition = 'good'
            recommendation = '继续使用'
        elif risk_score < 0.6:
            condition = 'warning'
            recommendation = '密切监控'
        elif risk_score < 0.8:
            condition = 'critical'
            recommendation = '准备更换'
        else:
            condition = 'replace'
            recommendation = '立即更换'
        
        return {
            'condition': condition,
            'risk_score': risk_score,
            'recommendation': recommendation,
            'key_factors': {
                'tear_ratio': tear_ratio,
                'spot_count': spot_count,
                'spot_density': spot_density,
                'roughness_difference': roughness_diff
            }
        }
    
    def predict_replacement_time(self) -> Dict[str, Any]:
        """
        预测剪刀更换时间
        
        Returns:
            更换时间预测结果
        """
        if len(self.feature_history) < 5:
            return {
                'prediction_available': False,
                'reason': 'insufficient_data',
                'estimated_hours_remaining': None
            }
        
        # 提取关键指标的时序数据
        tear_ratios = [f.get('tear_ratio', 0) for f in self.feature_history]
        spot_counts = [f.get('spot_count', 0) for f in self.feature_history]
        
        # 计算EMA
        tear_ema = self.calculate_ema(tear_ratios)
        spot_ema = self.calculate_ema(spot_counts)
        
        # 趋势分析
        tear_trend = self.detect_trend(tear_ema)
        spot_trend = self.detect_trend(spot_ema)
        
        # 预测逻辑
        if tear_trend['trend'] == 'increasing' and tear_trend['p_value'] < 0.05:
            # 基于撕裂面比例趋势预测
            current_ratio = tear_ema[-1]
            target_ratio = self.config['tear_ratio_threshold']
            
            if tear_trend['slope'] > 0:
                hours_to_threshold = (target_ratio - current_ratio) / tear_trend['slope']
                # 转换为实际时间（假设每小时一个观测）
                estimated_hours = max(0, hours_to_threshold)
            else:
                estimated_hours = float('inf')
        else:
            estimated_hours = None
        
        return {
            'prediction_available': estimated_hours is not None,
            'estimated_hours_remaining': estimated_hours,
            'tear_trend': tear_trend,
            'spot_trend': spot_trend,
            'current_tear_ratio': tear_ema[-1] if tear_ema else 0,
            'target_tear_ratio': self.config['tear_ratio_threshold']
        }
    
    def analyze_time_series(self) -> Dict[str, Any]:
        """
        分析时间序列数据
        
        Returns:
            时间序列分析结果
        """
        if len(self.feature_history) < 3:
            return {'error': 'insufficient_data'}
        
        # 提取时序特征
        tear_ratios = [f.get('tear_ratio', 0) for f in self.feature_history]
        spot_counts = [f.get('spot_count', 0) for f in self.feature_history]
        spot_densities = [f.get('spot_density', 0) for f in self.feature_history]
        
        # 计算EMA
        tear_ema = self.calculate_ema(tear_ratios)
        spot_ema = self.calculate_ema(spot_counts)
        
        # 变化点检测
        tear_changes = self.detect_change_points(tear_ratios)
        spot_changes = self.detect_change_points(spot_counts)
        
        # 趋势分析
        tear_trend = self.detect_trend(tear_ratios)
        spot_trend = self.detect_trend(spot_counts)
        
        return {
            'tear_ratios': tear_ratios,
            'spot_counts': spot_counts,
            'spot_densities': spot_densities,
            'tear_ema': tear_ema,
            'spot_ema': spot_ema,
            'tear_change_points': tear_changes,
            'spot_change_points': spot_changes,
            'tear_trend': tear_trend,
            'spot_trend': spot_trend,
            'data_length': len(self.feature_history)
        }
    
    def visualize_analysis(self, save_path: Optional[str] = None):
        """
        可视化分析结果
        
        Args:
            save_path: 保存路径
        """
        # 设置中文字体
        setup_chinese_font()
        
        analysis = self.analyze_time_series()
        
        if 'error' in analysis:
            print(f"无法进行可视化: {analysis['error']}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 时间轴（使用索引作为简化的时间轴）
        time_axis = list(range(len(self.feature_history)))
        
        # 撕裂面比例趋势
        axes[0, 0].plot(time_axis, analysis['tear_ratios'], 'b-', alpha=0.6, label='原始数据')
        axes[0, 0].plot(time_axis, analysis['tear_ema'], 'r-', linewidth=2, label='EMA')
        axes[0, 0].axhline(y=self.config['tear_ratio_threshold'], color='orange', 
                          linestyle='--', label='告警阈值')
        
        # 标记变化点
        for cp in analysis['tear_change_points']:
            if cp < len(time_axis):
                axes[0, 0].axvline(x=cp, color='red', alpha=0.7, linestyle=':')
        
        axes[0, 0].set_title('撕裂面比例变化趋势')
        axes[0, 0].set_xlabel('时间点')
        axes[0, 0].set_ylabel('撕裂面比例')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 白斑数量趋势
        axes[0, 1].plot(time_axis, analysis['spot_counts'], 'g-', alpha=0.6, label='原始数据')
        axes[0, 1].plot(time_axis, analysis['spot_ema'], 'purple', linewidth=2, label='EMA')
        axes[0, 1].axhline(y=self.config['spot_count_threshold'], color='orange', 
                          linestyle='--', label='告警阈值')
        
        # 标记变化点
        for cp in analysis['spot_change_points']:
            if cp < len(time_axis):
                axes[0, 1].axvline(x=cp, color='red', alpha=0.7, linestyle=':')
        
        axes[0, 1].set_title('白斑数量变化趋势')
        axes[0, 1].set_xlabel('时间点')
        axes[0, 1].set_ylabel('白斑数量')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 散点图：撕裂面比例 vs 白斑数量
        colors = ['green' if i < len(time_axis) * 0.7 else 'orange' if i < len(time_axis) * 0.9 else 'red' 
                 for i in range(len(time_axis))]
        
        scatter = axes[1, 0].scatter(analysis['tear_ratios'], analysis['spot_counts'], 
                                   c=colors, alpha=0.7, s=50)
        axes[1, 0].set_xlabel('撕裂面比例')
        axes[1, 0].set_ylabel('白斑数量')
        axes[1, 0].set_title('撕裂面比例 vs 白斑数量')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='早期'),
                          Patch(facecolor='orange', label='中期'),
                          Patch(facecolor='red', label='后期')]
        axes[1, 0].legend(handles=legend_elements)
        
        # 风险评分历史
        risk_scores = []
        for features in self.feature_history:
            evaluation = self.evaluate_blade_condition(features)
            risk_scores.append(evaluation['risk_score'])
        
        axes[1, 1].plot(time_axis, risk_scores, 'red', linewidth=2, marker='o')
        axes[1, 1].axhline(y=0.6, color='orange', linestyle='--', label='警告线')
        axes[1, 1].axhline(y=0.8, color='red', linestyle='--', label='危险线')
        axes[1, 1].set_title('剪刀磨损风险评分')
        axes[1, 1].set_xlabel('时间点')
        axes[1, 1].set_ylabel('风险评分')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
            plt.close()
    
    def generate_report(self) -> Dict[str, Any]:
        """
        生成分析报告
        
        Returns:
            分析报告
        """
        if not self.feature_history:
            return {'error': 'no_data'}
        
        # 当前状态评估
        current_evaluation = self.evaluate_blade_condition(self.feature_history[-1])
        
        # 时序分析
        time_series_analysis = self.analyze_time_series()
        
        # 更换时间预测
        replacement_prediction = self.predict_replacement_time()
        
        # 统计摘要
        tear_ratios = [f.get('tear_ratio', 0) for f in self.feature_history]
        spot_counts = [f.get('spot_count', 0) for f in self.feature_history]
        
        summary_stats = {
            'total_observations': len(self.feature_history),
            'tear_ratio_stats': {
                'mean': np.mean(tear_ratios),
                'std': np.std(tear_ratios),
                'min': np.min(tear_ratios),
                'max': np.max(tear_ratios),
                'current': tear_ratios[-1] if tear_ratios else 0
            },
            'spot_count_stats': {
                'mean': np.mean(spot_counts),
                'std': np.std(spot_counts),
                'min': np.min(spot_counts),
                'max': np.max(spot_counts),
                'current': spot_counts[-1] if spot_counts else 0
            }
        }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_evaluation': current_evaluation,
            'replacement_prediction': replacement_prediction,
            'time_series_analysis': time_series_analysis,
            'summary_statistics': summary_stats
        }
        
        return report


def test_simple_model():
    """测试简单模型"""
    import os
    from preprocessor import ImagePreprocessor
    from segmentation import SurfaceSegmentator
    from feature_extractor import FeatureExtractor
    from config import DATA_DIR, OUTPUT_DIR
    
    # 初始化所有组件
    preprocessor = ImagePreprocessor()
    segmentator = SurfaceSegmentator()
    extractor = FeatureExtractor()
    model = SimpleShearDetectionModel()
    
    # 测试图像路径
    test_image = os.path.join(DATA_DIR, 'Image_20250710125452500.bmp')
    
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
    try:
        print("开始模型验证测试...")
        
        # 模拟多个时间点的数据（通过添加噪声模拟不同状态）
        base_features = None
        
        for i in range(20):  # 模拟20个时间点
            print(f"处理时间点 {i+1}/20...")
            
            # 预处理
            roi_image, _ = preprocessor.preprocess_pipeline(test_image, target_size=(128, 512))
            
            # 分割
            tear_mask, shear_mask, _ = segmentator.segment_surface(roi_image, method='hybrid')
            
            # 特征提取
            features = extractor.extract_all_features(roi_image, tear_mask, shear_mask)
            
            # 第一次获取基础特征
            if base_features is None:
                base_features = features.copy()
            
            # 模拟时间演变：随着时间推移，撕裂面比例和白斑数量逐渐增加
            time_factor = i / 20.0
            noise = np.random.normal(0, 0.05)  # 添加噪声
            
            # 模拟撕裂面比例增长
            features['tear_ratio'] = min(0.9, base_features['tear_ratio'] + time_factor * 0.3 + noise)
            features['shear_ratio'] = 1 - features['tear_ratio']
            features['tear_to_shear_ratio'] = features['tear_ratio'] / features['shear_ratio'] if features['shear_ratio'] > 0 else float('inf')
            
            # 模拟白斑数量增长
            features['spot_count'] = max(0, int(base_features['spot_count'] + time_factor * 15 + noise * 5))
            features['spot_density'] = features['spot_count'] / max(1, features['tear_area']) * 1000
            
            # 添加时间戳
            features['timestamp'] = datetime.now().timestamp() + i * 3600  # 每小时一个观测
            
            # 添加到模型
            model.add_observation(features)
            
            # 评估当前状态
            evaluation = model.evaluate_blade_condition(features)
            print(f"  - 撕裂面比例: {features['tear_ratio']:.3f}")
            print(f"  - 白斑数量: {features['spot_count']}")
            print(f"  - 风险评分: {evaluation['risk_score']:.3f}")
            print(f"  - 状态: {evaluation['condition']}")
            print(f"  - 建议: {evaluation['recommendation']}")
        
        # 生成分析报告
        print("\n生成分析报告...")
        report = model.generate_report()
        
        # 保存报告
        report_path = os.path.join(OUTPUT_DIR, 'model_analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"分析报告已保存到: {report_path}")
        
        # 可视化分析结果
        vis_path = os.path.join(OUTPUT_DIR, 'model_analysis_visualization.png')
        model.visualize_analysis(vis_path)
        print(f"分析可视化已保存到: {vis_path}")
        
        # 打印关键结果
        print("\n=== 分析结果摘要 ===")
        current_eval = report['current_evaluation']
        print(f"当前状态: {current_eval['condition']}")
        print(f"风险评分: {current_eval['risk_score']:.3f}")
        print(f"建议: {current_eval['recommendation']}")
        
        replacement_pred = report['replacement_prediction']
        if replacement_pred['prediction_available']:
            print(f"预计剩余时间: {replacement_pred['estimated_hours_remaining']:.1f} 小时")
        else:
            print("暂无足够数据进行时间预测")
        
        stats = report['summary_statistics']
        print(f"撕裂面比例变化: {stats['tear_ratio_stats']['min']:.3f} -> {stats['tear_ratio_stats']['max']:.3f}")
        print(f"白斑数量变化: {stats['spot_count_stats']['min']:.0f} -> {stats['spot_count_stats']['max']:.0f}")
        
    except Exception as e:
        print(f"模型测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_model()

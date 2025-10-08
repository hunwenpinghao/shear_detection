"""
综合磨损指标计算模块
实现3种综合评分方法：加权平均、PCA主成分、多维度分组
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


class CompositeWearIndicator:
    """综合磨损指标计算器"""
    
    def __init__(self):
        """初始化综合指标计算器"""
        # 特征分组（4个维度）
        self.feature_groups = {
            'geometric': [
                'avg_rms_roughness', 
                'max_notch_depth', 
                'avg_waviness_amplitude', 
                'avg_trend_slope'
            ],
            'texture': [
                'avg_glcm_contrast', 
                'left_glcm_homogeneity', 
                'right_peak_density'
            ],
            'frequency': [
                'avg_gradient_energy', 
                'avg_high_freq_ratio', 
                'avg_spectral_centroid'
            ],
            'distribution': [
                'avg_peak_skewness', 
                'left_peak_kurtosis', 
                'tear_shear_area_ratio'
            ]
        }
        
        # 默认权重配置
        self.default_weights = {
            'geometric': 0.35,
            'texture': 0.25,
            'frequency': 0.25,
            'distribution': 0.15
        }
    
    def compute_weighted_score(self, features_df: pd.DataFrame, 
                               weights: Dict[str, float] = None) -> np.ndarray:
        """
        方法1: 可配置加权平均
        
        Args:
            features_df: 特征数据框
            weights: 各维度权重字典，如果为None则使用默认权重
            
        Returns:
            加权综合得分数组
        """
        if weights is None:
            weights = self.default_weights
        
        # 归一化所有特征
        scaler = MinMaxScaler()
        
        # 收集所有可用特征
        all_features = []
        for group_features in self.feature_groups.values():
            all_features.extend(group_features)
        
        # 过滤掉不存在的特征
        available_features = [f for f in all_features if f in features_df.columns]
        
        if len(available_features) == 0:
            return np.zeros(len(features_df))
        
        # 归一化特征
        features_normalized = pd.DataFrame(
            scaler.fit_transform(features_df[available_features]),
            columns=available_features,
            index=features_df.index
        )
        
        # 计算每个维度的得分
        dimension_scores = {}
        for dim_name, dim_features in self.feature_groups.items():
            # 过滤可用特征
            available_dim_features = [f for f in dim_features if f in features_normalized.columns]
            
            if len(available_dim_features) > 0:
                # 该维度的平均值
                dimension_scores[dim_name] = features_normalized[available_dim_features].mean(axis=1).values
            else:
                dimension_scores[dim_name] = np.zeros(len(features_df))
        
        # 加权求和
        weighted_score = np.zeros(len(features_df))
        for dim_name, score in dimension_scores.items():
            weight = weights.get(dim_name, 0.25)
            weighted_score += weight * score
        
        return weighted_score
    
    def compute_pca_score(self, features_df: pd.DataFrame, 
                         n_components: int = 3) -> Dict:
        """
        方法2: PCA主成分得分
        
        Args:
            features_df: 特征数据框
            n_components: 主成分数量
            
        Returns:
            字典，包含：
            - pca_score: 第一主成分得分（归一化到0-1）
            - explained_variance_ratio: 各主成分解释方差比例
            - loadings: 各特征在主成分上的载荷
            - components: 主成分矩阵
        """
        # 收集所有可用特征
        all_features = []
        for group_features in self.feature_groups.values():
            all_features.extend(group_features)
        
        # 过滤掉不存在的特征
        available_features = [f for f in all_features if f in features_df.columns]
        
        if len(available_features) < 2:
            return {
                'pca_score': np.zeros(len(features_df)),
                'explained_variance_ratio': np.array([]),
                'loadings': pd.DataFrame(),
                'components': np.array([]),
                'feature_names': []
            }
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df[available_features])
        
        # PCA降维
        n_components = min(n_components, len(available_features), len(features_df))
        pca = PCA(n_components=n_components)
        
        try:
            pca_transformed = pca.fit_transform(features_scaled)
            
            # 第一主成分作为综合得分
            pc1 = pca_transformed[:, 0]
            
            # 归一化到0-1
            if pc1.max() > pc1.min():
                pca_score = (pc1 - pc1.min()) / (pc1.max() - pc1.min())
            else:
                pca_score = np.zeros(len(pc1))
            
            # 载荷矩阵
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=available_features
            )
            
        except Exception as e:
            print(f"PCA计算出错: {e}")
            return {
                'pca_score': np.zeros(len(features_df)),
                'explained_variance_ratio': np.array([]),
                'loadings': pd.DataFrame(),
                'components': np.array([]),
                'feature_names': available_features
            }
        
        return {
            'pca_score': pca_score,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'loadings': loadings,
            'components': pca.components_,
            'feature_names': available_features
        }
    
    def compute_multi_dimensional_score(self, features_df: pd.DataFrame) -> Dict:
        """
        方法3: 多维度分组得分
        
        每个特征组计算子得分，最后加权组合
        
        Args:
            features_df: 特征数据框
            
        Returns:
            字典，包含各维度得分和总体得分
        """
        # 归一化
        scaler = MinMaxScaler()
        
        result = {}
        
        # 计算每个维度的得分
        for dim_name, dim_features in self.feature_groups.items():
            # 过滤可用特征
            available_features = [f for f in dim_features if f in features_df.columns]
            
            if len(available_features) > 0:
                # 归一化该维度的特征
                dim_normalized = scaler.fit_transform(features_df[available_features])
                
                # 该维度的平均得分
                dim_score = np.mean(dim_normalized, axis=1)
                result[f'{dim_name}_score'] = dim_score
            else:
                result[f'{dim_name}_score'] = np.zeros(len(features_df))
        
        # 总体得分（加权平均）
        overall_score = np.zeros(len(features_df))
        for dim_name in self.feature_groups.keys():
            weight = self.default_weights.get(dim_name, 0.25)
            overall_score += weight * result[f'{dim_name}_score']
        
        result['overall_score'] = overall_score
        
        return result
    
    def analyze_feature_importance(self, features_df: pd.DataFrame, 
                                   time_column: str = 'frame_id') -> pd.DataFrame:
        """
        特征重要性分析（基于方差和单调性）
        
        Args:
            features_df: 特征数据框
            time_column: 时间列名（用于计算单调性）
            
        Returns:
            特征重要性分析结果DataFrame，按重要性排序
        """
        # 收集所有可用特征
        all_features = []
        for group_features in self.feature_groups.values():
            all_features.extend(group_features)
        
        # 过滤掉不存在的特征
        available_features = [f for f in all_features if f in features_df.columns]
        
        if len(available_features) == 0:
            return pd.DataFrame()
        
        importance_data = []
        
        for feature in available_features:
            values = features_df[feature].values
            
            # 1. 变异系数 (CV = std / mean)
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if mean_val != 0:
                cv = abs(std_val / mean_val)
            else:
                cv = 0.0
            
            # 2. 单调性（Spearman相关系数 vs 时间）
            if time_column in features_df.columns:
                try:
                    correlation, pval = spearmanr(features_df[time_column], values)
                    monotonicity = abs(correlation)
                    
                    if np.isnan(monotonicity):
                        monotonicity = 0.0
                except:
                    monotonicity = 0.0
                    pval = 1.0
            else:
                # 如果没有时间列，用索引
                try:
                    correlation, pval = spearmanr(np.arange(len(values)), values)
                    monotonicity = abs(correlation)
                    
                    if np.isnan(monotonicity):
                        monotonicity = 0.0
                except:
                    monotonicity = 0.0
                    pval = 1.0
            
            # 3. 变化率（首尾差异）
            if len(values) > 1:
                first_val = np.mean(values[:max(1, len(values)//10)])  # 前10%
                last_val = np.mean(values[-max(1, len(values)//10):])   # 后10%
                
                if first_val != 0:
                    change_rate = abs((last_val - first_val) / first_val)
                else:
                    change_rate = abs(last_val - first_val)
            else:
                change_rate = 0.0
            
            # 4. 综合重要性分数（加权组合）
            # CV权重0.3, 单调性权重0.5, 变化率权重0.2
            importance_score = 0.3 * cv + 0.5 * monotonicity + 0.2 * change_rate
            
            # 确定特征所属维度
            feature_group = 'unknown'
            for group_name, group_features in self.feature_groups.items():
                if feature in group_features:
                    feature_group = group_name
                    break
            
            importance_data.append({
                'feature': feature,
                'group': feature_group,
                'cv': cv,
                'monotonicity': monotonicity,
                'change_rate': change_rate,
                'importance_score': importance_score,
                'p_value': pval
            })
        
        # 创建DataFrame并排序
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('importance_score', ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        return importance_df
    
    def compute_all_indicators(self, features_df: pd.DataFrame, 
                              weights: Dict[str, float] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        计算所有综合指标（便捷方法）
        
        Args:
            features_df: 特征数据框
            weights: 权重字典（可选）
            
        Returns:
            (增强后的特征DataFrame, 分析结果字典)
        """
        df = features_df.copy()
        
        # 方法1: 加权得分
        df['weighted_score'] = self.compute_weighted_score(df, weights)
        
        # 方法2: PCA得分
        pca_result = self.compute_pca_score(df)
        df['pca_score'] = pca_result['pca_score']
        
        # 方法3: 多维得分
        multi_scores = self.compute_multi_dimensional_score(df)
        for score_name, score_values in multi_scores.items():
            df[score_name] = score_values
        
        # 特征重要性分析
        importance_df = self.analyze_feature_importance(df)
        
        # 整理结果
        analysis_results = {
            'pca_result': pca_result,
            'multi_scores': multi_scores,
            'importance_df': importance_df
        }
        
        return df, analysis_results


"""
聚类分析测试模块
测试不同聚类算法将撕裂面和剪切面分离的效果
"""
import numpy as np
import os
import cv2
from feature_extractor import FeatureExtractor
from preprocessor import ImagePreprocessor
from config import DATA_DIR, OUTPUT_DIR

def test_clustering_methods():
    """测试所有聚类方法"""
    print("开始聚类分析测试...")
    
    # 初始化
    preprocessor = ImagePreprocessor()
    cluster_analyzer = FeatureExtractor()
    
    # 测试图像
    test_image_path = os.path.join(DATA_DIR, 'Image_20250710125452500.bmp')
    
    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        return
    
    # 预处理
    print("  - 预处理图像...")
    image, _ = preprocessor.preprocess_pipeline(test_image_path, target_size=(128, 512))
    
    # 测试四种聚类方法
    methods = ['horizontal_profiles', 'kmeans', 'hierarchical', 'dbscan']
    clustering_results = {}
    
    for method in methods:
        print(f"  - 测试 {method} 聚类方法...")
        try:
            result = cluster_analyzer.texture_clustering_analysis(image, method=method)
            clustering_results[method] = result
            
            print(f"    {method}: 成功率={result.get('clustering_result', {}).get('success', False)}")
            print(f"      分离质量: {result.get('separation_quality', 0):.3f}")
            print(f"      聚类稳定性: {result.get('clustering_stability', 0):.3f}")
            
        except Exception as e:
            print(f"    {method} 方法测试失败: {e}")
            clustering_results[method] = {'error': str(e)}
    
    # 可视化最佳结果
    print("  - 生成聚类可视化结果...")
    best_method = 'horizontal_profiles'  # 默认方法
    best_score = 0
    
    for method, result in clustering_results.items():
        if 'error' not in result:
            score = result.get('separation_quality', 0)
            if score > best_score:
                best_score = score
                best_method = method
    
    print(f"  选择最佳方法: {best_method} (得分: {best_score:.3f})")
    
    # 保存可视化结果
    if best_method in clustering_results and 'error' not in clustering_results[best_method]:
        # 获取斑块信息用于聚类可视化
        white_spots_info = cluster_analyzer.detect_all_white_spots(image)
        spot_image = white_spots_info.get('all_white_binary_mask', None)
        
        vis_path = os.path.join(OUTPUT_DIR, 'clustering_analysis_result.png')
        cluster_analyzer.visualize_clustering_results(image, clustering_results[best_method], vis_path, spot_image)
        print(f"  聚类可视化已保存到: {vis_path}")
        
        # 保存结果数据
        import json
        result_path = os.path.join(OUTPUT_DIR, 'clustering_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            # 将所有结果转换为可序列化的字典
            serializable_results = {}
            for method, result in clustering_results.items():
                if 'error' in result:
                    serializable_results[method] = {'error': result['error']}
                else:
                    serializable_results[method] = {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"  聚类结果已保存到: {result_path}")
    
    print("聚类分析测试完成!")

def compare_clustering_accuracy():
    """比较不同聚类方法的准确性和稳定性"""
    print("\n=== 聚类方法对比分析 ===")
    
    methods_accuracy = {
        'horizontal_profiles': "基于水平投影的线簇聚类",
        'kmeans': "K-Means空间聚类",
        'hierarchical': "层次聚类",
        'dbscan': "DBSCAN密度聚类"
    }
    
    print("聚类方法简介:")
    for method, description in methods_accuracy.items():
        print(f"  {method}: {description}")
    
    print("\n预期效果对比:")
    print("  1. horizontal_profiles: 最适合垂直条状纹理，能识别波纹边界")
    print("  2. kmeans: 适用于空间连续的区域分离")
    print("  3. hierarchical: 处理不规则边界和噪声较好")
    print("  4. dbscan: 自适应性最强，但参数调优复杂")

if __name__ == "__main__":
    test_clustering_methods()
    compare_clustering_accuracy()
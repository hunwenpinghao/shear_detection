"""
测试改进后的曲线分割检测方法
专门检测灰色分割线的算法，只考虑白色区域
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 导入相关模块
from segmentation import SurfaceSegmentator
from preprocessor import ImagePreprocessor
from config import PREPROCESS_CONFIG, VIS_CONFIG
from font_utils import setup_chinese_font

def test_improved_curve_detection():
    """测试改进的曲线检测算法"""
    setup_chinese_font()
    
    # 加载必要组件
    preprocessor = ImagePreprocessor()
    segmentator = SurfaceSegmentator()
    test_image = os.path.join('..', 'data', 'Image_20250710125452500.bmp')
    
    try:
        # 预处理图像 
        roi_image, _ = preprocessor.preprocess_pipeline(test_image, target_size=(128, 512))
        print('✓ 图像预处理完成')
        print(f'ROI图像尺寸: {roi_image.shape}')
        
        # 使用改进的曲线分割
        tear_mask, shear_mask, info = segmentator.segment_surface(roi_image, method='curved')
        boundary_positions = segmentator.detect_curved_boundary(roi_image)
        
        # 输出统计信息 
        print(f'\n=== 改进曲线分割结果 ===')
        print(f'撕裂面面积: {info["tear_area"]}')
        print(f'剪切面面积: {info["shear_area"]}')
        print(f'撕裂面比例: {info["tear_ratio"]:.3f}')
        print(f'剪切面比例: {info["shear_ratio"]:.3f}')
        print(f'边界线变化系数σ: {np.std(boundary_positions):.1f} pixels')
        
        # 可视化结果
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 原始图像
        axes[0, 0].imshow(roi_image, cmap='gray')
        axes[0, 0].set_title('原始ROI图像')
        axes[0, 0].axis('off')
        
        # 2. 灰色分割线检测显示
        rows = np.arange(len(boundary_positions))
        axes[0, 1].imshow(roi_image, cmap='gray')
        axes[0, 1].plot(boundary_positions, rows, 'r-', linewidth=3, label='检测到的灰色分割线')
        axes[0, 1].set_title('检测到的灰色分割线（红线显示）')
        axes[0, 1].axis('off')
        
        # 3. 撕裂面（仅白色区域）
        axes[1, 0].imshow(tear_mask, cmap='Reds')
        axes[1, 0].set_title('撕裂面（仅白色区域）')
        axes[1, 0].axis('off')
        
        # 4. 剪切面（仅白色区域）
        axes[1, 1].imshow(shear_mask, cmap='Blues')
        axes[1, 1].set_title('剪切面（仅白色区域）')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        os.makedirs('../output', exist_ok=True)
        result_path = '../output/improved_curve_segmentation.png'
        plt.savefig(result_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'\n✅ 改进曲线分割可视化已保存: {result_path}')
        
        # 输出边界线检测的详细信息
        print('\n=== 边界线质量分析 ===')
        white_ratio = np.sum(roi_image > PREPROCESS_CONFIG['white_threshold']) / (roi_image.shape[0] * roi_image.shape[1])
        print(f'白色区域占比: {white_ratio:.1%}')
        print(f'边界线检测范围: [{np.min(boundary_positions)}, {np.max(boundary_positions)}]')
        print(f'边界线轨迹稳定性: {"高" if np.std(boundary_positions) < 10 else "中" if np.std(boundary_positions) < 30 else "低"}'.format(np.std(boundary_positions)))
        
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_curve_detection()

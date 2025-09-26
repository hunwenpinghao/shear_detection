from segmentation import SurfaceSegmentator
from preprocessor import ImagePreprocessor
import numpy as np
import os

# 创建测试文件
preprocessor = ImagePreprocessor()
segmentator = SurfaceSegmentator()
test_image = os.path.join('..', 'data', 'Image_20250710125452500.bmp')

# 生成测试图像
roi_image, _ = preprocessor.preprocess_pipeline(test_image, target_size=(128, 512))
print('✓ 图像预处理成功')

# 测试曲线检测与可视化
tear_mask, shear_mask, info = segmentator.segment_surface(roi_image, method='curved')
boundary_positions = segmentator.detect_curved_boundary(roi_image)

# 建立输出目录
os.makedirs('../output', exist_ok=True)

# 调用片段可视化带边界位置
segmentator.visualize_segmentation(roi_image, tear_mask, shear_mask, 
                                  boundary_positions=boundary_positions,
                                  save_path='../output/curved_segmentation_result.png')

# 输出结果
print('\n=== 曲线拟合分割结果 ===')
tear_area = int(info.get("tear_area", 0))
shear_area = int(info.get("shear_area", 0))
tear_per = float(info.get("tear_ratio", 0.0))
shear_per = float(info.get("shear_ratio", 0.0))

print(f'撕裂面面积: {tear_area}')
print(f'剪切面面积: {shear_area}')
print(f'撕裂面比例: {tear_per:.3f}') 
print(f'剪切面比例: {shear_per:.3f}')
print(f'曲线散布标准差: {np.std(boundary_positions):.1f}pixels')
print('✅ 已保存曲线分割结果图像')

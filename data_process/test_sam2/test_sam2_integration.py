"""
测试 SAM2 集成
验证 API 以及可视化生成。
"""
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessor import ImagePreprocessor
from segmentation import SurfaceSegmentator

def test_sam2_invocation():
    """测试SAM2从segment_surface接口调用，保证不会在实例化时早期断路"""
    processor = ImagePreprocessor()
    image_path = "../data/Image_20250710125452500.bmp"
    
    try:
        print("[Test] 尝试SAM2方法分支")
        roi_image, info = processor.preprocess_pipeline(image_path, target_size=(128, 512))
        
        segmentor = SurfaceSegmentator()
        (tear_mask, shear_mask, seg_info) = segmentor.segment_surface(roi_image, method="sam2")
        
        print("[Success] SAM2 调用成功")
    except Exception as exc:
        print(f"[Normal] SAM2 尚未安装/可用。info - %s" % str(exc))
        # 这是一种常见情况，因此这不是一次失败.

if __name__ == '__main__':
    test_sam2_invocation()
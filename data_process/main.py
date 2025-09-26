"""
主程序 - 整合所有模块进行完整的剪刀磨损检测流程
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessor import ImagePreprocessor
from segmentation import SurfaceSegmentator  
from feature_extractor import FeatureExtractor
from simple_model import SimpleShearDetectionModel
from config import DATA_DIR, OUTPUT_DIR


class ShearDetectionPipeline:
    """完整的剪刀磨损检测流水线"""
    
    def __init__(self):
        """初始化所有组件"""
        self.preprocessor = ImagePreprocessor()
        self.segmentator = SurfaceSegmentator()
        self.extractor = FeatureExtractor()
        self.model = SimpleShearDetectionModel()
        
    def process_single_image(self, image_path: str, 
                           segmentation_method: str = 'hybrid',
                           target_size: Optional[tuple] = (128, 512)) -> dict:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            segmentation_method: 分割方法
            target_size: 目标尺寸
            
        Returns:
            处理结果字典
        """
        print(f"处理图像: {image_path}")
        
        # 1. 预处理
        print("  - 图像预处理...")
        roi_image, preprocess_info = self.preprocessor.preprocess_pipeline(
            image_path, target_size=target_size)
        
        # 2. 分割
        print("  - 表面分割...")
        tear_mask, shear_mask, segment_info = self.segmentator.segment_surface(
            roi_image, method=segmentation_method)
        
        # 3. 特征提取
        print("  - 特征提取...")
        features = self.extractor.extract_all_features(roi_image, tear_mask, shear_mask)
        
        # 4. 状态评估
        print("  - 状态评估...")
        evaluation = self.model.evaluate_blade_condition(features)
        
        # 5. 可视化（可选）
        vis_dir = os.path.join(OUTPUT_DIR, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 预处理可视化
        preprocess_vis_path = os.path.join(vis_dir, f'preprocess_{os.path.basename(image_path)}.png')
        self.preprocessor.visualize_preprocessing(image_path, preprocess_vis_path)
        
        # 分割可视化
        segment_vis_path = os.path.join(vis_dir, f'segment_{os.path.basename(image_path)}.png')
        # 曲线分割且启用
        boundary_positions = None
        if segmentation_method == 'curved':
            boundary_positions = self.segmentator.detect_curved_boundary(roi_image)
        self.segmentator.visualize_segmentation(roi_image, tear_mask, shear_mask, segment_vis_path, boundary_positions)
        
        # 特征可视化
        feature_vis_path = os.path.join(vis_dir, f'features_{os.path.basename(image_path)}.png')
        self.extractor.visualize_features(roi_image, tear_mask, shear_mask, features, feature_vis_path)
        
        result = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'preprocessing': preprocess_info,
            'segmentation': segment_info,
            'features': features,
            'evaluation': evaluation,
            'visualizations': {
                'preprocessing': preprocess_vis_path,
                'segmentation': segment_vis_path,
                'features': feature_vis_path
            }
        }
        
        return result
    
    def process_image_sequence(self, image_paths: list,
                             segmentation_method: str = 'hybrid') -> dict:
        """
        处理图像序列（模拟时序数据）
        
        Args:
            image_paths: 图像路径列表
            segmentation_method: 分割方法
            
        Returns:
            序列处理结果
        """
        print(f"处理图像序列，共 {len(image_paths)} 张图像")
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\n处理第 {i+1}/{len(image_paths)} 张图像")
            
            # 处理单张图像
            result = self.process_single_image(image_path, segmentation_method)
            results.append(result)
            
            # 添加到模型中进行时序分析
            self.model.add_observation(result['features'])
            
            # 打印当前状态
            evaluation = result['evaluation']
            print(f"  - 撕裂面比例: {result['features']['tear_ratio']:.3f}")
            print(f"  - 白斑数量: {result['features']['spot_count']}")
            print(f"  - 风险评分: {evaluation['risk_score']:.3f}")
            print(f"  - 状态: {evaluation['condition']}")
            print(f"  - 建议: {evaluation['recommendation']}")
        
        # 生成时序分析报告
        print("\n生成时序分析报告...")
        time_series_report = self.model.generate_report()
        
        # 时序可视化
        ts_vis_path = os.path.join(OUTPUT_DIR, 'time_series_analysis.png')
        self.model.visualize_analysis(ts_vis_path)
        
        sequence_result = {
            'sequence_info': {
                'total_images': len(image_paths),
                'processing_timestamp': datetime.now().isoformat(),
                'segmentation_method': segmentation_method
            },
            'individual_results': results,
            'time_series_analysis': time_series_report,
            'time_series_visualization': ts_vis_path
        }
        
        return sequence_result
    
    def run_demo(self):
        """运行演示程序"""
        print("=== 剪刀磨损检测演示程序 ===\n")
        
        # 查找测试图像
        test_image = os.path.join(DATA_DIR, 'Image_20250710125452500.bmp')
        
        if not os.path.exists(test_image):
            print(f"错误: 测试图像不存在 - {test_image}")
            print("请确保data目录下有测试图像文件")
            return
        
        print("1. 单张图像处理演示")
        print("-" * 40)
        
        # 单张图像处理
        single_result = self.process_single_image(test_image)
        
        # 保存单张图像结果
        single_result_path = os.path.join(OUTPUT_DIR, 'single_image_result.json')
        with open(single_result_path, 'w', encoding='utf-8') as f:
            json.dump(single_result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n单张图像处理结果已保存到: {single_result_path}")
        
        print("\n2. 时序模拟演示")
        print("-" * 40)
        
        # 模拟时序数据（使用同一张图像，通过添加噪声模拟不同时间点）
        # 在实际应用中，这里应该是真实的时序图像
        image_sequence = [test_image] * 10  # 模拟10个时间点
        
        sequence_result = self.process_image_sequence(image_sequence)
        
        # 保存时序结果
        sequence_result_path = os.path.join(OUTPUT_DIR, 'sequence_analysis_result.json')
        with open(sequence_result_path, 'w', encoding='utf-8') as f:
            json.dump(sequence_result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n时序分析结果已保存到: {sequence_result_path}")
        
        # 打印摘要
        print("\n=== 处理结果摘要 ===")
        final_evaluation = sequence_result['individual_results'][-1]['evaluation']
        ts_analysis = sequence_result['time_series_analysis']
        
        print(f"最终状态: {final_evaluation['condition']}")
        print(f"风险评分: {final_evaluation['risk_score']:.3f}")
        print(f"处理建议: {final_evaluation['recommendation']}")
        
        if ts_analysis.get('replacement_prediction', {}).get('prediction_available'):
            pred = ts_analysis['replacement_prediction']
            print(f"预计剩余时间: {pred['estimated_hours_remaining']:.1f} 小时")
        
        print(f"\n所有结果文件已保存到: {OUTPUT_DIR}")
        print("请查看可视化图片以了解详细分析结果")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='剪刀磨损检测系统')
    parser.add_argument('--mode', choices=['demo', 'single', 'sequence'], 
                       default='demo', help='运行模式')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--images', nargs='+', help='图像序列路径列表')
    parser.add_argument('--method', choices=['centerline', 'gradient', 'texture', 'hybrid', 'sam2', 'curved', 'boundary'],
                       default='hybrid', help='分割方法')
    
    args = parser.parse_args()
    
    # 创建流水线
    pipeline = ShearDetectionPipeline()
    
    try:
        if args.mode == 'demo':
            # 运行演示
            pipeline.run_demo()
            
        elif args.mode == 'single':
            # 处理单张图像
            if not args.image:
                print("错误: 单张图像模式需要指定 --image 参数")
                return
            
            if not os.path.exists(args.image):
                print(f"错误: 图像文件不存在 - {args.image}")
                return
            
            result = pipeline.process_single_image(args.image, args.method)
            
            # 保存结果
            result_path = os.path.join(OUTPUT_DIR, 'single_image_result.json')
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"处理完成，结果已保存到: {result_path}")
            
        elif args.mode == 'sequence':
            # 处理图像序列
            if not args.images:
                print("错误: 序列模式需要指定 --images 参数")
                return
            
            # 检查所有图像是否存在
            for img_path in args.images:
                if not os.path.exists(img_path):
                    print(f"错误: 图像文件不存在 - {img_path}")
                    return
            
            result = pipeline.process_image_sequence(args.images, args.method)
            
            # 保存结果
            result_path = os.path.join(OUTPUT_DIR, 'sequence_analysis_result.json')
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"序列处理完成，结果已保存到: {result_path}")
            
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

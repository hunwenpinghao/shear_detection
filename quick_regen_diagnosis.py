"""
快速重新生成帧诊断图脚本

只重新生成帧诊断图，不重新提取特征，适合在修改了visualizer后快速更新诊断图。

用法:
    python quick_regen_diagnosis.py --output_dir <输出目录> --diagnosis_interval 100

示例:
    python quick_regen_diagnosis.py --output_dir data_video12_20250819134002235/coil_wear_analysis --diagnosis_interval 100
"""

import os
import sys
import argparse
import glob
import cv2
import pandas as pd
from tqdm import tqdm

# 添加主项目的模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'wear_degree_analysis', 'src'))

from preprocessor import ImagePreprocessor
from geometry_features import GeometryFeatureExtractor
from visualizer import WearVisualizer
from utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description='快速重新生成帧诊断图（使用已有特征数据）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python quick_regen_diagnosis.py --output_dir data/analysis --diagnosis_interval 100
  python quick_regen_diagnosis.py --output_dir data_video12_20250819134002235/coil_wear_analysis
        """
    )
    
    parser.add_argument('--output_dir', required=True, help='分析输出目录（包含features/wear_features.csv的目录）')
    parser.add_argument('--diagnosis_interval', type=int, default=100, 
                       help='帧诊断图采样间隔（默认100）')
    parser.add_argument('--roi_dir', default=None,
                       help='ROI图像目录（如果不指定，会自动推测为output_dir的上级目录的roi_imgs）')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("快速重新生成帧诊断图")
    print(f"{'='*80}")
    
    # 检查输出目录
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        print(f"错误: 输出目录不存在: {output_dir}")
        return 1
    
    # 检查特征文件
    features_csv = os.path.join(output_dir, 'features', 'wear_features.csv')
    if not os.path.exists(features_csv):
        print(f"错误: 特征文件不存在: {features_csv}")
        print("提示: 请先运行 coil_wear_analysis.py 生成特征数据")
        return 1
    
    # 读取特征数据
    print(f"\n正在读取特征数据: {features_csv}")
    df = pd.read_csv(features_csv, encoding='utf-8-sig')
    print(f"✓ 成功读取 {len(df)} 帧的特征数据")
    
    # 推测或验证ROI目录
    if args.roi_dir is None:
        # 尝试自动推测ROI目录
        parent_dir = os.path.dirname(output_dir)
        roi_dir = os.path.join(parent_dir, 'roi_imgs')
        
        if not os.path.exists(roi_dir):
            print(f"\n错误: 无法找到ROI目录")
            print(f"尝试的路径: {roi_dir}")
            print("请使用 --roi_dir 参数手动指定ROI图像目录")
            return 1
        
        print(f"✓ 自动找到ROI目录: {roi_dir}")
    else:
        roi_dir = os.path.abspath(args.roi_dir)
        if not os.path.exists(roi_dir):
            print(f"错误: 指定的ROI目录不存在: {roi_dir}")
            return 1
    
    # 初始化处理器
    print("\n初始化图像处理器...")
    preprocessor = ImagePreprocessor()
    feature_extractor = GeometryFeatureExtractor()
    visualizer = WearVisualizer(output_dir)
    print("✓ 初始化完成")
    
    # 创建诊断图目录
    diagnosis_dir = os.path.join(output_dir, 'visualizations', 'frame_diagnosis')
    ensure_dir(diagnosis_dir)
    
    # 选择要重新生成的帧
    frame_ids = df['frame_id'].values
    sample_indices = list(range(0, len(df), args.diagnosis_interval))
    
    print(f"\n开始重新生成帧诊断图...")
    print(f"采样间隔: 每 {args.diagnosis_interval} 帧")
    print(f"将生成 {len(sample_indices)} 张诊断图")
    
    success_count = 0
    fail_count = 0
    
    for idx in tqdm(sample_indices, desc="生成诊断图"):
        try:
            frame_id = int(frame_ids[idx])
            
            # 读取原图
            filepath = os.path.join(roi_dir, f'frame_{frame_id:06d}_roi.png')
            if not os.path.exists(filepath):
                print(f"\n警告: 图像文件不存在: {filepath}")
                fail_count += 1
                continue
            
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"\n警告: 无法读取图像: {filepath}")
                fail_count += 1
                continue
            
            # 预处理（需要获取preprocessed_data）
            preprocessed = preprocessor.process(image)
            if not preprocessed['success']:
                print(f"\n警告: 预处理失败 frame {frame_id}")
                fail_count += 1
                continue
            
            # 重新提取特征（确保包含所有最新特征，如 centerline_rms_roughness）
            try:
                features = feature_extractor.extract_features(preprocessed)
                features['frame_id'] = frame_id
            except Exception as e:
                print(f"\n警告: 特征提取失败 frame {frame_id}: {e}")
                fail_count += 1
                continue
            
            # 生成诊断图
            diagnosis_path = os.path.join(diagnosis_dir, f"frame_{frame_id:06d}_diagnosis.png")
            visualizer.visualize_single_frame_diagnosis(
                image, preprocessed, features, frame_id, diagnosis_path
            )
            success_count += 1
            
        except Exception as e:
            print(f"\n错误: 处理帧 {idx} 时出错: {e}")
            fail_count += 1
            continue
    
    # 统计信息
    print(f"\n{'='*80}")
    print("重新生成完成！")
    print(f"{'='*80}")
    print(f"成功生成: {success_count} 张")
    print(f"失败: {fail_count} 张")
    print(f"诊断图保存在: {diagnosis_dir}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


""" 该文件用于运行整个剪刀磨损检测流水线，包括以下步骤：
1. 提取视频帧
2. 提取ROI区域
3. 斑块分析
4. 毛刺分析
6. 时间序列图表
用到的脚本如下：
data_process/frame_extractor.py
data_process/roi_extractor.py
data_adagaus_density_curve/adagaus_density_analyzer.py
data_burr_density_curve/burr_density_analyzer.py
coil_wear_analysis.py
analyze_spot_temporal.py
tear_surface_burr_density_analyzer.py

"""
import os
import sys
import subprocess
from typing import List


def run(skips: List[str], video_path: str, output_dir: str, interval_seconds: int=5):
    """运行完整的剪刀磨损检测流水线
    
    Args:
        skips: 要跳过的步骤列表，如 ['step1', 'step2']
        video_path: 输入视频路径
        output_dir: 输出目录路径
        interval_seconds: 帧提取间隔（秒）
    """
    # 获取当前Python解释器路径（更可靠，跨平台兼容）
    python_exe = sys.executable
    
    images_dir = os.path.join(output_dir, 'images')
    roi_dir = os.path.join(output_dir, 'roi_imgs')
    adagaus_dir = os.path.join(output_dir, 'adagaus_density_curve')
    burr_dir = os.path.join(output_dir, 'burr_density_curve')
    coil_wear_dir = os.path.join(output_dir, 'coil_wear_analysis')
    spot_temporal_dir = os.path.join(output_dir, 'spot_temporal_analysis')
    tear_surface_burr_dir = os.path.join(output_dir, 'white_patch_test')

    print(f"\n{'='*80}")
    print(f"开始运行剪刀磨损检测流水线")
    print(f"Python解释器: {python_exe}")
    print(f"视频路径: {video_path}")
    print(f"输出目录: {output_dir}")
    print(f"跳过步骤: {skips if skips else '无'}")
    print(f"{'='*80}\n")

    # step1 - 提取视频帧
    if 'step1' not in skips:
        print(">>> Step 1: 提取视频帧...")
        subprocess.run([python_exe, 'data_process/frame_extractor.py', video_path, images_dir, str(interval_seconds)], check=True)
        print("✓ Step 1 完成\n")
    else:
        print("⊗ Step 1: 跳过（提取视频帧）\n")
    
    # step2 - 提取ROI区域
    if 'step2' not in skips:
        print(">>> Step 2: 提取ROI区域...")
        subprocess.run([python_exe, 'data_process/roi_extractor.py', images_dir, roi_dir], check=True)
        print("✓ Step 2 完成\n")
    else:
        print("⊗ Step 2: 跳过（提取ROI区域）\n")
    
    # step3 - Adagaus密度分析
    if 'step3' not in skips:
        print(">>> Step 3: Adagaus密度分析...")
        subprocess.run([python_exe, 'data_adagaus_density_curve/adagaus_density_analyzer.py', '--roi_dir', roi_dir, '--output_dir', adagaus_dir], check=True)
        print("✓ Step 3 完成\n")
    else:
        print("⊗ Step 3: 跳过（Adagaus密度分析）\n")
    
    # step4 - 毛刺密度分析
    if 'step4' not in skips:
        print(">>> Step 4: 毛刺密度分析...")
        subprocess.run([python_exe, 'data_burr_density_curve/burr_density_analyzer.py', '--roi_dir', roi_dir, '--output_dir', burr_dir], check=True)
        print("✓ Step 4 完成\n")
    else:
        print("⊗ Step 4: 跳过（毛刺密度分析）\n")
    
    # step5 - 钢卷磨损分析
    if 'step5' not in skips:
        print(">>> Step 5: 钢卷磨损分析...")
        subprocess.run([python_exe, 'coil_wear_analysis.py', '--roi_dir', roi_dir, '--output_dir', coil_wear_dir], check=True)
        print("✓ Step 5 完成\n")
    else:
        print("⊗ Step 5: 跳过（钢卷磨损分析）\n")
    
    # step6 - 斑点时序分析
    if 'step6' not in skips:
        print(">>> Step 6: 斑点时序分析...")
        subprocess.run([python_exe, 'analyze_spot_temporal.py', '--roi_dir', roi_dir, '--output_dir', spot_temporal_dir], check=True)
        print("✓ Step 6 完成\n")
    else:
        print("⊗ Step 6: 跳过（斑点时序分析）\n")
    
    # step7 - 撕裂面毛刺密度分析
    if 'step7' not in skips:
        print(">>> Step 7: 撕裂面毛刺密度分析...")
        subprocess.run([python_exe, 'tear_surface_burr_density_analyzer.py', '--roi_dir', roi_dir, '--output_dir', tear_surface_burr_dir], check=True)
        print("✓ Step 7 完成\n")
    else:
        print("⊗ Step 7: 跳过（撕裂面毛刺密度分析）\n")
    
    print(f"\n{'='*80}")
    print(f"流水线运行完成！")
    print(f"所有结果已保存到: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    skips=['step1', 'step2']
    video_path='/Users/aibee/hwp/wphu个人资料/baogang/data_baogang/20250820/Video_20250820124904881.avi'
    output_dir='data_video4_20250820124904881'
    run(skips, video_path, output_dir)
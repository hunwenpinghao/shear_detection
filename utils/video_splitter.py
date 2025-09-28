#!/usr/bin/env python3
"""
视频分割工具
将视频平均分割成指定数量的片段
"""

import os
import sys
import cv2
import argparse
from pathlib import Path


def get_video_info(video_path):
    """获取视频信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }


def split_video(input_path, output_dir, num_segments=5):
    """
    将视频平均分割成指定数量的片段
    
    Args:
        input_path (str): 输入视频路径
        output_dir (str): 输出目录
        num_segments (int): 分割段数
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入视频文件不存在: {input_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频信息
    print("正在分析视频...")
    video_info = get_video_info(input_path)
    
    print(f"视频信息:")
    print(f"  分辨率: {video_info['width']}x{video_info['height']}")
    print(f"  帧率: {video_info['fps']:.2f} FPS")
    print(f"  总帧数: {video_info['frame_count']}")
    print(f"  时长: {video_info['duration']:.2f} 秒")
    
    # 计算每段的帧数
    frames_per_segment = video_info['frame_count'] // num_segments
    print(f"每段帧数: {frames_per_segment}")
    
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {input_path}")
    
    # 获取视频编码器 - 使用XVID编码器保持AVI格式
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    try:
        for i in range(num_segments):
            print(f"正在处理第 {i+1}/{num_segments} 段...")
            
            # 计算当前段的起始和结束帧
            start_frame = i * frames_per_segment
            end_frame = start_frame + frames_per_segment
            
            # 最后一段包含所有剩余帧
            if i == num_segments - 1:
                end_frame = video_info['frame_count']
            
            # 设置输出文件名 - 保持AVI格式
            output_filename = f"segment_{i+1:02d}.avi"
            output_path = os.path.join(output_dir, output_filename)
            
            # 创建视频写入器
            out = cv2.VideoWriter(
                output_path, 
                fourcc, 
                video_info['fps'], 
                (video_info['width'], video_info['height'])
            )
            
            # 跳转到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 写入当前段的帧
            current_frame = start_frame
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                current_frame += 1
            
            out.release()
            
            # 计算时间信息
            start_time = start_frame / video_info['fps']
            end_time = end_frame / video_info['fps']
            duration = end_time - start_time
            
            print(f"  段 {i+1}: 帧 {start_frame}-{end_frame-1}, 时间 {start_time:.2f}s-{end_time:.2f}s, 时长 {duration:.2f}s")
            print(f"  保存到: {output_path}")
    
    finally:
        cap.release()
    
    print(f"\n视频分割完成！共生成 {num_segments} 个片段，保存在: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将视频平均分割成指定数量的片段')
    parser.add_argument('input_video', help='输入视频路径')
    parser.add_argument('-o', '--output', help='输出目录', default=None)
    parser.add_argument('-n', '--num_segments', type=int, default=5, help='分割段数 (默认: 5)')
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output:
        output_dir = args.output
    else:
        # 从输入文件名生成输出目录名
        input_name = Path(args.input_video).stem
        output_dir = f"data_{input_name}"
    
    try:
        split_video(args.input_video, output_dir, args.num_segments)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

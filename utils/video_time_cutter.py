#!/usr/bin/env python3
"""
视频时间段切割工具
从视频中提取指定时间段的片段
"""

import os
import sys
import cv2
import argparse
from pathlib import Path


def time_to_seconds(time_str):
    """将时间字符串转换为秒数
    
    Args:
        time_str (str): 时间字符串，格式为 "HH:MM:SS" 或 "H:MM:SS"
    
    Returns:
        float: 秒数
    """
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"时间格式错误: {time_str}，应为 HH:MM:SS 格式")
    
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_time(seconds):
    """将秒数转换为时间字符串
    
    Args:
        seconds (float): 秒数
    
    Returns:
        str: 时间字符串 HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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


def cut_video_by_time(input_path, output_path, start_time, end_time):
    """
    从视频中提取指定时间段的片段
    
    Args:
        input_path (str): 输入视频路径
        output_path (str): 输出视频路径
        start_time (str): 开始时间，格式 "HH:MM:SS"
        end_time (str): 结束时间，格式 "HH:MM:SS"
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入视频文件不存在: {input_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取视频信息
    print("正在分析视频...")
    video_info = get_video_info(input_path)
    
    print(f"视频信息:")
    print(f"  分辨率: {video_info['width']}x{video_info['height']}")
    print(f"  帧率: {video_info['fps']:.2f} FPS")
    print(f"  总帧数: {video_info['frame_count']}")
    print(f"  总时长: {seconds_to_time(video_info['duration'])}")
    
    # 转换时间
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    
    print(f"\n切割时间段:")
    print(f"  开始时间: {start_time} ({start_seconds:.2f}秒)")
    print(f"  结束时间: {end_time} ({end_seconds:.2f}秒)")
    print(f"  片段时长: {end_seconds - start_seconds:.2f}秒")
    
    # 检查时间范围
    if start_seconds < 0:
        raise ValueError("开始时间不能为负数")
    if end_seconds > video_info['duration']:
        raise ValueError(f"结束时间 {end_time} 超过视频总时长 {seconds_to_time(video_info['duration'])}")
    if start_seconds >= end_seconds:
        raise ValueError("开始时间必须小于结束时间")
    
    # 计算帧数
    start_frame = int(start_seconds * video_info['fps'])
    end_frame = int(end_seconds * video_info['fps'])
    
    print(f"  开始帧: {start_frame}")
    print(f"  结束帧: {end_frame}")
    print(f"  总帧数: {end_frame - start_frame}")
    
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {input_path}")
    
    # 获取视频编码器 - 使用XVID编码器保持AVI格式
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # 创建视频写入器
    out = cv2.VideoWriter(
        output_path, 
        fourcc, 
        video_info['fps'], 
        (video_info['width'], video_info['height'])
    )
    
    try:
        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"\n正在提取视频片段...")
        current_frame = start_frame
        total_frames = end_frame - start_frame
        processed_frames = 0
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"警告: 在第 {current_frame} 帧处读取失败")
                break
            
            out.write(frame)
            current_frame += 1
            processed_frames += 1
            
            # 显示进度
            if processed_frames % 100 == 0:
                progress = (processed_frames / total_frames) * 100
                print(f"  进度: {processed_frames}/{total_frames} ({progress:.1f}%)")
    
    finally:
        cap.release()
        out.release()
    
    print(f"\n视频切割完成！")
    print(f"输出文件: {output_path}")
    print(f"实际处理帧数: {processed_frames}")
    print(f"实际时长: {processed_frames / video_info['fps']:.2f}秒")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从视频中提取指定时间段的片段')
    parser.add_argument('input_video', help='输入视频路径')
    parser.add_argument('start_time', help='开始时间 (HH:MM:SS)')
    parser.add_argument('end_time', help='结束时间 (HH:MM:SS)')
    parser.add_argument('-o', '--output', help='输出文件路径', default=None)
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        # 从输入文件名生成输出文件名
        input_name = Path(args.input_video).stem
        output_path = f"data_{input_name}/segment_one_cycle.avi"
    
    try:
        cut_video_by_time(args.input_video, output_path, args.start_time, args.end_time)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

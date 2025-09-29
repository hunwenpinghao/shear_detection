"""
视频抽帧工具
主要功能：
1. 将视频按每5秒抽取一帧
2. 保存到指定目录并按照帧数命名
3. 支持时间顺序排列
"""

import cv2
import os
import time
import numpy as np
from typing import Optional, Tuple
import sys
from tqdm import tqdm


class FrameExtractor:
    """视频抽帧器"""
    
    def __init__(self):
        self.video_cap = None
        self.fps = None
        self.frame_count = None
        self.duration = None
    
    def extract_frames_from_video(self, video_path: str, output_dir: str, 
                                  interval_seconds: float = 5.0, 
                                  image_format: str = 'jpg') -> bool:
        """
        从视频中按指定间隔抽取帧
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出图像目录路径
            interval_seconds: 抽帧间隔（秒），默认5秒
            image_format: 图像格式，默认jpg
            
        Returns:
            是否成功抽取帧
        """
        try:
            # 检查视频文件是否存在
            if not os.path.exists(video_path):
                print(f"错误：视频文件不存在 - {video_path}")
                return False
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 打开视频文件
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                print(f"错误：无法打开视频文件 - {video_path}")
                return False
            
            # 获取视频信息
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0
            
            print(f"视频信息：")
            print(f"  - 帧率: {self.fps:.2f} fps")
            print(f"  - 总帧数: {self.frame_count}")
            print(f"  - 时长: {self.duration:.2f} 秒")
            print(f"  - 抽帧间隔: {interval_seconds} 秒")
            # 计算预计抽取帧数
            expected_frames = int(self.duration / interval_seconds) + 1
            print(f"  - 预计抽取帧数: ~{expected_frames} 帧")
            
            # 计算抽帧间隔（帧数）
            frame_interval = int(self.fps * interval_seconds)
            if frame_interval <= 0:
                frame_interval = 1
                print(f"警告：计算的抽帧间隔过小，调整为每帧提取")
            
            print(f"  - 抽帧间隔（帧数）: {frame_interval}")
            
            frame_counter = 0
            saved_frame_number = 0
            success = True
            
            # 创建进度条
            progress_bar = tqdm(total=expected_frames, desc="抽帧进度", unit="帧")
            
            while success:
                # 设置视频位置到指定帧
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
                
                # 读取帧
                success, frame = self.video_cap.read()
                
                if success:
                    # 计算当前时间（秒）
                    current_time = frame_counter / self.fps
                    
                    # 生成输出文件名
                    filename = f"frame_{saved_frame_number:06d}.{image_format}"
                    output_path = os.path.join(output_dir, filename)
                    
                    # 保存帧
                    save_success = cv2.imwrite(output_path, frame)
                    
                    if save_success:
                        saved_frame_number += 1
                        # 更新进度条
                        progress_bar.update(1)
                        progress_bar.set_postfix({
                            '当前帧': f"{saved_frame_number:06d}",
                            '时间': f"{current_time:.1f}s"
                        })
                    else:
                        print(f"警告：无法保存帧到 {output_path}")
                    
                    # 跳到下一个抽帧位置
                    frame_counter += frame_interval
                    
                    # 检查是否超出视频长度
                    if frame_counter >= self.frame_count:
                        break
                else:
                    # 到达视频末尾
                    break
            
            # 关闭进度条
            progress_bar.close()
            
            # 关闭视频
            self.video_cap.release()
            
            print(f"\n抽帧完成！")
            print(f"成功保存 {saved_frame_number} 帧到：{output_dir}")
            
            return True
            
        except Exception as e:
            print(f"抽帧过程中发生错误：{str(e)}")
            if 'progress_bar' in locals():
                progress_bar.close()
            if self.video_cap:
                self.video_cap.release()
            return False
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频基本信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含视频信息的字典
        """
        try:
            if not os.path.exists(video_path):
                return {"error": "视频文件不存在"}
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "无法打开视频文件"}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration
            }
            
        except Exception as e:
            return {"error": str(e)}


def main(video_path: str=None, output_dir: str=None, interval_seconds: float=5.0):
    """主函数 - 处理指定的视频文件"""
    # 设置路径
    if video_path is None:
        video_path = "/Users/aibee/hwp/wphu个人资料/baogang/data_baogang/20250822/Video_20250821152112032.avi"
    if output_dir is None:
        output_dir = "data_video_20250821152112032/images"
    
    # 检查视频文件存在性
    if not os.path.exists(video_path):
        raise ValueError(f"错误：视频文件不存在 - {video_path}")
    
    # 创建帧提取器
    extractor = FrameExtractor()
    
    # 获取视频信息
    print("正在分析视频...")
    video_info = extractor.get_video_info(video_path)
    
    if "error" in video_info:
        print(f"无法分析视频：{video_info['error']}")
        return
    
    print(f"视频信息：")
    print(f"  - 分辨率: {video_info['width']}x{video_info['height']}")
    print(f"  - 帧率: {video_info['fps']:.2f} fps")
    print(f"  - 总帧数: {video_info['frame_count']}")
    print(f"  - 时长: {video_info['duration']:.2f} 秒")
    print()
    
    # 执行抽帧
    print(f"开始视频抽帧（间隔: {interval_seconds}秒）...")
    success = extractor.extract_frames_from_video(
        video_path=video_path,
        output_dir=output_dir,
        interval_seconds=interval_seconds,
        image_format='jpg'
    )
    
    if success:
        print("视频抽帧完成！")
    else:
        print("视频抽帧失败！")


if __name__ == "__main__":
    video_path = None
    output_dir = None
    interval_seconds = 5.0
    if len(sys.argv) > 1:
        video_path = str(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = str(sys.argv[2])
    if len(sys.argv) > 3:
        interval_seconds = float(sys.argv[3])
    main(video_path, output_dir, interval_seconds)
    

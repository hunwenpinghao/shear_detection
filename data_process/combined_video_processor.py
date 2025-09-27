#!/usr/bin/env python3
"""
组合视频处理器
主要功能：
1. 将ROI图、斑块图、毛刺图按帧横向拼接
2. 生成包含三种图像类型的组合视频
"""

import cv2
import numpy as np
import os
import glob
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import time
import platform
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("警告：PIL (Pillow) 未安装，中文字符支持可能受限")

class CombinedVideoProcessor:
    """组合视频处理器"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化组合视频处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.roi_dir = os.path.join(data_dir, 'roi_imgs')
        self.bankuai_dir = os.path.join(data_dir, 'bankuai')
        self.burr_dir = os.path.join(data_dir, 'burr_imgs')
        
    def get_frame_info(self, filename: str) -> Optional[int]:
        """
        从文件名提取帧号
        
        Args:
            filename: 文件名
            
        Returns:
            帧号或None
        """
        try:
            # 支持不同的文件命名格式
            if 'frame_' in filename:
                parts = filename.split('_')
                for part in parts:
                    if part.isdigit():
                        return int(part)
            return None
        except:
            return None
    
    def load_frame_images(self, frame_num: int) -> Dict[str, np.ndarray]:
        """
        加载特定帧的三种图像（ROI、斑块、毛刺）
        
        Args:
            frame_num: 帧号
            
        Returns:
            三种图像的字典
        """
        result = {'roi': None, 'bankuai': None, 'burr': None}
        
        # 构建可能的文件名格式
        frame_prefix = f"frame_{frame_num:06d}"
        
        # 尝试不同目录的文件
        directories = {
            'roi': self.roi_dir,
            'bankuai': self.bankuai_dir,
            'burr': self.burr_dir
        }
        
        for img_type, directory in directories.items():
            if not os.path.exists(directory):
                continue
                
            # 搜索匹配的文件
            pattern = os.path.join(directory, f"{frame_prefix}_*.png")
            files = glob.glob(pattern)
            
            # 如果没找到，尝试其他可能的命名格式
            if not files:
                for filename in os.listdir(directory):
                    if f"_{frame_num:06d}_" in filename or f"_{frame_num:05d}_" in filename:
                        files = [os.path.join(directory, filename)]
                        break
            
            if files:
                img_path = files[0]
                img = cv2.imread(img_path)
                if img is not None:
                    result[img_type] = img
                    # 转换为RGB以便后续处理
                    result[img_type] = cv2.cvtColor(result[img_type], cv2.COLOR_BGR2RGB)
        
        return result
    
    def resize_images_to_same_height(self, img_dict: Dict[str, np.ndarray], target_height: int = None) -> Dict[str, np.ndarray]:
        """
        将所有图像调整为相同高度
        
        Args:
            img_dict: 图像字典
            target_height: 目标高度，如果为None则使用最大高度
            
        Returns:
            调整后的图像字典
        """
        # 如果没有任何有效图像，返回空字典
        valid_images = {k: v for k, v in img_dict.items() if v is not None}
        if not valid_images:
            return img_dict
        
        # 计算目标高度
        if target_height is None:
            heights = [img.shape[0] for img in valid_images.values()]
            target_height = max(heights)
        
        # 调整所有图像
        for key, img in valid_images.items():
            if img.shape[0] != target_height:
                # 计算缩放比例，保持长宽比
                scale = target_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                
                img_resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
                img_dict[key] = img_resized
        
        return img_dict
    
    def concatenate_horizontal(self, img_dict: Dict[str, np.ndarray], spacing: int = 10) -> np.ndarray:
        """
        横向拼接图像
        
        Args:
            img_dict: 图像字典，期望包含 roi, bankuai, burr 三种类型
            spacing: 图像间的间距（像素）
            
        Returns:
            拼接后的图像
        """
        valid_images = {}
        type_names = []
        
        # 过滤有效图像并记录顺序
        type_order = ['roi', 'bankuai', 'burr']
        for img_type in type_order:
            if img_type in img_dict and img_dict[img_type] is not None:
                valid_images[img_type] = img_dict[img_type]
                type_names.append(img_type)
        
        if not valid_images:
            # 创建一个空白图像
            return np.zeros((100, 300, 3), dtype=np.uint8)
        
        # 统一高度
        target_height = min(img.shape[0] for img in valid_images.values())
        resized_images = []
        
        for img_type in type_names:
            img = valid_images[img_type]
            if img.shape[0] != target_height:
                scale = target_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                img_resized = cv2.resize(img, (new_width, target_height))
                resized_images.append(img_resized)
            else:
                resized_images.append(img)
        
        # 计算总宽度和创建画布
        total_width = sum(img.shape[1] for img in resized_images) + spacing * (len(resized_images) - 1)
        canvas = np.ones((target_height, total_width, 3), dtype=np.uint8) * 255  # 白色背景
        
        # 横向拼接图像
        x_offset = 0
        for i, img in enumerate(resized_images):
            canvas[:, x_offset:x_offset + img.shape[1]] = img
            x_offset += img.shape[1] + (spacing if i < len(resized_images) - 1 else 0)
        
        return canvas
    
    def add_labels_to_image(self, combined_img: np.ndarray, frame_info: Dict[str, Any]) -> np.ndarray:
        """
        为拼接图像添加标签（支持中文字符显示）
        
        Args:
            combined_img: 拼接后的图像
            frame_info: 帧信息
            
        Returns:
            带标签的图像
        """
        if not PIL_AVAILABLE:
            # 回退到OpenCV方案
            return self._add_labels_opencv(combined_img, frame_info)
        
        # 在图像顶部添加标题
        height, width = combined_img.shape[:2]
        
        # 创建标题区域
        title_height = 50
        labeled_img = np.zeros((height + title_height + 10, width, 3), dtype=np.uint8)
        labeled_img.fill(255)  # 白色背景
        
        # 复制原图到底部
        labeled_img[title_height + 10:, :] = combined_img
        
        # 转换为PIL图像以便处理中文字符
        pil_img = Image.fromarray(labeled_img)
        draw = ImageDraw.Draw(pil_img)
        
        # 尝试加载中文字体
        try:
            # 根据操作系统选择合适的中文字体
            if platform.system() == "Darwin":  # macOS
                font_paths = [
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/Arial Unicode MS.ttf",
                    "/System/Library/Fonts/STHeiti Medium.ttc"
                ]
            elif platform.system() == "Windows":
                font_paths = [
                    "C:/Windows/Fonts/simsun.ttc",  # 宋体
                    "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
                    "C:/Windows/Fonts/simhei.ttf"  # 黑体
                ]
            else:  # Linux
                font_paths = [
                    "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
                ]
            
            font_obj = None
            for font_path in font_paths:
                try:
                    font_obj = ImageFont.truetype(font_path, 20)
                    break
                except:
                    continue
            
            if font_obj is None:
                # 如果找不到中文字体，使用默认字体
                font_obj = ImageFont.load_default()
                print("警告：未找到中文字体，中文显示可能不正确")
                
        except Exception as e:
            print(f"字体加载失败: {e}")
            font_obj = ImageFont.load_default()
            
        # 获取较小的字体用于标签
        small_font = font_obj  # 可以调整尺寸
        try:
            if hasattr(font_obj, 'path'):
                small_font = ImageFont.truetype(font_obj.path, 16)
        except:
            small_font = font_obj
        
        # 添加帧号
        frame_num = frame_info.get('frame_num', 0)
        title_text = f"Frame {frame_num:06d}"
        
        # 计算文字位置
        bbox = draw.textbbox((0, 0), title_text, font=font_obj)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        
        # 绘制标题
        draw.text((x, 15), title_text, fill=(0, 0, 0), font=font_obj)
        
        # 添加分区域标签 - 支持中文
        roi_width = combined_img.shape[1] // 3
        segment_labels = ['ROI', 'SPOTS (斑块)', 'BURRS (毛刺)']
        
        for i, label in enumerate(segment_labels):
            bbox = draw.textbbox((0, 0), label, font=small_font)
            label_width = bbox[2] - bbox[0]
            x_pos = i * roi_width + (roi_width - label_width) // 2
            y_pos = 5
            
            draw.text((x_pos, y_pos), label, fill=(0, 0, 0), font=small_font)
        
        # 转换回numpy数组
        labeled_img = np.array(pil_img)
        
        return labeled_img
    
    def _add_labels_opencv(self, combined_img: np.ndarray, frame_info: Dict[str, Any]) -> np.ndarray:
        """
        使用OpenCV添加标签（不支持中文）
        """
        # 在图像顶部添加标题
        height, width = combined_img.shape[:2]
        
        # 创建标题区域
        title_height = 50
        labeled_img = np.zeros((height + title_height + 10, width, 3), dtype=np.uint8)
        labeled_img.fill(255)  # 白色背景
        
        # 复制原图到底部
        labeled_img[title_height + 10:, :] = combined_img
        
        # 添加帧号
        frame_num = frame_info.get('frame_num', 0)
        title_text = f"Frame {frame_num:06d}"
        
        # OpenCV 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 0, 0)  # 黑色
        thickness = 2
        
        text_size = cv2.getTextSize(title_text, font, font_scale, thickness)[0]
        x = (width - text_size[0]) // 2
        
        cv2.putText(labeled_img, title_text, (x, 30), font, font_scale, color, thickness)
        
        # 添加分区域标签 - 只用英文或拼音
        roi_width = combined_img.shape[1] // 3
        segment_labels = ['ROI', 'SPOTS', 'BURRS']  # 去掉中文，避免乱码
        
        for i, label in enumerate(segment_labels):
            x_pos = i * roi_width + (roi_width - cv2.getTextSize(label, font, 0.5, 1)[0][0]) // 2
            y_pos = 10
            cv2.putText(labeled_img, label, (x_pos, y_pos), font, 0.5, color, 1)
        
        return labeled_img
    
    def process_all_frames(self, output_video_path: str, fps: float = 2.39) -> Dict[str, Any]:
        """
        处理所有帧并生成组合视频
        
        Args:
            output_video_path: 输出视频路径
            fps: 视频帧率
            
        Returns:
            处理结果
        """
        try:
            print("=== 开始处理组合视频生成 ===")
            
            # 获取所有帧号
            frame_numbers = []
            
            # 从不同目录收集帧号
            for directory in [self.roi_dir, self.bankuai_dir, self.burr_dir]:
                if not os.path.exists(directory):
                    continue
                    
                files = [f for f in os.listdir(directory) if f.endswith('.png')]
                for filename in files:
                    frame_num = self.get_frame_info(filename)
                    if frame_num is not None:
                        frame_numbers.append(frame_num)
            
            if not frame_numbers:
                return {'success': False, 'error': '未找到任何有效的帧文件'}
            
            # 去重并排序
            frame_numbers = sorted(list(set(frame_numbers)))
            print(f"找到 {len(frame_numbers)} 个帧")
            
            # 处理第一帧以获取输出尺寸
            first_frame_data = self.load_frame_images(frame_numbers[0])
            first_frame_data = self.resize_images_to_same_height(first_frame_data)
            first_combined = self.concatenate_horizontal(first_frame_data)
            first_labeled = self.add_labels_to_image(first_combined, {'frame_num': frame_numbers[0]})
            
            # 创建视频写入器
            height, width = first_labeled.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)
            
            if not video_writer.isOpened():
                return {'success': False, 'error': '无法创建视频写入器'}
            
            print(f"视频输出尺寸: {width}x{height}")
            print(f"开始处理 {len(frame_numbers)} 帧...")
            
            success_count = 0
            
            for frame_num in tqdm(frame_numbers, desc="生成组合视频"):
                try:
                    # 加载帧图像
                    frame_data = self.load_frame_images(frame_num)
                    frame_data = self.resize_images_to_same_height(frame_data)
                    
                    # 限制大小并拼接
                    if any(img is not None for img in frame_data.values()):
                        combined_img = self.concatenate_horizontal(frame_data)
                        labeled_img = self.add_labels_to_image(combined_img, {'frame_num': frame_num})
                        
                        # 转换回BGR格式用于写入视频
                        labeled_img_bgr = cv2.cvtColor(labeled_img, cv2.COLOR_RGB2BGR)
                        video_writer.write(labeled_img_bgr)
                        success_count += 1
                    else:
                        print(f"警告：帧 {frame_num} 无有效图像数据")
                        
                except Exception as e:
                    print(f"处理帧 {frame_num} 时出错: {e}")
                    continue
            
            # 释放视频写入器
            video_writer.release()
            
            # 检查结果
            if os.path.exists(output_video_path):
                file_size = os.path.getsize(output_video_path)
                if file_size > 0:
                    print(f"\n组合视频生成成功: {output_video_path}")
                    print(f"成功处理帧数: {success_count}/{len(frame_numbers)}")
                    print(f"文件大小: {file_size / (1024 * 1024):.2f} MB")
                    
                    return {
                        'success': True,
                        'video_path': output_video_path,
                        'total_frames': len(frame_numbers),
                        'processed_frames': success_count,
                        'resolution': f"{width}x{height}",
                        'file_size_mb': file_size / (1024 * 1024),
                        'fps': fps
                    }
                else:
                    return {'success': False, 'error': '生成的视频文件为空'}
            else:
                return {'success': False, 'error': '视频文件未生成'}
                
        except Exception as e:
            return {'success': False, 'error': f'处理过程中出错: {str(e)}'}
    
    def process_single_frame(self, frame_num: int, output_path: str) -> Dict[str, Any]:
        """
        处理单帧的组合图像
        
        Args:
            frame_num: 帧号
            output_path: 输出路径
            
        Returns:
            处理结果
        """
        try:
            # 加载帧图像
            frame_data = self.load_frame_images(frame_num)
            frame_data = self.resize_images_to_same_height(frame_data)
            
            # 拼接图像
            combined_img = self.concatenate_horizontal(frame_data)
            labeled_img = self.add_labels_to_image(combined_img, {'frame_num': frame_num})
            
            # 保存图像
            labeled_img_bgr = cv2.cvtColor(labeled_img, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(output_path, labeled_img_bgr)
            
            if success:
                return {
                    'success': True,
                    'output_path': output_path,
                    'frame_num': frame_num,
                    'image_shapes': {k: v.shape if v is not None else None for k, v in frame_data.items()}
                }
            else:
                return {'success': False, 'error': f'无法保存图像到: {output_path}'}
                
        except Exception as e:
            return {'success': False, 'error': f'处理单帧时出错: {str(e)}'}


def main():
    """主函数"""
    # 初始化组合视频处理器
    processor = CombinedVideoProcessor()
    
    # 设置输出路径
    output_video_path = "data/组合视频_(ROI_斑块_毛刺).mp4"
    
    print("=== ROI图+斑块图+毛刺图 横向拼接视频生成 ===")
    print(f"ROI图像目录: {processor.roi_dir}")
    print(f"斑块图像目录: {processor.bankuai_dir}")
    print(f"毛刺图像目录: {processor.burr_dir}")
    print(f"输出视频路径: {output_video_path}")
    
    # 处理所有帧生成组合视频
    result = processor.process_all_frames(output_video_path, fps=2.39)
    
    if result['success']:
        print(f"\n✅ 组合视频生成成功！")
        print(f"📹 视频文件: {result['video_path']}")
        print(f"📊 帧数: {result['processed_frames']}/{result['total_frames']}")
        print(f"🔍 分辨率: {result['resolution']}")
        print(f"💾 大小: {result['file_size_mb']:.2f} MB")
        print(f"⏱️ 帧率: {result['fps']}")
    else:
        print(f"❌ 组合视频生成失败: {result['error']}")


if __name__ == "__main__":
    main()

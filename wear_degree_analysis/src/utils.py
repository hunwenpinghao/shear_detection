"""
工具函数模块
提供通用的图像处理和数据管理功能
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_json(data: Dict, filepath: str) -> None:
    """
    保存数据为JSON格式
    
    Args:
        data: 要保存的字典数据
        filepath: 保存路径
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict:
    """
    从JSON文件加载数据
    
    Args:
        filepath: JSON文件路径
        
    Returns:
        加载的字典数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def smooth_curve(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    使用移动平均平滑曲线
    
    Args:
        data: 输入数据序列
        window_size: 窗口大小
        
    Returns:
        平滑后的数据
    """
    if len(data) < window_size:
        return data
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='valid')
    
    # 补齐边界
    pad_left = (window_size - 1) // 2
    pad_right = window_size - 1 - pad_left
    result = np.concatenate([
        np.full(pad_left, smoothed[0]),
        smoothed,
        np.full(pad_right, smoothed[-1])
    ])
    
    return result


def compute_trend_slope(data: np.ndarray) -> float:
    """
    计算数据的线性趋势斜率
    
    Args:
        data: 输入数据序列
        
    Returns:
        线性拟合的斜率
    """
    if len(data) < 2:
        return 0.0
    
    x = np.arange(len(data))
    # 使用最小二乘法拟合
    coeffs = np.polyfit(x, data, 1)
    return float(coeffs[0])


def normalize_array(arr: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    归一化数组到[0, 1]范围
    
    Args:
        arr: 输入数组
        epsilon: 防止除零的小常数
        
    Returns:
        归一化后的数组
    """
    arr_min = arr.min()
    arr_max = arr.max()
    
    if arr_max - arr_min < epsilon:
        return np.zeros_like(arr)
    
    return (arr - arr_min) / (arr_max - arr_min)


def extract_frame_number(filename: str) -> int:
    """
    从文件名中提取帧编号
    例如: 'frame_000025_roi.png' -> 25
    
    Args:
        filename: 文件名
        
    Returns:
        帧编号
    """
    import re
    match = re.search(r'frame_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def get_frame_files(data_dir: str, max_frames: int = 100) -> List[str]:
    """
    获取指定数量的帧文件路径，按帧号排序
    
    Args:
        data_dir: 数据目录
        max_frames: 最大帧数
        
    Returns:
        排序后的文件路径列表
    """
    files = []
    for i in range(max_frames):
        filename = f"frame_{i:06d}_roi.png"
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            files.append(filepath)
        else:
            print(f"警告: 文件不存在 {filepath}")
    
    return files


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    计算数据的基本统计量
    
    Args:
        data: 输入数据
        
    Returns:
        包含统计量的字典
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75))
    }


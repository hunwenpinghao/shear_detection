#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
黑色条状物计数（针对 Adagaus 二值图）

思路：
1) 读取二值图（白=主体撕裂面，黑=背景/条状物）。
2) 逐行找到白色主体的最左边界列，向左扩展一个带状宽度 band_width（默认 30px），
   将该带内的黑色像素作为候选“黑色条状物”区域。
3) 对候选区域做连通域计数（8 连通），并可选按面积过滤（避免噪声点）。
4) 输出计数与可视化叠图，便于人工核验。
5) 扫描法：基于 Final Mask 的左右边缘，按行在左侧一定比例(如 0.3)处形成“弯曲扫描线”，输出梯度曲线与变化点。

使用：
python test_count_black_strips.py [image_path] [--band 30] [--min_area 10] [--max_area 5000] [--scan]
"""

import os
import sys
import cv2
import numpy as np
import argparse
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.ndimage import grey_closing


def setup_chinese_font():
    """设置中英文通用字体，尽量避免方框/缺字。
    优先顺序：Arial Unicode MS / Noto Sans CJK SC / PingFang / Microsoft YaHei / SimHei / WenQuanYi / DejaVu Sans。
    同时设置 font.family 与三大类字体回退。
    """
    try:
        from matplotlib import font_manager as fm
        available = {f.name for f in fm.fontManager.ttflist}
        preferred = [
            'Arial Unicode MS',
            'Noto Sans CJK SC',
            'PingFang SC',
            'Hiragino Sans GB',
            'STHeiti',
            'Microsoft YaHei',
            'SimHei',
            'WenQuanYi Micro Hei',
            'DejaVu Sans',
        ]
        chosen = [f for f in preferred if f in available]
        if not chosen:
            chosen = ['DejaVu Sans']

        # 设置家族与回退
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = chosen + ['DejaVu Sans']
        plt.rcParams['font.serif'] = chosen + ['DejaVu Serif']
        plt.rcParams['font.monospace'] = ['Menlo', 'Consolas', 'Courier New', 'DejaVu Sans Mono']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        # 最小兜底
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

def ensure_binary(gray: np.ndarray) -> np.ndarray:
    """确保输入为 0/255 二值图。"""
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    unique_vals = np.unique(gray)
    if unique_vals.size <= 2 and set(unique_vals.tolist()).issubset({0, 255}):
        return gray
    # 自动阈值：白色主体（中央竖向区域）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def build_left_band_mask(white_mask: np.ndarray, band_width: int) -> np.ndarray:
    """按行构建主体左侧的带状区域 mask。
    white_mask: 255 表示主体；其他为 0。
    返回：bool 数组，True 表示位于左侧带区域。
    """
    h, w = white_mask.shape
    band = np.zeros((h, w), dtype=bool)
    # 对每一行，找到该行白色主体的最左列索引
    for y in range(h):
        row = white_mask[y]
        xs = np.where(row == 255)[0]
        if xs.size == 0:
            continue
        left = int(xs.min())
        x0 = max(0, left - band_width)
        if x0 < left:
            band[y, x0:left] = True
    return band


def count_black_strips(
    binary_img: np.ndarray,
    band_width: int = 30,
    min_area: int = 10,
    max_area: int = 5000,
    h_close: int = 7,
    v_erode: int = 7,
    open_size: int = 3,
):
    """统计黑色条状物数量，返回计数与可视化掩码。
    参数：
      - binary_img: 0/255 二值图（白=主体，黑=背景）。
      - band_width: 主体左侧带宽（像素）。
      - min_area/max_area: 计数的面积过滤阈值。
    返回：(count, components_mask, labels)
    """
    white = (binary_img == 255).astype(np.uint8) * 255
    black = (binary_img == 0).astype(np.uint8)

    # 左带区域
    left_band = build_left_band_mask(white, band_width=band_width)
    candidates = np.logical_and(left_band, black.astype(bool)).astype(np.uint8)

    # 形态学流程：横向闭 → 纵向腐蚀 → 小开运算
    if h_close > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, h_close), 1))
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, k_close, iterations=1)
    if v_erode > 1:
        k_verode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, v_erode)))
        candidates = cv2.erode(candidates, k_verode, iterations=1)
    if open_size > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, k_open, iterations=1)

    # 连通域
    num_labels, labels = cv2.connectedComponents(candidates, connectivity=8)

    # 面积过滤并重建掩码
    components_mask = np.zeros_like(candidates)
    kept = 0
    for label_id in range(1, num_labels):
        area = int(np.sum(labels == label_id))
        if area < min_area or area > max_area:
            continue
        components_mask[labels == label_id] = 255
        kept += 1

    return kept, components_mask, labels


def estimate_scan_column_from_final_mask(binary_img: np.ndarray, frac: float = 0.25) -> int:
    """基于 Final Filled Mask 的白色主体区域计算扫描列位置。
    先计算 Final Filled Mask，然后在其白色主体区域内按 frac 比例确定扫描列。
    """
    # 计算 Final Filled Mask（模拟 AdagausExtractor 的逻辑）
    _, otsu_binary = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 检测左右边界（模拟 generate_final_mask 的逻辑）
    def detect_left_right_edges(binary):
        h, w = binary.shape
        left_edges = []
        right_edges = []
        
        for y in range(h):
            row = binary[y]
            white_pixels = np.where(row == 255)[0]
            if len(white_pixels) > 0:
                left_edges.append(white_pixels[0])
                right_edges.append(white_pixels[-1])
            else:
                left_edges.append(w)
                right_edges.append(-1)
        
        return np.array(left_edges), np.array(right_edges)
    
    def create_filled_mask(left_edges, right_edges, shape):
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for y in range(h):
            left = max(0, left_edges[y])
            right = min(w, right_edges[y] + 1)
            if left < right:
                mask[y, left:right] = 255
        
        return mask
    
    left_edges, right_edges = detect_left_right_edges(otsu_binary)
    final_mask = create_filled_mask(left_edges, right_edges, binary_img.shape)
    
    # 在 Final Mask 的白色主体区域内计算扫描列
    h, w = final_mask.shape
    xs = []
    for y in range(h):
        row = final_mask[y]
        cols = np.where(row == 255)[0]
        if cols.size == 0:
            continue
        left, right = int(cols.min()), int(cols.max())
        width = right - left + 1
        x = int(left + frac * width)
        x = min(max(0, x), w - 1)
        xs.append(x)
    
    if not xs:
        return w // 4
    return int(np.median(xs))


def _smooth_edges(left_edges: np.ndarray, right_edges: np.ndarray, w: int, sigma: float) -> tuple:
    """对上下方向平滑左右边缘，带插值填补无效行。"""
    h = left_edges.shape[0]
    yy = np.arange(h)
    # 有效性判断
    valid = (left_edges < w) & (right_edges >= 0) & (right_edges > left_edges)
    if not np.any(valid):
        return left_edges, right_edges
    # 插值填补
    left_filled = np.interp(yy, yy[valid], left_edges[valid].astype(float))
    right_filled = np.interp(yy, yy[valid], right_edges[valid].astype(float))
    # 高斯平滑
    if sigma and sigma > 0:
        left_sm = gaussian_filter1d(left_filled, sigma=sigma, mode='nearest')
        right_sm = gaussian_filter1d(right_filled, sigma=sigma, mode='nearest')
    else:
        left_sm, right_sm = left_filled, right_filled
    # 裁剪
    left_sm = np.clip(np.rint(left_sm), 0, w - 1).astype(int)
    right_sm = np.clip(np.rint(right_sm), 0, w - 1).astype(int)
    return left_sm, right_sm


def count_by_vertical_scanline(binary_img: np.ndarray, frac: float = 0.25, smooth_ksize: int = 9, use_gradient: bool = True, edge_sigma: float = 5.0):
    """沿弯曲扫描线自上而下统计变化，返回(估计数量, 扫描列x, 可视化序列)。
    - 若 use_gradient=True：沿弯曲扫描线计算梯度，统计峰的上升沿数量/2（去连续）。
    - 否则：沿弯曲扫描线统计二值上升沿(0→1)次数（不除2）。
    """
    if smooth_ksize % 2 == 0:
        smooth_ksize += 1
    # 基于 Final Mask 计算左右边缘并平滑后确定弯曲扫描线
    _, otsu_binary = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = binary_img.shape
    left_edges = np.full(h, w, dtype=int)
    right_edges = np.full(h, -1, dtype=int)
    for y in range(h):
        xs = np.where(otsu_binary[y] == 255)[0]
        if xs.size:
            left_edges[y] = xs.min()
            right_edges[y] = xs.max()
    left_edges, right_edges = _smooth_edges(left_edges, right_edges, w, edge_sigma)
    
    # 计算弯曲扫描线的实际位置
    scan_xs = []
    for y in range(h):
        l, r = left_edges[y], right_edges[y]
        if r > l:
            x = int(l + frac * (r - l + 1))
            x = min(max(0, x), w - 1)
            scan_xs.append(x)
        else:
            scan_xs.append(-1)  # 无效行
    
    # 取中位扫描列作为参考
    valid_xs = [x for x in scan_xs if x >= 0]
    scan_x = int(np.median(valid_xs)) if valid_xs else w // 4

    if use_gradient:
        # 沿弯曲扫描线计算梯度
        sm = cv2.GaussianBlur(binary_img, (smooth_ksize, smooth_ksize), 0)
        dy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        
        # 沿弯曲扫描线采样梯度值
        g_curve = []
        for y in range(h):
            if scan_xs[y] >= 0:
                g_curve.append(np.abs(dy[y, scan_xs[y]]))
            else:
                g_curve.append(0.0)
        g_curve = np.array(g_curve)
        
        # 更敏感的阈值设置
        if np.max(g_curve) > 0:
            # 使用更低的阈值，确保能检测到边缘
            thr = float(np.percentile(g_curve[g_curve > 0], 70))  # 只对非零值计算分位点
            if thr <= 0:
                thr = np.max(g_curve) * 0.3  # 如果分位点太低，用最大值的30%
        else:
            thr = 0
        
        peaks = (g_curve >= thr).astype(np.uint8)
        
        # 调试信息
        print(f"梯度统计: max={np.max(g_curve):.2f}, mean={np.mean(g_curve):.2f}, thr={thr:.2f}")
        print(f"峰值数量: {np.sum(peaks)}")
        
        # 压缩连续段
        transitions = int(np.sum((peaks[1:] > 0) & (peaks[:-1] == 0))) + int(peaks[0] > 0)
        est = max(0, transitions // 2)
        return est, scan_x, g_curve
    else:
        # 沿弯曲扫描线采样二值值
        col_curve = []
        for y in range(h):
            if scan_xs[y] >= 0:
                col_curve.append(1 if binary_img[y, scan_xs[y]] == 0 else 0)  # 黑色=1
            else:
                col_curve.append(0)
        col_curve = np.array(col_curve, dtype=np.uint8)
        
        # 1D 中值滤波去抖
        if smooth_ksize > 1:
            col_curve = cv2.medianBlur((col_curve * 255).astype(np.uint8), smooth_ksize) // 255
        
        # 调试信息：黑色像素数与上升沿次数
        rising_edges = int(np.sum((col_curve[1:] == 1) & (col_curve[:-1] == 0)))
        print(f"二值统计: 黑色像素数={np.sum(col_curve)}, 上升沿次数={rising_edges}")
        
        est = max(0, rising_edges)
        return est, scan_x, col_curve


def detect_gradient_contour_from_right_edge(
    binary_img: np.ndarray, 
    gradient_threshold: float = 0.3,
    search_width: int = 50,
    edge_sigma: float = 5.0,
    contraction_ratio: float = 0.1
) -> tuple:
    """
    检测final mask右边缘线往左第一条梯度等高线
    
    参数:
        binary_img: 二值图像 (0/255)
        gradient_threshold: 梯度阈值 (0-1)
        search_width: 从右边缘向左搜索的宽度
        edge_sigma: 边缘平滑的sigma值
        contraction_ratio: 边缘收缩比例 (0-1)
    
    返回:
        (contour_x, contour_y, right_edges, gradient_map)
        - contour_x: 等高线的x坐标数组
        - contour_y: 等高线的y坐标数组  
        - right_edges: 右边缘线坐标
        - gradient_map: 梯度图
    """
    h, w = binary_img.shape
    
    # 1. 计算Final Mask的左右边缘线
    _, otsu_binary = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    left_edges = np.full(h, w, dtype=int)
    right_edges = np.full(h, -1, dtype=int)
    
    for y in range(h):
        xs = np.where(otsu_binary[y] == 255)[0]
        if xs.size:
            left_edges[y] = xs.min()
            right_edges[y] = xs.max()
    
    # 2. 对左右边缘线进行平滑处理
    yy = np.arange(h)
    left_valid = left_edges < w
    right_valid = right_edges >= 0
    
    if np.any(left_valid):
        left_interp = np.interp(yy, yy[left_valid], left_edges[left_valid].astype(float))
        left_sm = gaussian_filter1d(left_interp, sigma=edge_sigma, mode='nearest')
        left_sm = np.clip(np.rint(left_sm), 0, w - 1).astype(int)
    else:
        left_sm = left_edges
        
    if np.any(right_valid):
        right_interp = np.interp(yy, yy[right_valid], right_edges[right_valid].astype(float))
        right_sm = gaussian_filter1d(right_interp, sigma=edge_sigma, mode='nearest')
        right_sm = np.clip(np.rint(right_sm), 0, w - 1).astype(int)
    else:
        right_sm = right_edges
    
    # 3. 计算收缩后的搜索区域
    # 左边缘向右收缩，右边缘向左收缩
    contracted_left = left_sm + np.round((right_sm - left_sm) * contraction_ratio).astype(int)
    contracted_right = right_sm - np.round((right_sm - left_sm) * contraction_ratio).astype(int)
    
    # 确保收缩后的区域有效
    contracted_left = np.clip(contracted_left, 0, w - 1)
    contracted_right = np.clip(contracted_right, 0, w - 1)
    
    # 4. 计算梯度图 (从右向左的梯度)
    # 使用Sobel算子计算水平梯度 (从右向左)
    sobel_x = cv2.Sobel(binary_img, cv2.CV_32F, 1, 0, ksize=3)
    # 取负值，因为我们要检测从白到黑的变化
    gradient_map = -sobel_x
    
    # 5. 在收缩后的区域内搜索梯度等高线
    contour_x = []
    contour_y = []
    
    for y in range(h):
        left_bound = contracted_left[y]
        right_bound = contracted_right[y]
        
        if left_bound >= right_bound or right_bound < 0 or left_bound >= w:
            continue
            
        # 在收缩后的区域内搜索梯度变化点
        max_gradient = 0
        best_x = right_bound
        
        for x in range(left_bound, right_bound + 1):
            if x >= 0 and x < w:
                grad_val = abs(gradient_map[y, x])
                if grad_val > max_gradient:
                    max_gradient = grad_val
                    best_x = x
        
        # 检查是否超过阈值
        if max_gradient > gradient_threshold * np.max(gradient_map):
            contour_x.append(best_x)
            contour_y.append(y)
    
    return np.array(contour_x), np.array(contour_y), right_sm, gradient_map


def interpolate_and_smooth_contour(
    contour_x: np.ndarray,
    contour_y: np.ndarray,
    image_height: int,
    image_width: int,
    right_edges: np.ndarray | None = None,
    closing_size: int = 11,
    smoothing_sigma: float = 3.0,
) -> tuple:
    """
    对等高线进行上下方向插值和平滑滤波
    
    参数:
        contour_x, contour_y: 原始等高线坐标
        image_height: 图像高度
        smoothing_sigma: 平滑滤波的sigma值
    
    返回:
        (smooth_x, smooth_y): 平滑后的等高线坐标
    """
    if len(contour_x) == 0 or len(contour_y) == 0:
        return np.array([]), np.array([])
    
    # 创建完整的y坐标范围
    y_full = np.arange(image_height)
    
    # 使用插值填补缺失的y坐标
    if len(contour_x) > 1:
        # 线性插值（以右边界为约束，后续做“填坑”处理）
        f_x = interp1d(contour_y, contour_x, kind='linear', bounds_error=False, fill_value='extrapolate')
        x_interp = f_x(y_full).astype(float)

        # 若提供右边界，先剪裁到不超过右边界
        if right_edges is not None and right_edges.shape[0] == image_height:
            x_interp = np.minimum(x_interp, right_edges.astype(float))

        # 先执行形态学 closing 沿 Y 方向“填平向左凹陷的坑”
        # 注意：closing 的 size 取奇数效果更好
        if closing_size < 1:
            closing_size = 1
        if closing_size % 2 == 0:
            closing_size += 1
        x_closed = grey_closing(x_interp, size=closing_size, mode='nearest')

        # 再执行高斯平滑
        x_smooth = gaussian_filter1d(x_closed, sigma=smoothing_sigma, mode='nearest') if smoothing_sigma > 0 else x_closed

        # 最终再次约束到右边界以内
        if right_edges is not None and right_edges.shape[0] == image_height:
            x_smooth = np.minimum(x_smooth, right_edges.astype(float))

        # 确保坐标在有效范围内（基于图像宽度）
        x_smooth = np.clip(x_smooth, 0, image_width - 1)

        return x_smooth, y_full
    else:
        # 如果只有一个点，返回原始坐标
        return contour_x, contour_y


def visualize_gradient_contour_detection(
    binary_img: np.ndarray,
    contour_x: np.ndarray,
    contour_y: np.ndarray,
    right_edges: np.ndarray,
    gradient_map: np.ndarray,
    smooth_x: np.ndarray,
    smooth_y: np.ndarray,
    save_path: str,
    left_edges: np.ndarray = None,
    contracted_left: np.ndarray = None,
    contracted_right: np.ndarray = None
):
    """
    可视化梯度等高线检测结果
    """
    h, w = binary_img.shape
    
    # 创建4个子图的布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 原始图像 + 边缘线 + 收缩区域 + 检测到的等高线
    ax1.imshow(binary_img, cmap='gray')
    
    # 绘制原始边缘线
    if left_edges is not None:
        ax1.plot(left_edges, np.arange(h), 'c-', linewidth=1, alpha=0.6, label='左边缘线')
    ax1.plot(right_edges, np.arange(h), 'g-', linewidth=2, alpha=0.8, label='右边缘线')
    
    # 绘制收缩后的搜索区域
    if contracted_left is not None and contracted_right is not None:
        ax1.plot(contracted_left, np.arange(h), 'm--', linewidth=1, alpha=0.7, label='收缩左边界')
        ax1.plot(contracted_right, np.arange(h), 'm--', linewidth=1, alpha=0.7, label='收缩右边界')
    
    if len(contour_x) > 0:
        ax1.plot(contour_x, contour_y, 'r.', markersize=3, alpha=0.8, label='检测到的等高线点')
        ax1.plot(smooth_x, smooth_y, 'b-', linewidth=2, alpha=0.9, label='平滑后的等高线')
    
    ax1.set_title('梯度等高线检测结果（含收缩区域）')
    # 将图例移至图外右侧，避免遮挡图像细节
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0., frameon=True)
    ax1.axis('off')
    
    # 子图2: 梯度图
    gradient_display = np.abs(gradient_map)
    im2 = ax2.imshow(gradient_display, cmap='hot', alpha=0.8)
    ax2.set_title('梯度图 (绝对值)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 子图3: 等高线x坐标随y的变化
    if len(contour_x) > 0:
        ax3.plot(contour_y, contour_x, 'r.', markersize=4, alpha=0.6, label='原始检测点')
        ax3.plot(smooth_y, smooth_x, 'b-', linewidth=2, alpha=0.9, label='平滑曲线')
        ax3.set_xlabel('Y坐标')
        ax3.set_ylabel('X坐标')
        ax3.set_title('等高线X坐标变化')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()  # 图像坐标系y轴向下
    else:
        ax3.text(0.5, 0.5, '未检测到等高线', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('等高线X坐标变化')
    
    # 子图4: 右边缘线与等高线的对比
    ax4.plot(np.arange(h), right_edges, 'g-', linewidth=2, alpha=0.8, label='右边缘线')
    if len(smooth_x) > 0:
        ax4.plot(smooth_y, smooth_x, 'b-', linewidth=2, alpha=0.9, label='平滑等高线')
        # 计算距离
        if len(smooth_x) == len(right_edges):
            distance = right_edges - smooth_x
            ax4_twin = ax4.twinx()
            ax4_twin.plot(smooth_y, distance, 'r--', linewidth=1, alpha=0.7, label='距离')
            ax4_twin.set_ylabel('右边缘到等高线的距离', color='r')
    
    ax4.set_xlabel('Y坐标')
    ax4.set_ylabel('X坐标')
    ax4.set_title('右边缘线与等高线对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()
    
    # 为右侧图例留出空间
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"梯度等高线检测可视化已保存: {save_path}")


def visualize_components(gray: np.ndarray, comp_mask: np.ndarray, save_path: str):
    """在原图/灰度背景上着色显示组件并保存。"""
    if len(gray.shape) == 2:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        vis = gray.copy()
    # 用青色高亮黑条
    vis[comp_mask == 255] = (255, 255, 0)
    cv2.imwrite(save_path, vis)


def visualize_scanline_analysis(
    binary_img: np.ndarray,
    scan_x: int,
    gradient_series: np.ndarray,
    grad_change_points: np.ndarray,
    binary_series: np.ndarray,
    bin_change_points: np.ndarray,
    save_path: str,
    title_prefix: str = "Scanline Analysis",
    frac: float = 0.25,
    edge_sigma: float = 5.0,
):
    """可视化扫描线分析：左侧图像+扫描线；右侧上下分别为梯度曲线与二值序列。"""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    h, w = binary_img.shape

    # 工具：将输入序列/变化点安全统一到长度 h 的 1D 数组
    def _to_len_h_1d(arr, dtype=np.uint8):
        a = np.asarray(arr)
        if a.ndim > 1:
            a = a.reshape(-1)
        out = np.zeros(h, dtype=dtype)
        n = min(h, a.shape[0])
        out[:n] = a[:n].astype(dtype)
        return out
    # 采用 GridSpec：三列布局，左侧大图，右侧并排二图（左：二值，右：梯度）
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 1.0, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])  # 左侧大图
    ax3 = fig.add_subplot(gs[0, 1])  # 中间：二值序列（按用户要求在左侧）
    ax2 = fig.add_subplot(gs[0, 2])  # 右侧：梯度曲线

    # 左图：原图 + 扫描线(弯曲) + 左右边缘线 + 变化点
    ax1.imshow(binary_img, cmap='gray')
    
    # 计算 Final Mask 及左右边缘，并绘制弯曲扫描线
    _, otsu_binary = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h_img, w_img = binary_img.shape
    left_edges = np.full(h_img, w_img, dtype=int)
    right_edges = np.full(h_img, -1, dtype=int)
    for y in range(h_img):
        xs = np.where(otsu_binary[y] == 255)[0]
        if xs.size:
            left_edges[y] = xs.min()
            right_edges[y] = xs.max()
    left_edges, right_edges = _smooth_edges(left_edges, right_edges, w_img, edge_sigma)
    # 画左右边缘（蓝、绿）
    ax1.plot(left_edges, np.arange(h_img), color='cyan', linewidth=1, alpha=0.9, label='Left edge')
    ax1.plot(right_edges, np.arange(h_img), color='lime', linewidth=1, alpha=0.9, label='Right edge')
    
    # 画弯曲扫描线（红）
    curve_x = []
    curve_y = []
    for y in range(h_img):
        l = left_edges[y]
        r = right_edges[y]
        if l < r and r >= 0 and l < w_img:
            width = r - l + 1
            x = int(l + frac * width)
            x = max(0, min(w_img - 1, x))
            curve_x.append(x)
            curve_y.append(y)
    if curve_x:
        ax1.plot(curve_x, curve_y, color='red', linewidth=2, alpha=0.9, label=f'Curved scan (frac={frac})')
    
    # 标记变化点（紫色横线）：使用“梯度变化点 ∪ 二值上升沿”
    grad_cp_1d = _to_len_h_1d(grad_change_points, dtype=np.uint8)
    bin_cp_1d = _to_len_h_1d(bin_change_points, dtype=np.uint8)
    y_coords = np.where((grad_cp_1d > 0) | (bin_cp_1d > 0))[0]
    if len(y_coords) > 0:
        for y in y_coords:
            ax1.hlines(y, xmin=0, xmax=w, colors='purple', linestyles='-', linewidth=0.5)
    
    ax1.set_title(f'{title_prefix} - Scan Line & Change Points')
    # 将左侧图例移至底部外侧
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=True, borderaxespad=0.)
    ax1.axis('off')

    # 中间：二值序列（黑=1，白=0）
    bin_series_1d = _to_len_h_1d(binary_series, dtype=np.uint8)
    y_coords_bin = np.arange(len(bin_series_1d))
    ax3.step(bin_series_1d, y_coords_bin, where='mid', color='black', linewidth=1.0, label='Binary (black=1)')
    bin_y = np.where(bin_cp_1d > 0)[0]
    if len(bin_y) > 0:
        for y in bin_y:
            ax3.hlines(y, xmin=0, xmax=1, colors='purple', linestyles='-', linewidth=0.6)
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_xlabel('Binary Value')
    ax3.set_ylabel('Y Position')
    ax3.set_title('Binary Sequence Along Scan Line')
    ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()

    # 右侧：梯度曲线
    grad_series_1d = _to_len_h_1d(gradient_series, dtype=float)
    y_coords = np.arange(len(grad_series_1d))
    ax2.plot(grad_series_1d, y_coords, 'b-', linewidth=1, alpha=0.8, label='Gradient magnitude')
    grad_y = np.where(grad_cp_1d > 0)[0]
    if len(grad_y) > 0:
        gmin = float(np.min(grad_series_1d)) if grad_series_1d.size else 0.0
        gmax = float(np.max(grad_series_1d)) if grad_series_1d.size else 1.0
        for y in grad_y:
            ax2.hlines(y, xmin=gmin, xmax=gmax, colors='purple', linestyles='-', linewidth=0.6)
    ax2.set_xlabel('Gradient Magnitude')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Gradient Profile Along Scan Line')
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # 调整布局并紧凑排版，底部为左图图例预留空间
    plt.tight_layout(rect=[0, 0.06, 0.9, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Count black strip-like components on the left of the white tear area")
    default_img = os.path.join(os.path.dirname(__file__), 'adagaus_imgs', 'frame_000000_adagaus.png')
    parser.add_argument('image', nargs='?', default=default_img, help='Binary image path (white tear on black background)')
    parser.add_argument('--band', type=int, default=30, help='Left band width in pixels')
    parser.add_argument('--min_area', type=int, default=10, help='Min component area to keep')
    parser.add_argument('--max_area', type=int, default=5000, help='Max component area to keep')
    parser.add_argument('--out', default=None, help='Output visualization path')
    parser.add_argument('--h_close', type=int, default=7, help='Horizontal closing kernel width')
    parser.add_argument('--v_erode', type=int, default=7, help='Vertical erosion kernel height')
    parser.add_argument('--open', type=int, default=3, help='Square opening kernel size')
    parser.add_argument('--scan', action='store_true', help='Enable vertical scanline counting (changes/2)')
    parser.add_argument('--scan_frac', type=float, default=0.25, help='Column position as fraction between left/right edges')
    parser.add_argument('--scan_no_grad', action='store_true', help='Use binary change counting instead of gradient peaks')
    parser.add_argument('--scan_hybrid', action='store_true', help='Use both gradient and binary change detection')
    parser.add_argument('--edge_sigma', type=float, default=5.0, help='Gaussian sigma for smoothing left/right edges along Y')
    parser.add_argument('--gradient_contour', action='store_true', help='Enable gradient contour detection from right edge')
    parser.add_argument('--gradient_threshold', type=float, default=0.3, help='Gradient threshold for contour detection (0-1)')
    parser.add_argument('--search_width', type=int, default=50, help='Search width from right edge for contour detection')
    parser.add_argument('--smoothing_sigma', type=float, default=3.0, help='Gaussian sigma for contour smoothing')
    parser.add_argument('--closing_size', type=int, default=11, help='Morphological closing size (odd int) to fill left-indented pits')
    parser.add_argument('--contraction_ratio', type=float, default=0.1, help='Edge contraction ratio for search region (0-1)')
    args = parser.parse_args()

    # 设置中文字体（在任何绘图前调用）
    setup_chinese_font()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"无法读取图像: {args.image}")
        sys.exit(1)

    binary = ensure_binary(gray)
    count, comp_mask, _ = count_black_strips(
        binary,
        band_width=args.band,
        min_area=args.min_area,
        max_area=args.max_area,
        h_close=args.h_close,
        v_erode=args.v_erode,
        open_size=args.open,
    )

    print(f"形态学-连通域计数: {count}")

    if args.scan:
        if args.scan_hybrid:
            # 混合模式：同时使用梯度和二值检测
            est_grad, scan_x, series_grad = count_by_vertical_scanline(
                binary, frac=args.scan_frac, smooth_ksize=9, use_gradient=True, edge_sigma=args.edge_sigma
            )
            est_bin, _, series_bin = count_by_vertical_scanline(
                binary, frac=args.scan_frac, smooth_ksize=9, use_gradient=False, edge_sigma=args.edge_sigma
            )
            est = max(est_grad, est_bin)  # 取较大值
            series = series_grad  # 用梯度序列做可视化
            print(f"扫描列 x={scan_x} 估计数量: 梯度={est_grad}, 二值={est_bin}, 最终={est}")
        else:
            est, scan_x, series = count_by_vertical_scanline(
                binary,
                frac=args.scan_frac,
                smooth_ksize=9,
                use_gradient=not args.scan_no_grad,
                edge_sigma=args.edge_sigma,
            )
            print(f"扫描列 x={scan_x} 估计数量: {est}")

        # 生成扫描线分析可视化
        scan_out_path = out_path.replace('.png', '_scanline_analysis.png') if args.out else None
        if scan_out_path is None:
            base, ext = os.path.splitext(args.image)
            scan_out_path = base + '_scanline_analysis.png'
        if args.scan_hybrid or (not args.scan_no_grad):
            grad_series = series_grad if args.scan_hybrid else series
            grad_cp = (grad_series >= np.percentile(grad_series, 85)).astype(np.uint8)
        else:
            grad_series = np.zeros_like(series)
            grad_cp = np.zeros_like(series)

        if args.scan_hybrid or args.scan_no_grad:
            # 二值序列与变化点（仅标注上升沿 0→1）
            if args.scan_hybrid:
                bin_series = series_bin
            else:
                bin_series = series
            bin_cp = ((bin_series[1:] == 1) & (bin_series[:-1] == 0)).astype(np.uint8)
            bin_cp = np.pad(bin_cp, (1, 0))
        else:
            bin_series = np.zeros_like(series)
            bin_cp = np.zeros_like(series)

        visualize_scanline_analysis(
            binary,
            scan_x,
            grad_series,
            grad_cp,
            bin_series,
            bin_cp,
            scan_out_path,
            frac=args.scan_frac,
            edge_sigma=args.edge_sigma,
        )
        print(f"扫描线分析图已保存: {scan_out_path}")

    # 梯度等高线检测功能
    if args.gradient_contour:
        print("\n开始梯度等高线检测...")
        contour_x, contour_y, right_edges, gradient_map = detect_gradient_contour_from_right_edge(
            binary,
            gradient_threshold=args.gradient_threshold,
            search_width=args.search_width,
            edge_sigma=args.edge_sigma,
            contraction_ratio=args.contraction_ratio
        )
        
        print(f"检测到 {len(contour_x)} 个等高线点")
        
        # 对等高线进行插值和平滑
        smooth_x, smooth_y = interpolate_and_smooth_contour(
            contour_x,
            contour_y,
            image_height=binary.shape[0],
            image_width=binary.shape[1],
            right_edges=right_edges,
            closing_size=getattr(args, 'closing_size', 11),
            smoothing_sigma=args.smoothing_sigma,
        )
        
        print(f"平滑后等高线长度: {len(smooth_x)}")
        
        # 重新计算收缩区域用于可视化
        _, otsu_binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = binary.shape
        left_edges = np.full(h, w, dtype=int)
        for y in range(h):
            xs = np.where(otsu_binary[y] == 255)[0]
            if xs.size:
                left_edges[y] = xs.min()
        
        # 平滑左右边缘
        yy = np.arange(h)
        left_valid = left_edges < w
        right_valid = right_edges >= 0
        
        if np.any(left_valid):
            left_interp = np.interp(yy, yy[left_valid], left_edges[left_valid].astype(float))
            left_sm = gaussian_filter1d(left_interp, sigma=args.edge_sigma, mode='nearest')
            left_sm = np.clip(np.rint(left_sm), 0, w - 1).astype(int)
        else:
            left_sm = left_edges
            
        if np.any(right_valid):
            right_interp = np.interp(yy, yy[right_valid], right_edges[right_valid].astype(float))
            right_sm = gaussian_filter1d(right_interp, sigma=args.edge_sigma, mode='nearest')
            right_sm = np.clip(np.rint(right_sm), 0, w - 1).astype(int)
        else:
            right_sm = right_edges
        
        # 计算收缩区域
        contracted_left = left_sm + np.round((right_sm - left_sm) * args.contraction_ratio).astype(int)
        contracted_right = right_sm - np.round((right_sm - left_sm) * args.contraction_ratio).astype(int)
        contracted_left = np.clip(contracted_left, 0, w - 1)
        contracted_right = np.clip(contracted_right, 0, w - 1)
        
        # 生成可视化到固定目录：data_adagaus_density_curve/gradient_contour_demo_output/contour_*.png
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        target_dir = os.path.join(os.path.dirname(__file__), 'gradient_contour_demo_output')
        os.makedirs(target_dir, exist_ok=True)
        contour_out_path = os.path.join(target_dir, f"contour_{base_name}.png")
            
        visualize_gradient_contour_detection(
            binary, contour_x, contour_y, right_edges, gradient_map,
            smooth_x, smooth_y, contour_out_path,
            left_edges=left_sm, contracted_left=contracted_left, contracted_right=contracted_right
        )
        print(f"梯度等高线检测可视化已保存: {contour_out_path}")

        # 统计指标：等高线占左边缘线到右边缘线长度的平均占比
        # 占比 = (smooth_x - left_sm) / (right_sm - left_sm)，逐行计算后对有效行求平均
        width_lr = (right_sm - left_sm).astype(float)
        valid_rows = (width_lr > 0) & (smooth_x.shape[0] == h)
        ratios = np.zeros(h, dtype=float)
        if np.any(valid_rows):
            # 防止除零
            safe_width = np.where(width_lr == 0, 1.0, width_lr)
            ratios = (smooth_x - left_sm) / safe_width
            ratios = np.clip(ratios, 0.0, 1.0)
            avg_ratio = float(np.mean(ratios[valid_rows]))
            print(f"等高线平均占比(相对左右边界宽度): {avg_ratio:.4f}")
        else:
            print("等高线平均占比(相对左右边界宽度): 无有效行")

    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.image)
        out_path = base + '_black_strips_overlay.png'
    visualize_components(binary, comp_mask, out_path)
    print(f"可视化叠图已保存: {out_path}")


if __name__ == '__main__':
    main()



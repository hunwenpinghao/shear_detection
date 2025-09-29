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
    args = parser.parse_args()

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

    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.image)
        out_path = base + '_black_strips_overlay.png'
    visualize_components(binary, comp_mask, out_path)
    print(f"可视化叠图已保存: {out_path}")


if __name__ == '__main__':
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from tqdm import tqdm
import sys
import platform
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import grey_closing

# 添加data_process目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_process'))
from adagaus_extractor import AdagausExtractor

# 设置中文字体
def setup_chinese_font():
    """设置中文字体"""
    import matplotlib.font_manager as fm
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = []
        
        preferred_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
        
        for font in preferred_fonts:
            if font in available_fonts:
                chinese_fonts.append(font)
        
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"设置中文字体: {chinese_fonts[0]}")
            return True
    
    elif system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("设置Windows中文字体")
        return True
    
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("设置Linux中文字体")
        return True
    
    print("无法设置中文字体，将使用英文标签")
    return False

class AdagausDensityAnalyzer:
    """Filtered Adaptive Gaussian 二值图密度分析器"""
    
    def __init__(self):
        self.results = []
        self.extractor = AdagausExtractor()
        
    def create_burr_visualization(self, background_image: np.ndarray, 
                                burr_binary: np.ndarray) -> np.ndarray:
        """
        创建毛刺可视化图像
        """
        try:
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            
            # 确保输入为灰度图
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 使用matplotlib生成可视化
            fig, ax = plt.subplots(1, 1, figsize=(6, 12))
            ax.imshow(gray_background, cmap='gray', alpha=0.7)
            ax.imshow(burr_binary, cmap='Blues', alpha=0.8)  # 使用蓝色表示毛刺
            ax.axis('off')
            
            # 保存到内存中的字节流
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # 读取图像数组并转换
            image = Image.open(buf)
            image_array = np.array(image)
            
            # 转换为OpenCV BGR格式
            if len(image_array.shape) == 3:
                bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            buf.close()
            plt.close(fig)
            return bgr
            
        except Exception as e:
            print(f"matplotlib方法失败，使用OpenCV回退: {e}")
            
            # OpenCV回退方法
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 背景alpha
            background_result = (gray_background * 0.7).astype(np.uint8)
            
            # 蓝色图层（毛刺）
            blue_result = np.zeros(gray_background.shape + (3,), dtype=np.uint8)
            burr_pixels = burr_binary > 0
            blue_result[burr_pixels, 2] = 255  # 蓝色通道
            
            # alpha混合
            blue_layer = blue_result.astype(np.float32)
            bg_layer = cv2.cvtColor(background_result, cv2.COLOR_GRAY2RGB).astype(np.float32)
            
            alpha_burr = 0.8
            alpha_bg = 0.7
            overlay = alpha_burr * blue_layer + alpha_bg * (1 - alpha_burr) * bg_layer
            return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def analyze_adagaus_density(self, image_path: str, final_mask: np.ndarray, filtered_adagaus: np.ndarray):
        """混合法统计 Final Mask 区域内黑色块数量（梯度法与二值上升沿法取最大）。
        同时输出基于混合计数的密度(数量/总像素)。"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        
        h, w = image.shape
        mask_pixels = int(np.sum(final_mask > 0))

        # 1) 基于 Final Mask 计算每行左/右边界
        left_edges = np.full(h, w, dtype=int)
        right_edges = np.full(h, -1, dtype=int)
        for y in range(h):
            cols = np.where(final_mask[y] > 0)[0]
            if cols.size:
                left_edges[y] = cols.min()
                right_edges[y] = cols.max()

        # 2) 沿弯曲扫描线统计（参数可按需调整）
        frac = 0.25        # 扫描线位于主体左侧比例
        edge_sigma = 5.0    # 上下方向边缘平滑
        smooth_ksize = 9    # 梯度平滑核
        x_window = 2        # 横向窗口半径

        # 2.1 平滑左右边缘（插值缺失行）
        yy = np.arange(h)
        valid = (left_edges < w) & (right_edges >= 0) & (right_edges > left_edges)
        if np.any(valid):
            left_interp = np.interp(yy, yy[valid], left_edges[valid].astype(float))
            right_interp = np.interp(yy, yy[valid], right_edges[valid].astype(float))
            left_sm = gaussian_filter1d(left_interp, sigma=edge_sigma, mode='nearest')
            right_sm = gaussian_filter1d(right_interp, sigma=edge_sigma, mode='nearest')
            left_sm = np.clip(np.rint(left_sm), 0, w - 1).astype(int)
            right_sm = np.clip(np.rint(right_sm), 0, w - 1).astype(int)
        else:
            left_sm, right_sm = left_edges, right_edges

        # 2.2 计算弯曲扫描线 x 坐标
        scan_xs = np.full(h, -1, dtype=int)
        for y in range(h):
            l, r = left_sm[y], right_sm[y]
            if r > l:
                x = int(l + frac * (r - l + 1))
                scan_xs[y] = max(0, min(w - 1, x))

        # 2.3 梯度法计数
        sm = cv2.GaussianBlur(filtered_adagaus, (smooth_ksize, smooth_ksize), 0)
        dy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        g_curve = np.zeros(h, dtype=float)
        for y in range(h):
            x = scan_xs[y]
            if x >= 0:
                g_curve[y] = abs(float(dy[y, x]))
        nz = g_curve[g_curve > 0]
        if nz.size:
            thr = np.percentile(nz, 70)
            if thr <= 0:
                thr = float(np.max(g_curve)) * 0.3
        else:
            thr = 0.0
        peaks = (g_curve >= thr).astype(np.uint8)
        transitions = int(np.sum((peaks[1:] > 0) & (peaks[:-1] == 0)) + (peaks[0] > 0))
        grad_count = max(0, transitions // 2)

        # 2.4 二值上升沿计数（带横向窗口）
        col_curve = np.zeros(h, dtype=np.uint8)
        for y in range(h):
            x = scan_xs[y]
            if x >= 0:
                if x_window > 0:
                    xa = max(0, x - x_window)
                    xb = min(w - 1, x + x_window)
                    is_black = np.any(filtered_adagaus[y, xa:xb + 1] == 0)
                    col_curve[y] = 1 if is_black else 0
                else:
                    col_curve[y] = 1 if filtered_adagaus[y, x] == 0 else 0
        rising_edges = int(np.sum((col_curve[1:] == 1) & (col_curve[:-1] == 0)))

        # 2.5 混合计数
        mixed_count = max(grad_count, rising_edges)

        # 2.6 计算“等高线平均占比”指标
        # 以收缩后的左右边界内，逐行搜索水平最大梯度位置得到等高线，再进行沿Y的插值、closing“填坑”、高斯平滑
        contraction_ratio = 0.10
        contracted_left = left_sm + np.round((right_sm - left_sm) * contraction_ratio).astype(int)
        contracted_right = right_sm - np.round((right_sm - left_sm) * contraction_ratio).astype(int)
        contracted_left = np.clip(contracted_left, 0, w - 1)
        contracted_right = np.clip(contracted_right, 0, w - 1)

        # 梯度（从右到左白->黑取负号）
        sobel_x = cv2.Sobel(filtered_adagaus, cv2.CV_32F, 1, 0, ksize=3)
        gradient_map = -sobel_x

        contour_x = np.full(h, -1, dtype=float)
        for y in range(h):
            lb = int(contracted_left[y])
            rb = int(contracted_right[y])
            if lb >= 0 and rb < w and rb > lb:
                row = np.abs(gradient_map[y, lb:rb + 1])
                if row.size > 0:
                    x_rel = int(np.argmax(row))
                    contour_x[y] = lb + x_rel

        # 插值缺失行
        yy = np.arange(h)
        valid = contour_x >= 0
        if np.any(valid):
            x_interp = np.interp(yy, yy[valid], contour_x[valid])
            # 不超过右边界
            x_interp = np.minimum(x_interp, right_sm.astype(float))
            # 形态学closing沿Y“填平向左凹陷”
            closing_size = 11
            if closing_size % 2 == 0:
                closing_size += 1
            x_closed = grey_closing(x_interp, size=closing_size, mode='nearest')
            # 高斯平滑
            x_smooth = gaussian_filter1d(x_closed, sigma=2.0, mode='nearest')
            x_smooth = np.minimum(x_smooth, right_sm.astype(float))
            x_smooth = np.clip(x_smooth, 0, w - 1)

            # 逐行占比
            width_lr = (right_sm - left_sm).astype(float)
            safe_width = np.where(width_lr == 0, 1.0, width_lr)
            ratios = (x_smooth - left_sm) / safe_width
            ratios = np.clip(ratios, 0.0, 1.0)
            valid_rows = width_lr > 0
            avg_contour_ratio = float(np.mean(ratios[valid_rows])) if np.any(valid_rows) else 0.0
        else:
            avg_contour_ratio = 0.0

        # 2.7 黑色像素总数（Final Mask 内）
        black_pixel_count = int(np.sum(((final_mask > 0) & (filtered_adagaus == 0)).astype(np.uint8)))
        # 单条黑色条状物平均像素数 D_b = N_pix / N_strips
        black_pixel_per_strip = (black_pixel_count / mixed_count) if mixed_count > 0 else 0.0

        frame_num = self.extract_frame_info(image_path)
        time_seconds = frame_num * 5 if frame_num > 0 else 0

        total_pixels = int(h * w)
        final_mask_coverage = (mask_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
        black_count_density = (mixed_count / total_pixels) if total_pixels > 0 else 0.0
        
        return {
            'frame_num': frame_num,
            'time_seconds': time_seconds,
            'final_mask_coverage': final_mask_coverage,
            'black_count': mixed_count,
            'black_count_density': black_count_density,
            'black_pixel_count': black_pixel_count,
            'black_pixel_per_strip': black_pixel_per_strip,
            'avg_contour_ratio': avg_contour_ratio,
            'image_shape': image.shape,
            'image_path': image_path
        }
    
    def apply_smoothing_filters(self, data, 
                               smoothing_method: str = 'gaussian',
                               window_size: int = 50,
                               sigma: float = 10.0):
        """对时间序列数据应用平滑滤波"""
        time_seconds = np.array([d['time_seconds'] for d in data])
        black_count_density = np.array([d['black_count_density'] for d in data])
        black_count = np.array([d['black_count'] for d in data])
        
        if smoothing_method == 'gaussian':
            smoothed_densities = gaussian_filter1d(black_count_density, sigma=sigma)
            smoothed_counts = gaussian_filter1d(black_count, sigma=sigma)
            
        elif smoothing_method == 'moving_avg':
            smoothed_densities = np.convolve(black_count_density, np.ones(window_size)/window_size, mode='same')
            smoothed_counts = np.convolve(black_count, np.ones(window_size)/window_size, mode='same')
            
        elif smoothing_method == 'savgol':
            window_length = min(window_size, len(black_count_density))
            if window_length % 2 == 0:
                window_length -= 1
            smoothed_densities = signal.savgol_filter(black_count_density, window_length, 3)
            smoothed_counts = signal.savgol_filter(black_count, window_length, 3)
            
        else:
            smoothed_densities = gaussian_filter1d(black_count_density, sigma=sigma)
            smoothed_counts = gaussian_filter1d(black_count, sigma=sigma)
        
        return time_seconds, smoothed_densities, smoothed_counts
    
    def extract_frame_info(self, filename: str) -> int:
        """从文件名提取帧号"""
        try:
            basename = os.path.basename(filename)
            # 提取frame_XXXXXX中的数字
            frame_num = int(basename.split('_')[1])
            return frame_num
        except (IndexError, ValueError):
            return -1
    
    def process_images(self, roi_dir, output_dir, use_contour_method=True):
        """按步骤处理所有ROI，生成并统计 Filtered Adaptive Gaussian。"""
        print("开始分析 Filtered Adaptive Gaussian 二值图密度...")
        print("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        adagaus_dir = os.path.join(output_dir, 'adagaus_imgs')
        os.makedirs(adagaus_dir, exist_ok=True)
        
        import glob
        roi_pattern = os.path.join(roi_dir, "*_roi.png")
        image_files = sorted(glob.glob(roi_pattern), key=self.extract_frame_info)
        if not image_files:
            print(f"在目录 {roi_dir} 中未找到ROI图像文件")
            return
        
        print(f"找到 {len(image_files)} 个ROI图像文件")
        
        # 如果需要使用撕裂面mask，则导入检测器
        if use_contour_method:
            import sys
            sys.path.append('/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_split')
            from shear_tear_detector import ShearTearDetector
            detector = ShearTearDetector(use_contour_method=True)
        
        for image_path in tqdm(image_files, desc="计算与统计 Adagaus", unit="图像"):
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
                
            # 根据方法选择不同的mask生成方式
            if use_contour_method:
                # 新方法：使用等高线方法生成撕裂面mask
                result = detector.detect_surfaces(gray, visualize=False)
                if result and 'intermediate_results' in result:
                    tear_mask = result['intermediate_results'].get('tear_mask', None)
                    if tear_mask is not None:
                        # 使用等高线方法生成的撕裂面mask
                        final_mask = tear_mask.astype(np.uint8) * 255
                    else:
                        # 回退到原始方法
                        final_mask = self.extractor._compute_final_filled_mask(gray)
                else:
                    # 回退到原始方法
                    final_mask = self.extractor._compute_final_filled_mask(gray)
            else:
                # 老方法：使用原始方法
                final_mask = self.extractor._compute_final_filled_mask(gray)
            
            adagaus = self.extractor._adaptive_gaussian_binary(gray)
            filtered = adagaus.copy()
            filtered[final_mask == 0] = 0
                
            frame_num = self.extract_frame_info(image_path)
            out_name = f"frame_{frame_num:06d}_adagaus.png"
            out_path = os.path.join(adagaus_dir, out_name)
            cv2.imwrite(out_path, filtered)
            
            analysis_result = self.analyze_adagaus_density(image_path, final_mask, filtered)
            if analysis_result:
                self.results.append(analysis_result)
        
        self.save_results(output_dir)
        self.create_visualizations(output_dir)
        
        print("=" * 60)
        print("Filtered Adaptive Gaussian 二值图密度分析完成!")
        print(f"所有结果已保存到: {output_dir}")
    
    def create_burr_visualization(self, background_image: np.ndarray, 
                                burr_binary: np.ndarray) -> np.ndarray:
        """创建毛刺可视化图像"""
        try:
            import io
            from PIL import Image
            
            # 确保输入为灰度图
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 使用matplotlib生成毛刺可视化图像
            fig, ax = plt.subplots(1, 1, figsize=(6, 12))
            ax.imshow(gray_background, cmap='gray', alpha=0.7)
            ax.imshow(burr_binary, cmap='Oranges', alpha=0.8)  # 使用橙色表示毛刺
            ax.axis('off')
            
            # 保存到内存中的字节流
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # 读取图像数组并转换
            image = Image.open(buf)
            image_array = np.array(image)
            
            # 转换为OpenCV BGR格式
            if len(image_array.shape) == 3:
                bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR) 
                
            buf.close()
            plt.close(fig)
            return bgr
            
        except Exception as e:
            print(f"matplotlib方法失败，使用OpenCV回退: {e}")
            
            # OpenCV回退方法 - 模拟橙色毛刺可视化
            if len(background_image.shape) == 3:
                gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_background = background_image.copy()
            
            # 背景透明度处理
            background_result = (gray_background * 0.7).astype(np.uint8)
            
            # 橙色毛刺图层 - 模拟橙色通道
            orange_result = np.zeros(gray_background.shape + (3,), dtype=np.uint8)
            burr_pixels = burr_binary > 0
            orange_result[burr_pixels, 0] = 255  # 红色通道 
            orange_result[burr_pixels, 1] = 165  # 绿色通道 (橙色=红+部分绿)
            orange_result[burr_pixels, 2] = 0    # 蓝色通道
            
            # Alpha混合
            orange_layer = orange_result.astype(np.float32)
            bg_layer = cv2.cvtColor(background_result, cv2.COLOR_GRAY2RGB).astype(np.float32)
            
            # 进行alpha混合
            alpha_burr = 0.8
            alpha_bg = 0.7
            overlay = alpha_burr * orange_layer + alpha_bg * (1 - alpha_burr) * bg_layer
            
            return cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    def save_results(self, output_dir):
        """保存分析结果"""
        
        # 保存JSON结果（转换numpy类型为Python原生类型）
        json_results = []
        for result in self.results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                elif hasattr(value, 'item'):  # numpy标量
                    json_result[key] = value.item()
                else:
                    json_result[key] = value
            json_results.append(json_result)
        
        json_path = os.path.join(output_dir, 'adagaus_density_analysis.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV结果
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = os.path.join(output_dir, 'adagaus_density_analysis.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"分析结果已保存到: {json_path}")
        print(f"CSV结果已保存到: {csv_path}")
    
    def create_visualizations(self, output_dir):
        """创建可视化图表"""
        
        if not self.results:
            print("没有分析结果，无法创建可视化")
            return
        
        # 按时间点排序
        sorted_results = sorted(self.results, key=lambda x: x['time_seconds'])
        
        # 提取数据
        time_seconds = np.array([r['time_seconds'] for r in sorted_results])
        mask_coverages = np.array([r['final_mask_coverage'] for r in sorted_results])
        black_counts = np.array([r['black_count'] for r in sorted_results])
        # 新密度定义：每条黑条平均像素数 D_b
        black_pixel_per_strip = np.array([r.get('black_pixel_per_strip', 0.0) for r in sorted_results])
        
        # 设置中文字体
        font_success = setup_chinese_font()
        
        # 应用平滑滤波（对数量使用通用方法；对 D_b 单独平滑）
        _, _, smoothed_black_count = self.apply_smoothing_filters(
            sorted_results, smoothing_method='gaussian', window_size=50, sigma=10.0)
        smoothed_db = gaussian_filter1d(black_pixel_per_strip, sigma=10.0)
        
        # 取等高线平均占比
        avg_ratios = np.array([r.get('avg_contour_ratio', 0.0) for r in sorted_results])

        # 创建图表（三行）：上-D_b；中-黑条数量；下-等高线平均占比
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14))
        
        # 新密度 D_b（每条黑条平均像素数）随时间（原始+平滑）
        ax1.plot(time_seconds, black_pixel_per_strip, 'b-', linewidth=0.8, alpha=0.3, label='Raw Data')
        ax1.plot(time_seconds, smoothed_db, 'b-', linewidth=2.5, alpha=0.9, label='Smoothed Curve')
        ax1.fill_between(time_seconds, smoothed_db, alpha=0.3, color='blue')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Pixels per Black Strip (D_b)')
        ax1.set_title('Average Pixels per Black Strip Over Time (Smoothed)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_db = np.mean(black_pixel_per_strip)
        ax1.axhline(y=mean_db, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_db:.1f}')
        ax1.legend()
        
        # 黑色块状数量随时间（原始+平滑）
        ax2.plot(time_seconds, black_counts, 'r-', linewidth=0.8, alpha=0.3, label='Raw Data')
        ax2.plot(time_seconds, smoothed_black_count, 'r-', linewidth=2.5, alpha=0.9, label='Smoothed Curve')
        ax2.fill_between(time_seconds, smoothed_black_count, alpha=0.3, color='red')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Number of Black Blocks')
        ax2.set_title('Black Block Count Over Time (Smoothed)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(time_seconds))
        
        # 添加统计信息
        mean_black = np.mean(black_counts)
        ax2.axhline(y=mean_black, color='blue', linestyle='--', alpha=0.7,
                   label=f'Mean: {mean_black:.1f}')
        ax2.legend()

        # 第三行：等高线平均占比曲线
        ax3.plot(time_seconds, avg_ratios, 'g-', linewidth=2.0, alpha=0.9, label='Avg Contour Ratio')
        ax3.fill_between(time_seconds, avg_ratios, alpha=0.15, color='green')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Avg Contour Ratio')
        ax3.set_title('Average Contour Position Ratio Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max(time_seconds))
        mean_ratio = np.mean(avg_ratios)
        ax3.axhline(y=mean_ratio, color='purple', linestyle='--', alpha=0.7,
                   label=f'Mean: {mean_ratio:.3f}')
        ax3.set_ylim(0.0, 1.0)
        ax3.legend()

        # 下：数量曲线保持不变
        
        # 添加滤波方法说明
        fig.suptitle('Smoothing Method: Gaussian Filter (σ=10, window=50)', fontsize=12, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 保存图表
        plot_path = os.path.join(output_dir, 'adagaus_density_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到: {plot_path}")
        
        # 创建综合统计图
        self.create_summary_plot(output_dir, sorted_results)
    
    def create_summary_plot(self, output_dir, sorted_results):
        """创建综合统计图"""
        
        # 提取数据
        time_points = [r['time_seconds'] for r in sorted_results]
        mask_coverages = [r['final_mask_coverage'] for r in sorted_results]
        black_counts = [r['black_count'] for r in sorted_results]
        
        # 创建综合图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Black Block Analysis Summary', fontsize=16)
        
        # 黑块密度与黑块数量对比（同一时间轴）
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(time_points, [r['black_count_density'] for r in sorted_results], 'b-o', linewidth=2, markersize=6, label='Black Block Density (count/pixel)')
        line2 = ax1_twin.plot(time_points, black_counts, 'r-s', linewidth=2, markersize=6, label='Black Block Count')
        
        ax1.set_xlabel('Time Point')
        ax1.set_ylabel('Black Block Density (count/pixel)', color='b')
        ax1_twin.set_ylabel('Black Block Count', color='r')
        ax1.set_title('Black Block Density vs Count')
        ax1.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 黑块密度分布
        black_densities = [r['black_count_density'] for r in sorted_results]
        ax2.hist(black_densities, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Black Block Density (count/pixel)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Black Block Densities')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存综合图
        summary_path = os.path.join(output_dir, 'adagaus_density_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"综合统计图已保存到: {summary_path}")

def main():
    """主函数"""
    import sys
    
    # 默认路径（可通过命令行参数覆盖）
    roi_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data/roi_imgs"
    output_dir = "/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_adagaus_density_curve"
    
    if len(sys.argv) > 1:
        roi_dir = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    # 创建分析器
    analyzer = AdagausDensityAnalyzer()
    
    # 处理图像（默认使用新方法）
    analyzer.process_images(roi_dir, output_dir, use_contour_method=True)

if __name__ == "__main__":
    main()

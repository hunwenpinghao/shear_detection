#!/usr/bin/env python3
"""
Adaptive Gaussian 二值化提取器（Filtered Adaptive Gaussian）
从 ROI 图像中计算自适应高斯二值化，并使用 Final Filled Mask 进行过滤后保存

结构参考: data_process/burr_extractor.py
"""

import cv2
import numpy as np
import os
import sys
import glob
from typing import Dict, Any
from tqdm import tqdm

# 路径设置，确保可导入同级与项目根下模块
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import PREPROCESS_CONFIG
from data_shear_split.generate_final_mask import detect_left_right_edges, create_filled_mask


class AdagausExtractor:
    """Adaptive Gaussian 二值化提取器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config is not None else PREPROCESS_CONFIG

    @staticmethod
    def _compute_final_filled_mask(gray_image: np.ndarray) -> np.ndarray:
        """基于 Otsu on Original 计算 Final Filled Mask。"""
        _, otsu_binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        left_edges, right_edges = detect_left_right_edges(otsu_binary)
        filled_mask = create_filled_mask(left_edges, right_edges, otsu_binary.shape)
        return filled_mask

    @staticmethod
    def _adaptive_gaussian_binary(gray_image: np.ndarray) -> np.ndarray:
        """计算自适应高斯二值化（与测试脚本一致：Gaussian + equalize + adaptive Gaussian）。"""
        if len(gray_image.shape) == 3:
            gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_image.copy()

        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        enhanced = cv2.equalizeHist(denoised)
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return binary

    def _process_single_roi(self, roi_path: str, output_path: str) -> Dict[str, Any]:
        """处理单张 ROI：计算 Adaptive Gaussian 后用 Final Mask 过滤并保存。"""
        try:
            gray = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return {'success': False, 'error': f'无法读取ROI图像: {roi_path}'}

            # 计算 Final Filled Mask
            filled_mask = self._compute_final_filled_mask(gray)

            # 计算 Adaptive Gaussian 二值图
            adagaus = self._adaptive_gaussian_binary(gray)

            # 应用 Final Filled Mask
            filtered = adagaus.copy()
            filtered[filled_mask == 0] = 0

            # 保存（灰度二值图）
            success = cv2.imwrite(output_path, filtered)
            if not success:
                return {'success': False, 'error': f'无法保存二值化图像到: {output_path}'}

            return {
                'success': True,
                'output_path': output_path,
                'white_pixels': int(np.sum(filtered == 255)),
                'black_pixels': int(np.sum(filtered == 0))
            }
        except Exception as e:
            return {'success': False, 'error': f'处理ROI图像时出错: {str(e)}'}

    def extract_from_roi_images(self, roi_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        从指定 ROI 目录提取 Filtered Adaptive Gaussian 结果。
        保存命名: frame_XXXXXX_adagaus.png
        """
        print("=== 从ROI图像提取 Filtered Adaptive Gaussian ===")
        print(f"输入ROI目录: {roi_dir}")
        print(f"输出目录: {output_dir}")

        try:
            if not os.path.exists(roi_dir):
                return {'success': False, 'error': f'输入ROI目录不存在: {roi_dir}'}

            os.makedirs(output_dir, exist_ok=True)

            roi_pattern = os.path.join(roi_dir, "*_roi.png")
            roi_files = sorted(glob.glob(roi_pattern),
                               key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            if not roi_files:
                return {'success': False, 'error': f'在目录 {roi_dir} 中未找到ROI图像文件'}

            print(f"找到 {len(roi_files)} 个ROI图像文件")
            print("开始计算 Filtered Adaptive Gaussian...")

            success_count = 0
            failed_count = 0

            for i, roi_path in enumerate(tqdm(roi_files, desc="提取Adap-Gaussian")):
                try:
                    basename = os.path.basename(roi_path)
                    frame_number = int(basename.split('_')[1].split('.')[0])
                    out_name = f"frame_{frame_number:06d}_adagaus.png"
                    out_path = os.path.join(output_dir, out_name)

                    # 已存在跳过
                    if os.path.exists(out_path):
                        success_count += 1
                        continue

                    result = self._process_single_roi(roi_path, out_path)
                    if result.get('success'):
                        success_count += 1
                    else:
                        failed_count += 1
                        print(f"失败 {roi_path}: {result.get('error', '未知错误')}")

                except Exception as e:
                    failed_count += 1
                    print(f"处理ROI图像时出错 {roi_path}: {e}")

                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(roi_files)} 帧，成功 {success_count}，失败 {failed_count}")

            print("\nFiltered Adaptive Gaussian 提取完成！")
            print(f"总帧数: {len(roi_files)}")
            print(f"成功: {success_count}")
            print(f"失败: {failed_count}")
            print(f"输出目录: {output_dir}")

            return {
                'success': True,
                'output_dir': output_dir,
                'total_frames': len(roi_files),
                'success_frames': success_count,
                'failed_frames': failed_count
            }

        except Exception as e:
            return {'success': False, 'error': f'批量提取时出错: {str(e)}'}

    def extract_from_all_rois(self, roi_dir: str, output_dir: str) -> Dict[str, Any]:
        """从默认目录提取全部 ROI 的 Filtered Adaptive Gaussian 结果。"""
        return self.extract_from_roi_images(roi_dir, output_dir)


def main():
    """主函数"""
    # Video1 from yinkuo
    # default_roi_dir = "data/roi_imgs"
    # default_out_dir = "data/adagaus_imgs"

    # Video2 from yinkuo
    default_roi_dir = "data_video2_20250821140339629/roi_imgs"
    default_out_dir = "data_video2_20250821140339629/adagaus_imgs"
    
    # first_cycle
    # default_roi_dir = "data_video_20250821152112032/first_cycle/roi_imgs"
    # default_out_dir = "data_video_20250821152112032/first_cycle/adagaus_imgs"

    # second_cycle
    # default_roi_dir = "data_video_20250821152112032/second_cycle/roi_imgs"
    # default_out_dir = "data_video_20250821152112032/second_cycle/adagaus_imgs"

    if len(sys.argv) >= 3:
        roi_dir = sys.argv[1]
        out_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        roi_dir = sys.argv[1]
        out_dir = default_out_dir
        print(f"使用默认输出目录: {out_dir}")
    else:
        roi_dir = default_roi_dir
        out_dir = default_out_dir
        print(f"使用默认输入ROI目录: {roi_dir}")
        print(f"使用默认输出目录: {out_dir}")

    if not os.path.exists(roi_dir):
        raise ValueError(f"错误：ROI目录不存在 - {roi_dir}")

    os.makedirs(out_dir, exist_ok=True)

    extractor = AdagausExtractor()
    result = extractor.extract_from_roi_images(roi_dir=roi_dir, output_dir=out_dir)

    if result.get('success'):
        print("\n✅ Filtered Adaptive Gaussian 提取完成！")
        print(f"输出目录: {result['output_dir']}")
        print(f"总帧数: {result['total_frames']}")
        print(f"成功: {result['success_frames']}")
        print(f"失败: {result['failed_frames']}")
    else:
        print(f"❌ 提取失败: {result.get('error')}")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量基于 Otsu on Original 生成左右边缘与最终填充的 Mask

输入: roi_dir (包含 ROI 的目录)
输出: 
  - 边缘可视化图 (红: 左边缘, 蓝: 右边缘)
  - 最终填充 Mask 图 (左边缘到右边缘之间)
  - 每张图的统计 JSON
  - 全量 CSV 汇总
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 使用无界面后端，便于批量保存
import matplotlib.pyplot as plt


def detect_left_right_edges(binary_image: np.ndarray):
    """逐行检测二值图的左/右边缘列位置。

    返回: left_edges, right_edges 列表，元素为 (row, col)，没有白像素则 col=-1。
    """
    left_edges = []
    right_edges = []

    for row in range(binary_image.shape[0]):
        row_pixels = binary_image[row, :]
        white_pixels = np.where(row_pixels == 255)[0]
        if len(white_pixels) > 0:
            left_edges.append((row, int(white_pixels[0])))
            right_edges.append((row, int(white_pixels[-1])))
        else:
            left_edges.append((row, -1))
            right_edges.append((row, -1))

    return left_edges, right_edges


def create_filled_mask(left_edges, right_edges, image_shape):
    """基于左右边缘线创建填充的 mask。"""
    mask = np.zeros(image_shape, dtype=np.uint8)
    for i, (left_row, left_col) in enumerate(left_edges):
        if left_col != -1 and i < len(right_edges):
            _, right_col = right_edges[i]
            if right_col != -1 and right_col >= left_col:
                mask[left_row, left_col:right_col + 1] = 255
    return mask


def draw_edges_on_binary(binary_image: np.ndarray, left_edges, right_edges) -> np.ndarray:
    """在二值图上绘制左右边缘点，返回 RGB 可视化图。"""
    vis = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    for row, col in left_edges:
        if col != -1:
            cv2.circle(vis, (col, row), 1, (255, 0, 0), -1)  # 红色
    for row, col in right_edges:
        if col != -1:
            cv2.circle(vis, (col, row), 1, (0, 0, 255), -1)  # 蓝色
    return vis


def compute_stats(image_name: str, otsu_binary: np.ndarray, left_edges, right_edges, filled_mask: np.ndarray) -> dict:
    """计算统计信息。"""
    stats = {
        "image": image_name,
        "width": int(otsu_binary.shape[1]),
        "height": int(otsu_binary.shape[0]),
        "otsu_on_original_mean": float(np.mean(otsu_binary)),
        "otsu_white_pixels": int(np.sum(otsu_binary == 255)),
        "otsu_black_pixels": int(np.sum(otsu_binary == 0)),
        "left_edge_points": int(len([e for e in left_edges if e[1] != -1])),
        "right_edge_points": int(len([e for e in right_edges if e[1] != -1])),
        "filled_white_pixels": int(np.sum(filled_mask == 255)),
        "filled_black_pixels": int(np.sum(filled_mask == 0)),
        "filled_ratio": float(np.sum(filled_mask == 255) / (filled_mask.shape[0] * filled_mask.shape[1] + 1e-7)),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    return stats


def process_one_image(image_path: str, output_dir: str) -> dict:
    """处理单张 ROI 图像，返回统计信息。"""
    image_name = os.path.basename(image_path)
    name_no_ext, _ = os.path.splitext(image_name)

    # 读取灰度
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"无法读取图像: {image_path}")

    # Otsu on Original
    _, otsu_binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 边缘与填充
    left_edges, right_edges = detect_left_right_edges(otsu_binary)
    filled_mask = create_filled_mask(left_edges, right_edges, otsu_binary.shape)

    # 保存可视化与 mask
    edges_vis = draw_edges_on_binary(otsu_binary, left_edges, right_edges)

    combined_path = os.path.join(output_dir, f"{name_no_ext}_combined.png")
    json_path = os.path.join(output_dir, f"{name_no_ext}_stats.json")

    # 生成四联图: 原图 -> Otsu -> otsu_with_edges -> final
    original_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    otsu_bgr = cv2.cvtColor(otsu_binary, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges_vis, cv2.COLOR_RGB2BGR)
    final_bgr = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)

    # 使用 matplotlib 1x4 横排子图展示，并保存
    h, w = original_bgr.shape[:2]
    titles = ["Original", "Otsu", "Edges", "Final"]

    # 转 RGB 以正确显示颜色
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    otsu_rgb = cv2.cvtColor(otsu_bgr, cv2.COLOR_BGR2RGB)
    edges_rgb = cv2.cvtColor(edges_bgr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(4 * w / 100.0, h / 100.0), dpi=100)
    ax_list = [axes[0], axes[1], axes[2], axes[3]]
    img_list = [original_rgb, otsu_rgb, edges_rgb, final_rgb]

    for ax, img, title in zip(ax_list, img_list, titles):
        ax.imshow(img)
        ax.set_title(title, color='red', fontsize=12)
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(combined_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 统计
    stats = compute_stats(image_name, otsu_binary, left_edges, right_edges, filled_mask)

    # 写单图 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def write_csv_summary(csv_path: str, records: list):
    """写入 CSV 汇总。"""
    headers = [
        "image", "width", "height",
        "otsu_on_original_mean", "otsu_white_pixels", "otsu_black_pixels",
        "left_edge_points", "right_edge_points",
        "filled_white_pixels", "filled_black_pixels", "filled_ratio", "timestamp"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def main():
    roi_dir = "data_shear_split/roi_images"
    output_dir = "data_shear_split/output/final_masks"

    if len(sys.argv) > 1:
        roi_dir = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    if not os.path.exists(roi_dir):
        raise RuntimeError(f"ROI 目录不存在: {roi_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 遍历图像
    image_files = [f for f in os.listdir(roi_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    image_files.sort()
    if not image_files:
        print("未在 ROI 目录中找到图像文件")
        return

    all_stats = []
    for fname in image_files:
        path = os.path.join(roi_dir, fname)
        try:
            stats = process_one_image(path, output_dir)
            all_stats.append(stats)
            print(f"处理完成: {fname}")
        except Exception as e:
            print(f"处理失败: {fname} - {e}")

    # 写 CSV 汇总
    csv_path = os.path.join(output_dir, "final_mask_summary.csv")
    write_csv_summary(csv_path, all_stats)
    print(f"CSV 汇总已保存: {csv_path}")


if __name__ == "__main__":
    main()



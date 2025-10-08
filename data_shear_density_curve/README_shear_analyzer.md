# 剪切面密度分析器 (Shear Density Analyzer)

## 概述

`shear_density_analyzer.py` 是专门用于分析剪切面斑块数量和密度变化曲线的工具。它基于现有的 `tear_density_analyzer.py` 修改而来，将分析目标从撕裂面改为剪切面。

## 主要功能

1. **剪切面检测**：使用 `ShearTearDetector` 检测图像中的剪切面区域
2. **斑块分析**：在剪切面区域内检测白色斑块
3. **密度计算**：计算剪切面区域密度和斑块密度
4. **时间序列分析**：分析剪切面斑块数量和密度随时间的变化
5. **可视化**：生成平滑的变化曲线图和统计图表

## 使用方法

### 基本用法

```bash
python shear_density_analyzer.py
```

### 指定输入输出目录

```bash
python shear_density_analyzer.py [roi_dir] [output_dir]
```

参数说明：
- `roi_dir`: ROI图像目录路径（默认：`/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data/roi_imgs`）
- `output_dir`: 输出结果目录路径（默认：`/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/data_shear_density_curve`）

## 输出结果

分析过程分为三个步骤，每个步骤都有对应的输出目录：

### Step 1: 剪切面Mask生成
- 目录：`step1_shear_masks/`
- 内容：
  - `shear_mask_contour_frame_XXXXXX.png`: 等高线方法生成的剪切面mask
  - `shear_mask_after_fill_frame_XXXXXX.png`: 填充后的剪切面mask

### Step 2: 剪切面区域提取
- 目录：`step2_shear_regions/`
- 内容：
  - `shear_region_frame_XXXXXX.png`: 提取的剪切面区域图像
  - `shear_patches_frame_XXXXXX.png`: 剪切面区域内的斑块检测结果

### Step 3: 分析结果和可视化
- 目录：`step3_patch_analysis/`
- 文件：
  - `shear_density_analysis.json`: JSON格式的分析结果
  - `shear_density_analysis.csv`: CSV格式的分析结果
  - `shear_density_analysis.png`: 剪切面密度和斑块数量变化曲线图
  - `shear_density_summary.png`: 综合统计图表

## 主要修改点

与原始的 `tear_density_analyzer.py` 相比，主要修改包括：

1. **类名**：`TearDensityAnalyzer` → `ShearDensityAnalyzer`
2. **分析方法**：`analyze_tear_density` → `analyze_shear_density`
3. **Mask提取**：从检测结果中提取 `shear_mask` 而不是 `tear_mask`
4. **分割值**：剪切面在分割图像中的值为255（白色），撕裂面为128（灰色）
5. **输出命名**：所有输出文件名和目录名都从 "tear" 改为 "shear"
6. **可视化标签**：图表标题和标签从"撕裂面"改为"剪切面"

## 平滑滤波

支持三种平滑滤波方法：

1. **高斯滤波**（默认）：使用 `scipy.ndimage.gaussian_filter1d`
2. **移动平均滤波**：使用卷积操作
3. **Savitzky-Golay滤波**：使用 `scipy.signal.savgol_filter`

默认参数：
- 滤波方法：高斯滤波
- 窗口大小：50
- 标准差：10.0

## 测试

运行测试脚本验证功能：

```bash
python test_shear_analyzer.py
```

## 依赖项

- OpenCV (cv2)
- NumPy
- Matplotlib
- Pandas
- SciPy
- scikit-image
- tqdm

## 注意事项

1. 确保输入目录包含格式为 `frame_XXXXXX_roi.png` 的ROI图像文件
2. 脚本会自动创建输出目录结构
3. 分析过程可能需要较长时间，具体取决于图像数量和分辨率
4. 建议在运行前确保有足够的磁盘空间存储中间结果和最终输出

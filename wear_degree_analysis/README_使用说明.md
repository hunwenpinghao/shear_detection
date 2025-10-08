# 剪刀磨损程度综合分析系统 - 使用说明

## 📋 概述

本系统整合了所有剪刀磨损分析功能到一个统一的主脚本 `main_analysis.py` 中，支持处理不同视频的数据。

### ✨ 主要功能

1. **特征提取** - 自动提取所有帧的磨损特征
2. **基础可视化** - 时序趋势、相关性分析、磨损递进
3. **增强可视化** - 峰值连线、周期对比、累积磨损、首尾对比
4. **深度趋势分析** - 峰值包络、分段趋势、低通滤波
5. **按卷分析** - 箱线图、统计图、热力图、雷达图
6. **最佳指标评估** - 评估并推荐最能反映磨损的指标
7. **平滑长期趋势** - 多种方法提取长期磨损趋势

## 🚀 快速开始

### 基础用法

```bash
# 使用默认设置（处理 ../data/roi_imgs 目录）
python main_analysis.py

# 查看完整帮助
python main_analysis.py --help
```

### 指定数据目录

```bash
# 处理指定目录的数据
python main_analysis.py \
  --data_dir /path/to/your/roi_images \
  --output_dir /path/to/output

# 例如：处理 Video2 的数据
python main_analysis.py \
  --data_dir ../data_video2/roi_imgs \
  --output_dir ./results_video2
```

### 只运行特定模块

```bash
# 只运行基础可视化和按卷分析
python main_analysis.py --modules basic coil

# 只运行增强可视化
python main_analysis.py --modules enhanced

# 运行所有模块（默认）
python main_analysis.py --modules all
```

### 使用已有特征数据

如果已经提取过特征，可以跳过特征提取步骤：

```bash
python main_analysis.py \
  --skip_extraction \
  --features_csv ./results/features/wear_features.csv \
  --modules enhanced coil
```

### 自定义钢卷参数

```bash
# 指定钢卷数量和起始编号
python main_analysis.py \
  --n_coils 12 \
  --coil_start_id 1

# 例如：分析第1-8卷
python main_analysis.py \
  --n_coils 8 \
  --coil_start_id 1
```

## 📊 输出结果

所有结果保存在 `results/` 目录下：

```
results/
├── features/
│   ├── wear_features.csv              # 特征数据（CSV格式）
│   ├── wear_features.json             # 特征数据（JSON格式）
│   ├── indicator_evaluation.csv       # 指标评估结果
│   └── trend_analysis_summary.csv     # 趋势分析摘要
│
├── visualizations/
│   ├── # 基础可视化
│   ├── temporal_trends.png
│   ├── feature_correlations.png
│   ├── wear_progression.png
│   │
│   ├── # 增强可视化
│   ├── peaks_trend.png
│   ├── cycle_comparison.png
│   ├── cumulative_wear.png
│   ├── first_last_comparison.png
│   │
│   ├── # 深度分析
│   ├── envelope_analysis.png
│   ├── segment_analysis.png
│   ├── longterm_trend.png
│   │
│   ├── # 按卷分析
│   ├── coil_by_coil_boxplot.png
│   ├── coil_by_coil_bars.png
│   ├── coil_heatmap.png
│   ├── coil_progression_detailed.png
│   ├── coil_radar_comparison.png
│   │
│   ├── # 指标评估
│   ├── best_indicators_comparison.png
│   ├── recommended_indicators.png
│   │
│   ├── # 平滑趋势
│   ├── smooth_method1_envelope.png
│   ├── smooth_method2_peaks.png
│   ├── smooth_method3_global.png
│   └── smooth_comparison_final.png
│
└── analysis_report.md                 # 综合分析报告
```

## 🔧 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | `../data/roi_imgs` | 数据目录（包含frame_*_roi.png图像） |
| `--output_dir` | str | 当前目录 | 输出目录 |
| `--max_frames` | int | 自动检测 | 最大处理帧数 |
| `--n_coils` | int | 9 | 钢卷数量 |
| `--coil_start_id` | int | 4 | 起始钢卷编号 |
| `--skip_extraction` | flag | False | 跳过特征提取 |
| `--features_csv` | str | None | 已有特征文件路径 |
| `--modules` | list | `['all']` | 要运行的分析模块 |

### 可用模块

- `basic` - 基础可视化
- `enhanced` - 增强可视化  
- `deep` - 深度趋势分析
- `coil` - 按卷分析
- `indicator` - 最佳指标评估
- `smooth` - 平滑长期趋势
- `all` - 运行所有模块（默认）

## 💡 使用场景示例

### 场景1：处理新视频数据

```bash
# 1. 将视频帧提取到 roi_imgs 目录
# 2. 运行完整分析
python main_analysis.py \
  --data_dir ./my_video/roi_imgs \
  --output_dir ./my_video_analysis \
  --n_coils 10 \
  --coil_start_id 1
```

### 场景2：快速查看某个模块的结果

```bash
# 已经提取过特征，只想重新生成按卷分析图表
python main_analysis.py \
  --skip_extraction \
  --features_csv ./results/features/wear_features.csv \
  --modules coil
```

### 场景3：对比不同视频

```bash
# 视频1
python main_analysis.py \
  --data_dir ./video1/roi_imgs \
  --output_dir ./analysis_video1 \
  --n_coils 9

# 视频2  
python main_analysis.py \
  --data_dir ./video2/roi_imgs \
  --output_dir ./analysis_video2 \
  --n_coils 12

# 然后对比两个输出目录的结果
```

### 场景4：批量处理

创建脚本 `batch_process.sh`:

```bash
#!/bin/bash

# 处理多个视频
videos=("Video1" "Video2" "Video3")

for video in "${videos[@]}"; do
    echo "Processing $video..."
    python main_analysis.py \
        --data_dir "./data_${video}/roi_imgs" \
        --output_dir "./results_${video}" \
        --n_coils 9 \
        --coil_start_id 4
done

echo "All videos processed!"
```

运行：
```bash
chmod +x batch_process.sh
./batch_process.sh
```

## 📁 数据目录结构要求

数据目录应包含以下格式的图像文件：

```
data_dir/
├── frame_000000_roi.png
├── frame_000001_roi.png
├── frame_000002_roi.png
├── ...
└── frame_NNNNNN_roi.png
```

**注意**：
- 文件名格式必须为 `frame_NNNNNN_roi.png`（其中N为6位数字）
- 图像应为ROI（感兴趣区域）裁剪后的灰度图

## ⚙️ 系统要求

### Python 版本
- Python 3.7+

### 依赖库
```bash
pip install numpy pandas opencv-python matplotlib scipy tqdm
```

或使用 requirements.txt:
```bash
pip install -r ../requirements.txt
```

## 🔍 故障排查

### 问题1：找不到图像文件

**错误信息**：`⚠ 警告: 数据目录未找到图像文件`

**解决方法**：
- 检查 `--data_dir` 参数是否正确
- 确认目录下有 `frame_*_roi.png` 格式的文件
- 检查文件权限

### 问题2：内存不足

**症状**：处理大量帧时系统卡死

**解决方法**：
- 使用 `--max_frames` 参数限制处理帧数
- 先提取特征，再分批运行可视化模块

```bash
# 先提取特征
python main_analysis.py --modules basic

# 再运行其他模块
python main_analysis.py \
  --skip_extraction \
  --features_csv ./results/features/wear_features.csv \
  --modules enhanced deep coil
```

### 问题3：中文显示乱码

**解决方法**：
确保系统安装了中文字体：
- macOS: 已内置
- Linux: `sudo apt-get install fonts-wqy-microhei`
- Windows: 通常已内置

## 📝 旧脚本说明

为了保持兼容性，以下独立脚本仍然保留：

- `enhanced_visualization.py` - 增强可视化
- `deep_trend_analysis.py` - 深度趋势分析
- `coil_by_coil_analysis.py` - 按卷分析
- `best_indicator_analysis.py` - 最佳指标评估
- `smooth_longterm_trend.py` - 平滑长期趋势

这些脚本可以独立运行，但推荐使用新的 `main_analysis.py`。

## 📮 联系与反馈

如有问题或建议，请联系开发团队。

---

**最后更新**: 2025-10-02


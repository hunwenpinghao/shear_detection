# 剪刀磨损按卷分析工具

## 📋 概述

这是一个通用的剪刀磨损分析工具，能够：
- ✨ **自动检测钢卷边界** - 基于统计变化点检测，无需手动指定
- 📊 **多特征融合分析** - 综合梯度能量、缺口深度、粗糙度等特征
- 🎯 **智能分割** - 自适应识别不同长度的钢卷
- 📈 **完整可视化** - 自动生成5种分析图表

## 🚀 快速开始

### 基本用法

```bash
python coil_wear_analysis.py \
  --roi_dir <ROI图像目录> \
  --output_dir <输出目录>
```

### 实际示例

```bash
# 分析视频1
python coil_wear_analysis.py \
  --roi_dir data_video_20250821152112032/first_cycle/roi_imgs \
  --output_dir data_video_20250821152112032/first_cycle/coil_analysis \
  --name "第一周期"

# 分析视频2  
python coil_wear_analysis.py \
  --roi_dir data_video2_20250821140339629/roi_imgs \
  --output_dir data_video2_20250821140339629/coil_analysis \
  --name "Video2分析"
```

## ⚙️ 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--roi_dir` | ✅ 是 | - | ROI图像目录路径 |
| `--output_dir` | ✅ 是 | - | 输出目录路径 |
| `--name` | ❌ 否 | "视频分析" | 分析名称 |
| `--min_coils` | ❌ 否 | 5 | 最小钢卷数（用于限制检测范围） |
| `--max_coils` | ❌ 否 | 15 | 最大钢卷数（用于限制检测范围） |

## 🔍 工作原理

### 1. 特征提取
对每一帧图像提取以下特征：
- RMS粗糙度
- 梯度能量（锐度）
- 最大缺口深度
- 峰密度
- 面积比

### 2. 自动检测算法

```
输入: 帧序列特征数据
│
├─ 多特征平滑处理 (Savitzky-Golay滤波)
│  └─ 去除噪声，保留趋势
│
├─ 特征标准化
│  └─ 消除量纲影响
│
├─ Pelt变化点检测
│  ├─ 检测特征突变点
│  └─ 识别钢卷切换边界
│
└─ 输出: 钢卷边界 + 卷号分配
```

**核心算法**: Pelt (Pruned Exact Linear Time)
- 时间复杂度: O(n log n)
- 优点: 快速、准确、无需预设断点数
- 原理: 最小化段内方差，最大化段间差异

### 3. 为什么不需要手动指定钢卷数？

传统方法的问题：
- ❌ 不同视频的钢卷数可能不同
- ❌ 每个钢卷的长度可能不均匀
- ❌ 人工指定容易出错

自动检测的优势：
- ✅ 基于数据本身的特征变化
- ✅ 自适应识别不同长度的钢卷
- ✅ 更符合实际生产情况

### 示例：自动检测结果

```
✓ 检测到 10 个钢卷
边界位置: [0, 15, 30, 45, 65, 85, 105, 120, 140, 170]

每卷帧数分布:
  第1-3卷: 15帧  ← 较短的钢卷
  第4-6卷: 20帧  ← 标准长度
  第9卷: 30帧    ← 自动识别长钢卷
  第10卷: 13帧   ← 最后一卷
```

## 📊 输出内容

分析完成后，会在输出目录生成：

```
output_dir/
├── features/
│   ├── wear_features.csv              # 原始特征数据
│   └── wear_features_with_coils.csv   # 带钢卷编号的特征
│
├── visualizations/
│   ├── coil_by_coil_boxplot.png       # 箱线图对比
│   ├── coil_by_coil_bars.png          # 柱状图统计
│   ├── coil_heatmap.png               # 热力图
│   ├── coil_radar_comparison.png      # 雷达图对比
│   └── coil_progression_detailed.png  # 逐卷趋势（推荐！）
│
└── analysis_report.md                  # 分析报告
```

## 📈 核心特征解释

### 1. RMS粗糙度
- **含义**: 边界起伏程度
- **磨损表现**: 递增 → 边缘变粗糙

### 2. 梯度能量（锐度）
- **含义**: 刀口锐利程度
- **磨损表现**: 下降 → 刀口变钝 ✓

### 3. 最大缺口深度
- **含义**: 边界最大凹陷
- **磨损表现**: 递增 → 缺口加深 ✓

### 4. 右侧峰密度
- **含义**: 剪切面微缺口数量
- **磨损表现**: 递增 → 微损伤增多

## 🎯 判断标准

**磨损递增的证据**：
1. ✅ 梯度能量显著下降（-20%以上）
2. ✅ 最大缺口深度显著递增（+20%以上）
3. ✅ 逐卷递增次数 > 50%

**示例结果分析**：
```
【梯度能量】
  变化率: -36.4%  ← 锐度大幅下降
  逐卷递增次数: 3/8 = 38%
  ✓ 刀口磨钝，符合磨损预期

【最大缺口深度】
  变化率: +24.6%  ← 缺口显著加深
  逐卷递增次数: 5/8 = 62%
  ✓ 磨损加重，符合预期
```

## 🔧 高级用法

### 调整检测范围

如果知道钢卷数大致范围，可以调整：

```bash
python coil_wear_analysis.py \
  --roi_dir data/roi_imgs \
  --output_dir data/analysis \
  --min_coils 8 \
  --max_coils 12  # 限制在8-12个钢卷
```

### 批量处理

```bash
#!/bin/bash
# 批量处理多个视频

videos=(
  "video1/roi_imgs"
  "video2/roi_imgs"
  "video3/roi_imgs"
)

for video in "${videos[@]}"; do
  python coil_wear_analysis.py \
    --roi_dir "$video" \
    --output_dir "${video%/roi_imgs}/analysis" \
    --name "$(basename $(dirname $video))"
done
```

## 📚 技术栈

- **变化点检测**: `ruptures` (Pelt算法)
- **特征提取**: `opencv-python`, `scipy`
- **数据处理**: `pandas`, `numpy`
- **可视化**: `matplotlib`
- **标准化**: `scikit-learn`

## 🐛 常见问题

### Q1: 检测失败怎么办？
A: 脚本会自动回退到均匀分割（9个钢卷）。如果经常失败：
1. 检查特征提取成功率
2. 调整 `min_coils` 和 `max_coils` 范围
3. 检查ROI图像质量

### Q2: 为什么有些视频成功率低？
A: 可能原因：
- 图像质量差
- ROI提取不准确
- 预处理参数需要调整

### Q3: 如何验证检测结果？
A: 查看生成的可视化图表：
- `coil_progression_detailed.png` - 观察趋势是否合理
- 分析报告中的边界位置 - 是否符合预期

## 📝 版本历史

### v2.0 (当前版本)
- ✨ 完全自动化，移除手动指定参数
- 🔍 基于Pelt算法的智能检测
- 📊 多特征融合提高准确性
- 🎯 自适应钢卷长度

### v1.0 (已废弃)
- ❌ 需要手动指定钢卷数
- ❌ 均匀分割，不够灵活

## 🤝 贡献

欢迎提出改进建议！

## 📧 联系方式

如有问题，请查看 `wear_degree_analysis/README_使用说明.md` 获取更多技术细节。


# 毛刺密度分析系统

## 概述
本系统用于分析ROI图像中的毛刺密度随时间的变化，生成毛刺密度曲线图。

## 功能特点
- 自动检测ROI图像中的毛刺特征
- 生成毛刺mask和可视化图像
- 计算毛刺密度和数量统计
- 生成平滑的密度变化曲线
- 保存中间处理步骤的图像

## 目录结构
```
data_burr_density_curve/
├── burr_density_analyzer.py    # 主分析脚本
├── README.md                   # 说明文档
├── requirements.txt            # 依赖包
├── step1_burr_masks/          # 第一步：毛刺mask
├── step2_burr_regions/        # 第二步：毛刺区域和可视化
├── step3_density_analysis/    # 第三步：密度分析结果
├── burr_density_analysis.json # 分析结果JSON
├── burr_density_analysis.csv  # 分析结果CSV
├── burr_density_analysis.png  # 密度曲线图
└── burr_density_summary.png   # 综合统计图
```

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行分析
```bash
python burr_density_analyzer.py
```

## 处理步骤

### 第一步：生成毛刺mask
- 对每个ROI图像进行毛刺检测
- 生成二值毛刺mask
- 保存到 `step1_burr_masks/` 目录

### 第二步：提取毛刺区域
- 应用毛刺mask过滤原图
- 生成毛刺区域图像
- 创建毛刺可视化图像
- 保存到 `step2_burr_regions/` 目录

### 第三步：密度分析
- 计算毛刺密度和数量
- 生成时间序列数据
- 应用平滑滤波
- 生成密度曲线图
- 保存到 `step3_density_analysis/` 目录

## 输出文件

### 数据文件
- `burr_density_analysis.json`: 详细分析结果
- `burr_density_analysis.csv`: 表格格式结果

### 图像文件
- `burr_density_analysis.png`: 毛刺密度随时间变化曲线
- `burr_density_summary.png`: 综合统计图表

### 中间文件
- `step1_burr_masks/`: 毛刺检测mask
- `step2_burr_regions/`: 毛刺区域和可视化图像

## 技术特点
- 使用高斯滤波平滑曲线
- 支持中文字体显示
- 自动保存中间处理步骤
- 生成详细的统计报告

# 撕裂面密度分析

## 功能说明

本工具用于分析撕裂面的斑块数量和密度随时间的变化曲线。

## 主要功能

1. **撕裂面检测**: 使用现有的撕裂面检测算法获取撕裂面mask
2. **斑块分析**: 统计撕裂面中的斑块数量、面积、分布等特征
3. **密度计算**: 计算撕裂面在图像中的密度百分比
4. **时间序列分析**: 分析撕裂面特征随时间的变化趋势
5. **可视化**: 生成多种图表展示分析结果

## 文件结构

```
data_replot_tear_density_curve/
├── tear_density_analyzer.py    # 主分析脚本
├── requirements.txt            # 依赖包列表
├── README.md                  # 说明文档
└── output/                    # 输出结果目录
    ├── tear_density_analysis.json      # JSON格式分析结果
    ├── tear_density_analysis.csv       # CSV格式分析结果
    ├── tear_density_analysis.png       # 主要分析图表
    └── tear_density_summary.png        # 综合统计图
```

## 使用方法

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 运行分析:
```bash
python tear_density_analyzer.py
```

## 分析指标

- **撕裂面密度**: 撕裂面像素占总像素的百分比
- **斑块数量**: 撕裂面中连通组件的数量
- **平均斑块面积**: 所有斑块的平均面积
- **斑块间距**: 斑块中心点之间的平均距离
- **斑块面积分布**: 斑块面积的统计分布

## 输出结果

- JSON文件: 包含所有分析数据的结构化结果
- CSV文件: 便于Excel等工具进一步分析
- PNG图表: 可视化分析结果，包括时间序列图和统计分布图

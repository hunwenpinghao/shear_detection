# 剪刀磨损检测 SAM2 版本 更新报告

## 添加内容总结 
+ SAM2_MODULE: sam2_segmentator.py 基于 facebook/sam2.1-hiera-tiny
- SAM2_CONFIG 在 config.py 新增
- 主 API segment_surface(params) 增加 sam2 分支
- 处理类：segment_by_sam2 方法
- 可视化绘图函数已增列
- 术语翻译与模板完备
- 已知限制：需要安装 sam2 可来实际使用

使用时可以运行line命令切换modelname:

python main.py --method sam2

## 命令使用

- `python -m main --method curved`  用于曲线分割（之前的状态）
- `python -m main --method sam2`    使用新的SAM2 method（最新）

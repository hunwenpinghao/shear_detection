项目背景：我们有一些类似`data/Image_20250710125452500.bmp`格式的线阵相机拍摄的到的图片。该图片中白色条状物拍摄的是钢材被圆盘剪刀剪出的截面，理论上来说，该条状物可以分为“撕裂面”和“剪切面”，图中两个面左右两边宽度不等，我的任务是设计一个cv算法学习根据图片截面的特征判断是否该更换剪刀。 根据经验更换剪刀时一般撕裂面：剪切面的比例会变大。且撕裂面会出现较多的白色斑块，一般几个小时会更换一次，我想判断出最佳换剪刀的时间。 要求： 算法要实时检测；需要学到撕裂面和剪切面的比例变化曲线；需要学到撕裂面斑块数目变化曲线或趋势； 请你帮我设计一个算法方案，要根据这种图的特点进行分析选择最合适的算法。并写出具体的数据集制作方法和制作规模；设计好评测指标；大概给出可能的性能评估；

代码方案设计草稿：
```Markdown
下面是一套从**思路 → 算法设计 → 数据集制作 → 训练/评估 → 部署与运行时决策**的完整方案，专门针对你这种线阵相机拍到的 `.bmp` 截面图（“白色条状物”包含 **撕裂面** 与 **剪切面** 两类区域），目标是**实时检测并给出剪刀更换时机**（基于撕裂面/剪切面比例和撕裂面白斑数量的时序变化）。

我把方案分成若干模块，每个模块给出可立刻实施的细节、超参建议、评测指标以及可能的性能预估。若要我直接给训练代码/训练脚本、或根据你现有样本做一次原型（例如用你容器里的 `.bmp` 做小试验），我可以立刻做，但本回答先把算法与工程蓝图梳理完整。

# 1 关键设计思路（要点）

- 目标量化指标
  - 撕裂面 / 剪切面 比例：R(t)=Atear(t)Ashear(t) 或可用归一化比例 Ptear(t)=AtearAtear+Ashear。
  - 撕裂面白色斑块数（或斑块密度）：B(t)（connected components 数或大于阈值的白点像素占比）。
- 两条并行路线（短期 + 长期）
  - **短期（可立刻部署，轻量）**：传统 CV 规则化流水线（阈值、形态学、连通域等）提取两个区域与斑块计数。优点快速、易解释、实时延迟低。
  - **长期（更稳健）**：轻量语义分割 / 二分类像素级网络（例如小型 U-Net / DeepLab-MobileNet / SegFormer-lite）用于将像素分为：`background` / `撕裂面` / `剪切面`。网络输出更健壮，适应光照、灰度漂移、噪声。
- 时序建模与决策
  - 对 Ptear(t) 与 B(t) 做滑窗/平滑（EMA）并存储时间序列；
  - 使用统计/变化点检测（CUSUM、Bayesian online changepoint、or simple threshold + hysteresis）来触发“预警/更换建议”；
  - 可选：用小型时间序列模型（ARIMA / LightGBM on windows / LSTM）学习趋势并预测“到何时会超过阈值”以得到更佳换刀时机（提前预警）。

# 2 详细算法流水线（最终部署建议：短期规则+长期模型并行，输出融合）

## 2.1 输入与预处理（通用）

- 输入：单幅线阵截面图（灰度或单通道白底黑背景常见）
- 预处理：
  1. 读入并转为灰度（若是单通道可跳过）。
  2. 直方图/CLAHE 自适应对比度增强（对抗光照漂移）。
  3. 高斯去噪（kernel 3~5）。
  4. 如线阵一维特性明显，先裁 ROI：选定图像中心竖条或横条，减少计算量。
  5. 标准化到固定分辨率（例如长边保持原始像素，短边 resize 到 256 或 512 以保证速度/精度平衡）。

## 2.2 传统 CV 管道（轻量可实时）

- 二值化：自适应阈值（局部 Otsu 或 Sauvola），或直接 Otsu 后结合固定阈值。
- 形态学：闭运算消除小空洞，开运算去除细小噪点（kernel 3~7）。
- 连通域与轮廓提取：找到白色连通区域 → 根据形状（长宽比、面积）把主要条状物分离出来；条状物左右分成两个走向（依据中轴线或边界点），然后在条状物内部进一步用边界/梯度区分“撕裂面”与“剪切面”：
  - 分割策略（启发式）：沿条状物中心线计算像素强度梯度或边界曲率，剪切面通常更平整、边界连续；撕裂面边界更锯齿/不规则且含斑点。
  - 如果两个面左右明显，用中轴或轮廓切分（根据几何先验：两个面左右两侧宽度不等）。
- 斑块计数：在撕裂面区域内做开操作后找连通小块，按面积阈值（>A_min）计数，得到 B(t)。
- 输出：Atear,Ashear,Ptear,B。

优点：实现快（CPU即可），可在数十到数百 FPS（取决于分辨率与CPU）运行。

## 2.3 深度学习语义分割管道（长期，稳健）

- 标签：像素级三类：`0 background`, `1 tear`, `2 shear`。
- 网络备选（轻量、易部署）：
  - MobileNetV3-UNet-like（encoder: MobileNetV3 small，decoder: lightweight upsampling）；
  - DeepLabV3-MobileNet 或 SegFormer-lite；
  - 若你要极致速度：ENet / FastSCNN。
- 损失：`CrossEntropy + DiceLoss`（处理类不平衡）。
- 后处理：连通域、面积过滤，输出同上指标。网络输出更好的边界、能容忍光照/噪声变化。
- 训练/推理：
  - 输入尺寸：512×128 或 1024×256，取决于线阵图像的形状（宽高比）。
  - batch size 与 lr 按 GPU 规格调整。
  - 模型导出：ONNX → TensorRT / OpenVINO / TVM，以实现低延迟部署。

## 2.4 时序决策（如何判断“该换刀”）

- 实时计算滑动窗口（例如 last 5~60 分钟的窗口，或 last N 帧）：
  - 使用指数移动平均（EMA）： P^tear(t)=αPtear(t)+(1−α)P^tear(t−1) ，α 取 0.1~0.3。
  - 计算 B(t) 的移动均值与斜率（用线性回归拟合窗口内点）。
- 变化点/告警策略（组合）：
  1. **阈值+滞后**：若 Ptear 连续超过阈值 TP（例如 0.6）且持续超过 Tdur（例如 5 分钟），触发警告。
  2. **增长率**：若短时段内 Ptear 增幅 > Rdelta（例如 10%）触发预警。
  3. **斑块数**：若 B(t) 超过阈值或增速显著，触发更高等级警报。
  4. **融合逻辑**：两者同时异常（高 Ptear 且高 B）则立即建议更换；单一异常则建议观察/短期复测。
- 进一步：用 CUSUM 或 Bayesian change point 检测更鲁棒地捕捉突变点并减少误报。

# 3 数据集制作（关键与详尽步骤）

> 目标：训练稳健语义分割模型并验证时序决策。数据集要覆盖“正常刀具”→“磨损/待换刀”全过程（多个刀具/班次/光照/材料差异）。

## 3.1 标注格式

- 每张图片保存：
  - 原图 `.bmp` 或 `.png`；
  - 像素标签 mask（单通道 uint8）：0=background, 1=tear, 2=shear；
  - 元数据 `.json`：`{"timestamp": "...", "tool_id": "...", "operator": "...", "shift": "...", "temperature": "...", "notes": "..."}`
- 另外，为时序任务每 N 帧记录一个“人工更换标签”：`change_time`（精确到帧或分钟）。

## 3.2 数据量（推荐）

- **最小可用集（快速原型）**：
  - 1,000 ~ 2,000 帧（多把刀，多次更换周期），尽量包含不同磨损阶段。
- **建议生产级集**：
  - 5,000 ~ 20,000 帧：覆盖 10~20 把刀、每把刀在使用从新到报废（或换刀）的完整周期中均抽取样本。
  - 每把刀至少 300~1,000 帧，且在更换前后高密度采样（例如每分钟 1~5 帧更高采样率）。
- **标注分配**：
  - Train/Val/Test 按刀具周期拆分（非常重要）：例如用部分刀具的完整周期作为训练，留出不同时间/不同刀具用于测试，避免过拟合到单把刀的特性。
```

第一步：
先给出预处理步骤的代码，要求： 
  1.我们首先应该提取图片的有效部分，因为从图片来看图片存在大面积黑色背景，白色条状物仅占比很小的部分，所以，我们需要利用你上面提到的各种预处理手段将图片处理成主要包含条状物的图片部分，然后处理成利于用算法去学习或统计的格式； 
  2.你应该先给出具体处理步骤以及原理，再给出具体的代码实现，并用上面的图片进行测试；

--------- 后面步骤先不做 --------

---

## 最新更新记录

### 2025-10-11: 主分析脚本优化 - 钢卷边界检测大幅加速

**🚀 重大改进：新增波谷检测法**

根据用户建议，实现了基于**二次滤波 + 波谷检测**的新方法，取代原有的慢速 Pelt 搜索算法：

#### 核心原理
1. **二次平滑滤波**：过滤假波谷和噪声
   - 第一次：大窗口（201帧）滤波
   - 第二次：中窗口（151帧）滤波
2. **波谷检测**：识别局部最小值点（钢卷交界处）
3. **山丘分割**：相邻波谷之间为一个钢卷

#### 性能对比

| 方法 | 速度 | 适用场景 | 说明 |
|------|------|----------|------|
| **波谷检测法（推荐）** | ⚡⚡⚡ 极快 | 通用场景 | 二次滤波+波谷，直观高效 |
| Pelt 算法 | 🐌 慢 | 复杂边界 | 需搜索penalty，耗时但精确 |
| 指定钢卷数 | ⚡⚡⚡⚡⚡ 最快 | 已知钢卷数 | 跳过检测，速度最快 |

**实测数据（33333帧）**：
- 原 Pelt 搜索：~30秒（搜索15个penalty值）
- 新波谷检测：~3秒（提速**10倍**）
- 指定钢卷数：~2秒（提速**15倍**）

#### 使用方法

```bash
# 方法1：波谷检测法（推荐，默认）
python coil_wear_analysis.py \
  --roi_dir data/roi_imgs \
  --output_dir data/analysis

# 方法2：指定钢卷数（最快）
python coil_wear_analysis.py \
  --roi_dir data/roi_imgs \
  --output_dir data/analysis \
  --n_coils 10

# 方法3：Pelt算法（较慢但可能更精确）
python coil_wear_analysis.py \
  --roi_dir data/roi_imgs \
  --output_dir data/analysis \
  --detection_method pelt

# 自定义波谷检测参数
python coil_wear_analysis.py \
  --roi_dir data/roi_imgs \
  --output_dir data/analysis \
  --detection_method valley \
  --min_coils 8 --max_coils 12
```

#### 输出示例

**波谷检测法输出**：
```
检测钢卷边界...
🔍 自动检测模式：钢卷数范围 5-15个
📊 使用波谷检测法（推荐，快速且直观）
🌊 使用波谷检测法识别钢卷边界...
使用综合磨损指数
第一次平滑：窗口大小 201
第二次平滑：窗口大小 151
波谷检测参数：最小距离=2000帧, 显著性阈值=0.145
检测到 9 个波谷
✓ 检测到 10 个钢卷
边界位置（波谷）: [2156, 4890, 7623, 10356, 13089, 16822, 19555, 22288, 25021, 27754]
  第1卷: 2156帧 (帧 0 → 2156)
  第2卷: 2734帧 (帧 2156 → 4890)
  第3卷: 2733帧 (帧 4890 → 7623)
  ...
```

**优势总结**：
- ✅ 速度快：无需搜索参数
- ✅ 直观：物理意义明确（波谷=钢卷交界）
- ✅ 鲁棒：二次滤波过滤噪声
- ✅ 灵活：自动调整或手动指定范围

---

### 2025-10-11: 时间序列分析脚本优化 - 参数化增强 + Bug修复

**功能说明**：
- 为 `analyze_spot_temporal.py` 和 `analyze_split_temporal_filter_tear.py` 添加完整的命令行参数支持
- 支持自定义输入输出路径、平滑方法和可视化间隔
- 提供灵活的配置选项，提升脚本复用性
- 修复尺寸不匹配导致的索引错误

**🐛 Bug修复**：
- **问题**：`analyze_split_temporal_filter_tear.py` 在处理某些帧时报错：`boolean index did not match indexed array along dimension 0`
- **原因**：`shear_tear_detector.detect_surfaces()` 返回的 `segmented_image` 与输入的 `roi_image` 尺寸不一致
- **修复位置1**：在第一步缓存 `segmented_image` 前添加尺寸检查和调整（根本性修复）
- **修复位置2**：在 `filter_tear_region_with_shear_mask()` 方法中添加尺寸检查（双重保险）
- **修复方法**：使用 `cv2.resize()` 和 `INTER_NEAREST` 插值自动调整不匹配的分割图像尺寸
- **影响**：确保所有帧都能正常处理，不会因尺寸不一致而中断

#### 涉及脚本

1. **`analyze_spot_temporal.py`** - 整体ROI斑块时间序列分析
   - 分析整个ROI图像的斑块数量和密度随时间变化
   - 不区分撕裂面和剪切面，统计所有斑块

2. **`analyze_split_temporal_filter_tear.py`** - 撕裂面斑块时间序列分析
   - 使用剪切面mask过滤出撕裂面区域
   - 专门分析撕裂面的斑块数量和密度
   - 生成多步骤可视化（mask、过滤区域、斑块检测）

#### 新增参数

**共同参数**：
```bash
--roi_dir          # 必需：ROI图像目录路径
--output_dir       # 必需：输出目录路径
--smoothing_method # 平滑方法（gaussian/moving_avg/savgol/median，默认：gaussian）
--window_size      # 滤波窗口大小（默认：50）
--sigma            # 高斯滤波标准差（默认：10.0）
--viz_interval     # 可视化保存间隔（默认：100，0=不保存可视化）
```

**analyze_split_temporal_filter_tear.py 特有参数**：
```bash
--skip_step1       # 跳过第一步的可视化保存（仍会进行检测以构建缓存）
```

#### 使用方法

**analyze_spot_temporal.py**（整体ROI分析）：
```bash
# 基本使用
python analyze_spot_temporal.py \
  --roi_dir data/roi_imgs \
  --output_dir output/spot_analysis

# 指定平滑参数和可视化间隔（每100帧保存一次）
python analyze_spot_temporal.py \
  --roi_dir data/roi_imgs \
  --output_dir output/spot_analysis \
  --smoothing_method gaussian \
  --sigma 10.0 \
  --window_size 50 \
  --viz_interval 100

# 使用Savitzky-Golay滤波
python analyze_spot_temporal.py \
  --roi_dir data/roi_imgs \
  --output_dir output/spot_analysis \
  --smoothing_method savgol \
  --window_size 51

# 不保存每帧可视化（仅生成数据和曲线）
python analyze_spot_temporal.py \
  --roi_dir data/roi_imgs \
  --output_dir output/spot_analysis \
  --viz_interval 0
```

**analyze_split_temporal_filter_tear.py**（撕裂面过滤分析）：
```bash
# 基本使用
python analyze_split_temporal_filter_tear.py \
  --roi_dir data/roi_imgs \
  --output_dir output/tear_filter_analysis

# 指定平滑参数和可视化间隔
python analyze_split_temporal_filter_tear.py \
  --roi_dir data/roi_imgs \
  --output_dir output/tear_filter_analysis \
  --smoothing_method gaussian \
  --sigma 10.0 \
  --window_size 50 \
  --viz_interval 100

# 跳过第一步（第一步已运行过，想直接运行第二步和第三步）
python analyze_split_temporal_filter_tear.py \
  --roi_dir data/roi_imgs \
  --output_dir output/tear_filter_analysis \
  --skip_step1

# 跳过第一步 + 自定义可视化间隔
python analyze_split_temporal_filter_tear.py \
  --roi_dir data/roi_imgs \
  --output_dir output/tear_filter_analysis \
  --skip_step1 \
  --viz_interval 200

# 不保存任何可视化（仅生成数据和曲线）
python analyze_split_temporal_filter_tear.py \
  --roi_dir data/roi_imgs \
  --output_dir output/tear_filter_analysis \
  --viz_interval 0
```

#### 输出目录结构

**analyze_spot_temporal.py 输出**：
```
output/spot_analysis/
├── spot_visualizations/          # 每帧斑块检测可视化（按viz_interval采样）
│   ├── frame_000000_spots.png
│   ├── frame_000100_spots.png
│   └── ...
├── spot_temporal_analysis_smoothed.png    # 时间序列曲线图
├── spot_statistics_summary.png            # 统计摘要图
├── spot_temporal_data.csv                 # 数据CSV文件
└── spot_temporal_data.json                # 数据JSON文件
```

**analyze_split_temporal_filter_tear.py 输出**：
```
output/tear_filter_analysis/
├── step1_shear_tear_masks/              # 剪切面和撕裂面mask（按viz_interval采样）
│   ├── shear_mask_frame_000000.png
│   ├── tear_mask_frame_000000.png
│   └── ...
├── step2_filtered_tear_regions/         # 过滤后的撕裂面区域（按viz_interval采样）
│   ├── filtered_tear_region_frame_000000.png
│   ├── tear_patches_frame_000000.png
│   ├── tear_patch_visualization_frame_000000.png
│   └── ...
├── step3_tear_patch_analysis/
│   ├── tear_spot_temporal_analysis_smoothed.png    # 时间序列曲线图
│   ├── tear_spot_statistics_summary.png            # 统计摘要图
│   ├── tear_spot_temporal_data.csv                 # 数据CSV文件
│   └── tear_spot_temporal_data.json                # 数据JSON文件
```

#### 技术实现

**可视化采样策略**：
- ⚠️ **重要**：所有帧都会完整计算，只有可视化保存按间隔采样
- 中间结果（如mask、分割结果）缓存在内存中供后续步骤使用
- 数值分析（斑块数量、密度）覆盖所有帧
- 仅磁盘I/O（图像保存）按间隔进行

**⚡ 性能优化（第二步加速）**：
- **优化前**：每帧都调用 `spot_processor.process_single_roi_spots()`
  - 读取图像
  - 检测斑块
  - **创建matplotlib可视化**（耗CPU/内存）
  - **保存到磁盘**（耗时I/O）
  - 不需要的帧再删除（又一次I/O）
  
- **优化后**：分离计算和可视化
  - 所有帧：直接调用 `feature_extractor.detect_all_white_spots()`（仅计算）
  - 采样帧：才创建和保存可视化
  - 避免不必要的matplotlib渲染和磁盘I/O
  
- **预期提速**：第二步处理速度提升 **5-10倍**（取决于viz_interval）

**步骤控制（`--skip_step1`）**：
- **适用场景**：第一步已经运行过，想直接从第二步开始
- **工作原理**：
  - 仍会进行第一步的检测计算（构建内存缓存）
  - 但跳过第一步的可视化保存（不创建step1_shear_tear_masks目录）
  - 第二步和第三步正常运行
- **性能提升**：跳过第一步可视化保存，节省约50%的磁盘I/O时间

**输出消息示例**：

*正常模式*：
```
第一步：生成剪切面和撕裂面mask...
第一步完成: 计算了 1000 帧，保存了 10 帧可视化到: step1_shear_tear_masks

第二步：过滤撕裂面区域...
第二步完成: 计算了 1000 帧，保存了 10 帧可视化到: step2_filtered_tear_regions
```

*跳过第一步模式*：
```
⚠️  跳过第一步可视化保存（仍会进行检测以构建缓存）
第一步：生成剪切面和撕裂面mask（跳过可视化保存）...
第一步完成: 计算了 1000 帧（已跳过可视化保存）

第二步：过滤撕裂面区域...
第二步完成: 计算了 1000 帧，保存了 10 帧可视化到: step2_filtered_tear_regions
```

#### 性能提升

以1000帧为例，`viz_interval=100`：
- **analyze_spot_temporal.py**：
  - 原：1000张可视化图（约500MB）
  - 现：10张可视化图（约5MB）
  - 磁盘空间节省：~99%
  
- **analyze_split_temporal_filter_tear.py**：
  - 原：3000+张可视化图（Step1×2 + Step2×3）
  - 现：30+张可视化图
  - 磁盘空间节省：~99%

---

### 2025-10-11: 主分析脚本优化 - 可视化采样间隔

**功能说明**：
- `coil_wear_analysis.py` 新增可视化采样间隔参数
- 支持控制帧诊断图和白斑标注图的生成频率

**技术改进**：
- 新增 `--diagnosis_interval` 参数（默认100）：控制帧诊断图采样间隔
- 新增 `--marker_interval` 参数（默认100）：控制白斑标注图采样间隔
- 移除旧的硬编码逻辑（原为"前10帧和每100帧"）
- 增加统计信息输出，显示实际生成的诊断图数量

**使用方法**：
```bash
# 基本用法（默认每100帧）
python coil_wear_analysis.py \
  --roi_dir data/roi_imgs \
  --output_dir data/analysis \
  --name "视频分析"

# 自定义诊断图采样间隔（每50帧）
python coil_wear_analysis.py \
  --roi_dir data/roi_imgs \
  --output_dir data/analysis \
  --diagnosis_interval 50

# 自定义白斑标注图采样间隔（每200帧）
python coil_wear_analysis.py \
  --roi_dir data/roi_imgs \
  --output_dir data/analysis \
  --marker_interval 200

# 同时自定义两个间隔
python coil_wear_analysis.py \
  --roi_dir data/roi_imgs \
  --output_dir data/analysis \
  --diagnosis_interval 50 \
  --marker_interval 50
```

**输出说明**：
- 帧诊断图：保存在 `visualizations/frame_diagnosis/`，按 `diagnosis_interval` 采样
- 白斑标注图：保存在 `visualizations/white_patch_markers/`，按 `marker_interval` 采样（最多20张）
- 所有帧的特征提取仍会完整执行
- 终端会显示实际生成的诊断图数量

**性能提升**：
- 1000帧，diagnosis_interval=100：生成约10张诊断图（原为110张，减少90%）
- 白斑标注图：已有限制最多20张，现支持自定义间隔
- 节省磁盘空间和图像生成时间

---

### 2025-10-10: 密度分析器统一优化 - 可视化采样间隔

**功能说明**：
- 为所有密度分析器添加可视化采样间隔参数，避免生成过多可视化文件
- 统一使用 `--viz_interval` 参数（默认100），与 `tear_surface_white_patch_analyzer.py` 的 `--marker_interval` 保持一致的设计理念

**涉及脚本**：
1. ✅ `adagaus_density_analyzer.py` - Adagaus二值图密度分析
2. ✅ `burr_density_analyzer.py` - 毛刺密度分析
3. ✅ `shear_density_analyzer.py` - 剪切面密度分析
4. ✅ `tear_density_analyzer.py` - 撕裂面密度分析
5. ✅ `tear_texture_density_analyzer.py` - 撕裂面纹理密度分析
6. ✅ `tear_texture_entropy_analyzer.py` - 撕裂面纹理熵分析（参数保留但不使用）

**技术改进**：
- 新增 `--viz_interval` 命令行参数（默认100），控制可视化生成频率
- 在中间步骤循环中添加采样逻辑（`if idx % visualization_interval == 0`）
- ⚠️ **重要**：所有帧都会执行完整计算（mask生成、特征检测等），仅可视化图像保存按采样间隔
- 使用内存缓存机制，避免重复计算（例如mask_cache、adagaus_cache等）
- 增加详细的统计信息输出，区分"计算"和"保存"的数量
- 统一使用 `argparse` 替代旧的命令行参数解析方式

**采样策略说明**：
```python
# 所有帧都计算（例如生成mask）
for idx, image_path in enumerate(image_files):
    mask = compute_mask(image)        # ← 所有帧都执行
    mask_cache[frame_num] = mask      # ← 缓存结果供后续步骤使用
    
    if idx % visualization_interval == 0:
        save_mask_image(mask)         # ← 仅采样帧保存可视化
```

**使用方法**：

```bash
# 1. Adagaus密度分析器
python data_adagaus_density_curve/adagaus_density_analyzer.py \
  --roi_dir data/roi_imgs \
  --output_dir data_adagaus_density_curve \
  --viz_interval 50

# 2. 毛刺密度分析器
python data_burr_density_curve/burr_density_analyzer.py \
  --roi_dir data/roi_imgs \
  --output_dir data_burr_density_curve \
  --viz_interval 50

# 3. 剪切面密度分析器
python data_shear_density_curve/shear_density_analyzer.py \
  --roi_dir data/roi_imgs \
  --output_dir data_shear_density_curve \
  --viz_interval 50

# 4. 撕裂面密度分析器
python data_tear_density_curve/tear_density_analyzer.py \
  --roi_dir data/roi_imgs \
  --output_dir data_tear_density_curve \
  --viz_interval 50

# 5. 撕裂面纹理密度分析器
python data_texture_density_curve/tear_texture_density_analyzer.py \
  --roi_dir data/roi_imgs \
  --output_dir data_texture_density_curve \
  --viz_interval 50

# 6. 撕裂面纹理熵分析器（不生成中间可视化）
python data_texture_density_curve/tear_texture_entropy_analyzer.py \
  --roi_dir data/roi_imgs \
  --output_dir data_texture_density_curve
```

**各脚本可视化输出**：

| 脚本 | 可视化类型 | 输出目录 | 默认行为 |
|-----|-----------|---------|---------|
| adagaus_density_analyzer | 橙色叠加图 | step3_filtered_results/ | 每100帧 |
| burr_density_analyzer | 蓝色毛刺叠加图 | step3_burr_analysis/ | 每100帧 |
| shear_density_analyzer | 剪切面区域+斑块图 | step2_shear_regions/ | 默认跳过* |
| tear_density_analyzer | 撕裂面区域+斑块图 | step2_tear_regions/ | 默认跳过* |
| tear_texture_density_analyzer | 撕裂面区域图 | step2_tear_regions/ | 每100帧 |
| tear_texture_entropy_analyzer | 无中间可视化 | - | 不适用 |

\*注：这些脚本的第二步默认被 `skip_step2=True` 跳过，但如果启用，会按采样间隔生成可视化

**重要说明（采样策略详解）**：

**以 adagaus_density_analyzer 为例**，所有三个步骤的中间图像都按采样间隔保存：
- `step1_tear_masks/` - 撕裂面mask（用于检查分割质量）
- `step2_adagaus_analysis/` - 原始Adagaus二值图（用于检查检测效果）
- `step3_filtered_results/` - 过滤后的二值图 + 橙色叠加可视化图像

**关键**：所有帧的数值分析仍会完整执行，仅中间可视化图像按采样保存

**输出示例（1000帧，interval=100）**：
```
开始分析 Filtered Adaptive Gaussian 二值图密度...
============================================================
可视化采样间隔: 每 100 帧
找到 1000 个ROI图像文件

第一步：生成撕裂面mask...
第一步完成，撕裂面mask已保存到: step1_tear_masks
  - 共计算 1000 个撕裂面mask（所有帧）        ← 所有帧都计算
  - 共保存 10 个可视化mask（采样间隔: 100）   ← 仅采样帧保存

第二步：生成原始Adagaus二值图...
第二步完成，原始Adagaus二值图已保存到: step2_adagaus_analysis
  - 共计算 1000 个Adagaus二值图（所有帧）      ← 所有帧都计算
  - 共保存 10 个可视化二值图（采样间隔: 100）  ← 仅采样帧保存

第三步：根据撕裂面mask过滤二值图...
第三步完成，过滤后的二值图已保存到: step3_filtered_results
  - 共生成 10 个过滤后的二值图（采样间隔: 100）  ← 仅采样帧保存
  - 共生成 10 张可视化图像（采样间隔: 100）     ← 仅采样帧保存
  - 所有 1000 帧的数值分析已完成                ← 所有帧都分析
```

**各脚本步骤详情**：

| 脚本 | Step 1 | Step 2 | Step 3 | 说明 |
|-----|--------|--------|--------|------|
| **adagaus_density_analyzer** | 撕裂面mask | 原始Adagaus二值图 | 过滤后二值图+可视化 | 所有步骤都采样 ✅ |
| **burr_density_analyzer** | 撕裂面mask | 原始毛刺图 | 撕裂面区域+毛刺可视化 | 所有步骤都采样 ✅ |
| **shear_density_analyzer** | 剪切面mask | 剪切面区域+斑块图 | 分析+可视化 | 步骤1-2采样 ✅ |
| **tear_density_analyzer** | 撕裂面mask | 撕裂面区域+斑块图 | 分析+可视化 | 步骤1-2采样 ✅ |
| **tear_texture_density_analyzer** | 撕裂面mask | 撕裂面区域 | 分析+可视化 | 步骤1-2采样 ✅ |
| **tear_texture_entropy_analyzer** | - | - | 分析+可视化 | 无中间步骤 |

**性能提升**：
- 磁盘空间占用从数GB降至数百MB（减少90%+）
- 图像文件数量从~4000张降至~40张（减少99%）
- 图像保存时间显著减少（减少70-80%）
- ⚠️ **处理时间**：由于所有帧仍需完整计算，总处理时间仅略有减少
- ✅ **数值分析**：所有1000帧的计算结果完全保留，不受任何影响

**适用场景**：
- 处理大量帧时节省磁盘空间（最主要优势）
- 可视化图像仅用于抽查质量，不需要每帧都生成
- 所有帧的数值分析结果不受影响
- 批量处理多个视频时减少磁盘I/O负担

**⚠️ 关键设计说明**：
- **所有帧都计算**：为了保证后续步骤能正常使用中间结果（如mask、二值图等）
- **仅可视化按采样**：只有保存到磁盘的可视化图像按采样间隔
- **内存缓存机制**：中间结果缓存在内存中（mask_cache、adagaus_cache等）
- **优势权衡**：主要节省磁盘空间和I/O时间，而非计算时间

**如果需要同时减少计算时间**：
可以考虑直接对输入图像进行采样，例如：
```bash
# 方案1：先用frame_extractor时提取时增加间隔
python data_process/frame_extractor.py  # 修改interval参数

# 方案2：手动筛选ROI图像到新目录
mkdir data/roi_imgs_sampled
cp data/roi_imgs/frame_*00_roi.png data/roi_imgs_sampled/  # 每100帧
```

---

### 2025-10-08: 新增水平梯度能量特征

**功能说明**：
- 在原有的"总梯度能量"基础上，新增了"水平梯度能量"特征
- 水平梯度只计算x方向（水平方向）的Sobel梯度，专门用于捕捉垂直边缘（刀口边缘）的锐度变化
- 相比总梯度（包含x和y两个方向），水平梯度对剪刀磨损导致的边缘钝化更加敏感

**技术原理**：
```python
# 总梯度能量（原有）
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
energy = mean(grad_x² + grad_y²)  # 包含所有方向

# 水平梯度能量（新增）
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
energy = mean(grad_x²)  # 仅水平方向
```

**物理意义**：
- 水平梯度主要反映垂直边缘（剪刀的竖直切割面）的锐度
- 刀口锋利时：垂直边缘清晰，水平梯度大
- 刀口磨钝时：垂直边缘模糊，水平梯度显著下降
- **下降趋势**表示刀口钝化，磨损加重

**新增可视化**：
在运行 `coil_wear_analysis.py` 后会自动生成 `horizontal_gradient_comparison.png`，包含4个子图：
1. **时序对比图**：总梯度 vs 水平梯度的完整时序变化（原始+平滑）
2. **按卷统计图**：各钢卷的平均梯度能量柱状图对比
3. **归一化趋势图**：归一化后的梯度能量变化，更清晰展示趋势差异
4. **变化率统计**：首尾变化率对比，负值表示下降（磨损加重）

**使用方法**：
```bash
# 运行磨损分析脚本会自动包含新特征
python coil_wear_analysis.py --roi_dir data/roi_imgs --output_dir data/analysis --name "测试"

# 输出文件中会包含：
# - features/wear_features.csv 中的 avg_horizontal_gradient 列
# - visualizations/horizontal_gradient_comparison.png 专项对比图
```

**特征命名**：
- `left_horizontal_gradient`: 左侧（撕裂面）水平梯度能量
- `right_horizontal_gradient`: 右侧（剪切面）水平梯度能量  
- `avg_horizontal_gradient`: 左右平均水平梯度能量

第二步：
确认模型，并写出demo;

第三步：
确认数据集制作步骤，并写出每个步骤的脚本；




第四步：
快速原型验证，要求：
需要验证方案的可行性，验证数据：

```Markdown
data/Video_20250821110325928.avi
00:00-00:17 第4卷
00:17:00 开始第5卷
00:34:00 第6卷 // 前⾯⽐较糊，后⾯还好，也可能是速度稳定了
00:57:00 第7卷
01:16:00 8卷
01:34:00 第9卷
01:57:00 第10卷
02:20:00第11卷
02:39:00 - 02:58:00 第12卷
换圆盘剪

data/Video_20250821140339629.avi
00:00:00 第1卷
00:18:37 第2卷
00:36:45 第3卷
00:57:00 第4卷
01:17:07 第5卷
```

这两个视频完整记录了第一个剪刀到第二个剪刀的更换过程；第一个剪刀剪了12卷；大约2小时58分钟；

验证期望：我们的模型应该能够画出整个工作过程的大体曲线。
曲线一：纵轴是“撕裂面”/“剪切面”比值变化；横轴是时间；
曲线二：纵轴是“白色毛刺斑块的数量或密度”；横轴是时间；

---

## 📢 最新更新 (2025年10月8日)

### ⭐ coil_wear_analysis.py 功能增强

**新增功能：**

1. **平滑长期趋势分析**
   - 集成3种高级平滑方法：移动最大值包络线、周期峰值样条插值、全局二次拟合
   - 生成图表：`smooth_longterm_trends.png`
   
2. **深度趋势分析**
   - 峰值包络线分析（自动判断趋势方向）
   - 分段趋势分析（统计递增段占比）
   - 低通滤波长期趋势（去除高频噪声）
   - 生成图表：`deep_envelope_analysis.png`, `deep_segment_analysis.png`, `deep_longterm_filtered.png`

**使用方法：**
```bash
python coil_wear_analysis.py \
  --roi_dir data/roi_imgs \
  --output_dir data_analysis \
  --name "视频分析"
```

**功能对比：**

| 特性 | coil_wear_analysis.py | main_analysis.py |
|-----|----------------------|------------------|
| **钢卷边界检测** | ✅ 自动检测（基于ruptures） | ❌ 需手动指定参数 |
| **综合指标** | ✅ 3种方法（加权/PCA/多维度） | ❌ 无 |
| **平滑趋势** | ✅ 已集成（单脚本） | ✅ 需调用子脚本 |
| **深度分析** | ✅ 已集成（单脚本） | ✅ 需调用子脚本 |
| **运行方式** | 🎯 一键运行，自动化程度高 | 🔧 模块化，灵活性高 |

**详细文档：**
- 📖 [CHANGELOG_coil_wear_analysis.md](CHANGELOG_coil_wear_analysis.md) - 技术更新日志
- 📘 [USAGE_EXAMPLE.md](USAGE_EXAMPLE.md) - 详细使用指南

---

## 📊 撕裂面白色斑块分析 (新增)

基于用户观察："撕裂面白色斑块随剪刀磨损增加而增多"，本功能提供专门的白斑量化分析。

### 功能特点

1. **4种检测方法对比**
   - 方法1：固定阈值法（灰度值 > 200）
   - 方法2：Otsu自适应阈值法（带最小阈值约束170，平衡检测精度）
   - 方法3：相对亮度法（均值 + 1.5σ）
   - 方法4：形态学Top-Hat法（局部对比度）

2. **8种量化指标**
   - 白色区域面积占比（%）
   - 白色斑块数量（连通域计数）
   - 平均亮度和亮度标准差
   - 单个白斑平均面积 - 白斑面积占比/白斑数量
   - 综合指标 - 白斑数量+亮度标准差
   - 亮度直方图熵 - 反映白斑亮度分布的复杂度
   - **斑块面积分布熵（新增）** - 反映白斑大小分布的均匀性

3. **双模式分析**
   - **集成模式**：自动集成到主分析流程（32个白斑特征）
   - **独立模式**：详细的方法对比和推荐报告

### 使用方法

**集成模式（自动，推荐）：**
```bash
python coil_wear_analysis.py \
  --roi_dir data/roi_imgs \
  --output_dir data/coil_analysis \
  --name "视频分析"
```
输出包含：
- `white_patch_analysis.png` - 白斑分析专项图（3行对比）
- `white_patch_temporal_curves_4x8.png` - **完整时序曲线**（8×4布局，32条曲线）
- `white_patch_markers/` - **标注图目录**（3×2布局，含直方图）
- `white_patch_recommendation.md` - **方法推荐报告**
- `wear_features.csv` - 包含32个白斑特征列（4方法×8指标）
- `analysis_report.md` - 含白斑变化结论

**优势：** 一键运行，自动生成所有白斑分析结果，无需单独运行独立脚本

**独立模式（可选，用于单独白斑分析）：**
```bash
python tear_surface_white_patch_analyzer.py \
  --roi_dir data/roi_imgs \
  --output_dir data/white_patch_analysis \
  --marker_interval 50
```
输出包含：
- `method_comparison.png` - 4种方法检测效果对比（抽样6帧，6×6布局）
- `temporal_curves_4x8.png` - 32条时序曲线（4方法×8指标）
- `coil_statistics.png` - 按卷统计分析（如提供卷号CSV）
- `method_recommendation.md` - 方法推荐报告
- `white_patch_features.csv` - 详细特征数据（32列）
- `white_patch_markers/` - 标注图目录（3×2布局，含直方图）

**使用场景：** 只需要白斑分析，不需要其他磨损指标时使用

### 模式对比

| 特性 | 集成模式 | 独立模式 |
|-----|---------|---------|
| **运行方式** | coil_wear_analysis.py | tear_surface_white_patch_analyzer.py |
| **特征提取** | ✅ 全部磨损特征 + 白斑特征 | ⚠️ 仅白斑特征 |
| **钢卷检测** | ✅ 自动检测钢卷边界 | ❌ 需提供卷号CSV |
| **白斑时序图** | ✅ 8×4完整版 | ✅ 8×4完整版 |
| **白斑标注图** | ✅ 3×2含直方图 | ✅ 3×2含直方图 |
| **方法对比图** | ❌ 无 | ✅ 6×6抽样对比 |
| **方法推荐** | ✅ 自动生成 | ✅ 自动生成 |
| **综合分析** | ✅ 磨损+白斑综合报告 | ❌ 仅白斑报告 |
| **推荐使用** | 🎯 **推荐**（功能最全） | 🔧 白斑专项研究 |

### 输出示例

**时序分析：**
- 第一行：4种方法的白斑面积占比变化曲线
- 第二行：4种方法的白斑数量变化曲线
- 第三行：4种方法的平均亮度变化曲线
- 第四行：4种方法的亮度标准差变化曲线
- 第五行：4种方法的单个白斑平均面积变化曲线
- 第六行：4种方法的综合指标变化曲线 - 数量+亮度标准差
- 第七行：4种方法的亮度直方图熵变化曲线 - 反映亮度分布复杂度
- **第八行：4种方法的斑块面积分布熵变化曲线（新增）** - 反映白斑大小分布均匀性

**白斑标注可视化（增强版）：**
- **3×2布局**：上方4个标注图 + 下方2个直方图对比
- **上方标注图（2×2）**：
  - 红色圆圈标注每个白斑（圆圈大小反映面积）
  - 绿色点标记白斑质心
  - 黄色标签显示白斑数量
- **左下直方图**：撕裂面亮度分布对比
  - 灰色曲线：撕裂面整体亮度分布（基准）
  - 4条彩色曲线：各方法检测到的白斑亮度分布
  - 可看出各方法选择的亮度范围
- **右下直方图**：白斑面积分布对比
  - 4条彩色曲线：各方法检测到的斑块面积分布
  - 显示每个方法检测到的白斑数量
  - 可看出白斑大小的分布特征（大、中、小斑块的比例）
- **集成模式**：每隔100帧生成一张，最多20张
- **独立模式**：可通过 `--marker_interval` 参数自定义

**方法推荐：**
基于单调性、稳定性和灵敏度的综合评分，自动推荐最佳检测方法。

### 参数说明

独立脚本参数：
- `--roi_dir`: ROI图像目录（必需）
- `--output_dir`: 输出目录（必需）
- `--coil_csv`: 包含卷号信息的CSV文件（可选）
- `--marker_interval`: 白斑标注采样间隔，默认100（每隔100帧生成一张标注图）

### 技术细节

**特征命名规则：**
- `white_area_ratio_m1/m2/m3/m4` - 白斑面积占比（%）
- `white_patch_count_m1/m2/m3/m4` - 白斑数量（个）
- `white_avg_brightness_m1/m2/m3/m4` - 平均亮度
- `white_brightness_std_m1/m2/m3/m4` - 亮度标准差
- `white_avg_patch_area_m1/m2/m3/m4` - 单个白斑平均面积（%）
- `white_composite_index_m1/m2/m3/m4` - 综合指标（数量+亮度std）
- `white_brightness_entropy_m1/m2/m3/m4` - 亮度直方图熵
- `white_patch_area_entropy_m1/m2/m3/m4` - **斑块面积分布熵（新增）**

**适用场景：**
- 撕裂面质量评估
- 剪刀磨损状态监控
- 与其他磨损指标综合判断

**指标详解：**

1. **白斑面积占比** - 白斑总面积占撕裂面的百分比
2. **白斑数量** - 连通域计数，反映白斑碎片化程度
3. **平均亮度** - 白斑区域的平均灰度值
4. **亮度标准差** - 白斑亮度的离散程度
5. **单个白斑平均面积** - 面积/数量，反映白斑平均尺寸
6. **综合指标** - 数量+亮度std，快速评估严重程度
7. **亮度直方图熵** - 反映亮度分布复杂度（0-5，越高越复杂）
8. **斑块面积分布熵** - 反映白斑大小分布均匀性（0-4.3，越高白斑大小差异越大）

**方法对比：**

| 方法 | 阈值策略 | 优点 | 适用场景 |
|-----|---------|------|---------|
| 方法1 | 固定200 | 稳定可靠 | 光照稳定 |
| 方法2 | Otsu+170约束 | 自适应且有下限 | 通用场景 |
| 方法3 | 均值+1.5σ | 相对统计 | 局部亮度变化 |
| 方法4 | Top-Hat | 检测凸起 | 小亮点检测 |

**阈值调整：**
- Otsu方法当前最小阈值：170
- 如需调整：修改 `geometry_features.py` (562行) 和 `tear_surface_white_patch_analyzer.py` (110行)
- 建议范围：160-185

---

## 工具说明

### 视频抽帧工具 (frame_extractor.py)

**位置：** `data_process/frame_extractor.py`

**功能：** 从视频文件中按指定时间间隔（默认5秒）抽取关键帧

**使用方法：**
```bash
python data_process/frame_extractor.py
```

**重要改进（2025-10-10）：**
- ✅ **修复超长视频中断问题**：原方法使用跳帧定位在处理超长视频时会失败（通常在24小时位置中断）
- ✅ **改用顺序读取**：更可靠的帧读取方式，适用于40万+帧的超长视频
- ✅ **智能错误恢复机制**：遇到损坏帧时采用渐进式跳跃策略
  - 前3次失败：跳过2个间隔（约10秒）
  - 4-6次失败：跳过10个间隔（约50秒）
  - 7次以上：跳过50个间隔（约4分钟）
  - 超过20次连续失败：停止处理并报告最后成功位置
- ✅ **实时进度显示**：显示当前帧数和处理时间

**技术细节：**
- **旧方法缺陷**：使用 `cap.set(cv2.CAP_PROP_POS_FRAMES)` 跳帧，在大型视频中不可靠
- **新方法优势**：顺序读取所有帧，按间隔保存，处理速度约16帧/秒
- **支持格式**：AVI、MP4等常见视频格式
- **损坏处理**：智能跳过视频损坏区域，避免卡死在解码错误
- **容错能力**：可处理包含损坏片段的视频，最大限度提取可用帧

---

### 长期趋势图拆分工具 (split_longterm_trend_charts.py)

**位置：** `split_longterm_trend_charts.py`

**功能：** 将包含多个子图的长期趋势图拆分为单独的图表文件，x轴拉长以便更清楚地查看随时间的变化曲线

**特点：**
- ✅ **支持批量处理**：可同时处理多个目录
- ✅ **自动化输出**：自动创建输出目录并保存图表
- ✅ **高分辨率**：默认200 DPI，可自定义
- ✅ **完整分析**：包含原始数据、散点、线性趋势线和趋势方向标注

**使用方法：**
```bash
# 处理单个目录（使用默认路径）
python split_longterm_trend_charts.py --input_dir data/coil_wear_analysis

# 处理多个目录
python split_longterm_trend_charts.py --input_dir data/coil_wear_analysis data_video7_20250909110956225/coil_wear_analysis

# 批量处理所有目录（使用 shell 脚本）
./batch_split_trends.sh

# 自定义分辨率
python split_longterm_trend_charts.py --input_dir data/coil_wear_analysis --dpi 300
```

**输入要求：**
- 需要 `features/wear_features.csv` 文件（由 coil_wear_analysis.py 生成）
- CSV 文件必须包含以下列：
  - `frame_id`：帧编号
  - `avg_rms_roughness`：平均RMS粗糙度
  - `max_notch_depth`：最大缺口深度
  - `right_peak_density`：剪切面峰密度
  - `avg_gradient_energy`：平均梯度能量
  - `tear_shear_area_ratio`：撕裂/剪切面积比

**输出文件：**
保存到 `<input_dir>/visualizations/individual_trends/` 目录：
- `all_trends_6x1.png` - **6×1总图**（80×29英寸，约6.5MB，包含综合磨损指标 + 5个特征）
- `avg_rms_roughness_trend.png` - 平均RMS粗糙度单独图（60×6英寸，约0.9MB）
- `max_notch_depth_trend.png` - 最大缺口深度单独图（60×6英寸，约1.1MB）
- `right_peak_density_trend.png` - 剪切面峰密度单独图（60×6英寸，约1.0MB）
- `avg_gradient_energy_trend.png` - 平均梯度能量单独图（60×6英寸，约1.1MB）
- `tear_shear_area_ratio_trend.png` - 撕裂/剪切面积比单独图（60×6英寸，约0.8MB）

**综合指标说明：**
- 综合磨损指标是将5个特征分别归一化到0-1后取平均值
- 该指标综合反映了剪刀的整体磨损状况
- 值越大表示磨损越严重
- 位于6×1总图的第一个位置，便于快速了解整体趋势

**参数说明：**
- `--input_dir`：输入主目录路径（可指定多个）
- `--csv_path`：CSV文件相对路径（默认：features/wear_features.csv）
- `--output_subdir`：输出子目录相对路径（默认：visualizations/individual_trends）
- `--dpi`：输出图片分辨率（默认：200）

**集成说明：**
此功能已集成到 `coil_wear_analysis.py` 中，运行磨损分析时会自动生成单独的趋势图。如果只需要重新生成趋势图而不重新分析，可以单独运行此脚本。

---

**模型信息：** Claude Sonnet 4.5 (claude-sonnet-4-20250514)  
**更新日期：** 2025年10月13日

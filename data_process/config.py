"""
配置文件 - 包含所有算法参数和路径设置
"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
PROCESSED_DIR = os.path.join(OUTPUT_DIR, 'processed')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True) 
os.makedirs(RESULTS_DIR, exist_ok=True)

# 预处理参数
PREPROCESS_CONFIG = {
    # ROI提取参数
    'roi_threshold': 30,  # 二值化阈值，用于分离白色条状物和黑色背景
    'min_contour_area': 1000,  # 最小轮廓面积，过滤小噪点
    'gaussian_kernel': 5,  # 高斯滤波核大小
    'morphology_kernel': 3,  # 形态学操作核大小
    
    # 条状物分割参数
    'segment_method': 'gradient',  # 分割方法: 'gradient', 'texture', 'hybrid', 'boundary', 'curved', 'sam2'
    'gradient_threshold': 0.3,  # 梯度阈值
    'texture_window': 15,  # 纹理分析窗口大小
    'white_threshold': 50,  # 白色区域检测阈值，用于曲线分割 
    
    # 特征提取参数
    'white_spot_threshold': 200,  # 白斑检测阈值
    'min_spot_area': 5,  # 最小白斑面积
    'max_spot_area': 500,  # 最大白斑面积
}

# SAM2分割模型配置参数
SAM2_CONFIG = {
    'model_name': 'facebook/sam2.1-hiera-tiny',
    'device': 'auto',  # auto/cuda/cpu
    'batch_size': 1,
    'confidence_threshold': 0.6,  # 分割结果置信度阈值
    'min_mask_area': 100,   # 最小生成mask的有效面积
    'prompt_point_density': 0.05,  # 提示点密度（图像宽*高*这个比率决定采样多少个点）
    'enable_tiling': False   # 大图像切片
}

# 模型验证参数
MODEL_CONFIG = {
    'time_window': 60,  # 时间窗口长度（分钟）
    'ema_alpha': 0.2,  # 指数移动平均参数
    'tear_ratio_threshold': 0.6,  # 撕裂面比例告警阈值
    'spot_count_threshold': 10,  # 白斑数量告警阈值
    'change_detection_sensitivity': 0.1,  # 变化检测敏感度
}

# 可视化参数
VIS_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 100,
    'colors': {
        'background': 'black',
        'tear_surface': 'red',
        'shear_surface': 'blue',
        'white_spots': 'yellow'
    },
    'font_family': 'DejaVu Sans',  # 使用支持中文的字体
    'chinese_font': 'SimHei'  # 中文字体
}

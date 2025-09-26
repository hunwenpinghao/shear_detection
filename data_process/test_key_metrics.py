"""
测试关键指标显示效果
"""
import matplotlib.pyplot as plt
import numpy as np
from font_utils import setup_chinese_font

def test_key_metrics_display():
    """测试关键指标显示效果"""
    # 设置中文字体
    setup_chinese_font()
    
    # 模拟特征数据
    features = {
        'tear_to_shear_ratio': 0.818,
        'spot_density': 0.063,
        'average_spot_size': 99.1,
        'tear_area_roughness': 0.111,
        'shear_area_roughness': 0.940,
        'roughness_difference': 0.830
    }
    
    # 创建测试图像
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 关键指标文本显示（与feature_extractor.py中相同的代码）
    key_metrics = [
        f"撕裂面/剪切面比值: {features.get('tear_to_shear_ratio', 0):.3f}",
        f"白斑密度: {features.get('spot_density', 0):.3f}",
        f"平均白斑大小: {features.get('average_spot_size', 0):.1f}",
        f"撕裂面粗糙度: {features.get('tear_area_roughness', 0):.3f}",
        f"剪切面粗糙度: {features.get('shear_area_roughness', 0):.3f}",
        f"粗糙度差异: {features.get('roughness_difference', 0):.3f}"
    ]
    
    # 使用与原始代码相同的参数
    ax.text(0.1, 0.9, '\n'.join(key_metrics), 
           transform=ax.transAxes, fontsize=14,
           verticalalignment='top')
    ax.set_title('关键指标显示测试', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # 添加边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('gray')
    
    plt.tight_layout()
    
    # 保存测试图像
    save_path = '/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/output/key_metrics_test.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"关键指标显示测试图像已保存到: {save_path}")
    print("请检查图像中的关键指标名称是否正常显示")
    
    # 打印指标内容供参考
    print("\n关键指标内容:")
    for metric in key_metrics:
        print(f"  - {metric}")

if __name__ == "__main__":
    test_key_metrics_display()

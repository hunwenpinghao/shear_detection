"""
测试中文字体显示效果
"""
import matplotlib.pyplot as plt
import numpy as np
from font_utils import setup_chinese_font

def test_chinese_display():
    """测试中文显示效果"""
    # 设置中文字体
    setup_chinese_font()
    
    # 创建测试图像
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 测试各种中文标题
    test_titles = [
        '原始图像',
        '对比度增强', 
        '去噪处理',
        'ROI提取'
    ]
    
    # 生成一些测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x)
    y4 = np.sin(x) + np.cos(x)
    
    data = [y1, y2, y3, y4]
    
    for i, (ax, title, y_data) in enumerate(zip(axes.flat, test_titles, data)):
        ax.plot(x, y_data, 'b-', linewidth=2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('时间点', fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存测试图像
    save_path = '/Users/aibee/hwp/wphu个人资料/baogang/shear_detection/output/chinese_font_test.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"中文字体测试图像已保存到: {save_path}")
    print("请检查图像中的中文标题是否正常显示")

if __name__ == "__main__":
    test_chinese_display()

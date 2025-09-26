"""
字体设置工具模块
解决matplotlib中文显示问题
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os


def setup_chinese_font():
    """
    设置matplotlib中文字体
    """
    # 获取系统信息
    system = platform.system()
    
    # 尝试设置中文字体
    chinese_fonts = []
    
    if system == "Darwin":  # macOS
        chinese_fonts = [
            'PingFang SC',
            'Hiragino Sans GB', 
            'STHeiti',
            'SimHei',
            'Arial Unicode MS'
        ]
    elif system == "Windows":
        chinese_fonts = [
            'SimHei',
            'Microsoft YaHei',
            'SimSun',
            'KaiTi'
        ]
    else:  # Linux
        chinese_fonts = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC',
            'DejaVu Sans'
        ]
    
    # 查找可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用字体: {font}")
            return font
    
    # 如果没有找到中文字体，使用默认字体并给出警告
    print("警告: 未找到合适的中文字体，中文可能显示为方块")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'


def get_available_fonts():
    """
    获取系统可用的字体列表
    """
    fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [f for f in fonts if any(keyword in f.lower() for keyword in 
                   ['chinese', 'cjk', 'han', 'hei', 'song', 'kai', 'fang', 'ping'])]
    return chinese_fonts


def test_chinese_display():
    """
    测试中文显示效果
    """
    setup_chinese_font()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, '中文测试：剪刀磨损检测系统', 
            ha='center', va='center', fontsize=16)
    ax.set_title('字体测试')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # 测试字体设置
    print("可用中文字体:")
    chinese_fonts = get_available_fonts()
    for font in chinese_fonts[:10]:  # 只显示前10个
        print(f"  - {font}")
    
    print(f"\n总共找到 {len(chinese_fonts)} 个中文字体")
    
    # 设置字体
    selected_font = setup_chinese_font()
    print(f"已选择字体: {selected_font}")

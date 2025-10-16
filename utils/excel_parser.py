"""
Excel文件解析工具
用于解析包含中文描述行的Excel文件，将其转换为字典格式
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional


def parse_baogang_excel(file_path: str, 
                        skip_index_column: bool = True,
                        convert_timestamps: bool = True,
                        clean_whitespace: bool = True) -> Dict[str, Any]:
    """
    解析宝钢Excel文件格式
    
    文件格式说明：
    - 第0行：中文描述（英文列名对应的中文说明）
    - 第1行开始：实际数据
    
    Args:
        file_path: Excel文件路径
        skip_index_column: 是否跳过第一列索引列（Unnamed: 0）
        convert_timestamps: 是否将时间戳字符串转换为datetime对象
        clean_whitespace: 是否清理数据中的空白字符（如制表符、空格）
    
    Returns:
        包含以下键的字典：
        - 'column_mapping': 英文列名到中文描述的映射
        - 'data': 数据列表，每行为一个字典
        - 'metadata': 元数据信息（文件路径、行数等）
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 获取列名
    columns = df.columns.tolist()
    
    # 第0行是中文描述
    chinese_descriptions = df.iloc[0].to_dict()
    
    # 创建英文到中文的列名映射
    column_mapping = {}
    for col in columns:
        if skip_index_column and 'Unnamed' in col:
            continue
        column_mapping[col] = str(chinese_descriptions[col])
    
    # 从第1行开始提取数据
    data_rows = []
    for idx in range(1, len(df)):
        row_dict = {}
        for col in columns:
            if skip_index_column and 'Unnamed' in col:
                continue
            
            value = df.iloc[idx][col]
            
            # 清理空白字符
            if clean_whitespace and isinstance(value, str):
                value = value.strip()
            
            # 转换时间戳
            if convert_timestamps and 'time' in col.lower() and isinstance(value, str):
                try:
                    # 格式：YYYYMMDDHHMMSS
                    if len(value) == 14:
                        value = datetime.strptime(value, '%Y%m%d%H%M%S')
                except Exception:
                    pass  # 保持原始值
            
            row_dict[col] = value
        
        data_rows.append(row_dict)
    
    # 构建返回结果
    result = {
        'column_mapping': column_mapping,
        'data': data_rows,
        'metadata': {
            'file_path': file_path,
            'total_rows': len(data_rows),
            'columns': list(column_mapping.keys())
        }
    }
    
    return result


def parse_baogang_excel_with_chinese_keys(file_path: str,
                                          skip_index_column: bool = True,
                                          convert_timestamps: bool = True,
                                          clean_whitespace: bool = True) -> Dict[str, Any]:
    """
    解析宝钢Excel文件格式，使用中文列名作为键
    
    Args:
        file_path: Excel文件路径
        skip_index_column: 是否跳过第一列索引列（Unnamed: 0）
        convert_timestamps: 是否将时间戳字符串转换为datetime对象
        clean_whitespace: 是否清理数据中的空白字符
    
    Returns:
        包含以下键的字典：
        - 'data': 数据列表，每行为一个字典（使用中文列名作为键）
        - 'metadata': 元数据信息
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 获取列名
    columns = df.columns.tolist()
    
    # 第0行是中文描述
    chinese_descriptions = df.iloc[0].to_dict()
    
    # 创建英文到中文的映射
    en_to_cn = {}
    for col in columns:
        if skip_index_column and 'Unnamed' in col:
            continue
        en_to_cn[col] = str(chinese_descriptions[col])
    
    # 从第1行开始提取数据
    data_rows = []
    for idx in range(1, len(df)):
        row_dict = {}
        for en_col, cn_col in en_to_cn.items():
            value = df.iloc[idx][en_col]
            
            # 清理空白字符
            if clean_whitespace and isinstance(value, str):
                value = value.strip()
            
            # 转换时间戳
            if convert_timestamps and 'time' in en_col.lower() and isinstance(value, str):
                try:
                    if len(value) == 14:
                        value = datetime.strptime(value, '%Y%m%d%H%M%S')
                except Exception:
                    pass
            
            row_dict[cn_col] = value
        
        data_rows.append(row_dict)
    
    result = {
        'data': data_rows,
        'metadata': {
            'file_path': file_path,
            'total_rows': len(data_rows),
            'columns': list(en_to_cn.values())
        }
    }
    
    return result


def print_parsed_data(parsed_dict: Dict[str, Any], max_rows: int = 5):
    """
    打印解析后的数据（用于调试）
    
    Args:
        parsed_dict: parse_baogang_excel 返回的字典
        max_rows: 最多打印的数据行数
    """
    print("=" * 80)
    print("列名映射（英文 -> 中文）:")
    print("-" * 80)
    if 'column_mapping' in parsed_dict:
        for en, cn in parsed_dict['column_mapping'].items():
            print(f"  {en:30s} -> {cn}")
    
    print("\n" + "=" * 80)
    print(f"数据预览（前{max_rows}行）:")
    print("-" * 80)
    for i, row in enumerate(parsed_dict['data'][:max_rows]):
        print(f"\n行 {i+1}:")
        for key, value in row.items():
            print(f"  {key:30s}: {value}")
    
    print("\n" + "=" * 80)
    print("元数据:")
    print("-" * 80)
    for key, value in parsed_dict['metadata'].items():
        print(f"  {key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 默认测试文件
        file_path = '/Users/aibee/hwp/wphu个人资料/baogang/data_baogang/T_ODS_FV_BY_PDO_20251014144143947.xlsx'
    
    print("解析方式1：使用英文列名")
    result1 = parse_baogang_excel(file_path)
    print_parsed_data(result1, max_rows=3)
    
    print("\n\n" + "="*100 + "\n\n")
    
    print("解析方式2：使用中文列名")
    result2 = parse_baogang_excel_with_chinese_keys(file_path)
    print_parsed_data(result2, max_rows=3)


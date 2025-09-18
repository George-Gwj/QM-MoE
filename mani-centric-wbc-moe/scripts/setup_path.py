#!/usr/bin/env python3
"""
路径设置模块
确保使用正确的mani-centric-wbc模块
"""

import os
import sys

def setup_correct_path():
    """设置正确的Python路径"""
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取mani-centric-wbc的根目录
    compose_root = os.path.dirname(current_dir)
    
    # 将mani-centric-wbc添加到Python路径的最前面
    if compose_root not in sys.path:
        sys.path.insert(0, compose_root)
    
    # 设置环境变量，确保使用正确的模块
    os.environ['PYTHONPATH'] = compose_root + ':' + os.environ.get('PYTHONPATH', '')
    
    print(f"设置Python路径: {compose_root}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path[:3]}...")  # 只显示前3个路径

if __name__ == "__main__":
    setup_correct_path()
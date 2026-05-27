@echo off
chcp 65001 >nul
echo ========================================
echo 基于深度学习的表情识别系统
echo ========================================
echo.

echo [1/3] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未检测到Python，请先安装Python 3.7+
    pause
    exit /b 1
)
python --version
echo.

echo [2/3] 检查并安装依赖包...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖包，请稍候...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo ❌ 依赖安装失败，请检查网络连接
        pause
        exit /b 1
    )
) else (
    echo ✓ 依赖包已安装
)
echo.

echo [3/3] 启动表情识别系统...
echo.
python gui.py

if errorlevel 1 (
    echo.
    echo ❌ 程序运行出错
    pause
)

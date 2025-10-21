@echo off
chcp 65001 >nul
echo ============================================
echo    FP8 vs FP16 对比测试启动器
echo ============================================
echo.

REM 设置HuggingFace镜像
set HF_ENDPOINT=https://hf-mirror.com
echo ✅ 已设置 HuggingFace 镜像

REM 设置CUDA内存优化
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo ✅ 已启用 CUDA 内存优化

echo.
echo 🔬 启动对比测试工具...
echo.

python 对比测试_FP8_vs_FP16.py

if errorlevel 1 (
    echo.
    echo ❌ 测试过程出错！
    pause
    exit /b 1
)

echo.
echo ✅ 测试完成！
echo 📊 查看报告: output\comparison_report.txt
echo 📁 查看样本对比: output\comparison_fp8_vs_fp16\
echo.
pause


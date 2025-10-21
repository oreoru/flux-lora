@echo off
chcp 65001 >nul
title 一键下载 FLUX 模型

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              一键下载 FLUX.1-dev 模型                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 💡 提示：将尝试多种下载方式，请选择：
echo.
echo [1] 使用 ModelScope 下载（国内推荐，速度快）
echo [2] 使用 HuggingFace 镜像下载（备选）
echo [3] 查看手动下载指南
echo [0] 退出
echo.

set /p choice="请选择 [1-3]: "

if "%choice%"=="1" goto modelscope
if "%choice%"=="2" goto hfmirror
if "%choice%"=="3" goto manual
if "%choice%"=="0" goto end

echo 无效选择，请重新运行
pause
goto end

:modelscope
echo.
echo ════════════════════════════════════════════════════════════════
echo 🇨🇳 使用 ModelScope 下载（国内推荐）
echo ════════════════════════════════════════════════════════════════
echo.
python download_flux_modelscope.py
pause
goto end

:hfmirror
echo.
echo ════════════════════════════════════════════════════════════════
echo 🌐 使用 HuggingFace 镜像下载
echo ════════════════════════════════════════════════════════════════
echo.
echo 📝 请确保已设置 HF_TOKEN
echo    如果还没设置，请先运行: setup_token_offline.bat
echo.
pause
python download_flux_model.py
pause
goto end

:manual
echo.
echo ════════════════════════════════════════════════════════════════
echo 📖 打开手动下载指南
echo ════════════════════════════════════════════════════════════════
echo.
start 使用本地模型_README.txt
goto end

:end


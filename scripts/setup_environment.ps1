# COAT + FLUX LoRA 环境设置脚本 (Windows PowerShell)

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "COAT + FLUX LoRA 环境设置" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# 检查Python版本
Write-Host "检查Python版本..." -ForegroundColor Yellow
$pythonVersion = python --version
Write-Host "Python版本: $pythonVersion"

# 创建虚拟环境
Write-Host ""
Write-Host "创建虚拟环境..." -ForegroundColor Yellow
if (-Not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "✅ 虚拟环境已创建" -ForegroundColor Green
} else {
    Write-Host "⚠️  虚拟环境已存在" -ForegroundColor Yellow
}

# 激活虚拟环境
Write-Host ""
Write-Host "激活虚拟环境..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# 检查CUDA
Write-Host ""
Write-Host "检查CUDA..." -ForegroundColor Yellow
try {
    $nvidiaInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    Write-Host $nvidiaInfo
    
    $cudaVersion = (nvidia-smi | Select-String "CUDA Version").ToString().Split()[8]
    Write-Host "CUDA版本: $cudaVersion"
} catch {
    Write-Host "⚠️  未检测到NVIDIA GPU" -ForegroundColor Yellow
}

# 安装PyTorch
Write-Host ""
Write-Host "安装PyTorch..." -ForegroundColor Yellow
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
Write-Host ""
Write-Host "安装其他依赖..." -ForegroundColor Yellow
pip install -r requirements.txt

# 验证安装
Write-Host ""
Write-Host "验证安装..." -ForegroundColor Yellow
python -c @"
import torch
print(f'✅ PyTorch版本: {torch.__version__}')
print(f'✅ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print(f'✅ FP8支持: {hasattr(torch, \"float8_e4m3fn\")}')
"@

# 克隆ai-toolkit (可选)
Write-Host ""
$response = Read-Host "是否克隆ai-toolkit? (y/n)"
if ($response -eq "y" -or $response -eq "Y") {
    if (-Not (Test-Path "ai-toolkit")) {
        Write-Host "克隆ai-toolkit..." -ForegroundColor Yellow
        git clone https://github.com/ostris/ai-toolkit.git
        Write-Host "✅ ai-toolkit已克隆" -ForegroundColor Green
    } else {
        Write-Host "⚠️  ai-toolkit已存在" -ForegroundColor Yellow
    }
}

# 创建目录结构
Write-Host ""
Write-Host "创建目录结构..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "datasets\clothing" | Out-Null
New-Item -ItemType Directory -Force -Path "output" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "✅ 环境设置完成!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步:" -ForegroundColor Yellow
Write-Host "1. 将服装图片和标注放入 datasets\clothing\"
Write-Host "2. 阅读快速开始: type QUICKSTART_CN.md"
Write-Host "3. 运行训练或基准测试"
Write-Host ""
Write-Host "激活虚拟环境: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan








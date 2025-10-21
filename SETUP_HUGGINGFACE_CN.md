# 🤗 HuggingFace访问配置指南

## 问题说明

FLUX.1-dev 是一个**门控模型**（gated model），需要：
1. ✅ HuggingFace账号
2. ✅ 申请访问权限
3. ✅ 使用Token登录

## 🎯 解决步骤

### 步骤1：创建HuggingFace账号（如果没有）

访问：https://huggingface.co/join

### 步骤2：申请FLUX.1-dev访问权限

1. **访问模型页面**：
   ```
   https://huggingface.co/black-forest-labs/FLUX.1-dev
   ```

2. **点击"Agree and access repository"按钮**
   - 阅读许可协议
   - 勾选同意
   - 提交申请（通常立即批准）

### 步骤3：获取Access Token

1. **访问Token设置页面**：
   ```
   https://huggingface.co/settings/tokens
   ```

2. **创建新Token**：
   - 点击"New token"
   - Token名称：`flux-lora-training`（随意命名）
   - Token类型：选择 **"Read"** 或 **"Write"**（推荐Read）
   - 点击"Generate token"

3. **复制Token**（只显示一次！）：
   ```
   hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### 步骤4：登录HuggingFace（4种方法）

#### 方法A：使用Python脚本（最简单，推荐）✅

```powershell
# 运行简化登录脚本
python hf_login_simple.py

# 按提示粘贴你的Token
```

#### 方法B：手动创建Token文件（最可靠）✅

```powershell
# 1. 创建目录
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.cache\huggingface"

# 2. 创建Token文件（用你的实际Token替换）
"hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" | Out-File -FilePath "$env:USERPROFILE\.cache\huggingface\token" -Encoding utf8 -NoNewline

# 3. 验证
type "$env:USERPROFILE\.cache\huggingface\token"
```

或者直接编辑文件：
- 打开记事本
- 粘贴你的Token（只有Token，没有其他内容）
- 另存为：`C:\Users\14155\.cache\huggingface\token`
- 编码选择UTF-8

#### 方法C：使用环境变量（临时）

```powershell
# 设置环境变量（仅当前会话有效）
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 然后立即运行训练
python train_flux_lora_with_coat.py
```

#### 方法D：在配置文件中设置

编辑 `ai_toolkit_integration/coat_config.yaml`:
```yaml
config:
  # 添加Token
  huggingface_token: "hf_your_token_here"
```

### 步骤5：验证登录

```powershell
# 检查是否已登录
huggingface-cli whoami

# 应该显示你的用户名
```

## 🚀 快速一键设置脚本

```powershell
# 复制这段代码到PowerShell运行

Write-Host "📋 HuggingFace FLUX.1-dev 访问设置" -ForegroundColor Cyan
Write-Host ""

# 步骤1：检查是否已登录
Write-Host "检查登录状态..." -ForegroundColor Yellow
$whoami = huggingface-cli whoami 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 已登录: $whoami" -ForegroundColor Green
} else {
    Write-Host "❌ 未登录" -ForegroundColor Red
    Write-Host ""
    Write-Host "请按以下步骤操作：" -ForegroundColor Yellow
    Write-Host "1. 访问: https://huggingface.co/black-forest-labs/FLUX.1-dev" -ForegroundColor Cyan
    Write-Host "2. 点击 'Agree and access repository'" -ForegroundColor Cyan
    Write-Host "3. 访问: https://huggingface.co/settings/tokens" -ForegroundColor Cyan
    Write-Host "4. 创建新Token（Read权限即可）" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "然后运行: huggingface-cli login" -ForegroundColor Green
    Write-Host "并粘贴你的Token" -ForegroundColor Green
}

Write-Host ""
Write-Host "完成后运行训练命令。" -ForegroundColor Cyan
```

## 📝 在配置文件中使用Token

编辑 `ai_toolkit_integration/coat_config.yaml`:

```yaml
job: extension
config:
  name: flux_lora_coat
  process:
    - type: sd_trainer
      training_folder: "output"
      device: cuda:0
      
      # 添加HuggingFace Token
      huggingface_token: "hf_your_token_here"  # ← 添加这行
      
      network:
        type: lora
        # ... 其他配置
```

或者在环境变量中设置：

```powershell
# 在train_flux_lora_with_coat.py运行前设置
$env:HF_TOKEN = "hf_your_token_here"
python train_flux_lora_with_coat.py
```

## 常见问题

### Q1: 申请访问需要多久？

通常**立即批准**。点击"Agree"后就可以访问了。

### Q2: Token找不到了怎么办？

重新创建一个新Token：
```
https://huggingface.co/settings/tokens
```
Token可以有多个，旧的可以删除。

### Q3: 提示"401 Unauthorized"

说明：
- Token无效或过期
- 没有申请模型访问权限
- Token权限不足（至少需要Read）

**解决**：
```powershell
# 重新登录
huggingface-cli login --token YOUR_TOKEN

# 验证
huggingface-cli whoami
```

### Q4: 能否离线使用？

可以！模型下载后会缓存在：
```
C:\Users\14155\.cache\huggingface\hub\
```

首次需要联网下载，之后可以离线使用。

### Q5: 使用代理

```powershell
# 设置代理
$env:HTTP_PROXY = "http://proxy.example.com:8080"
$env:HTTPS_PROXY = "http://proxy.example.com:8080"

# 或使用HuggingFace镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## 🎯 完整流程示例

```powershell
# 1. 访问并申请权限（浏览器操作）
# https://huggingface.co/black-forest-labs/FLUX.1-dev

# 2. 获取Token（浏览器操作）
# https://huggingface.co/settings/tokens

# 3. 登录
huggingface-cli login
# 粘贴Token: hf_xxx...

# 4. 验证
huggingface-cli whoami
# 输出：你的用户名

# 5. 开始训练
cd "C:\Users\14155\Desktop\flux lora"
python train_flux_lora_with_coat.py
```

## 下一步

登录成功后，运行：

```powershell
# 测试COAT集成
python check_coat_integration.py

# 开始训练
python train_flux_lora_with_coat.py
```

---

**重要提醒**：
- ✅ Token要妥善保管（不要分享给他人）
- ✅ 首次运行会下载约23GB模型文件（耐心等待）
- ✅ 下载后会缓存，后续无需重新下载

🚀 **完成这些步骤后，就可以开始COAT加速训练了！**


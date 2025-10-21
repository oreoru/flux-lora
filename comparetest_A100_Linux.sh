#!/bin/bash
# FP8 vs FP16 对比测试脚本 - A100 40GB Linux版

echo "============================================"
echo "   FP8 vs FP16 对比测试"
echo "   A100 40GB - Linux系统"
echo "============================================"
echo ""

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "📋 测试计划:"
echo "   1. 运行 FP16 基准训练"
echo "   2. 运行 FP8 COAT 训练"
echo "   3. 对比生成样本"
echo ""

read -p "按 Enter 键开始测试..."

# 记录开始时间
START_TIME=$(date +%s)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  第 1 阶段: FP16 基准训练"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

FP16_START=$(date +%s)
python3 train_flux_lora_with_coat.py ai_toolkit_integration/fp16_baseline_config_a100.yaml
FP16_EXIT=$?
FP16_END=$(date +%s)
FP16_DURATION=$((FP16_END - FP16_START))

if [ $FP16_EXIT -ne 0 ]; then
    echo ""
    echo "⚠️  FP16 训练失败！是否继续 FP8 测试？ (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  第 2 阶段: FP8 COAT 训练"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

FP8_START=$(date +%s)
python3 train_flux_lora_with_coat.py ai_toolkit_integration/coat_config_a100.yaml
FP8_EXIT=$?
FP8_END=$(date +%s)
FP8_DURATION=$((FP8_END - FP8_START))

# 总时间
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  测试完成 - 结果汇总"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 格式化时间
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    
    if [ $hours -gt 0 ]; then
        echo "${hours}小时 ${minutes}分钟 ${secs}秒"
    elif [ $minutes -gt 0 ]; then
        echo "${minutes}分钟 ${secs}秒"
    else
        echo "${secs}秒"
    fi
}

FP16_TIME=$(format_time $FP16_DURATION)
FP8_TIME=$(format_time $FP8_DURATION)
TOTAL_TIME=$(format_time $TOTAL_DURATION)

echo "FP16 训练时长: $FP16_TIME ($FP16_DURATION 秒)"
echo "FP8 训练时长:  $FP8_TIME ($FP8_DURATION 秒)"
echo "总耗时:        $TOTAL_TIME"
echo ""

if [ $FP16_DURATION -gt 0 ] && [ $FP8_DURATION -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $FP16_DURATION / $FP8_DURATION" | bc)
    TIME_SAVED=$((FP16_DURATION - FP8_DURATION))
    TIME_SAVED_FMT=$(format_time $TIME_SAVED)
    
    echo "🚀 加速比: ${SPEEDUP}x"
    echo "⏱️  节省时间: $TIME_SAVED_FMT"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  生成样本位置"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "FP16 样本: output/flux_lora_clothing_fp16_baseline_a100/samples/"
echo "FP8 样本:  output/flux_lora_clothing_coat_a100/samples/"
echo ""

# 创建对比文件夹
echo "📁 创建对比文件夹..."
COMP_DIR="output/comparison_fp8_vs_fp16_a100"
mkdir -p "$COMP_DIR/fp16_baseline"
mkdir -p "$COMP_DIR/fp8_coat"

# 复制样本
if [ -d "output/flux_lora_clothing_fp16_baseline_a100/samples" ]; then
    cp output/flux_lora_clothing_fp16_baseline_a100/samples/*.png "$COMP_DIR/fp16_baseline/" 2>/dev/null
fi

if [ -d "output/flux_lora_clothing_coat_a100/samples" ]; then
    cp output/flux_lora_clothing_coat_a100/samples/*.png "$COMP_DIR/fp8_coat/" 2>/dev/null
fi

echo "✅ 对比文件夹: $COMP_DIR"
echo ""

# 生成报告
REPORT="$COMP_DIR/comparison_report.txt"
cat > "$REPORT" << EOF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FP8 vs FP16 训练对比报告
  A100 40GB - Linux系统
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

生成时间: $(date '+%Y-%m-%d %H:%M:%S')

═══════════════════════════════════════════
📊 FP16 基准训练
═══════════════════════════════════════════

配置文件: ai_toolkit_integration/fp16_baseline_config_a100.yaml
训练时长: $FP16_TIME ($FP16_DURATION 秒)
训练状态: $([ $FP16_EXIT -eq 0 ] && echo "✅ 成功" || echo "❌ 失败")

═══════════════════════════════════════════
📊 FP8 COAT 训练
═══════════════════════════════════════════

配置文件: ai_toolkit_integration/coat_config_a100.yaml
训练时长: $FP8_TIME ($FP8_DURATION 秒)
训练状态: $([ $FP8_EXIT -eq 0 ] && echo "✅ 成功" || echo "❌ 失败")

═══════════════════════════════════════════
📈 性能对比
═══════════════════════════════════════════

FP16 训练时长: $FP16_TIME
FP8 训练时长:  $FP8_TIME
加速比:        ${SPEEDUP}x
节省时间:      $TIME_SAVED_FMT

$(if (( $(echo "$SPEEDUP >= 1.5" | bc -l) )); then
    echo "🎉 结论: FP8 显著加速训练！"
elif (( $(echo "$SPEEDUP >= 1.2" | bc -l) )); then
    echo "✅ 结论: FP8 有效加速训练。"
else
    echo "⚠️  结论: 加速效果不明显。"
fi)

═══════════════════════════════════════════
🎨 生成样本对比
═══════════════════════════════════════════

FP16 样本: $COMP_DIR/fp16_baseline/
FP8 样本:  $COMP_DIR/fp8_coat/

💡 建议: 请手动对比相同步数的生成图片，评估质量差异。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

echo "📊 对比报告已生成: $REPORT"
echo ""
cat "$REPORT"

echo ""
echo "🎉 对比测试完成！"


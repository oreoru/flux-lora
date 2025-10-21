#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FP8 vs FP16 对比测试脚本
比较 COAT FP8 和传统 FP16 训练的效果
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
import shutil

class ComparisonTest:
    def __init__(self):
        self.root = Path(__file__).parent
        self.results = {
            'fp16': {},
            'fp8': {},
            'comparison': {}
        }
    
    def print_banner(self, text):
        """打印横幅"""
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60 + "\n")
    
    def run_training(self, config_name, config_path, label):
        """运行训练"""
        self.print_banner(f"开始 {label} 训练")
        
        print(f"📝 配置文件: {config_path}")
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行训练
        cmd = f"python train_flux_lora_with_coat.py {config_path}"
        print(f"\n🚀 执行命令: {cmd}\n")
        
        exit_code = os.system(cmd)
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        # 保存结果
        result = {
            'config': config_path,
            'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': duration,
            'duration_formatted': self.format_duration(duration),
            'exit_code': exit_code,
            'success': exit_code == 0
        }
        
        if exit_code == 0:
            print(f"\n✅ {label} 训练完成!")
            print(f"⏱️  耗时: {result['duration_formatted']}")
        else:
            print(f"\n❌ {label} 训练失败 (退出码: {exit_code})")
        
        return result
    
    def format_duration(self, seconds):
        """格式化时长"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}小时 {minutes}分钟 {secs}秒"
        elif minutes > 0:
            return f"{minutes}分钟 {secs}秒"
        else:
            return f"{secs}秒"
    
    def compare_samples(self):
        """对比生成的样本图片"""
        self.print_banner("对比生成样本")
        
        fp16_samples = Path("output/flux_lora_clothing_fp16_baseline/samples")
        fp8_samples = Path("output/flux_lora_clothing_coat/samples")
        
        if not fp16_samples.exists() or not fp8_samples.exists():
            print("⚠️  样本文件夹不存在，跳过对比")
            return
        
        # 统计样本数量
        fp16_count = len(list(fp16_samples.glob("*.png")))
        fp8_count = len(list(fp8_samples.glob("*.png")))
        
        print(f"📊 FP16 样本数量: {fp16_count}")
        print(f"📊 FP8 样本数量:  {fp8_count}")
        
        # 创建对比文件夹
        comparison_dir = Path("output/comparison_fp8_vs_fp16")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制样本到对比文件夹
        print(f"\n📁 创建对比文件夹: {comparison_dir}")
        
        # 复制 FP16 样本
        fp16_dest = comparison_dir / "fp16_baseline"
        fp16_dest.mkdir(exist_ok=True)
        if fp16_samples.exists():
            for img in fp16_samples.glob("*.png"):
                shutil.copy2(img, fp16_dest / img.name)
        
        # 复制 FP8 样本
        fp8_dest = comparison_dir / "fp8_coat"
        fp8_dest.mkdir(exist_ok=True)
        if fp8_samples.exists():
            for img in fp8_samples.glob("*.png"):
                shutil.copy2(img, fp8_dest / img.name)
        
        print(f"✅ 样本已复制到对比文件夹")
        print(f"   - FP16: {fp16_dest}")
        print(f"   - FP8:  {fp8_dest}")
        
        return {
            'fp16_count': fp16_count,
            'fp8_count': fp8_count,
            'comparison_dir': str(comparison_dir)
        }
    
    def generate_report(self):
        """生成对比报告"""
        self.print_banner("生成对比报告")
        
        report_path = Path("output/comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            f.write("  FP8 vs FP16 训练对比报告\n")
            f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # FP16 结果
            if 'fp16' in self.results:
                f.write("═══════════════════════════════════════════\n")
                f.write("📊 FP16 基准训练\n")
                f.write("═══════════════════════════════════════════\n\n")
                
                fp16 = self.results['fp16']
                f.write(f"配置文件: {fp16.get('config', 'N/A')}\n")
                f.write(f"开始时间: {fp16.get('start_time', 'N/A')}\n")
                f.write(f"结束时间: {fp16.get('end_time', 'N/A')}\n")
                f.write(f"训练耗时: {fp16.get('duration_formatted', 'N/A')}\n")
                f.write(f"训练状态: {'✅ 成功' if fp16.get('success') else '❌ 失败'}\n\n")
            
            # FP8 结果
            if 'fp8' in self.results:
                f.write("═══════════════════════════════════════════\n")
                f.write("📊 FP8 COAT 训练\n")
                f.write("═══════════════════════════════════════════\n\n")
                
                fp8 = self.results['fp8']
                f.write(f"配置文件: {fp8.get('config', 'N/A')}\n")
                f.write(f"开始时间: {fp8.get('start_time', 'N/A')}\n")
                f.write(f"结束时间: {fp8.get('end_time', 'N/A')}\n")
                f.write(f"训练耗时: {fp8.get('duration_formatted', 'N/A')}\n")
                f.write(f"训练状态: {'✅ 成功' if fp8.get('success') else '❌ 失败'}\n\n")
            
            # 对比分析
            if 'fp16' in self.results and 'fp8' in self.results:
                f.write("═══════════════════════════════════════════\n")
                f.write("📈 性能对比\n")
                f.write("═══════════════════════════════════════════\n\n")
                
                fp16_time = self.results['fp16'].get('duration_seconds', 0)
                fp8_time = self.results['fp8'].get('duration_seconds', 0)
                
                if fp16_time > 0 and fp8_time > 0:
                    speedup = fp16_time / fp8_time
                    time_saved = fp16_time - fp8_time
                    
                    f.write(f"FP16 训练时长: {self.format_duration(fp16_time)}\n")
                    f.write(f"FP8 训练时长:  {self.format_duration(fp8_time)}\n")
                    f.write(f"加速比:        {speedup:.2f}x\n")
                    f.write(f"节省时间:      {self.format_duration(time_saved)}\n\n")
                    
                    if speedup >= 1.5:
                        f.write("🎉 结论: FP8 显著加速训练！\n")
                    elif speedup >= 1.2:
                        f.write("✅ 结论: FP8 有效加速训练。\n")
                    else:
                        f.write("⚠️  结论: 加速效果不明显。\n")
            
            # 样本对比
            if 'comparison' in self.results:
                f.write("\n═══════════════════════════════════════════\n")
                f.write("🎨 生成样本对比\n")
                f.write("═══════════════════════════════════════════\n\n")
                
                comp = self.results['comparison']
                f.write(f"FP16 样本数量: {comp.get('fp16_count', 0)}\n")
                f.write(f"FP8 样本数量:  {comp.get('fp8_count', 0)}\n")
                f.write(f"对比文件夹:    {comp.get('comparison_dir', 'N/A')}\n\n")
                
                f.write("💡 建议: 请手动对比相同步数的生成图片，评估质量差异。\n")
                f.write("   例如: 对比 000500_0.png, 001000_0.png 等\n")
            
            f.write("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
        print(f"✅ 报告已保存: {report_path}")
        
        # 打印报告内容
        with open(report_path, 'r', encoding='utf-8') as f:
            print("\n" + f.read())
    
    def run_sequential_test(self):
        """运行顺序测试（先 FP16，后 FP8）"""
        self.print_banner("FP8 vs FP16 顺序对比测试")
        
        print("📋 测试计划:")
        print("   1. 运行 FP16 基准训练")
        print("   2. 运行 FP8 COAT 训练")
        print("   3. 对比生成样本")
        print("   4. 生成对比报告")
        
        input("\n按 Enter 键开始测试...")
        
        # 测试 1: FP16 基准
        self.results['fp16'] = self.run_training(
            'fp16',
            'ai_toolkit_integration/fp16_baseline_config.yaml',
            'FP16 基准'
        )
        
        if not self.results['fp16']['success']:
            print("\n⚠️  FP16 训练失败，是否继续 FP8 测试？")
            choice = input("输入 y 继续，其他键退出: ")
            if choice.lower() != 'y':
                return
        
        # 测试 2: FP8 COAT
        self.results['fp8'] = self.run_training(
            'fp8',
            'ai_toolkit_integration/coat_config.yaml',
            'FP8 COAT'
        )
        
        # 对比样本
        self.results['comparison'] = self.compare_samples()
        
        # 生成报告
        self.generate_report()
        
        self.print_banner("测试完成")
    
    def run_parallel_test(self):
        """运行并行测试（需要多GPU）"""
        print("⚠️  并行测试需要多GPU支持，当前暂未实现")
        print("💡 建议使用顺序测试模式")


def main():
    print("\n" + "="*60)
    print("  🔬 FP8 vs FP16 对比测试工具")
    print("="*60)
    
    print("\n选择测试模式:")
    print("1. 顺序测试 (推荐) - 先运行 FP16，再运行 FP8")
    print("2. 仅运行 FP16 基准")
    print("3. 仅运行 FP8 COAT")
    print("4. 仅对比现有结果")
    print("5. 退出")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    tester = ComparisonTest()
    
    if choice == "1":
        tester.run_sequential_test()
    
    elif choice == "2":
        tester.results['fp16'] = tester.run_training(
            'fp16',
            'ai_toolkit_integration/fp16_baseline_config.yaml',
            'FP16 基准'
        )
    
    elif choice == "3":
        tester.results['fp8'] = tester.run_training(
            'fp8',
            'ai_toolkit_integration/coat_config.yaml',
            'FP8 COAT'
        )
    
    elif choice == "4":
        tester.results['comparison'] = tester.compare_samples()
        tester.generate_report()
    
    else:
        print("👋 已退出")

if __name__ == "__main__":
    main()


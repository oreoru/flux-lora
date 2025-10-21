#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 显存监控工具
实时查看 GPU 使用情况
"""

import sys
import torch
import time
import os

# 确保输出立即显示
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

def format_bytes(bytes_value):
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def print_gpu_info():
    """打印GPU信息"""
    print("\n" + "="*60)
    print("  GPU 显存监控")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用！")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"\n检测到 {device_count} 个 GPU 设备\n")
    
    for i in range(device_count):
        print(f"━━━ GPU {i}: {torch.cuda.get_device_name(i)} ━━━")
        
        # 获取显存信息
        total_memory = torch.cuda.get_device_properties(i).total_memory
        reserved_memory = torch.cuda.memory_reserved(i)
        allocated_memory = torch.cuda.memory_allocated(i)
        free_memory = reserved_memory - allocated_memory
        
        print(f"  总显存:     {format_bytes(total_memory)}")
        print(f"  已预留:     {format_bytes(reserved_memory)} ({reserved_memory/total_memory*100:.1f}%)")
        print(f"  已分配:     {format_bytes(allocated_memory)} ({allocated_memory/total_memory*100:.1f}%)")
        print(f"  未分配:     {format_bytes(free_memory)}")
        print(f"  系统空闲:   {format_bytes(total_memory - reserved_memory)} ({(total_memory-reserved_memory)/total_memory*100:.1f}%)")
        
        # 显存使用率
        usage_percent = (allocated_memory / total_memory) * 100
        
        # 进度条
        bar_length = 40
        filled_length = int(bar_length * usage_percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\n  使用率: [{bar}] {usage_percent:.1f}%")
        
        # 警告
        if usage_percent > 90:
            print("  ⚠️  警告: 显存使用率超过 90%，可能即将耗尽！")
        elif usage_percent > 80:
            print("  ⚠️  注意: 显存使用率较高")
        elif usage_percent < 10:
            print("  ✅ 显存充足")
        
        # CUDA 能力
        capability = torch.cuda.get_device_capability(i)
        print(f"\n  CUDA 能力: {capability[0]}.{capability[1]}")
        
        # FP8 支持
        if capability[0] >= 9 or (capability[0] == 8 and capability[1] >= 9):
            print("  ✅ 支持硬件 FP8 加速")
        else:
            print("  ⚠️  不支持硬件 FP8 (使用软件模拟)")
        
        print()
    
    return True

def continuous_monitor(interval=2):
    """连续监控模式"""
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print_gpu_info()
            print(f"\n⏱️  每 {interval} 秒更新一次 (按 Ctrl+C 退出)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n✅ 监控已停止")

def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        print("🔍 启动连续监控模式...")
        continuous_monitor()
    else:
        print_gpu_info()
        print("\n💡 提示: 使用 'python check_gpu_memory.py --monitor' 启动连续监控\n")

if __name__ == "__main__":
    main()


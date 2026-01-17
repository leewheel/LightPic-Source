#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z-Image 系统诊断脚本
用于诊断 ComfyUI 和 WebUI 的崩溃问题
"""

import sys
import os
import platform
import subprocess
import json
from datetime import datetime

def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_python_version():
    """检查 Python 版本"""
    print_section("Python 版本检查")
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    
    # 检查是否为推荐的版本
    version_info = sys.version_info
    if version_info.major == 3 and version_info.minor >= 10:
        print("✓ Python 版本符合要求 (3.10+)")
    else:
        print("⚠ 警告: 建议使用 Python 3.10 或更高版本")

def check_gpu():
    """检查 GPU 状态"""
    print_section("GPU 状态检查")
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}:")
                print(f"  名称: {props.name}")
                print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
                print(f"  计算能力: {props.major}.{props.minor}")
                
                # 检查当前显存使用
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  已分配显存: {allocated:.2f} GB")
                print(f"  已保留显存: {reserved:.2f} GB")
        else:
            print("⚠ 警告: CUDA 不可用，将使用 CPU 模式")
    except ImportError:
        print("✗ 错误: PyTorch 未安装")
    except Exception as e:
        print(f"✗ 错误: 检查 GPU 时发生异常: {e}")

def check_nvidia_driver():
    """检查 NVIDIA 驱动"""
    print_section("NVIDIA 驱动检查")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ NVIDIA 驱动已安装")
            print("\n" + result.stdout)
        else:
            print("⚠ 警告: nvidia-smi 命令执行失败")
    except FileNotFoundError:
        print("⚠ 警告: 未找到 nvidia-smi 命令")
    except Exception as e:
        print(f"⚠ 警告: 检查 NVIDIA 驱动时发生异常: {e}")

def check_memory():
    """检查系统内存"""
    print_section("系统内存检查")
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"总内存: {mem.total / 1024**3:.2f} GB")
        print(f"可用内存: {mem.available / 1024**3:.2f} GB")
        print(f"已使用内存: {mem.used / 1024**3:.2f} GB ({mem.percent}%)")
        
        if mem.available < 4 * 1024**3:
            print("⚠ 警告: 可用内存不足 4GB，可能导致崩溃")
        else:
            print("✓ 内存充足")
    except ImportError:
        print("⚠ 警告: psutil 未安装，无法检查内存")
    except Exception as e:
        print(f"✗ 错误: 检查内存时发生异常: {e}")

def check_disk_space():
    """检查磁盘空间"""
    print_section("磁盘空间检查")
    
    try:
        import shutil
        current_dir = os.getcwd()
        total, used, free = shutil.disk_usage(current_dir)
        
        print(f"当前目录: {current_dir}")
        print(f"总空间: {total / 1024**3:.2f} GB")
        print(f"已使用: {used / 1024**3:.2f} GB")
        print(f"可用空间: {free / 1024**3:.2f} GB")
        
        if free < 10 * 1024**3:
            print("⚠ 警告: 可用磁盘空间不足 10GB")
        else:
            print("✓ 磁盘空间充足")
    except Exception as e:
        print(f"✗ 错误: 检查磁盘空间时发生异常: {e}")

def check_environment_variables():
    """检查环境变量"""
    print_section("环境变量检查")
    
    important_vars = [
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ENABLE_CUDA_MEM_EFFICIENCY",
        "PYTORCH_NO_CUDA_MEMORY_CACHING",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "CUBLAS_WORKSPACE_CONFIG",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
    ]
    
    for var in important_vars:
        value = os.environ.get(var, "未设置")
        print(f"{var}: {value}")

def check_comfyui_files():
    """检查 ComfyUI 文件"""
    print_section("ComfyUI 文件检查")
    
    # 检查 ComfyUI 目录
    comfyui_dir = os.path.join(os.getcwd(), "ComfyUI_windows_portable")
    if os.path.exists(comfyui_dir):
        print(f"✓ ComfyUI 目录存在: {comfyui_dir}")
        
        # 检查关键文件
        main_py = os.path.join(comfyui_dir, "ComfyUI", "main.py")
        if os.path.exists(main_py):
            print(f"✓ main.py 存在")
        else:
            print(f"✗ main.py 不存在")
        
        # 检查启动脚本
        bat_files = [
            "run_nvidia_gpu.bat",
            "run_nvidia_gpu_stable.bat",
            "run_cpu.bat"
        ]
        
        for bat in bat_files:
            bat_path = os.path.join(comfyui_dir, bat)
            if os.path.exists(bat_path):
                print(f"✓ {bat} 存在")
            else:
                print(f"⚠ {bat} 不存在")
    else:
        print(f"⚠ ComfyUI 目录不存在: {comfyui_dir}")

def check_model_files():
    """检查模型文件"""
    print_section("模型文件检查")
    
    model_dirs = [
        "ckpts",
        "lora",
        "Mod/vae",
        "Mod/transformer"
    ]
    
    for model_dir in model_dirs:
        model_path = os.path.join(os.getcwd(), model_dir)
        if os.path.exists(model_path):
            files = os.listdir(model_path)
            print(f"✓ {model_dir}: {len(files)} 个文件")
        else:
            print(f"⚠ {model_dir}: 目录不存在")

def provide_recommendations():
    """提供修复建议"""
    print_section("修复建议")
    
    recommendations = [
        "1. 如果遇到 Exit Code: -1073741819 崩溃:",
        "   - 确保已应用 low_vram_fix.py 补丁",
        "   - 使用 run_nvidia_gpu_stable.bat 启动 ComfyUI",
        "   - 检查 NVIDIA 驱动是否为最新版本",
        "",
        "2. 如果显存不足:",
        "   - 使用更小的模型",
        "   - 减少 batch size",
        "   - 启用 CPU offload",
        "",
        "3. 如果内存不足:",
        "   - 关闭其他应用程序",
        "   - 增加系统虚拟内存",
        "   - 清理临时文件",
        "",
        "4. 如果磁盘空间不足:",
        "   - 清理 outputs 目录",
        "   - 删除不需要的模型文件",
        "",
        "5. 通用建议:",
        "   - 定期重启计算机",
        "   - 更新显卡驱动",
        "   - 使用稳定的启动脚本"
    ]
    
    for rec in recommendations:
        print(rec)

def save_diagnostic_report():
    """保存诊断报告"""
    print_section("保存诊断报告")
    
    report_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"diagnostic_report_{timestamp}.txt")
    
    try:
        # 重定向输出到文件
        original_stdout = sys.stdout
        
        with open(report_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            
            # 重新运行所有检查
            print(f"Z-Image 系统诊断报告")
            print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"操作系统: {platform.system()} {platform.release()}")
            print(f"处理器: {platform.processor()}")
            
            check_python_version()
            check_gpu()
            check_nvidia_driver()
            check_memory()
            check_disk_space()
            check_environment_variables()
            check_comfyui_files()
            check_model_files()
            provide_recommendations()
        
        sys.stdout = original_stdout
        print(f"✓ 诊断报告已保存: {report_file}")
        
    except Exception as e:
        sys.stdout = original_stdout
        print(f"✗ 保存诊断报告时发生异常: {e}")

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  Z-Image 系统诊断工具")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        check_python_version()
        check_gpu()
        check_nvidia_driver()
        check_memory()
        check_disk_space()
        check_environment_variables()
        check_comfyui_files()
        check_model_files()
        provide_recommendations()
        save_diagnostic_report()
        
        print("\n" + "=" * 70)
        print("  诊断完成")
        print("=" * 70)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n请查看 logs 目录中的诊断报告文件以获取详细信息。")
        
    except KeyboardInterrupt:
        print("\n\n诊断被用户中断")
    except Exception as e:
        print(f"\n\n诊断过程中发生异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""
高级诊断脚本 - 详细追踪崩溃原因
基于 GitHub issues: 
- https://github.com/Comfy-Org/ComfyUI/issues/10700
- https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/592
"""
import os
import sys
import time
import traceback
import logging
from datetime import datetime
import subprocess

# 设置日志
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"advanced_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_system_info():
    """检查系统信息"""
    logger.info("=" * 80)
    logger.info("系统信息检查")
    logger.info("=" * 80)
    
    # Python 版本
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"Python 可执行文件: {sys.executable}")
    
    # 操作系统
    import platform
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"处理器: {platform.processor()}")
    
    # 内存信息
    import psutil
    mem = psutil.virtual_memory()
    logger.info(f"总内存: {mem.total / (1024**3):.2f} GB")
    logger.info(f"可用内存: {mem.available / (1024**3):.2f} GB")
    logger.info(f"内存使用率: {mem.percent}%")
    
    # 磁盘信息
    disk = psutil.disk_usage('/')
    logger.info(f"磁盘总空间: {disk.total / (1024**3):.2f} GB")
    logger.info(f"磁盘可用空间: {disk.free / (1024**3):.2f} GB")
    
    # 环境变量
    logger.info("\n关键环境变量:")
    cuda_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '未设置')
    logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {cuda_alloc_conf}")
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    torch_allow_tf32 = os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', '未设置')
    logger.info(f"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: {torch_allow_tf32}")
    
    cublas_workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG', '未设置')
    logger.info(f"CUBLAS_WORKSPACE_CONFIG: {cublas_workspace}")

def check_gpu_info():
    """检查GPU信息"""
    logger.info("\n" + "=" * 80)
    logger.info("GPU 信息检查")
    logger.info("=" * 80)
    
    try:
        import torch
        logger.info(f"PyTorch 版本: {torch.__version__}")
        logger.info(f"CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA 版本: {torch.version.cuda}")
            logger.info(f"cuDNN 版本: {torch.backends.cudnn.version()}")
            logger.info(f"cuDNN 启用: {torch.backends.cudnn.enabled}")
            
            device_count = torch.cuda.device_count()
            logger.info(f"GPU 数量: {device_count}")
            
            for i in range(device_count):
                logger.info(f"\nGPU {i}:")
                logger.info(f"  名称: {torch.cuda.get_device_name(i)}")
                logger.info(f"  计算能力: {torch.cuda.get_device_capability(i)}")
                logger.info(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
                
                # 检查 bfloat16 支持
                try:
                    bf16_supported = torch.cuda.is_bf16_supported()
                    logger.info(f"  bfloat16 支持: {bf16_supported}")
                except Exception as e:
                    logger.warning(f"  bfloat16 支持检查失败: {e}")
                
                # 检查当前显存使用
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory
                reserved_memory = torch.cuda.memory_reserved(i)
                allocated_memory = torch.cuda.memory_allocated(i)
                
                logger.info(f"  已分配显存: {allocated_memory / (1024**3):.2f} GB")
                logger.info(f"  已保留显存: {reserved_memory / (1024**3):.2f} GB")
                logger.info(f"  可用显存: {(total_memory - reserved_memory) / (1024**3):.2f} GB")
                
                # 检查内存分配器
                try:
                    allocator_backend = torch.cuda.get_allocator_backend()
                    logger.info(f"  内存分配器: {allocator_backend}")
                except Exception as e:
                    logger.warning(f"  内存分配器检查失败: {e}")
                
                # 检查 cudaMallocAsync
                cuda_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
                if 'cudaMallocAsync' in cuda_alloc_conf:
                    logger.warning(f"  ⚠️  cudaMallocAsync 已启用！这可能导致崩溃")
                else:
                    logger.info(f"  ✓ cudaMallocAsync 已禁用")
    except Exception as e:
        logger.error(f"GPU 检查失败: {e}")
        logger.error(traceback.format_exc())

def check_nvidia_driver():
    """检查NVIDIA驱动"""
    logger.info("\n" + "=" * 80)
    logger.info("NVIDIA 驱动检查")
    logger.info("=" * 80)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("nvidia-smi 输出:")
            logger.info(result.stdout)
        else:
            logger.error(f"nvidia-smi 执行失败: {result.stderr}")
    except Exception as e:
        logger.error(f"nvidia-smi 检查失败: {e}")

def check_comfyui_files():
    """检查ComfyUI文件"""
    logger.info("\n" + "=" * 80)
    logger.info("ComfyUI 文件检查")
    logger.info("=" * 80)
    
    comfyui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ComfyUI_windows_portable", "ComfyUI")
    
    # 检查关键文件
    key_files = [
        "main.py",
        "cuda_malloc.py",
        "cuda_malloc_fix.py",
        "comfy/model_management.py",
        "comfy/sd.py"
    ]
    
    for file in key_files:
        file_path = os.path.join(comfyui_dir, file)
        if os.path.exists(file_path):
            logger.info(f"✓ {file} 存在")
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            logger.info(f"  文件大小: {file_size} bytes")
        else:
            logger.warning(f"✗ {file} 不存在")

def check_model_files():
    """检查模型文件"""
    logger.info("\n" + "=" * 80)
    logger.info("模型文件检查")
    logger.info("=" * 80)
    
    comfyui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ComfyUI_windows_portable")
    
    # 检查模型目录
    model_dirs = [
        "models/checkpoints",
        "models/vae",
        "models/clip",
        "models/loras"
    ]
    
    for model_dir in model_dirs:
        dir_path = os.path.join(comfyui_dir, model_dir)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            logger.info(f"✓ {model_dir}: {len(files)} 个文件")
            # 列出前5个文件
            for i, file in enumerate(files[:5]):
                file_path = os.path.join(dir_path, file)
                file_size = os.path.getsize(file_path) / (1024**2)
                logger.info(f"  - {file} ({file_size:.2f} MB)")
            if len(files) > 5:
                logger.info(f"  ... 还有 {len(files) - 5} 个文件")
        else:
            logger.warning(f"✗ {model_dir} 不存在")

def test_torch_operations():
    """测试PyTorch操作"""
    logger.info("\n" + "=" * 80)
    logger.info("PyTorch 操作测试")
    logger.info("=" * 80)
    
    try:
        import torch
        
        # 测试基本张量操作
        logger.info("测试基本张量操作...")
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        z = x @ y
        logger.info(f"✓ 基本张量操作成功: {z.shape}")
        
        # 测试CUDA操作
        if torch.cuda.is_available():
            logger.info("\n测试CUDA操作...")
            device = torch.device("cuda:0")
            
            # 测试float16
            logger.info("测试 float16...")
            x_fp16 = torch.randn(1000, 1000, dtype=torch.float16).to(device)
            y_fp16 = torch.randn(1000, 1000, dtype=torch.float16).to(device)
            z_fp16 = x_fp16 @ y_fp16
            logger.info(f"✓ float16 操作成功: {z_fp16.shape}")
            
            # 测试bfloat16
            logger.info("测试 bfloat16...")
            try:
                x_bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16).to(device)
                y_bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16).to(device)
                z_bf16 = x_bf16 @ y_bf16
                logger.info(f"✓ bfloat16 操作成功: {z_bf16.shape}")
            except Exception as e:
                logger.error(f"✗ bfloat16 操作失败: {e}")
                logger.error(traceback.format_exc())
            
            # 测试VAE编码/解码
            logger.info("\n测试 VAE 操作...")
            try:
                # 创建一个简单的VAE测试
                from comfy.sd import VAE
                logger.info("✓ VAE 模块导入成功")
            except Exception as e:
                logger.error(f"✗ VAE 模块导入失败: {e}")
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"PyTorch 操作测试失败: {e}")
        logger.error(traceback.format_exc())

def main():
    """主函数"""
    logger.info("开始高级诊断...")
    logger.info(f"日志文件: {log_file}")
    
    try:
        check_system_info()
        check_gpu_info()
        check_nvidia_driver()
        check_comfyui_files()
        check_model_files()
        test_torch_operations()
        
        logger.info("\n" + "=" * 80)
        logger.info("诊断完成")
        logger.info("=" * 80)
        logger.info(f"详细日志已保存到: {log_file}")
        
    except Exception as e:
        logger.error(f"诊断过程中发生错误: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

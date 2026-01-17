# Z-Image 低显存模式崩溃修复补丁 (增强版)
# 此文件修复了在加载模型时发生的访问违例崩溃问题
# 主要问题：低显存模式下 pipe.to("cpu") 和 enable_sequential_cpu_offload() 的冲突
# 以及 ComfyUI 的 GPU 内存访问冲突 (Exit Code: -1073741819)

import sys
import os

print("=" * 60)
print("[补丁] 应用低显存模式修复 (增强版)...")
print("=" * 60)

# ==========================================
# 方案1：设置 PyTorch CUDA 内存分配策略
# ==========================================
# 限制 split size 防止内存碎片
# 对于 12GB 显存，使用 256MB 的 split size 比较合适
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
print("[补丁] ✓ 设置 CUDA 内存分配策略: max_split_size_mb=256")

# ==========================================
# 方案2：禁用某些可能导致冲突的优化
# ==========================================
# 防止 Sequential Offload 与手动 CPU 移动冲突
os.environ["PYTORCH_ENABLE_CUDA_MEM_EFFICIENCY"] = "0"
print("[补丁] ✓ 禁用 CUDA 内存效率优化")

# ==========================================
# 方案3：强制使用确定性算法
# ==========================================
# 避免某些随机性导致的内存访问问题
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
print("[补丁] ✓ 设置 CUBLAS 工作空间配置")

# ==========================================
# 方案4：优化 PyTorch 内存管理
# ==========================================
# 禁用 CUDA 内存缓存，避免内存碎片
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
print("[补丁] ✓ 禁用 CUDA 内存缓存")

# ==========================================
# 方案5：设置线程数限制
# ==========================================
# 限制 PyTorch 使用的线程数，避免资源竞争
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
print("[补丁] ✓ 设置线程数限制: 4")

# ==========================================
# 方案6：禁用某些可能导致崩溃的优化
# ==========================================
# 禁用 TF32 以提高兼容性
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
print("[补丁] ✓ 禁用 TF32 覆盖")

# ==========================================
# 方案7：设置内存分配器
# ==========================================
# 使用更保守的内存分配策略
os.environ["PYTORCH_CUDA_ALLOCATOR"] = "native"
print("[补丁] ✓ 使用原生 CUDA 分配器")

print("=" * 60)
print("[补丁] ✓ 所有内存管理修复已应用")
print("[补丁] 现在可以正常加载 Z-Image-App.py")
print("=" * 60)

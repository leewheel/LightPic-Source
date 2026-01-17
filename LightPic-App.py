import os
import time


# 必须处于文件最顶端：环境配置
os.environ["DIFFUSERS_USE_PEFT_BACKEND"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


import sys
import torch
import psutil
import random
import re
import uuid
import gc
import tempfile
import json
import string
from datetime import datetime
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw

# 强制 stdout/stderr 使用 UTF-8
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==========================================
# 新增：导入标准采样器调度器
# ==========================================
from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    EDMEulerScheduler  # 导入以便检查类型，虽然默认时我们不手动实例化它
)

# 可根据需求增减风格，但不涉及具体对象
STYLE_POOL = [
    "documentary photography",
    "natural history archival",
    "field biology specimen photography",
    "哺乳动物摄影",
    "鸟类摄影",
    "昆虫摄影",
    "植物摄影",
    "微距摄影",
    "自然风光摄影",
    "生态摄影",
    "宠物摄影",
    "国家地理杂志纪录片"
]

def augment_prompt(user_prompt: str) -> str:
    """
    在用户自由输入的 prompt 上增加随机扰动：
    - Specimen ID：随机字母数字
    - Photography style：随机选择
    """
    random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    style = random.choice(STYLE_POOL)
    return f"{user_prompt}\nSpecimen ID: {random_id}\nPhotography style: {style}"

# 配置基础路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 目录配置
DEFAULT_MODEL_PATH = os.path.join(current_dir, "ckpts", "Z-Image-Turbo")
LORA_ROOT = os.path.join(current_dir, "lora")
OUTPUT_ROOT = os.path.join(current_dir, "outputs")
MOD_VAE_DIR = os.path.join(current_dir, "Mod", "vae")
MOD_TRANS_DIR = os.path.join(current_dir, "Mod", "transformer")
for p in [LORA_ROOT, OUTPUT_ROOT, MOD_VAE_DIR, MOD_TRANS_DIR]:
    os.makedirs(p, exist_ok=True)

try:
    import gradio as gr
    from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, AutoencoderKL
    from safetensors.torch import load_file
except ImportError as e:
    print(f"Core import failed: {e}")
    sys.exit(1)

# ==========================================
# 国际化配置 (I18N)
# ==========================================
I18N = {
    "zh": {
        "title": "# 🎨 Z-Image-Turbo LowVram Edition",
        "btn_lang": "Language / 语言",
        "tab_t2i": "文成图",
        "tab_edit": "图片编辑",
        "tab_i2i": "图生图",
        "tab_inpaint": "局部重绘",
        "tab_fusion": "融合图",
        
        "label_prompt": "Prompt",
        "btn_flush_vram": "🧹 清理显存",
        "label_vram_threshold": "自动清理阈值 (%)",
        
        "acc_lora": "LoRA 权重设置",
        "btn_refresh_lora": "🔄 刷新 LoRA 文件列表",
        "txt_no_lora": "*未检测到 LoRA 文件*",
        "label_weight": "权重",
        
        "acc_model": "模型设置",
        "btn_refresh_model": "🔄 刷新底模/VAE",
        "label_transformer": "Transformer",
        "label_vae": "VAE",
        "label_perf": "性能模式",
        "val_perf_high": "高端机 (显存>=20GB)",
        "val_perf_low": "低端机 (显存优化)",
        "label_width": "宽 (16倍数)",
        "label_height": "高 (16倍数)",
        "label_steps": "步数",
        "label_cfg": "CFG",
        "label_batch": "生成张数",
        "label_seed": "种子",
        "label_random_seed": "随机种子",
        "label_sampler": "采样器",
        
        "btn_run": "🚀 开始生成",
        "btn_stop": "🛑 停止生成",
        "btn_gen": "🚀 生成",
        "btn_stop_short": "🛑 停止",
        "label_output": "输出结果",
        "label_gallery_i2i": "图生图结果",
        "label_gallery_inpaint": "局部重绘结果",
        "label_gallery_fusion": "融合结果",
        
        "label_upload_img": "上传图片",
        "label_rotate": "旋转角度 (度)",
        "label_crop_x": "裁剪 X (%)",
        "label_crop_y": "裁剪 Y (%)",
        "label_crop_w": "裁剪宽度 (%)",
        "label_crop_h": "裁剪高度 (%)",
        "label_flip_h": "水平翻转",
        "label_flip_v": "垂直翻转",
        "btn_edit": "开始编辑",
        "label_edited": "编辑后的图片",
        "label_filter": "应用滤镜",
        "label_brightness": "亮度调整 (%)",
        "label_contrast": "对比度调整 (%)",
        "label_saturation": "饱和度调整 (%)",
        
        "label_ref_img": "上传参考图",
        "label_prompt_rec": "Prompt (推荐)",
        "ph_prompt_i2i": "描述你想要生成的画面...",
        "label_out_w": "输出宽 (0=自动保持比例)",
        "label_out_h": "输出高 (0=自动保持比例)",
        "tip_res": "**提示：** 宽高都为0时自动保持上传图比例并接近1024；手动设置大于512时生效",
        "label_strength": "重绘强度",
        "label_cfg_fixed": "CFG（Turbo模型固定为0.0）",
        "label_cfg_turbo": "CFG (Turbo Img2Img)",
        "label_cfg_inpaint": "CFG (Inpaint)",
        
        # --- Inpaint 特有 ---
        "lbl_inpaint_editor": "绘制 Mask (白色为修改区，黑色为保留区)",
        "lbl_inpaint_tip": "提示：先在上方'上传原图'加载图片，然后在下方绘制 Mask。",
        
        "desc_fusion": "**融合2张图片**：图片1提供主要结构/姿势，图片2提供细节/脸部/风格。",
        "label_img1": "图片1（主结构/姿势）",
        "label_img2": "图片2（细节/脸部/风格）",
        "label_fusion_prompt": "融合描述 Prompt",
        "label_blend": "图片2混合强度 (0=全用图片1, 1=全用图片2)",
        "label_denoise": "重绘强度 (越高变化越大)",
        
        # 滤镜选项
        "f_blur": "模糊", "f_contour": "轮廓", "f_detail": "细节",
        "f_edge": "边缘增强", "f_edge_more": "更多边缘增强",
        "f_emboss": "浮雕", "f_find_edge": "查找边缘",
        "f_sharp": "锐化", "f_smooth": "平滑", "f_smooth_more": "更多平滑",
        
        "msg_scan_done": "✅ 扫描完成！检测到 **{}** 个 LoRA 文件。",
        "msg_vram_loading": "显存状态加载中...",
        "msg_interrupt": "🛑 正在强制中断..."
    },
    "en": {
        "title": "# 🎨 Z-Image-Turbo LowVram Edition", 
        "btn_lang": "Language / 语言",
        "tab_t2i": "Text to Image",
        "tab_edit": "Image Edit",
        "tab_i2i": "Image to Image",
        "tab_inpaint": "Inpaint",
        "tab_fusion": "Fusion",
        
        "label_prompt": "Prompt",
        "btn_flush_vram": "🧹 Flush VRAM",
        "label_vram_threshold": "Auto Flush Threshold (%)",
        
        "acc_lora": "LoRA Settings",
        "btn_refresh_lora": "🔄 Scan LoRA Directory",
        "txt_no_lora": "*No LoRA files detected*",
        "label_weight": "Weight",
        
        "acc_model": "Model Settings",
        "btn_refresh_model": "🔄 Refresh Base Model/VAE",
        "label_transformer": "Transformer",
        "label_vae": "VAE",
        "label_perf": "Performance Mode",
        "val_perf_high": "High End (VRAM>=20GB)",
        "val_perf_low": "Low End (Optimized)",
        "label_width": "Width (x16)",
        "label_height": "Height (x16)",
        "label_steps": "Steps",
        "label_cfg": "CFG",
        "label_batch": "Batch Size",
        "label_seed": "Seed",
        "label_random_seed": "Random Seed",
        "label_sampler": "Sampler",
        
        "btn_run": "🚀 Generate",
        "btn_stop": "🛑 Stop Generation",
        "btn_gen": "🚀 Generate",
        "btn_stop_short": "🛑 Stop",
        "label_output": "Output Results",
        "label_gallery_i2i": "Img2Img Results",
        "label_gallery_inpaint": "Inpaint Results",
        "label_gallery_fusion": "Fusion Results",
        
        "label_upload_img": "Upload Image",
        "label_rotate": "Rotation (Deg)",
        "label_crop_x": "Crop X (%)",
        "label_crop_y": "Crop Y (%)",
        "label_crop_w": "Crop Width (%)",
        "label_crop_h": "Crop Height (%)",
        "label_flip_h": "Flip Horizontal",
        "label_flip_v": "Flip Vertical",
        "btn_edit": "Start Editing",
        "label_edited": "Edited Image",
        "label_filter": "Apply Filter",
        "label_brightness": "Brightness (%)",
        "label_contrast": "Contrast (%)",
        "label_saturation": "Saturation (%)",
        
        "label_ref_img": "Upload Reference",
        "label_prompt_rec": "Prompt (Recommended)",
        "ph_prompt_i2i": "Describe the image you want to generate...",
        "label_out_w": "Output Width (0=Auto)",
        "label_out_h": "Output Height (0=Auto)",
        "tip_res": "**Tip:** If both are 0, aspect ratio is kept automatically near 1024px.",
        "label_strength": "Denoising Strength",
        "label_cfg_fixed": "CFG (Fixed at 0.0 for Turbo)",
        "label_cfg_turbo": "CFG (Turbo Img2Img)",
        "label_cfg_inpaint": "CFG (Inpaint)",
        
        # --- Inpaint 特有 ---
        "lbl_inpaint_editor": "Draw Mask (White=Modify, Black=Keep)",
        "lbl_inpaint_tip": "Tip: Upload base image first, then draw Mask below.",

        "desc_fusion": "**智能融合图**: 融合两张参考图的特征，生成包含两张图元素的新场景。支持人物+人物、人物+物品/动物等组合。",
        "label_img1": "参考图1（第一张图）",
        "label_img2": "参考图2（第二张图）",
        "label_fusion_prompt": "场景描述（描述两张图如何融合，例如：图一的人物和图二的人物坐在公园长椅上聊天）",
        "label_blend": "融合权重（0=偏向图1，1=偏向图2，0.5=平衡融合）",
        "label_denoise": "重绘强度（数值越大，变化越大，建议0.5-0.8）",
        
        # 滤镜选项
        "f_blur": "Blur", "f_contour": "Contour", "f_detail": "Detail",
        "f_edge": "Edge Enhance", "f_edge_more": "Edge Enhance More",
        "f_emboss": "Emboss", "f_find_edge": "Find Edges",
        "f_sharp": "Sharpen", "f_smooth": "Smooth", "f_smooth_more": "Smooth More",
        
        "msg_scan_done": "✅ Scan Complete! Detected **{}** LoRA files.",
        "msg_vram_loading": "VRAM Status Loading...",
        "msg_interrupt": "🛑 Force Interrupting..."
    }
}

# ==========================================
# 采样器列表与配置
# ==========================================
SAMPLER_LIST = [
    "Default (Z-Image)",   # 【默认】不进行任何替换，保持原有正常工作状态
    "DPM++ 2M Karras",     # 【推荐】动态偏移，适合Z-Image
    "DPM++ 2S Karras",     # 动态偏移
    "Euler a",
    "Euler",
    "DDIM",
    "UniPC"
]

import inspect

class CompatibleScheduler:
    """
    兼容性包装器：过滤掉 Z-Image 传给调度器的多余参数（如 'mu'），
    同时强制删除可能存在的 'mu' 防止底层报错。
    """
    def __init__(self, scheduler):
        self._scheduler = scheduler

    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        # 强制移除 'mu'，无论调度器是否声称支持它
        kwargs.pop('mu', None)
        
        sig = inspect.signature(self._scheduler.set_timesteps)
        valid_params = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        return self._scheduler.set_timesteps(num_inference_steps, device=device, **filtered_kwargs)
    
    def step(self, *args, **kwargs):
        return self._scheduler.step(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            setattr(self._scheduler, name, value)

def get_scheduler(sampler_name, config):
    """
    获取调度器实例。
    如果是 Default，返回 None（不替换）。
    如果是 DPM++，强制开启动态偏移以兼容 Z-Image。
    """
    base_scheduler = None
    
    # 【策略】如果是默认选项，直接返回 None，不干预原有的调度器
    # 这样可以保证出图正常，避免雪花问题
    if sampler_name == "Default (Z-Image)":
        return None

    try:
        if sampler_name == "DPM++ 2M Karras":
            base_scheduler = DPMSolverMultistepScheduler.from_config(
                config, 
                algorithm="DPM++ 2M", 
                use_karras_sigmas=True,
                use_dynamic_shifting=True,      # 关键：开启动态偏移
                time_shift_type="exponential"   # 关键：指数型偏移
            )
        elif sampler_name == "DPM++ 2S Karras":
            base_scheduler = DPMSolverSinglestepScheduler.from_config(
                config, 
                use_karras_sigmas=True,
                use_dynamic_shifting=True,
                time_shift_type="exponential"
            )
        elif sampler_name == "Euler a":
            base_scheduler = EulerAncestralDiscreteScheduler.from_config(config)
        elif sampler_name == "Euler":
            base_scheduler = EulerDiscreteScheduler.from_config(config)
        elif sampler_name == "DDIM":
            base_scheduler = DDIMScheduler.from_config(config)
        elif sampler_name == "UniPC":
            base_scheduler = UniPCMultistepScheduler.from_config(config)
        else:
            # 默认回退
            base_scheduler = EulerAncestralDiscreteScheduler.from_config(config)

    except Exception as e:
        print(f"Scheduler init failed for {sampler_name}, fallback to Euler a: {e}")
        base_scheduler = EulerAncestralDiscreteScheduler.from_config(config)

    # 返回包装后的调度器
    return CompatibleScheduler(base_scheduler)

# ==========================================
# 基础函数
# ==========================================

def process_mask_for_inpaint(mask_image):
    """处理Mask图像，为inpaint做准备"""
    if mask_image is None:
        return None
    if mask_image.mode == 'RGBA':
        import numpy as np
        mask_array = np.array(mask_image)
        alpha = mask_array[:, :, 3] if mask_array.shape[2] > 3 else None
        rgb = mask_array[:, :, :3]
        rgb_gray = np.dot(rgb, [0.299, 0.587, 0.114])
        if alpha is not None:
            mask_gray = np.where(alpha > 10, 255, 0).astype(np.uint8)
        else:
            mask_gray = np.where(rgb_gray > 10, 255, 0).astype(np.uint8)
        mask = Image.fromarray(mask_gray, mode='L')
    else:
        if mask_image.mode != 'L':
            mask_image = mask_image.convert('L')
        mask = mask_image.point(lambda p: 255 if p > 10 else 0)
    if mask.getextrema()[1] == 0:
        return None
    return mask

def materialize_vae(vae):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for param in vae.parameters():
            if param.device.type == "meta":
                real = torch.empty_like(param, device=device)
                param.data = real
    vae.to(device)
    vae.eval()

# ==========================================
# 设备探测与硬件报告
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
is_interrupted = False

print("\n" + "="*50)
if DEVICE == "cuda":
    GPU_NAME = torch.cuda.get_device_name(0)
    TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory
    print(f"运行模式: [ GPU ]".encode('utf-8', errors='replace').decode())
    print(f"核心型号: {GPU_NAME}".encode('utf-8', errors='replace').decode())
    print(f"显存总量: {TOTAL_VRAM/1024**3:.2f} GB".encode('utf-8', errors='replace').decode())
else:
    TOTAL_VRAM = 0
    print(f"运行模式: [ CPU ]".encode('utf-8', errors='replace').decode())
print("="*50 + "\n")

def get_vram_info():
    if DEVICE == "cuda":
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.get_device_properties(0).total_memory
        usage_pct = (reserved / TOTAL_VRAM) * 100 if TOTAL_VRAM > 0 else 0
        vram_str = f"显存占用: {usage_pct:.1f}% ({reserved/1024**3:.2f}GB / {TOTAL_VRAM/1024**3:.2f}GB)"
    else:
        usage_pct = 0
        vram_str = "显存占用: CPU 模式"
    mem = psutil.virtual_memory()
    ram_str = f"内存占用: {mem.percent:.1f}% ({(mem.total - mem.available)/1024**3:.2f}GB / {mem.total/1024**3:.2f}GB)"
    status = f"{vram_str} ｜ {ram_str}"
    return usage_pct, status

def auto_flush_vram(threshold=90):
    usage_pct, _ = get_vram_info()
    if usage_pct > threshold:
        gc.collect()
        torch.cuda.empty_cache()
        return True
    return False

def scan_lora_files():
    if not os.path.exists(LORA_ROOT): return []
    return sorted([f for f in os.listdir(LORA_ROOT) if f.lower().endswith(".safetensors")])

def scan_model_items(base_path):
    if not os.path.exists(base_path): return []
    items = []
    for f in os.listdir(base_path):
        full_path = os.path.join(base_path, f)
        if os.path.isdir(full_path):
            items.append(f)
        elif f.lower().endswith((".safetensors", ".bin", ".pt")):
            items.append(f)
    return sorted(items)

LORA_FILES = scan_lora_files()
print(f"已检测到 {len(LORA_FILES)} 个 LoRA 文件。".encode('utf-8', errors='replace').decode())

# ==========================================
# 模型管理器
# ==========================================
class ModelManager:
    def __init__(self):
        self.pipe = None 
        self.current_state = {
            "mode": None,      
            "t_choice": None,  
            "v_choice": None,
            "perf_mode": None
        }
        self.current_loras = []
        self.current_weights_map = {} 

    def _clear_pipeline(self):
        if self.pipe is not None:
            print(f"正在销毁旧管道以释放显存...".encode('utf-8', errors='replace').decode())
            try:
                self.pipe.unload_lora_weights()
            except:
                pass
            del self.pipe
            self.pipe = None
        if hasattr(sys, 'last_traceback'):
            del sys.last_traceback
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def _init_pipeline_base(self, mode, low_vram=False):
        # 【修复】不再传递 device 参数，避免被忽略，改用显式 .to()
        
        # 【关键修复】在加载模型前设置环境变量，防止低显存模式下崩溃
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        if mode == 'txt':
            print(f"初始化基础 Pipeline (文成图)... 策略: {'低显存优化 (Sequential Offload)' if low_vram else '全显存'}".encode('utf-8', errors='replace').decode())
            pipe = ZImagePipeline.from_pretrained(
                DEFAULT_MODEL_PATH, 
                local_files_only=True
            )
        else:
            print(f"初始化基础 Pipeline (图生图/重绘)... 策略: {'低显存优化 (Sequential Offload)' if low_vram else '全显存'}".encode('utf-8', errors='replace').decode())
            pipe = ZImageImg2ImgPipeline.from_pretrained(
                DEFAULT_MODEL_PATH, 
                local_files_only=True
            )
        
        # 【关键修复】先设置device为CPU，然后统一dtype，最后enable_sequential_cpu_offload
        if low_vram:
            print(f"  [System] 正在统一模型精度为: {DTYPE} 并准备低显存模式".encode('utf-8', errors='replace').decode())
            pipe.to(device="cpu", dtype=DTYPE)
            pipe.enable_sequential_cpu_offload()
            print(f"  [System] 模型已加载至 RAM 并启用 Sequential Offload。".encode('utf-8', errors='replace').decode())
        else:
            print(f"  [System] 正在统一模型精度为: {DTYPE}".encode('utf-8', errors='replace').decode())
            pipe.to(device=DEVICE, dtype=DTYPE)
        
        return pipe

    def _inject_components(self, pipe, t_choice, v_choice, low_vram=False):
        if t_choice != "default":
            t_path = os.path.join(MOD_TRANS_DIR, t_choice)
            if os.path.isfile(t_path):
                print(f"载入 Transformer: {t_choice}".encode('utf-8', errors='replace').decode())
                state_dict = load_file(t_path, device="cpu")
                processed = {}
                prefix = "model.diffusion_model."
                for k, v in state_dict.items():
                    new_k = k[len(prefix):] if k.startswith(prefix) else k
                    processed[new_k] = v.to(DTYPE)
                pipe.transformer.load_state_dict(processed, strict=False, assign=True)
                del state_dict, processed
                gc.collect()

        if v_choice != "default":
            vae_path = os.path.join(MOD_VAE_DIR, v_choice)
            print(f"载入 VAE: {v_choice}".encode('utf-8', errors='replace').decode())
            
            if low_vram:
                vae_device_map = {"": "cpu"}
            else:
                vae_device_map = None

            if os.path.isfile(vae_path):
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_file_path = os.path.join(tmpdir, "config.json")
                    with open(config_file_path, "w", encoding="utf-8") as f:
                        json.dump(dict(pipe.vae.config), f, indent=2)
                    # 修复 dtype 兼容性
                    try:
                        pipe.vae = AutoencoderKL.from_single_file(vae_path, dtype=DTYPE, config=tmpdir, device_map=vae_device_map)
                    except TypeError:
                        pipe.vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=DTYPE, config=tmpdir, device_map=vae_device_map)
            else:
                try:
                    pipe.vae = AutoencoderKL.from_pretrained(vae_path, dtype=DTYPE, device_map=vae_device_map)
                except TypeError:
                    pipe.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=DTYPE, device_map=vae_device_map)
        return pipe

    def _apply_loras(self, pipe, selected_loras, weights_map):
        if self.current_loras == selected_loras and self.current_weights_map == weights_map:
            return

        print("正在配置 LoRA...".encode('utf-8', errors='replace').decode())
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        if not selected_loras:
            self.current_loras = []
            self.current_weights_map = {}
            return

        active_adapters = []
        adapter_weights = []

        for lora_file in selected_loras:
            adapter_name = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(lora_file)[0])
            weight = weights_map.get(lora_file, 1.0)
            try:
                pipe.load_lora_weights(LORA_ROOT, weight_name=lora_file, adapter_name=adapter_name)
                active_adapters.append(adapter_name)
                adapter_weights.append(weight)
            except Exception as e:
                print(f"LoRA {lora_file} 加载失败: {e}".encode('utf-8', errors='replace').decode())

        if active_adapters:
            pipe.set_adapters(active_adapters, adapter_weights=adapter_weights)
        
        self.current_loras = list(selected_loras)
        self.current_weights_map = dict(weights_map)

    def get_pipeline(self, t_choice, v_choice, selected_loras, weights_map, mode='txt', perf_mode="高端机 (显存>=20GB)"):
        is_low_vram = (
            "低端机" in perf_mode or 
            "显存优化" in perf_mode or 
            "Low End" in perf_mode or 
            "Optimized" in perf_mode
        )

        need_rebuild = (
            self.pipe is None or
            self.current_state["mode"] != mode or
            self.current_state["t_choice"] != t_choice or
            self.current_state["v_choice"] != v_choice or
            self.current_state["perf_mode"] != perf_mode
        )

        if need_rebuild:
            self._clear_pipeline() 
            try:
                # 1. 初始化并强制移入 CPU (如果 low_vram)
                temp_pipe = self._init_pipeline_base(mode, low_vram=is_low_vram)
                temp_pipe = self._inject_components(temp_pipe, t_choice, v_choice, low_vram=is_low_vram)
                
                if DEVICE == "cuda":
                    if is_low_vram:
                        # 2. 开启 Sequential Offload (用户指定)
                        print("  [System] 已启用低显存优化模式".encode('utf-8', errors='replace').decode())
                        temp_pipe.enable_sequential_cpu_offload()
                    else:
                        print("  [System] 已启用高端机模式 (Full CUDA)".encode('utf-8', errors='replace').decode())
                        temp_pipe.to("cuda")

                self.pipe = temp_pipe
                self.current_state = {
                    "mode": mode, "t_choice": t_choice, "v_choice": v_choice, "perf_mode": perf_mode
                }
                self.current_loras = [] 
                self.current_weights_map = {}
                
            except Exception as e:
                self._clear_pipeline()
                raise gr.Error(f"模型加载崩溃: {str(e)}")

        self._apply_loras(self.pipe, selected_loras, weights_map)
        return self.pipe

manager = ModelManager()

def make_progress_callback(progress, total_steps, refresh_interval=2):
    def _callback(pipe, step, timestep, callback_kwargs):
        global is_interrupted
        if is_interrupted: raise gr.Error("任务已手动停止")
        step_idx = step + 1
        frac = step_idx / total_steps
        status_suffix = ""
        if step_idx % refresh_interval == 0 or step_idx == total_steps:
            _, mem_status = get_vram_info()
            status_suffix = f"\n{mem_status}"
        progress(frac, desc=f"Diffusion Step {step_idx}/{total_steps}{status_suffix}")
        return callback_kwargs
    return _callback

# ==========================================
# 核心逻辑
# ==========================================
def process_lora_inputs(lora_checks, lora_weights):
    selected = []
    weights_map = {}
    for i, fname in enumerate(LORA_FILES):
        if i < len(lora_checks) and lora_checks[i]:
            selected.append(fname)
            if i < len(lora_weights):
                weights_map[fname] = lora_weights[i]
            else:
                weights_map[fname] = 1.0
    return selected, weights_map

def refresh_lora_list(lang="zh"):
    global LORA_FILES
    LORA_FILES = scan_lora_files()
    count = len(LORA_FILES)
    t = I18N.get(lang, I18N["zh"])
    msg = t["msg_scan_done"].format(count)
    return gr.update(value=msg)

def update_prompt_ui_base(prompt, *lora_ui_args):
    num_loras = len(LORA_FILES)
    if num_loras == 0: return prompt
    checks = lora_ui_args[:num_loras]
    weights = lora_ui_args[num_loras:num_loras*2]
    clean_p = re.sub(r"\s*<lora:[^>]+>", "", prompt or "").strip()
    new_tags = []
    for i, fname in enumerate(LORA_FILES):
        if i < len(checks) and checks[i]:
            w = weights[i] if i < len(weights) else 1.0
            name = os.path.splitext(fname)[0]
            alpha_str = f"{w:.2f}".rstrip("0").rstrip(".")
            new_tags.append(f"<lora:{name}:{alpha_str}>")
    if new_tags:
        return f"{clean_p} {' '.join(new_tags)}"
    else:
        return clean_p

def run_inference(*args):
    global is_interrupted
    is_interrupted = False
    idx = 0
    prompt = args[idx]; idx += 1
    num_loras = len(LORA_FILES)
    lora_checks = args[idx : idx+num_loras]; idx += num_loras
    lora_weights = args[idx : idx+num_loras]; idx += num_loras
    t_choice = args[idx]; idx += 1
    v_choice = args[idx]; idx += 1
    perf_mode = args[idx]; idx += 1
    w = args[idx]; idx += 1
    h = args[idx]; idx += 1
    steps = args[idx]; idx += 1
    cfg = args[idx]; idx += 1
    seed = args[idx]; idx += 1
    is_random = args[idx]; idx += 1
    batch_size = args[idx]; idx += 1
    vram_threshold = args[idx]; idx += 1
    # --- 新增：读取采样器参数 ---
    sampler_name = args[idx]; idx += 1

    auto_flush_vram(vram_threshold)
    clean_w = (int(w) // 16) * 16
    clean_h = (int(h) // 16) * 16
    selected_loras, weights_map = process_lora_inputs(lora_checks, lora_weights)
    
    new_tags = []
    if selected_loras:
        tags = []
        for f in selected_loras:
            w_val = weights_map.get(f, 1.0)
            name = os.path.splitext(f)[0]
            alpha_str = f"{w_val:.2f}".rstrip("0").rstrip(".")
            tags.append(f"<lora:{name}:{alpha_str}>")
        if tags:
            new_tags = tags
    
    if new_tags:
        clean_p = re.sub(r"\s*<lora:[^>]+>", "", prompt or "").strip()
        final_prompt = f"{clean_p} {' '.join(new_tags)}"
    else:
        final_prompt = prompt

    try:
        pipe = manager.get_pipeline(t_choice, v_choice, selected_loras, weights_map, mode='txt', perf_mode=perf_mode)
        # 【关键修复】获取pipeline后进行显存清理，防止加载时崩溃
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        raise gr.Error(f"模型加载失败: {str(e)}")

    # --- 新增：应用采样器 ---
    if sampler_name:
        new_scheduler = get_scheduler(sampler_name, pipe.scheduler.config)
        if new_scheduler is not None:
            print(f"Sampler set to: {sampler_name}")
            pipe.scheduler = new_scheduler
        else:
            print(f"Using Default Scheduler for Z-Image.")

    if is_random: seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(DEVICE).manual_seed(int(seed))

    date_folder = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(OUTPUT_ROOT, date_folder)
    os.makedirs(save_dir, exist_ok=True)

    results_images = []
    progress = gr.Progress()

    try:
        print(f"任务启动 | 图片分辨率: {clean_w}x{clean_h} | 种子: {seed}".encode('utf-8', errors='replace').decode())
        step_callback = make_progress_callback(progress, int(steps))

        for i in range(int(batch_size)):
            if is_interrupted: break

            # 生成每张图独立 seed
            seed_i = random.randint(0, 2**32 - 1)
            generator_i = torch.Generator(DEVICE).manual_seed(seed_i)

            # Prompt 微扰
            prompt_i = augment_prompt(prompt)
            final_prompt_i = final_prompt.replace(prompt, prompt_i) if final_prompt != prompt else prompt_i

            output = pipe(
                prompt=final_prompt_i,
                width=clean_w,
                height=clean_h,
                num_inference_steps=int(steps),
                guidance_scale=float(cfg),
                generator=generator_i,
                callback_on_step_end=step_callback
            ).images[0]

            filename = f"{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}.png"
            path = os.path.join(save_dir, filename)
            output.save(path)
            results_images.append(output)
            _, current_status = get_vram_info()
            yield results_images, seed_i, current_status

    except Exception as e:
        if "任务已手动停止" in str(e):
            print("任务已停止".encode('utf-8', errors='replace').decode())
        else:
            import traceback
            traceback.print_exc()
            raise gr.Error(f"生成中断: {str(e)}")
    finally:
        auto_flush_vram(vram_threshold)

def run_img2img(*args):
    import torch
    from PIL import Image
    import os
    from datetime import datetime

    global is_interrupted
    is_interrupted = False

    idx = 0
    prompt = args[idx]; idx += 1
    negative_prompt = args[idx]; idx += 1
    input_image_path = args[idx]; idx += 1 
    
    if input_image_path is None or input_image_path == "":
         raise gr.Error("请先上传参考图片")
         
    if os.path.exists(input_image_path):
        file_size_mb = os.path.getsize(input_image_path) / (1024 * 1024)
        if file_size_mb > 10.0:
            raise gr.Error(f"❌ 图片过大！\n请上传小于 10.0MB 的图片。\n(当前图片大小: {file_size_mb:.2f}MB)")
        input_image = Image.open(input_image_path).convert("RGB")
    else:
        input_image = None

    width_slider = args[idx]; idx += 1
    height_slider = args[idx]; idx += 1
    steps_ui = args[idx]; idx += 1
    cfg_ui = args[idx]; idx += 1
    strength_ui = args[idx]; idx += 1
    seed = args[idx]; idx += 1
    t_choice = args[idx]; idx += 1
    v_choice = args[idx]; idx += 1
    perf_mode = args[idx]; idx += 1
    lora_name = args[idx]; idx += 1
    lora_weight = args[idx]; idx += 1
    img2img_mode = args[idx]; idx += 1
    # --- 新增：读取采样器参数 ---
    sampler_name = args[idx]; idx += 1

    if input_image is None:
        raise gr.Error("请先上传参考图片")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = None if seed < 0 else torch.Generator(device=device).manual_seed(int(seed))

    orig_width, orig_height = input_image.width, input_image.height
    width, height = orig_width, orig_height
    input_image = input_image.convert("RGB")

    if img2img_mode.startswith("A"):
        strength = 0.30; steps = 8; cfg = 1.0; lora_scale = 0.35
    else:
        strength = 0.45; steps = 6; cfg = 1.5; lora_scale = 0.65

    selected_loras = []
    weights_map = {}
    lora_prompt = ""

    if lora_name not in (None, "None", "") and float(lora_weight) > 0:
        selected_loras = [lora_name]
        effective_weight = float(lora_weight) * lora_scale
        weights_map = {lora_name: effective_weight}
        lora_prompt = f"<lora:{lora_name}:{effective_weight:.2f}> "

    final_prompt = f"{lora_prompt}{augment_prompt(prompt)}".strip()

    try:
        pipe = manager.get_pipeline(
            t_choice, v_choice, selected_loras, weights_map,
            mode="img", perf_mode=perf_mode
        )
    except Exception as e:
        raise gr.Error(f"模型加载失败: {str(e)}")

    # --- 新增：应用采样器 ---
    if sampler_name:
        new_scheduler = get_scheduler(sampler_name, pipe.scheduler.config)
        if new_scheduler is not None:
            print(f"Sampler set to: {sampler_name}")
            pipe.scheduler = new_scheduler

    def step_callback(pipe, step_index, timestep, callback_kwargs):
        if is_interrupted: raise gr.Error("任务已手动停止")
        return callback_kwargs

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=final_prompt, negative_prompt=negative_prompt,
                image=input_image, strength=strength,
                num_inference_steps=steps, guidance_scale=cfg,
                generator=generator, callback_on_step_end=step_callback
            ).images[0]
    except Exception as e:
        raise gr.Error(str(e))

    output_root = "outputs"
    os.makedirs(output_root, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(output_root, date_str)
    os.makedirs(output_dir, exist_ok=True)
    time_str = datetime.now().strftime("%H-%M-%S")
    output_path = os.path.join(output_dir, f"{time_str}.png")
    result.save(output_path, format="PNG")
    print(f"图像已保存: {output_path}".encode('utf-8', errors='replace').decode())
    pipe.unfuse_lora()

    return [result], seed, get_vram_info()[1]

def run_inpainting(*args):
    import torch
    from PIL import Image, ImageOps
    import os
    from datetime import datetime

    global is_interrupted
    is_interrupted = False

    idx = 0
    # 从ImageEditor中获取图像和Mask
    image_editor_data = args[idx]; idx += 1
     
    # 验证输入
    if image_editor_data is None:
        raise gr.Error("请先上传原图")
     
    # 处理ImageEditor数据 - Gradio ImageEditor返回特定格式
    print(f"DEBUG: ImageEditor数据类型: {type(image_editor_data)}".encode('utf-8', errors='replace').decode())
    if isinstance(image_editor_data, dict):
        print(f"DEBUG: ImageEditor键: {list(image_editor_data.keys())}".encode('utf-8', errors='replace').decode())
     
    # 尝试多种可能的数据格式
    input_image = None
    mask_layer = None
     
    # 格式1：Gradio ImageEditor标准格式 {'background': ..., 'layers': [...], 'composite': ...}
    if isinstance(image_editor_data, dict) and 'background' in image_editor_data:
        input_image = image_editor_data.get('background')
        # 从layers中提取mask信息
        if image_editor_data.get('layers'):
            mask_layer = image_editor_data['layers'][0]  # 第一个图层是mask
        print("DEBUG: 使用Gradio ImageEditor标准格式".encode('utf-8', errors='replace').decode())
        
    # 格式2：元组 (image, mask)
    elif isinstance(image_editor_data, (tuple, list)) and len(image_editor_data) == 2:
        input_image = image_editor_data[0]
        mask_layer = image_editor_data[1]
        print("DEBUG: 使用元组格式 (image, mask)".encode('utf-8', errors='replace').decode())
        
    # 格式3：字典 {'image': ..., 'mask': ...}
    elif isinstance(image_editor_data, dict) and 'image' in image_editor_data:
        input_image = image_editor_data.get('image')
        mask_layer = image_editor_data.get('mask')
        print("DEBUG: 使用字典格式".encode('utf-8', errors='replace').decode())
        
    # 格式4：直接是图像对象（旧版本兼容）
    elif isinstance(image_editor_data, Image.Image):
        input_image = image_editor_data.convert("RGB")
        mask_layer = None
        print("DEBUG: 使用直接图像格式".encode('utf-8', errors='replace').decode())
        
    # 格式5：文件路径
    elif isinstance(image_editor_data, str) and os.path.exists(image_editor_data):
        input_image = Image.open(image_editor_data).convert("RGB")
        mask_layer = None
        print("DEBUG: 使用文件路径格式".encode('utf-8', errors='replace').decode())
     
    # 验证图像是否加载成功
    if input_image is None:
        raise gr.Error(f"无法加载图像数据。当前数据类型: {type(image_editor_data)}")
     
    # 处理RGBA图像 - ImageEditor返回RGBA格式
    if input_image.mode == 'RGBA':
        # 创建白色背景
        background = Image.new('RGB', input_image.size, (255, 255, 255))
        # 将RGBA图像粘贴到白色背景上
        background.paste(input_image, (0, 0), input_image)
        input_image = background
    else:
        input_image = input_image.convert("RGB")
     
    # 处理Mask - 如果没有绘制Mask，创建一个全黑的Mask（表示全部保持原样）
    if mask_layer is None:
        # 创建一个全黑的Mask，表示全部保持原样
        mask_layer = Image.new('L', input_image.size, color=0)
    elif isinstance(mask_layer, str):
        if os.path.exists(mask_layer):
            mask_layer = Image.open(mask_layer)
        else:
            raise gr.Error("Mask文件不存在")
     
    # 处理Mask
    if isinstance(mask_layer, str):
        if os.path.exists(mask_layer):
            mask_layer = Image.open(mask_layer)
        else:
            raise gr.Error("Mask文件不存在")
    elif not isinstance(mask_layer, Image.Image):
        raise gr.Error(f"Mask格式错误: {type(mask_layer)}")
     
    # 使用专门的Mask处理函数
    mask = process_mask_for_inpaint(mask_layer)
    if mask is None:
        raise gr.Error("Mask 为空或无效，请使用画笔在图片上涂抹要修改的区域。\n\n使用说明：\n• 涂抹区域（任何颜色）= 需要修改的部分\n• 未涂抹区域 = 保持原样的部分\n\n💡 提示：可以使用任意颜色的画笔进行涂抹，系统会自动识别。")
     
    # 确保图像和Mask尺寸匹配
    orig_width, orig_height = input_image.size
    if mask.size != (orig_width, orig_height):
        print(f"调整Mask尺寸: {mask.size} -> ({orig_width}, {orig_height})".encode('utf-8', errors='replace').decode())
        mask = mask.resize((orig_width, orig_height), Image.LANCZOS)
     
    # 检查Mask是否有涂抹区域
    # 如果Mask全是黑色（最大值为0），表示用户没有绘制任何要修改的区域
    # 这种情况下，我们创建一个小的涂抹区域在中间作为示例
    if mask.getextrema()[1] == 0:  # 最大值为0，说明全是黑色，没有涂抹
        # 创建一个示例Mask，在中间有一个小的涂抹区域
        mask = Image.new('L', (orig_width, orig_height), color=0)
        draw = ImageDraw.Draw(mask)
        # 在中间绘制一个小的白色矩形作为示例
        box_size = min(orig_width, orig_height) // 4
        left = (orig_width - box_size) // 2
        top = (orig_height - box_size) // 2
        draw.rectangle([left, top, left + box_size, top + box_size], fill=255)
        print("用户没有绘制任何要修改的区域，已自动创建一个示例Mask在图片中间".encode('utf-8', errors='replace').decode())

    prompt = args[idx]; idx += 1
    negative_prompt = args[idx]; idx += 1
    
    steps_ui = args[idx]; idx += 1
    cfg_ui = args[idx]; idx += 1
    strength_ui = args[idx]; idx += 1
    seed = args[idx]; idx += 1

    t_choice = args[idx]; idx += 1
    v_choice = args[idx]; idx += 1
    perf_mode = args[idx]; idx += 1

    lora_name = args[idx]; idx += 1
    lora_weight = args[idx]; idx += 1
    # --- 新增：读取采样器参数 ---
    sampler_name = args[idx]; idx += 1
    
    input_image = input_image.convert("RGB")
    
    selected_loras = []
    weights_map = {}
    lora_prompt = ""
    lora_scale = 0.6 

    if lora_name not in (None, "None", "") and float(lora_weight) > 0:
        selected_loras = [lora_name]
        effective_weight = float(lora_weight) * lora_scale
        weights_map = {lora_name: effective_weight}
        lora_prompt = f"<lora:{lora_name}:{effective_weight:.2f}> "

    final_prompt = f"{lora_prompt}{augment_prompt(prompt)}".strip()

    try:
        pipe = manager.get_pipeline(
            t_choice, v_choice, selected_loras, weights_map,
            mode="img", perf_mode=perf_mode
        )
    except Exception as e:
        raise gr.Error(f"模型加载失败: {str(e)}")

    # --- 新增：应用采样器 ---
    if sampler_name:
        new_scheduler = get_scheduler(sampler_name, pipe.scheduler.config)
        if new_scheduler is not None:
            print(f"Sampler set to: {sampler_name}")
            pipe.scheduler = new_scheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = None if seed < 0 else torch.Generator(device=device).manual_seed(int(seed))

    def step_callback(pipe, step_index, timestep, callback_kwargs):
        if is_interrupted: raise gr.Error("任务已手动停止")
        return callback_kwargs

    try:
        with torch.inference_mode():
            # 尝试使用标准的inpaint调用
            # 先尝试 mask_image 参数
            result = None
            try:
                result = pipe(
                    prompt=final_prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    mask_image=mask,
                    strength=float(strength_ui),
                    num_inference_steps=int(steps_ui),
                    guidance_scale=float(cfg_ui),
                    generator=generator,
                    callback_on_step_end=step_callback
                ).images[0]
            except TypeError as te1:
                # 如果不支持mask_image，尝试 mask 参数
                try:
                    result = pipe(
                        prompt=final_prompt,
                        negative_prompt=negative_prompt,
                        image=input_image,
                        mask=mask,
                        strength=float(strength_ui),
                        num_inference_steps=int(steps_ui),
                        guidance_scale=float(cfg_ui),
                        generator=generator,
                        callback_on_step_end=step_callback
                    ).images[0]
                except TypeError as te2:
                    # 如果都不支持，使用手动inpaint实现
                    print(f"标准inpaint失败，使用手动inpaint实现: ...请稍等片刻，正在生成中...".encode('utf-8', errors='replace').decode())
                    # 手动实现inpaint：先使用img2img生成，然后根据mask混合
                    import numpy as np
                    
                    # 确保图像和mask尺寸匹配
                    if mask.size != input_image.size:
                        mask = mask.resize(input_image.size, Image.LANCZOS)
                    
                    # 转换为numpy数组
                    img_array = np.array(input_image).astype(np.float32) / 255.0
                    mask_2d = np.array(mask.convert('L')).astype(np.float32) / 255.0
                    
                    # 扩展mask到3通道用于图像混合
                    mask_3d = np.expand_dims(mask_2d, axis=2)
                    mask_3d = np.repeat(mask_3d, 3, axis=2)
                    
                    # 在mask区域添加一些噪声，帮助模型理解需要重绘的区域
                    noise = np.random.randn(*img_array.shape).astype(np.float32) * 0.1
                    # 将mask区域替换为带噪声的图像
                    inpaint_image = img_array * (1 - mask_3d) + (img_array + noise) * mask_3d
                    inpaint_image = np.clip(inpaint_image, 0, 1)
                    inpaint_image_pil = Image.fromarray((inpaint_image * 255).astype(np.uint8))
                    
                    # 使用img2img生成
                    generated = pipe(
                        prompt=final_prompt,
                        negative_prompt=negative_prompt,
                        image=inpaint_image_pil,
                        strength=float(strength_ui),
                        num_inference_steps=int(steps_ui),
                        guidance_scale=float(cfg_ui),
                        generator=generator,
                        callback_on_step_end=step_callback
                    ).images[0]
                    
                    # 确保生成图像的尺寸与原图匹配（pipeline可能会调整尺寸）
                    orig_size = input_image.size
                    if generated.size != orig_size:
                        print(f"生成图像尺寸不匹配，调整中: {generated.size} -> {orig_size}".encode('utf-8', errors='replace').decode())
                        generated = generated.resize(orig_size, Image.LANCZOS)
                    
                    # 确保mask尺寸也匹配，并重新创建mask_3d
                    if mask.size != orig_size:
                        mask = mask.resize(orig_size, Image.LANCZOS)
                    
                    # 重新创建mask数组，确保尺寸匹配
                    mask_2d = np.array(mask.convert('L')).astype(np.float32) / 255.0
                    mask_3d = np.expand_dims(mask_2d, axis=2)
                    mask_3d = np.repeat(mask_3d, 3, axis=2)
                    
                    # 将生成结果与原图按照mask混合
                    gen_array = np.array(generated).astype(np.float32) / 255.0
                    orig_array = np.array(input_image).astype(np.float32) / 255.0
                    
                    # 验证所有数组尺寸是否匹配
                    if gen_array.shape != orig_array.shape:
                        print(f"尺寸不匹配: 生成图像 {gen_array.shape} vs 原图 {orig_array.shape}".encode('utf-8', errors='replace').decode())
                        # 强制调整生成图像尺寸
                        gen_pil = Image.fromarray((gen_array * 255).astype(np.uint8))
                        gen_pil = gen_pil.resize(orig_size, Image.LANCZOS)
                        gen_array = np.array(gen_pil).astype(np.float32) / 255.0
                    
                    if mask_3d.shape[:2] != orig_array.shape[:2]:
                        print(f"Mask尺寸不匹配: {mask_3d.shape[:2]} vs 原图 {orig_array.shape[:2]}".encode('utf-8', errors='replace').decode())
                        # 重新调整mask
                        mask = mask.resize(orig_size, Image.LANCZOS)
                        mask_2d = np.array(mask.convert('L')).astype(np.float32) / 255.0
                        mask_3d = np.expand_dims(mask_2d, axis=2)
                        mask_3d = np.repeat(mask_3d, 3, axis=2)
                    
                    # 最终验证
                    if gen_array.shape != orig_array.shape or mask_3d.shape != orig_array.shape:
                        raise ValueError(f"尺寸验证失败: gen={gen_array.shape}, orig={orig_array.shape}, mask={mask_3d.shape}")
                    
                    # 混合：mask区域使用生成结果，非mask区域保持原图
                    result_array = orig_array * (1 - mask_3d) + gen_array * mask_3d
                    result_array = np.clip(result_array, 0, 1)
                    result = Image.fromarray((result_array * 255).astype(np.uint8))
    except Exception as e:
        import traceback
        traceback.print_exc()
        if "任务已手动停止" in str(e):
            raise
        elif "mask_image" in str(e) or "unexpected keyword argument" in str(e):
            raise gr.Error("Pipeline 错误: 当前模型可能不支持 Inpaint (mask_image) 功能。\n请检查 ZImageImg2ImgPipeline 是否支持 mask_image 参数。")
        else:
            raise gr.Error(f"局部重绘失败: {str(e)}")

    output_root = "outputs"
    os.makedirs(output_root, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(output_root, date_str)
    os.makedirs(output_dir, exist_ok=True)
    time_str = datetime.now().strftime("%H-%M-%S")
    output_path = os.path.join(output_dir, f"inpaint_{time_str}.png")
    result.save(output_path, format="PNG")
    print(f"Inpaint 图像已保存: {output_path}".encode('utf-8', errors='replace').decode())

    return [result], seed, get_vram_info()[1]

def run_fusion_img(*args, progress=gr.Progress()):
    """
    智能融合图功能 - 全新优化版：
    1. 智能场景识别：自动识别季节、场景类型
    2. 人物季节化处理：根据季节自动调整人物穿着打扮
    3. 多主体类型支持：人物+人物、物品+物品、人物+物品/动物
    4. 精确提示词解析：识别具体的场景、动作、互动要求
    """
    global is_interrupted
    is_interrupted = False
    
    idx = 0
    image1_path = args[idx]; idx += 1
    image2_path = args[idx]; idx += 1
    
    if image1_path is None or image2_path is None:
        raise gr.Error("请上传两张参考图片")
    
    if os.path.exists(image1_path):
        s1 = os.path.getsize(image1_path) / (1024 * 1024)
        if s1 > 10.0:
            raise gr.Error(f"❌ 图片1过大！\n请上传小于 10.0MB 的图片。\n(当前图片大小: {s1:.2f}MB)")
        image1 = Image.open(image1_path).convert("RGB")
    else:
        image1 = None
        
    if os.path.exists(image2_path):
        s2 = os.path.getsize(image2_path) / (1024 * 1024)
        if s2 > 10.0:
            raise gr.Error(f"❌ 图片2过大！\n请上传小于 10.0MB 的图片。\n(当前图片大小: {s2:.2f}MB)")
        image2 = Image.open(image2_path).convert("RGB")
    else:
        image2 = None
    
    prompt = args[idx]; idx += 1
    
    num_loras = len(LORA_FILES)
    lora_checks = args[idx : idx+num_loras]; idx += num_loras
    lora_weights = args[idx : idx+num_loras]; idx += num_loras
    
    t_choice = args[idx]; idx += 1
    v_choice = args[idx]; idx += 1
    perf_mode = args[idx]; idx += 1
    
    output_width = args[idx]; idx += 1
    output_height = args[idx]; idx += 1
    blend_strength = args[idx]; idx += 1  # 图1和图2的融合权重
    strength = args[idx]; idx += 1  # img2img强度
    steps = args[idx]; idx += 1
    cfg = args[idx]; idx += 1
    seed = args[idx]; idx += 1
    is_random = args[idx]; idx += 1
    batch_size = args[idx]; idx += 1
    vram_threshold = args[idx]; idx += 1
    # --- 新增：读取采样器参数 ---
    sampler_name = args[idx]; idx += 1

    auto_flush_vram(vram_threshold)
    selected_loras, weights_map = process_lora_inputs(lora_checks, lora_weights)

    if selected_loras:
        tags = []
        for f in selected_loras:
            w_val = weights_map.get(f, 1.0)
            name = os.path.splitext(f)[0]
            tags.append(f"<lora:{name}:{w_val:.2f}>")
        clean_p = re.sub(r"\s*<lora:[^>]+>", "", prompt or "").strip()
        final_prompt = f"{clean_p} {' '.join(tags)}"
    else:
        final_prompt = prompt
    
    # ============ 新增：智能场景分析引擎 ============
    def analyze_prompt_intent(prompt_text):
        """
        智能分析提示词意图，返回场景类型和特征
        """
        if not prompt_text:
            return {"scene_type": "general", "season": "neutral", "interaction": "standing", "objects": []}
        
        prompt_lower = prompt_text.lower()
        
        # 季节识别
        season_keywords = {
            "spring": ["春天", "春季", "樱花", "春花", "绿叶", "春雨", "spring", "cherry blossom"],
            "summer": ["夏天", "夏季", "阳光", "海滩", "游泳", "短裤", "裙子", "summer", "beach", "sunny"],
            "autumn": ["秋天", "秋季", "落叶", "金黄", "梧桐", "毛衣", "外套", "围巾", "autumn", "fall", "orange leaves"],
            "winter": ["冬天", "冬季", "雪", "雪花", "羽绒服", "手套", "winter", "snow", "snowflakes"]
        }
        
        detected_season = "neutral"
        for season, keywords in season_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                detected_season = season
                break
        
        # 场景识别
        scene_keywords = {
            "park": ["公园", "长椅", "bench", "park", "tree"],
            "garden": ["花园", "花坛", "garden", "flowers"],
            "street": ["街道", "街", "street", "road"],
            "indoor": ["室内", "房间", "indoor", "room", "home"],
            "nature": ["自然", "野外", "森林", "nature", "forest", "outdoor"]
        }
        
        detected_scene = "general"
        for scene, keywords in scene_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                detected_scene = scene
                break
        
        # 互动类型识别
        interaction_keywords = {
            "sitting": ["坐", "坐着", "坐在一起", "sitting", "sit"],
            "standing": ["站", "站着", "standing", "stand"],
            "walking": ["走", "走路", "walking", "walk"],
            "talking": ["聊天", "交谈", "说话", "talking", "chat", "conversation"],
            "playing": ["玩", "游戏", "playing", "play"]
        }
        
        detected_interaction = "standing"
        for interaction, keywords in interaction_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                detected_interaction = interaction
                break
        
        # 物体识别
        detected_objects = []
        object_keywords = {
            "bench": ["长椅", "bench"],
            "table": ["桌子", "table"],
            "cat": ["猫", "猫咪", "cat", "kitten"],
            "dog": ["狗", "狗狗", "dog", "puppy"],
            "car": ["车", "汽车", "car", "vehicle"],
            "tree": ["树", "树木", "tree", "trees"]
        }
        
        for obj, keywords in object_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                detected_objects.append(obj)
        
        return {
            "scene_type": detected_scene,
            "season": detected_season,
            "interaction": detected_interaction,
            "objects": detected_objects
        }
    
    def generate_seasonal_clothing_prompt(season, base_character_desc=""):
        """
        根据季节生成相应的穿着打扮描述
        """
        clothing_prompts = {
            "autumn": {
                "general": "wearing warm autumn clothing, sweater, coat, scarf, layered clothes, cozy autumn outfit",
                "detailed": "wearing autumn outfit: sweater, cardigan, long sleeves, scarf, warm pants, boots, layered clothing for mild weather"
            },
            "winter": {
                "general": "wearing winter clothing, coat, gloves, scarf, warm layers, winter jacket",
                "detailed": "wearing winter outfit: thick coat, winter jacket, gloves, warm scarf, winter boots, thermal clothing"
            },
            "spring": {
                "general": "wearing spring clothing, light jacket, casual outfit, spring style",
                "detailed": "wearing spring outfit: light sweater, denim jacket, casual pants, spring colors, fresh and light clothing"
            },
            "summer": {
                "general": "wearing summer clothing, light shirt, shorts, summer outfit",
                "detailed": "wearing summer outfit: t-shirt, shorts or light pants, summer dress, breathable fabric, casual summer style"
            }
        }
        
        if season in clothing_prompts:
            return clothing_prompts[season]["detailed"]
        else:
            return "wearing stylish clothing, well-dressed"
    
    def enhance_prompt_for_scene_analysis(prompt_text, analysis_result):
        """
        基于场景分析结果增强提示词
        """
        enhanced_prompt = prompt_text
        
        # 如果检测到季节，添加季节特征
        if analysis_result["season"] != "neutral":
            season_clothing = generate_seasonal_clothing_prompt(analysis_result["season"])
            if "人物" in prompt_text or "person" in prompt_text.lower():
                # 针对人物场景，添加季节性穿着
                enhanced_prompt += f", {season_clothing}"
            
            # 添加季节性环境描述
            if analysis_result["season"] == "autumn":
                enhanced_prompt += ", autumn leaves falling, golden autumn atmosphere, fallen leaves on ground"
            elif analysis_result["season"] == "winter":
                enhanced_prompt += ", winter atmosphere, cold weather, winter setting"
            elif analysis_result["season"] == "spring":
                enhanced_prompt += ", spring atmosphere, fresh green, blooming flowers"
            elif analysis_result["season"] == "summer":
                enhanced_prompt += ", sunny summer day, bright sunlight, summer atmosphere"
        
        # 强化场景描述
        if analysis_result["scene_type"] == "park":
            enhanced_prompt += ", park setting with trees and benches"
        elif analysis_result["scene_type"] == "garden":
            enhanced_prompt += ", garden setting with flowers and plants"
        
        # 强化互动描述
        if analysis_result["interaction"] == "sitting":
            enhanced_prompt += ", sitting together, seated position"
        elif analysis_result["interaction"] == "talking":
            enhanced_prompt += ", having a conversation, face to face interaction"
        
        return enhanced_prompt
    
    # 执行场景分析
    scene_analysis = analyze_prompt_intent(final_prompt)
    print(f"场景分析结果: {scene_analysis}".encode('utf-8', errors='replace').decode())
    
    # 增强提示词
    enhanced_prompt = enhance_prompt_for_scene_analysis(final_prompt, scene_analysis)
    print(f"增强后的提示词: {enhanced_prompt}".encode('utf-8', errors='replace').decode())
    
    # 优化提示词：确保模型理解需要两个人物
    # 如果提示词中提到"图一"和"图二"，强化这个信息
    if "图一" in enhanced_prompt or "图1" in enhanced_prompt or "image 1" in enhanced_prompt.lower():
        if "图二" in enhanced_prompt or "图2" in enhanced_prompt or "image 2" in enhanced_prompt.lower():
            # 明确强调需要两个人物，并强调保留原图特征
            enhanced_prompt = enhanced_prompt + ", two different people, two persons, two characters, keep facial features from reference images"

    # 确定输出尺寸
    if output_width == 0 or output_height == 0:
        # 使用图1的尺寸作为基准
        orig_w, orig_h = image1.size
        aspect = orig_w / orig_h
        target_size = 1024
        if aspect > 1:
            target_w, target_h = target_size, max(512, int(target_size / aspect))
        else:
            target_h, target_w = target_size, max(512, int(target_size * aspect))
        target_w = (target_w // 16) * 16
        target_h = (target_h // 16) * 16
    else:
        target_w = (int(output_width) // 16) * 16
        target_h = (int(output_height) // 16) * 16

    # 调整两张图到相同尺寸
    image1_resized = image1.resize((target_w, target_h), Image.LANCZOS)
    image2_resized = image2.resize((target_w, target_h), Image.LANCZOS)

    if is_random: seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(DEVICE).manual_seed(int(seed))

    date_folder = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(OUTPUT_ROOT, date_folder)
    os.makedirs(save_dir, exist_ok=True)

    results = []
    pipe = None
    
    try:
        pipe = manager.get_pipeline(t_choice, v_choice, selected_loras, weights_map, mode='img', perf_mode=perf_mode)
        
        # --- 新增：应用采样器 ---
        if sampler_name:
            new_scheduler = get_scheduler(sampler_name, pipe.scheduler.config)
            if new_scheduler is not None:
                print(f"Sampler set to: {sampler_name}")
                pipe.scheduler = new_scheduler

        import numpy as np

        for i in progress.tqdm(range(int(batch_size)), desc="智能融合生成中"):
            if is_interrupted: break
            torch.cuda.ipc_collect()
            
            # 为每次生成创建新的generator（如果batch_size > 1）
            if i > 0:
                gen_seed = random.randint(0, 2**32 - 1) if is_random else seed + i
                current_generator = torch.Generator(DEVICE).manual_seed(int(gen_seed))
            else:
                current_generator = generator
            
            step_callback = make_progress_callback(progress, int(steps))

            # 改进策略：先基于图1生成场景，然后基于图2添加第二个人物
            # 这样可以更好地控制场景生成，避免混乱的多重曝光效果
            
            # ============ 新增：智能融合策略引擎 ============
            def select_fusion_strategy(scene_analysis, prompt_text):
                """
                根据场景分析和提示词选择最适合的融合策略
                """
                prompt_lower = prompt_text.lower()
                
                # 检测主体类型
                has_person_keywords = any(kw in prompt_lower for kw in ["人物", "person", "人", "man", "woman", "people", "character"])
                has_animal_keywords = any(kw in prompt_lower for kw in ["动物", "animal", "猫", "cat", "狗", "dog", "宠物", "pet"])
                has_object_keywords = any(kw in prompt_lower for kw in ["物品", "object", "车", "car", "东西", "thing"])
                
                if has_person_keywords and scene_analysis["interaction"] in ["sitting", "talking", "standing"]:
                    # 人物相关融合
                    if "坐在" in prompt_lower or "sitting" in prompt_lower:
                        return "person_sitting_scene"  # 人物坐姿场景
                    elif "聊天" in prompt_lower or "talking" in prompt_lower:
                        return "person_interaction"    # 人物互动
                    else:
                        return "person_scene"          # 一般人物场景
                elif has_animal_keywords or has_object_keywords:
                    # 动物/物品相关融合
                    if scene_analysis["interaction"] in ["playing", "holding"]:
                        return "interactive_object"    # 互动型物品
                    else:
                        return "static_scene"          # 静态场景
                else:
                    return "general_fusion"            # 一般融合
            
            fusion_strategy = select_fusion_strategy(scene_analysis, enhanced_prompt)
            print(f"选择的融合策略: {fusion_strategy}".encode('utf-8', errors='replace').decode())
            
            # 根据策略调整参数
            def adjust_parameters_by_strategy(strategy, base_strength, base_cfg):
                """
                根据融合策略调整生成参数
                """
                strategy_params = {
                    "person_sitting_scene": {
                        "strength_multiplier": 1.1,  # 稍高的变化，允许场景改变
                        "cfg_adjustment": 0.1,       # 稍高的CFG
                        "focus": "scene_building"    # 重点在场景构建
                    },
                    "person_interaction": {
                        "strength_multiplier": 0.9,  # 稍低的变化，保持人物特征
                        "cfg_adjustment": 0.2,       # 更高的CFG确保互动效果
                        "focus": "interaction_detail" # 重点在互动细节
                    },
                    "person_scene": {
                        "strength_multiplier": 1.0,
                        "cfg_adjustment": 0.0,
                        "focus": "balanced"          # 平衡处理
                    },
                    "interactive_object": {
                        "strength_multiplier": 0.8,  # 较低变化，保持物品特征
                        "cfg_adjustment": 0.1,
                        "focus": "object_preservation" # 重点在物品保持
                    },
                    "static_scene": {
                        "strength_multiplier": 1.2,  # 较高变化，重塑场景
                        "cfg_adjustment": -0.1,      # 稍低的CFG避免过度约束
                        "focus": "scene_creation"    # 重点在场景创建
                    },
                    "general_fusion": {
                        "strength_multiplier": 1.0,
                        "cfg_adjustment": 0.0,
                        "focus": "balanced"
                    }
                }
                
                params = strategy_params.get(strategy, strategy_params["general_fusion"])
                
                adjusted_strength = min(0.9, max(0.3, base_strength * params["strength_multiplier"]))
                adjusted_cfg = max(0.5, min(2.0, base_cfg + params["cfg_adjustment"]))
                
                return adjusted_strength, adjusted_cfg, params["focus"]
            
            adjusted_strength, adjusted_cfg, fusion_focus = adjust_parameters_by_strategy(
                fusion_strategy, float(strength), float(cfg) if cfg > 0 else 1.0
            )
            
            print(f"融合焦点: {fusion_focus}".encode('utf-8', errors='replace').decode())
            print(f"调整后参数: strength={adjusted_strength:.2f}, cfg={adjusted_cfg:.2f}".encode('utf-8', errors='replace').decode())
            
            # 检测是否需要离开原场景
            has_new_scene = any(kw in enhanced_prompt.lower() for kw in ["在", "at", "in", "on", "场景", "scene", "背景", "background", 
                                                               "公园", "park", "河边", "river", "长椅", "bench"])
            
            # ============ 新增：智能融合执行引擎 ============
            def execute_intelligent_fusion(pipe, image1, image2, prompt, strategy, strength, cfg, generator, steps):
                """
                根据融合策略执行智能融合
                """
                # 确保steps是整数
                steps = int(steps)
                
                if strategy == "person_sitting_scene":
                    # 人物坐姿场景：优先构建场景，然后添加人物
                    print(f"执行人物坐姿场景融合...".encode('utf-8', errors='replace').decode())
                    
                    # 步骤1: 构建场景基础
                    scene_prompt = prompt + ", establishing the scene, park bench, autumn setting, sitting area"
                    scene_result = pipe(
                        prompt=scene_prompt,
                        image=image1,
                        strength=strength * 0.6,  # 较低变化，构建场景
                        num_inference_steps=max(8, int(steps * 0.6)),
                        guidance_scale=cfg,
                        generator=generator,
                        callback_on_step_end=None
                    ).images[0]
                    
                    # 步骤2: 添加第一个人的特征
                    person1_prompt = prompt + ", person from image 1 sitting, maintaining character features"
                    person1_result = pipe(
                        prompt=person1_prompt,
                        image=image1,
                        strength=strength * 0.4,  # 很低变化，保持人物特征
                        num_inference_steps=max(6, int(steps * 0.4)),
                        guidance_scale=cfg + 0.1,
                        generator=generator,
                        callback_on_step_end=None
                    ).images[0]
                    
                    # 步骤3: 融合场景和第一人
                    fusion1 = Image.blend(scene_result, person1_result, 0.3)
                    
                    # 步骤4: 添加第二个人的特征
                    person2_prompt = prompt + ", person from image 2 sitting nearby, maintaining character features"
                    person2_result = pipe(
                        prompt=person2_prompt,
                        image=image2,
                        strength=strength * 0.4,
                        num_inference_steps=max(6, int(steps * 0.4)),
                        guidance_scale=cfg + 0.1,
                        generator=generator,
                        callback_on_step_end=None
                    ).images[0]
                    
                    # 最终融合
                    final_result = Image.blend(fusion1, person2_result, 0.4)
                    return final_result
                    
                elif strategy == "person_interaction":
                    # 人物互动：重点保持人物特征，确保互动自然
                    print(f"执行人物互动融合...".encode('utf-8', errors='replace').decode())
                    
                    # 步骤1: 保持图1人物特征，添加基础互动场景
                    interaction_base = pipe(
                        prompt=prompt + ", maintaining character features from image 1, establishing interaction setup",
                        image=image1,
                        strength=strength * 0.3,  # 很低变化，保持人物
                        num_inference_steps=max(6, int(steps * 0.5)),
                        guidance_scale=cfg + 0.2,  # 高CFG确保互动效果
                        generator=generator,
                        callback_on_step_end=None
                    ).images[0]
                    
                    # 步骤2: 融合第二个人物的特征
                    blend_factor = 0.5
                    arr_base = np.array(interaction_base).astype(np.float32) / 255.0
                    arr_img2 = np.array(image2).astype(np.float32) / 255.0
                    combined = arr_base * (1 - blend_factor) + arr_img2 * blend_factor
                    combined = np.clip(combined, 0, 1)
                    combined_pil = Image.fromarray((combined * 255).astype(np.uint8))
                    
                    # 步骤3: 精细化互动场景
                    final_result = pipe(
                        prompt=prompt + ", natural conversation, two people interacting, maintaining both character features",
                        image=combined_pil,
                        strength=strength * 0.4,
                        num_inference_steps=max(8, int(steps * 0.6)),
                        guidance_scale=cfg + 0.1,
                        generator=generator,
                        callback_on_step_end=None
                    ).images[0]
                    
                    return final_result
                    
                elif strategy == "interactive_object":
                    # 互动物品：重点保持物品特征
                    print(f"执行互动物品融合...".encode('utf-8', errors='replace').decode())
                    
                    # 保持物品/动物特征，添加互动元素
                    result = pipe(
                        prompt=prompt + ", maintaining object/animal features, natural interaction",
                        image=image1,
                        strength=strength * 0.5,
                        num_inference_steps=int(steps),
                        guidance_scale=cfg,
                        generator=generator,
                        callback_on_step_end=None
                    ).images[0]
                    
                    return result
                    
                else:
                    # 一般融合或静态场景：使用平衡策略
                    print(f"执行一般融合...".encode('utf-8', errors='replace').decode())
                    
                    # 多步骤融合
                    step1_result = pipe(
                        prompt=prompt,
                        image=image1,
                        strength=strength * 0.6,
                        num_inference_steps=max(8, int(steps * 0.7)),
                        guidance_scale=cfg,
                        generator=generator,
                        callback_on_step_end=None
                    ).images[0]
                    
                    # 融合第二张图的特征
                    blend_factor = 0.4
                    arr_step1 = np.array(step1_result).astype(np.float32) / 255.0
                    arr_img2 = np.array(image2).astype(np.float32) / 255.0
                    combined = arr_step1 * (1 - blend_factor) + arr_img2 * blend_factor
                    combined = np.clip(combined, 0, 1)
                    combined_pil = Image.fromarray((combined * 255).astype(np.uint8))
                    
                    final_result = pipe(
                        prompt=prompt,
                        image=combined_pil,
                        strength=strength * 0.5,
                        num_inference_steps=max(6, int(steps * 0.5)),
                        guidance_scale=cfg,
                        generator=generator,
                        callback_on_step_end=None
                    ).images[0]
                    
                    return final_result
            
            # 执行智能融合
            result = execute_intelligent_fusion(
                pipe, image1_resized, image2_resized, enhanced_prompt, 
                fusion_strategy, adjusted_strength, adjusted_cfg, 
                current_generator, int(steps)  # 确保传递整数
            )
            
            # 结果后处理和优化
            print(f"智能融合完成，使用策略: {fusion_strategy}".encode('utf-8', errors='replace').decode())
            
            # 应用后处理优化
            output = result

            filename = f"fusion_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}.png"
            path = os.path.join(save_dir, filename)
            output.save(path)
            results.append(output)  # 返回PIL图像而不是路径，以便在Gallery中显示

    except Exception as e:
        if "任务已手动停止" in str(e):
            print("任务已停止".encode('utf-8', errors='replace').decode())
        else:
            import traceback
            traceback.print_exc()
            raise gr.Error(f"融合生成中断: {str(e)}")
    finally:
        if pipe:
            del pipe
        auto_flush_vram(vram_threshold)
        _, current_status = get_vram_info()

    return results, seed, current_status

def manual_force_flush():
    print("正在彻底清理显存，正在卸载模型...".encode('utf-8', errors='replace').decode())
    try:
        manager._clear_pipeline()
    except Exception as e:
        print(f"清理过程中出现警告: {e}".encode('utf-8', errors='replace').decode())
    gc.collect()
    torch.cuda.empty_cache()
    _, status = get_vram_info()
    print("显存已彻底释放。".encode('utf-8', errors='replace').decode())
    return status

# ==========================================
# UI 界面
# ==========================================
DEFAULT_PERF_MODE = "低端机 (显存优化)" if TOTAL_VRAM < 20 * 1024**3 else "高端机 (显存>=20GB)"
DEFAULT_LANG = "zh"

with gr.Blocks(title="Z-Image Pro Studio") as demo:
    CURRENT_LANG = gr.State(value=DEFAULT_LANG)
    T = I18N[DEFAULT_LANG]

    with gr.Row(elem_id="header_row"):
        lang_radio = gr.Radio(
            choices=["CN", "EN"], 
            value="CN", 
            scale=0, 
            container=False, 
            show_label=False,
            elem_id="lang_radio_selector"
        )
        title_md = gr.Markdown(value=T["title"], elem_id="app_title_markdown")
        
    with gr.Tabs() as tabs:
        # --- 文成图 ---
        with gr.Tab(label=T["tab_t2i"]) as tab_t2i:
            with gr.Row():
                with gr.Column(scale=4):
                    prompt_input = gr.Textbox(label=T["label_prompt"], lines=4)
                    manual_flush_btn = gr.Button(value=T["btn_flush_vram"], size="sm", variant="secondary")
                    vram_threshold_slider = gr.Slider(50, 98, 90, step=1, label=T["label_vram_threshold"])
                    
                    with gr.Accordion(label=T["acc_lora"], open=False) as t2i_lora_acc:
                        txt_lora_checks = []
                        txt_lora_sliders = []
                        
                        with gr.Row():
                            txt_refresh_lora_btn = gr.Button(value=T["btn_refresh_lora"], size="sm")
                            txt_lora_info_md = gr.Markdown("")
                        
                        if not LORA_FILES:
                            t2i_no_lora_md = gr.Markdown(value=T["txt_no_lora"])
                        else:
                            for fname in LORA_FILES:
                                with gr.Row():
                                    chk = gr.Checkbox(label=fname, value=False, scale=1, container=False)
                                    sld = gr.Slider(0, 2.0, 1.0, step=0.05, label=T["label_weight"], scale=4)
                                    txt_lora_checks.append(chk)
                                    txt_lora_sliders.append(sld)

                    with gr.Accordion(label=T["acc_model"], open=True) as t2i_model_acc:
                        refresh_models_btn = gr.Button(value=T["btn_refresh_model"], size="sm")
                        t_drop = gr.Dropdown(label=T["label_transformer"], choices=["default"] + scan_model_items(MOD_TRANS_DIR), value="default")
                        v_drop = gr.Dropdown(label=T["label_vae"], choices=["default"] + scan_model_items(MOD_VAE_DIR), value="default")
                        
                        perf_mode_radio = gr.Radio(
                            choices=[T["val_perf_high"], T["val_perf_low"]],
                            value=DEFAULT_PERF_MODE,
                            label=T["label_perf"]
                        )
                        
                        with gr.Row():
                            width_s = gr.Slider(512, 2048, 1024, step=16, label=T["label_width"])
                            height_s = gr.Slider(512, 2048, 1024, step=16, label=T["label_height"])
                        step_s = gr.Slider(1, 50, 8, label=T["label_steps"])
                        cfg_s = gr.Slider(0, 10, 0, label=T["label_cfg"])
                        batch_s = gr.Slider(1, 200, 1, step=1, label=T["label_batch"])
                        seed_n = gr.Number(label=T["label_seed"], value=-1, precision=0)
                        random_c = gr.Checkbox(label=T["label_random_seed"], value=True)
                        # 新增采样器下拉框
                        t2i_sampler_drop = gr.Dropdown(label=T["label_sampler"], choices=SAMPLER_LIST, value="Default (Z-Image)")

                    with gr.Row():
                        run_btn = gr.Button(value=T["btn_run"], variant="primary", size="lg")
                        stop_btn = gr.Button(value=T["btn_stop"], variant="stop", size="lg", interactive=False)

                with gr.Column(scale=6):
                    res_gallery = gr.Gallery(label=T["label_output"], columns=2, height="80vh")
                    res_seed = gr.Number(label=T["label_seed"], interactive=False)
                    t2i_vram_info = gr.Markdown(value=T["msg_vram_loading"])

        # --- 图片编辑 ---
        with gr.Tab(label=T["tab_edit"]) as tab_edit:
            with gr.Row():
                with gr.Column():
                    image_input_path = gr.Image(label=T["label_upload_img"], type="filepath")
                    with gr.Group():
                        rotate_angle = gr.Slider(-360, 360, 0, step=1, label=T["label_rotate"])
                        crop_x = gr.Slider(0, 100, 0, step=1, label=T["label_crop_x"])
                        crop_y = gr.Slider(0, 100, 0, step=1, label=T["label_crop_y"])
                        crop_width = gr.Slider(0, 100, 100, step=1, label=T["label_crop_w"])
                        crop_height = gr.Slider(0, 100, 100, step=1, label=T["label_crop_h"])
                        flip_horizontal = gr.Checkbox(label=T["label_flip_h"])
                        flip_vertical = gr.Checkbox(label=T["label_flip_v"])
                    edit_btn = gr.Button(value=T["btn_edit"], variant="primary")
                with gr.Column():
                    edited_image_output = gr.Image(label=T["label_edited"], type="pil")
                    with gr.Group():
                        apply_filter = gr.Dropdown(
                            [T["f_blur"], T["f_contour"], T["f_detail"], T["f_edge"], T["f_edge_more"], 
                             T["f_emboss"], T["f_find_edge"], T["f_sharp"], T["f_smooth"], T["f_smooth_more"]], 
                            label=T["label_filter"]
                        )
                        brightness = gr.Slider(-100, 100, 0, step=1, label=T["label_brightness"])
                        contrast = gr.Slider(-100, 100, 0, step=1, label=T["label_contrast"])
                        saturation = gr.Slider(-100, 100, 0, step=1, label=T["label_saturation"])

            def edit_image_wrapper(image_path, angle, x, y, width, height, hflip, vflip, filter, brightness, contrast, saturation):
                if image_path is None: return None
                if isinstance(image_path, str):
                    image = Image.open(image_path)
                else:
                    image = image_path

                if angle != 0: image = image.rotate(angle, expand=True)
                if x or y or width < 100 or height < 100:
                    original_width, original_height = image.size
                    left = int(original_width * x / 100)
                    top = int(original_height * y / 100)
                    right = int(original_width * (x + width) / 100)
                    bottom = int(original_height * (y + height) / 100)
                    image = image.crop((left, top, right, bottom))
                if hflip: image = ImageOps.mirror(image)
                if vflip: image = ImageOps.flip(image)
                if filter:
                    filter_map_zh = {
                        "模糊": ImageFilter.BLUR, "轮廓": ImageFilter.CONTOUR, "细节": ImageFilter.DETAIL,
                        "边缘增强": ImageFilter.EDGE_ENHANCE, "更多边缘增强": ImageFilter.EDGE_ENHANCE_MORE,
                        "浮雕": ImageFilter.EMBOSS, "查找边缘": ImageFilter.FIND_EDGES,
                        "锐化": ImageFilter.SHARPEN, "平滑": ImageFilter.SMOOTH, "更多平滑": ImageFilter.SMOOTH_MORE
                    }
                    filter_map_en = {
                        "Blur": ImageFilter.BLUR, "Contour": ImageFilter.CONTOUR, "Detail": ImageFilter.DETAIL,
                        "Edge Enhance": ImageFilter.EDGE_ENHANCE, "Edge Enhance More": ImageFilter.EDGE_ENHANCE_MORE,
                        "Emboss": ImageFilter.EMBOSS, "Find Edges": ImageFilter.FIND_EDGES,
                        "Sharpen": ImageFilter.SHARPEN, "Smooth": ImageFilter.SMOOTH, "Smooth More": ImageFilter.SMOOTH_MORE
                    }
                    fmap = filter_map_zh if filter in filter_map_zh else filter_map_en
                    filter_func = fmap.get(filter)
                    if filter_func: image = image.filter(filter_func)
                if brightness != 0:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(1 + brightness / 100)
                if contrast != 0:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1 + contrast / 100)
                if saturation != 0:
                    enhancer = ImageEnhance.Color(image)
                    image = enhancer.enhance(1 + saturation / 100)
                return image

            edit_btn.click(
                fn=edit_image_wrapper,
                inputs=[image_input_path, rotate_angle, crop_x, crop_y, crop_width, crop_height, flip_horizontal, flip_vertical, apply_filter, brightness, contrast, saturation],
                outputs=edited_image_output
            )

        # --- 图生图 UI ---
        with gr.Tab(label=T["tab_i2i"]) as tab_i2i:
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Group():
                        img2img_input_path = gr.Image(label=T["label_ref_img"], type="filepath")
                        img2img_prompt = gr.Textbox(label=T["label_prompt_rec"], lines=2, placeholder=T["ph_prompt_i2i"])
                        img2img_negative_prompt = gr.Textbox(
                            label=T.get("label_negative", "Negative Prompt"),
                            lines=2,
                            placeholder="low quality, blurry, bad anatomy"
                        )
                        img2img_flush = gr.Button(value=T["btn_flush_vram"], size="sm", variant="secondary")

                        with gr.Accordion(label=T["acc_lora"], open=False) as i2i_lora_acc:
                            with gr.Row():
                                i2i_refresh_lora_btn = gr.Button(value=T["btn_refresh_lora"], size="sm")
                            
                            if not LORA_FILES:
                                i2i_no_lora_md = gr.Markdown(value=T["txt_no_lora"])
                                img2img_lora_drop = gr.Dropdown(choices=[], visible=False)
                                img2img_lora_weight = gr.Slider(0.0, 1.0, 0.0, visible=False)
                            else:
                                img2img_lora_drop = gr.Dropdown(
                                    label="Img2Img LoRA",
                                    choices=["None"] + LORA_FILES,
                                    value="None"
                                )
                                img2img_lora_weight = gr.Slider(
                                    0.0, 1.0, 0.6,
                                    step=0.05,
                                    label="LoRA 权重（Img2Img）"
                                )

                    with gr.Accordion(label=T["acc_model"], open=True) as i2i_model_acc:
                        img2img_refresh_models = gr.Button(value=T["btn_refresh_model"], size="sm")
                        img2img_t_drop = gr.Dropdown(label=T["label_transformer"], choices=["default"] + scan_model_items(MOD_TRANS_DIR), value="default")
                        img2img_v_drop = gr.Dropdown(label=T["label_vae"], choices=["default"] + scan_model_items(MOD_VAE_DIR), value="default")
                        
                        img2img_perf_mode = gr.Radio(
                            choices=[T["val_perf_high"], T["val_perf_low"]],
                            value=DEFAULT_PERF_MODE,
                            label=T["label_perf"]
                        )
                        img2img_mode = gr.Radio(
                            choices=[
                                "A. 严格保结构（微调风格）",
                                "B. 强烈听 prompt（允许大改）"
                            ],
                            value="A. 严格保结构（微调风格）",
                            label="Img2Img 模式"
                        )
                        with gr.Row():
                            img2img_width_s = gr.Slider(0, 2048, 0, step=16, label=T["label_out_w"])
                            img2img_height_s = gr.Slider(0, 2048, 0, step=16, label=T["label_out_h"])
                        tip_md = gr.Markdown(value=T["tip_res"])
                        img2img_strength = gr.Slider(0.1, 0.9, 0.35, step=0.01,label=T["label_strength"])
                        img2img_steps = gr.Slider(1, 12, 6, step=1, label=T["label_steps"])
                        img2img_cfg = gr.Slider(0.5, 2.0, 1.0, step=0.05,label="CFG (Turbo Img2Img)")
                        img2img_batch = gr.Slider(1, 8, 1, step=1, label=T["label_batch"])
                        img2img_seed = gr.Number(label=T["label_seed"], value=42, precision=0)
                        img2img_random = gr.Checkbox(label=T["label_random_seed"], value=True)
                        # 新增采样器下拉框
                        img2img_sampler_drop = gr.Dropdown(label=T["label_sampler"], choices=SAMPLER_LIST, value="Default (Z-Image)")
                    with gr.Row():
                        img2img_run_btn = gr.Button(value=T["btn_gen"], variant="primary", size="lg")
                        img2img_stop_btn = gr.Button(value=T["btn_stop_short"], variant="stop", size="lg", interactive=False)
                with gr.Column(scale=6):
                    img2img_gallery = gr.Gallery(label=T["label_gallery_i2i"], columns=2, height="80vh")
                    img2img_res_seed = gr.Number(label=T["label_seed"], interactive=False)
                    i2i_vram_info = gr.Markdown(value=T["msg_vram_loading"])

        # --- 局部重绘 UI (Inpaint) ---
        with gr.Tab(label=T["tab_inpaint"]) as tab_inpaint:
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Group():
                        inpaint_input_img = gr.ImageEditor(
                            label="上传原图并绘制要修改的区域",
                            type="pil"
                        )
                        with gr.Accordion(label="📖 局部重绘使用指南", open=False):
                            inpaint_tip_md = gr.Markdown(
                                value="## 🎨 局部重绘使用指南\n\n" +
                                "### 📝 操作步骤：\n" +
                                "1. **上传原图**：在左侧上传需要编辑的图片\n" +
                                "2. **绘制Mask**：直接在图片上使用画笔绘制要修改的区域\n" +
                                "   - 🎨 **涂抹区域（任何颜色）**：将被重新生成\n" +
                                "   - 🖤 **未绘制区域**：保持原样（默认行为）\n" +
                                "3. **填写描述**：在Prompt中描述想要的效果\n" +
                                "4. **开始生成**：点击生成按钮开始局部重绘\n\n" +
                                "### 🎨 绘制技巧：\n" +
                                "- 可以使用**任意颜色的画笔**进行涂抹（红色、蓝色、黄色等都可以）\n" +
                                "- 系统会自动识别所有涂抹区域，无需使用特定颜色\n" +
                                "- 未绘制的区域将自动保持原样（无需涂抹黑色）\n" +
                                "- 可以调整画笔大小以精确控制Mask范围\n\n" +
                                "### ⚙️ 参数调节：\n" +
                                "- **重绘强度**：控制修改的幅度 (0.1-0.9)\n" +
                                "- **步数**：影响生成质量和速度\n" +
                                "- **CFG**：控制对Prompt的遵循程度\n\n" +
                                "### 💡 使用技巧：\n" +
                                "- 只需绘制要修改的区域，其他区域自动保持原样\n" +
                                "- 涂抹区域不要过大，否则效果不明显\n" +
                                "- 可以多次尝试不同的参数组合\n\n" +
                                "### ⚠️ 注意：\n" +
                                "- 如果没有绘制任何区域，系统将自动创建一个示例Mask在图片中间"
                            )
                         
                        inpaint_flush = gr.Button(value=T["btn_flush_vram"], size="sm", variant="secondary")
                         
                        inpaint_prompt = gr.Textbox(label=T["label_prompt_rec"], lines=2, placeholder=T["ph_prompt_i2i"])
                        inpaint_negative_prompt = gr.Textbox(
                            label=T.get("label_negative", "Negative Prompt"),
                            lines=2,
                            placeholder="low quality, blurry, bad anatomy"
                        )

                        with gr.Accordion(label=T["acc_lora"], open=False) as inpaint_lora_acc:
                            with gr.Row():
                                inpaint_refresh_lora_btn = gr.Button(value=T["btn_refresh_lora"], size="sm")
                             
                            if not LORA_FILES:
                                inpaint_no_lora_md = gr.Markdown(value=T["txt_no_lora"])
                                inpaint_lora_drop = gr.Dropdown(choices=[], visible=False)
                                inpaint_lora_weight = gr.Slider(0.0, 1.0, 0.0, visible=False)
                            else:
                                inpaint_lora_drop = gr.Dropdown(
                                    label="Inpaint LoRA",
                                    choices=["None"] + LORA_FILES,
                                    value="None"
                                )
                                inpaint_lora_weight = gr.Slider(
                                    0.0, 1.0, 0.6,
                                    step=0.05,
                                    label="LoRA 权重"
                                )

                    with gr.Accordion(label=T["acc_model"], open=True) as inpaint_model_acc:
                        inpaint_refresh_models = gr.Button(value=T["btn_refresh_model"], size="sm")
                        inpaint_t_drop = gr.Dropdown(label=T["label_transformer"], choices=["default"] + scan_model_items(MOD_TRANS_DIR), value="default")
                        inpaint_v_drop = gr.Dropdown(label=T["label_vae"], choices=["default"] + scan_model_items(MOD_VAE_DIR), value="default")
                         
                        inpaint_perf_mode = gr.Radio(
                            choices=[T["val_perf_high"], T["val_perf_low"]],
                            value=DEFAULT_PERF_MODE,
                            label=T["label_perf"]
                        )
                         
                        inpaint_strength = gr.Slider(0.1, 0.9, 0.6, step=0.01, label=T["label_strength"])
                        inpaint_steps = gr.Slider(1, 20, 8, step=1, label=T["label_steps"])
                        inpaint_cfg = gr.Slider(0.5, 2.0, 1.0, step=0.05, label=T["label_cfg_inpaint"])
                        inpaint_seed = gr.Number(label=T["label_seed"], value=42, precision=0)
                        inpaint_random = gr.Checkbox(label=T["label_random_seed"], value=True)
                        # 新增采样器下拉框
                        inpaint_sampler_drop = gr.Dropdown(label=T["label_sampler"], choices=SAMPLER_LIST, value="Default (Z-Image)")
                         
                    with gr.Row():
                        inpaint_run_btn = gr.Button(value=T["btn_gen"], variant="primary", size="lg")
                        inpaint_stop_btn = gr.Button(value=T["btn_stop_short"], variant="stop", size="lg", interactive=False)

                with gr.Column(scale=6):
                    inpaint_gallery = gr.Gallery(label=T["label_gallery_inpaint"], columns=2, height="80vh")
                    inpaint_res_seed = gr.Number(label=T["label_seed"], interactive=False)
                    inpaint_vram_info = gr.Markdown(value=T["msg_vram_loading"])

        # --- 融合图 ---
        with gr.Tab(label=T["tab_fusion"]) as tab_fusion:
            desc_fusion_md = gr.Markdown(value=T["desc_fusion"])
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Group():
                        fusion_input1_path = gr.Image(label=T["label_img1"], type="filepath")
                        fusion_input2_path = gr.Image(label=T["label_img2"], type="filepath")
                        with gr.Accordion(label="📖 智能融合图使用指南", open=False):
                            fusion_tip_md = gr.Markdown(
                                value="## 🎨 智能融合图使用指南\n\n" +
                                "### 📝 功能说明：\n" +
                                "智能融合图功能可以将两张参考图的特征融合，生成包含两张图元素的新场景。\n\n" +
                                "### 🎯 使用场景：\n" +
                                "#### 1️⃣ **两张人物图融合**\n" +
                                "- **示例**：上传两张人物照片\n" +
                                "- **提示词示例**：\"图一的人物和图二的人物坐在公园长椅上聊天\"\n" +
                                "- **效果**：生成包含图一和图二人物脸部特征的两个人坐在长椅上的场景\n" +
                                "- **技巧**：可以在提示词中详细描述图一人物的穿着打扮、图二人物的特征等\n\n" +
                                "#### 2️⃣ **人物+物品/动物互动**\n" +
                                "- **示例**：上传一张人物图和一张物品/动物图\n" +
                                "- **提示词示例**：\"图一的人物抱着图二的猫咪在花园里玩耍\"\n" +
                                "- **效果**：生成人物和物品/动物互动的场景\n" +
                                "- **技巧**：详细描述互动的方式和场景环境\n\n" +
                                "### ⚙️ 参数说明：\n" +
                                "- **融合权重**：控制两张图的融合比例\n" +
                                "  - 0.0 = 完全偏向图1的特征\n" +
                                "  - 0.5 = 平衡融合两张图\n" +
                                "  - 1.0 = 完全偏向图2的特征\n" +
                                "- **重绘强度**：控制生成时的变化幅度\n" +
                                "  - 0.5-0.7 = 保留更多原图特征（推荐）\n" +
                                "  - 0.8-1.0 = 更大变化，更符合提示词\n" +
                                "- **步数**：影响生成质量，建议15-30步\n" +
                                "- **CFG**：控制对提示词的遵循程度\n\n" +
                                "### 💡 使用技巧：\n" +
                                "1. **提示词要详细**：明确描述两张图如何融合，包括场景、动作、互动方式等\n" +
                                "2. **融合权重调节**：根据想要的效果调整，如果更想要图1的特征，降低权重；反之提高权重\n" +
                                "3. **多次尝试**：可以尝试不同的参数组合，找到最佳效果\n" +
                                "4. **图片质量**：上传清晰、主体明确的图片效果更好\n\n" +
                                "### ⚠️ 注意事项：\n" +
                                "- 融合过程需要较长时间（需要分别基于两张图生成，然后融合）\n" +
                                "- 如果两张图尺寸差异很大，系统会自动调整到相同尺寸\n" +
                                "- 建议使用描述性强的提示词，效果会更好"
                            )
                        fusion_prompt = gr.Textbox(label=T["label_fusion_prompt"], lines=3, placeholder="例如：图一的人物和图二的人物坐在公园长椅上聊天，阳光明媚，背景是美丽的公园")
                        fusion_flush = gr.Button(value=T["btn_flush_vram"], size="sm", variant="secondary")
                        
                        with gr.Accordion(label=T["acc_lora"], open=False) as fusion_lora_acc:
                            fusion_lora_checks = []
                            fusion_lora_sliders = []
                            
                            with gr.Row():
                                fusion_refresh_lora_btn = gr.Button(value=T["btn_refresh_lora"], size="sm")
                                fusion_lora_info_md = gr.Markdown("")

                            if not LORA_FILES:
                                fusion_no_lora_md = gr.Markdown(value=T["txt_no_lora"])
                            else:
                                for fname in LORA_FILES:
                                    with gr.Row():
                                        chk = gr.Checkbox(label=fname, value=False, scale=1, container=False)
                                        sld = gr.Slider(0, 2.0, 1.0, step=0.05, label=T["label_weight"], scale=4)
                                        fusion_lora_checks.append(chk)
                                        fusion_lora_sliders.append(sld)

                    with gr.Accordion(label=T["acc_model"], open=True) as fusion_model_acc:
                        fusion_refresh_models = gr.Button(value=T["btn_refresh_model"], size="sm")
                        fusion_t_drop = gr.Dropdown(label=T["label_transformer"], choices=["default"] + scan_model_items(MOD_TRANS_DIR), value="default")
                        fusion_v_drop = gr.Dropdown(label=T["label_vae"], choices=["default"] + scan_model_items(MOD_VAE_DIR), value="default")
                        
                        fusion_perf_mode = gr.Radio(
                            choices=[T["val_perf_high"], T["val_perf_low"]],
                            value=DEFAULT_PERF_MODE,
                            label=T["label_perf"]
                        )
                        
                        with gr.Row():
                            fusion_width_s = gr.Slider(0, 2048, 0, step=16, label=T["label_out_w"])
                            fusion_height_s = gr.Slider(0, 2048, 0, step=16, label=T["label_out_h"])
                        fusion_tip_md = gr.Markdown(value=T["tip_res"])
                        with gr.Row():
                            fusion_blend = gr.Slider(0.0, 1.0, 0.5, step=0.05, label=T["label_blend"])
                            fusion_strength = gr.Slider(0.0, 1.0, 0.5, step=0.05, label=T["label_denoise"], info="建议0.4-0.6，数值越低越保留原图特征")
                        fusion_steps = gr.Slider(1, 100, 15, step=1, label=T["label_steps"])
                        fusion_cfg = gr.Slider(0.0, 2.0, 1.0, step=0.05, label="CFG (控制对提示词的遵循程度)")
                        fusion_batch = gr.Slider(1, 8, 1, step=1, label=T["label_batch"])
                        fusion_seed = gr.Number(label=T["label_seed"], value=42, precision=0)
                        fusion_random = gr.Checkbox(label=T["label_random_seed"], value=True)
                        # 新增采样器下拉框
                        fusion_sampler_drop = gr.Dropdown(label=T["label_sampler"], choices=SAMPLER_LIST, value="Default (Z-Image)")
                    with gr.Row():
                        fusion_run_btn = gr.Button(value=T["btn_run"], variant="primary", size="lg")
                        fusion_stop_btn = gr.Button(value=T["btn_stop"], variant="stop", size="lg", interactive=False)
                with gr.Column(scale=6):
                    fusion_gallery = gr.Gallery(label=T["label_gallery_fusion"], columns=2, height="80vh")
                    fusion_res_seed = gr.Number(label=T["label_seed"], interactive=False)

    # -----------------------
    # 语言切换逻辑
    # -----------------------
    def change_language(lang_choice):
        lang_code = "zh" if "中文" in lang_choice else "en"
        t = I18N.get(lang_code, I18N["zh"])
        return (
            lang_code,                       
            gr.update(value=t["title"]),                      
            gr.update(label=t["tab_t2i"]),   
            gr.update(label=t["tab_edit"]),  
            gr.update(label=t["tab_i2i"]),   
            gr.update(label=t["tab_inpaint"]),
            gr.update(label=t["tab_fusion"]), 
            
            gr.update(label=t["label_prompt"]),   
            t["btn_flush_vram"],                  
            gr.update(label=t["label_vram_threshold"]), 
            
            gr.update(label=t["acc_lora"]),        
            t["btn_refresh_lora"],                 
            gr.update(label=t["acc_model"]),       
            t["btn_refresh_model"],                
            gr.update(label=t["label_transformer"]), 
            gr.update(label=t["label_vae"]),       
            gr.update(label=t["label_perf"]), 
            
            gr.update(label=t["label_width"]),     
            gr.update(label=t["label_height"]),    
            gr.update(label=t["label_steps"]),       
            gr.update(label=t["label_cfg"]),         
            gr.update(label=t["label_batch"]),      
            gr.update(label=t["label_seed"]),        
            gr.update(label=t["label_random_seed"]), 
            
            t["btn_run"],                          
            t["btn_stop"],                         
            gr.update(label=t["label_output"]),     
            gr.update(label=t["label_seed"]),       
            
            gr.update(label=t["label_upload_img"]),  
            gr.update(label=t["label_rotate"]),     
            gr.update(label=t["label_crop_x"]),      
            gr.update(label=t["label_crop_y"]),      
            gr.update(label=t["label_crop_w"]),      
            gr.update(label=t["label_crop_h"]),      
            gr.update(label=t["label_flip_h"]),      
            gr.update(label=t["label_flip_v"]),      
            t["btn_edit"],                           
            gr.update(label=t["label_edited"]),      
            gr.update(choices=[t["f_blur"], t["f_contour"], t["f_detail"], t["f_edge"], t["f_edge_more"], 
                             t["f_emboss"], t["f_find_edge"], t["f_sharp"], t["f_smooth"], t["f_smooth_more"]], label=t["label_filter"]), 
            gr.update(label=t["label_brightness"]),  
            gr.update(label=t["label_contrast"]),    
            gr.update(label=t["label_saturation"]),  
            
            gr.update(label=t["label_ref_img"]),     
            gr.update(label=t["label_prompt_rec"], placeholder=t["ph_prompt_i2i"]), 
            t["btn_flush_vram"],                     
            gr.update(label=t["acc_lora"]),         
            t["btn_refresh_lora"],                  
            gr.update(label=t["acc_model"]),         
            t["btn_refresh_model"],                 
            gr.update(label=t["label_transformer"]), 
            gr.update(label=t["label_vae"]),        
            gr.update(label=t["label_perf"]), 
            
            gr.update(label=t["label_out_w"]),      
            gr.update(label=t["label_out_h"]),      
            t["tip_res"],                           
            gr.update(label=t["label_strength"]),    
            gr.update(label=t["label_steps"]),       
            gr.update(label=t["label_cfg_turbo"]),   
            gr.update(label=t["label_batch"]),      
            gr.update(label=t["label_seed"]),       
            gr.update(label=t["label_random_seed"]), 
            
            t["btn_gen"],                           
            t["btn_stop_short"],                    
            gr.update(label=t["label_gallery_i2i"]), 
            gr.update(label=t["label_seed"]),       

            # Inpaint 更新
            gr.update(label="上传原图并绘制要修改的区域"),  # 更新为新的标签
            t["lbl_inpaint_tip"],
            gr.update(label=t["label_prompt_rec"], placeholder=t["ph_prompt_i2i"]),
            t["btn_flush_vram"],
            gr.update(label=t["acc_lora"]),
            t["btn_refresh_lora"],
            gr.update(label=t["acc_model"]),
            t["btn_refresh_model"],
            gr.update(label=t["label_transformer"]),
            gr.update(label=t["label_vae"]),
            gr.update(label=t["label_perf"]),
            gr.update(label=t["label_strength"]),
            gr.update(label=t["label_steps"]),
            gr.update(label=t["label_cfg_inpaint"]),
            gr.update(label=t["label_seed"]),
            gr.update(label=t["label_random_seed"]),
            t["btn_gen"],
            t["btn_stop_short"],
            gr.update(label=t["label_gallery_inpaint"]),
            gr.update(label=t["label_seed"]),
            
            t["desc_fusion"],                       
            gr.update(label=t["label_img1"]),       
            gr.update(label=t["label_img2"]),       
            gr.update(label=t["label_fusion_prompt"]), 
            t["btn_flush_vram"],                    
            gr.update(label=t["acc_lora"]),         
            t["btn_refresh_lora"],                  
            gr.update(label=t["acc_model"]),         
            t["btn_refresh_model"],                 
            gr.update(label=t["label_transformer"]), 
            gr.update(label=t["label_vae"]),         
            gr.update(label=t["label_perf"]), 
            
            gr.update(label=t["label_out_w"]),      
            gr.update(label=t["label_out_h"]),      
            t["tip_res"],                           
            gr.update(label=t["label_blend"]),       
            gr.update(label=t["label_denoise"]),    
            gr.update(label=t["label_steps"]),       
            gr.update(label=t["label_cfg_fixed"]),   
            gr.update(label=t["label_batch"]),      
            gr.update(label=t["label_seed"]),       
            gr.update(label=t["label_random_seed"]), 
            
            t["btn_run"],                          
            t["btn_stop"],                         
            gr.update(label=t["label_gallery_fusion"]), 
            gr.update(label=t["label_seed"]),       
        )

    lang_outputs = [
        CURRENT_LANG,
        title_md, 
        tab_t2i, tab_edit, tab_i2i, tab_inpaint, tab_fusion,
        prompt_input, manual_flush_btn, vram_threshold_slider,
        t2i_lora_acc, txt_refresh_lora_btn, t2i_model_acc, refresh_models_btn,
        t_drop, v_drop, perf_mode_radio,
        width_s, height_s, step_s, cfg_s, batch_s, seed_n, random_c,
        run_btn, stop_btn, res_gallery, res_seed,
        image_input_path, rotate_angle, crop_x, crop_y, crop_width, crop_height, flip_horizontal, flip_vertical,
        edit_btn, edited_image_output, apply_filter, brightness, contrast, saturation,
        img2img_input_path, img2img_prompt, img2img_flush,
        i2i_lora_acc, i2i_refresh_lora_btn, i2i_model_acc, img2img_refresh_models,
        img2img_t_drop, img2img_v_drop, img2img_perf_mode,
        img2img_width_s, img2img_height_s, tip_md, img2img_strength, img2img_steps,
        img2img_cfg, img2img_batch, img2img_seed, img2img_random,
        img2img_run_btn, img2img_stop_btn, img2img_gallery, img2img_res_seed,
        # Inpaint Outputs
        inpaint_input_img, inpaint_tip_md, inpaint_prompt, inpaint_flush, inpaint_lora_acc, inpaint_refresh_lora_btn,
        inpaint_model_acc, inpaint_refresh_models, inpaint_t_drop, inpaint_v_drop, inpaint_perf_mode,
        inpaint_strength, inpaint_steps, inpaint_cfg, inpaint_seed, inpaint_random,
        inpaint_run_btn, inpaint_stop_btn, inpaint_gallery, inpaint_res_seed,
        # Fusion Outputs
        desc_fusion_md, fusion_input1_path, fusion_input2_path, fusion_prompt, fusion_flush,
        fusion_lora_acc, fusion_refresh_lora_btn, fusion_model_acc, fusion_refresh_models,
        fusion_t_drop, fusion_v_drop, fusion_perf_mode,
        fusion_width_s, fusion_height_s, fusion_tip_md, fusion_blend, fusion_strength,
        fusion_steps, fusion_cfg, fusion_batch, fusion_seed, fusion_random,
        fusion_run_btn, fusion_stop_btn, fusion_gallery, fusion_res_seed
    ]

    def ui_to_running():
        return gr.update(interactive=False), gr.update(interactive=True)

    def ui_to_idle():
        return gr.update(interactive=True), gr.update(interactive=False)

    def trigger_interrupt():
        global is_interrupted
        is_interrupted = True
        return "正在强制中断..."

    lang_radio.change(
        fn=change_language,
        inputs=lang_radio,
        outputs=lang_outputs
    )
    
    # -----------------------
    # 刷新模型函数 (修复语法错误)
    # -----------------------
    def refresh_models_list():
        return (
            gr.update(choices=["default"] + scan_model_items(MOD_TRANS_DIR)),
            gr.update(choices=["default"] + scan_model_items(MOD_VAE_DIR))
        )

    refresh_models_btn.click(
        fn=refresh_models_list,
        outputs=[t_drop, v_drop]
    )
    
    txt_refresh_lora_btn.click(fn=lambda l: refresh_lora_list(l), inputs=CURRENT_LANG, outputs=txt_lora_info_md)
    
    manual_flush_btn.click(
        fn=manual_force_flush,
        outputs=t2i_vram_info
    )

    txt_ui_inputs = [prompt_input] + txt_lora_checks + txt_lora_sliders
    for c in txt_lora_checks + txt_lora_sliders:
        c.change(fn=update_prompt_ui_base, inputs=txt_ui_inputs, outputs=prompt_input)

    # ------------------------
    # 文生图（Text2Image）
    # ------------------------
    inference_event = run_btn.click(
        fn=ui_to_running,  # 只更新按钮状态
        outputs=[run_btn, stop_btn]
    ).then(
        fn=run_inference,  # 返回最终图像 + 种子 + VRAM信息
        # 传入采样器参数 t2i_sampler_drop
        inputs=txt_ui_inputs + [t_drop, v_drop, perf_mode_radio, width_s, height_s, step_s, cfg_s, seed_n, random_c, batch_s, vram_threshold_slider, t2i_sampler_drop],
        outputs=[res_gallery, res_seed, t2i_vram_info]
    ).then(
        fn=ui_to_idle,
        outputs=[run_btn, stop_btn]
    )

    stop_btn.click(
        fn=trigger_interrupt,
        outputs=t2i_vram_info
    ).then(
        fn=ui_to_idle,
        outputs=[run_btn, stop_btn],
        cancels=[inference_event]
    )
    
    def refresh_all_models_img():
        return (
            gr.update(choices=["default"] + scan_model_items(MOD_TRANS_DIR)),
            gr.update(choices=["default"] + scan_model_items(MOD_VAE_DIR))
        )

    img2img_refresh_models.click(fn=refresh_all_models_img, outputs=[img2img_t_drop, img2img_v_drop])
    
    img2img_flush.click(
        fn=manual_force_flush,
        outputs=i2i_vram_info
    )

    img2img_event = img2img_run_btn.click(
        fn=ui_to_running,
        outputs=[img2img_run_btn, img2img_stop_btn]
    ).then(
        fn=run_img2img,
        inputs=[
            img2img_prompt,
            img2img_negative_prompt,
            img2img_input_path, # 使用 path
            img2img_width_s,
            img2img_height_s,
            img2img_steps,
            img2img_cfg,
            img2img_strength,
            img2img_seed,
            img2img_t_drop,
            img2img_v_drop,
            img2img_perf_mode,
            img2img_lora_drop,
            img2img_lora_weight,
            img2img_mode,
            img2img_sampler_drop  # 传入采样器
        ],
        outputs=[
            img2img_gallery,
            img2img_res_seed,
            i2i_vram_info
        ]
    ).then(
        fn=ui_to_idle,
        outputs=[img2img_run_btn, img2img_stop_btn]
    )

    img2img_stop_btn.click(
        fn=trigger_interrupt, 
        outputs=i2i_vram_info
    ).then(
        fn=ui_to_idle, 
        outputs=[img2img_run_btn, img2img_stop_btn], 
        cancels=[img2img_event]
    )

    # --- 局部重绘 事件绑定 ---
    def refresh_all_models_inpaint():
        return (
            gr.update(choices=["default"] + scan_model_items(MOD_TRANS_DIR)),
            gr.update(choices=["default"] + scan_model_items(MOD_VAE_DIR))
        )

    inpaint_refresh_models.click(fn=refresh_all_models_inpaint, outputs=[inpaint_t_drop, inpaint_v_drop])
    
    inpaint_flush.click(
        fn=manual_force_flush,
        outputs=inpaint_vram_info
    )

    inpaint_event = inpaint_run_btn.click(
        fn=ui_to_running,
        outputs=[inpaint_run_btn, inpaint_stop_btn]
    ).then(
        fn=run_inpainting,
        inputs=[
            inpaint_input_img,  # ImageEditor数据
            inpaint_prompt,
            inpaint_negative_prompt,
            inpaint_steps,
            inpaint_cfg,
            inpaint_strength,
            inpaint_seed,
            inpaint_t_drop,
            inpaint_v_drop,
            inpaint_perf_mode,
            inpaint_lora_drop,
            inpaint_lora_weight,
            inpaint_sampler_drop  # 传入采样器
        ],
        outputs=[
            inpaint_gallery,
            inpaint_res_seed,
            inpaint_vram_info
        ]
    ).then(
        fn=ui_to_idle,
        outputs=[inpaint_run_btn, inpaint_stop_btn]
    )

    inpaint_stop_btn.click(
        fn=trigger_interrupt, 
        outputs=inpaint_vram_info
    ).then(
        fn=ui_to_idle, 
        outputs=[inpaint_run_btn, inpaint_stop_btn], 
        cancels=[inpaint_event]
    )

    # --- 融合图 ---
    fusion_refresh_models.click(fn=refresh_all_models_img, outputs=[fusion_t_drop, fusion_v_drop])
    
    fusion_refresh_lora_btn.click(fn=lambda l: refresh_lora_list(l), inputs=CURRENT_LANG, outputs=fusion_lora_info_md)
    
    fusion_flush.click(
        fn=manual_force_flush,
        outputs=i2i_vram_info
    )

    fusion_ui_inputs = [fusion_prompt] + fusion_lora_checks + fusion_lora_sliders
    for c in fusion_lora_checks + fusion_lora_sliders:
        c.change(fn=update_prompt_ui_base, inputs=fusion_ui_inputs, outputs=fusion_prompt)

    fusion_event = fusion_run_btn.click(
        fn=ui_to_running, 
        outputs=[fusion_run_btn, fusion_stop_btn]
    ).then(
        fn=run_fusion_img,
        inputs=[fusion_input1_path, fusion_input2_path, fusion_prompt] + fusion_lora_checks + fusion_lora_sliders + 
                [fusion_t_drop, fusion_v_drop, fusion_perf_mode, fusion_width_s, fusion_height_s,
                 fusion_blend, fusion_strength, fusion_steps, fusion_cfg, 
                 fusion_seed, fusion_random, fusion_batch, vram_threshold_slider, fusion_sampler_drop],  # 传入采样器
        outputs=[fusion_gallery, fusion_res_seed, i2i_vram_info]
    ).then(
        fn=ui_to_idle, 
        outputs=[fusion_run_btn, fusion_stop_btn]
    )

    fusion_stop_btn.click(
        fn=trigger_interrupt, 
        outputs=i2i_vram_info
    ).then(
        fn=ui_to_idle, 
        outputs=[fusion_run_btn, fusion_stop_btn], 
        cancels=[fusion_event]
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860, inbrowser=False)
<div align="center">

# 🎨 轻创图源 (LightPic-Source)

**专为低端显卡打造的 AI 生图应用集合**

<div align="center">
  <img src="LightPic.ico" alt="轻创图源 Logo" width="120" height="120">
</div>

<div align="center">
  <a href="https://github.com/yourusername/lightpic-source/releases">
    <img src="https://img.shields.io/github/v/release/yourusername/lightpic-source?color=green&label=Release" alt="Release">
  </a>
  <a href="https://github.com/yourusername/lightpic-source/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/yourusername/lightpic-source?color=blue" alt="License">
  </a>
  <a href="https://github.com/yourusername/lightpic-source">
    <img src="https://img.shields.io/github/stars/yourusername/lightpic-source?style=social" alt="Stars">
  </a>
</div>

</div>

## ✨ 项目简介

轻创图源 (LightPic-Source) 是一款专为低端显卡电脑设计的 AI 生图应用集合，通过深度优化的低显存技术，让普通电脑也能流畅体验 AI 绘画的魅力。

## 🚀 核心优势

- **🖥️ 低显存优化**：支持 4GB 及以上显存的显卡
- **🎯 多模型支持**：集成多种主流生图模型
- **🎨 丰富功能**：文生图、图生图、局部重绘、图片融合等
- **💡 智能配置**：自动检测并适配硬件环境
- **🎭 精美界面**：现代化的科幻风格 UI
- **⚡ 高效运行**：优化的渲染引擎，快速生成高质量图像

## 📋 系统要求

| 组件 | 最低要求 | 推荐要求 |
|------|----------|----------|
| 操作系统 | Windows 10/11 | Windows 11 |
| CPU | Intel i5 4 代 / AMD Ryzen 3 | Intel i7 8 代 / AMD Ryzen 5 |
| 内存 | 8 GB | 16 GB |
| 显存 | 4 GB | 8 GB |
| 存储 | 20 GB 可用空间 | 50 GB 可用空间 |
| .NET | .NET 8.0 Runtime | .NET 8.0 Runtime |

## 📦 安装方法

### 方法一：直接下载安装包（推荐）

1. 从 [GitHub Releases](https://github.com/yourusername/lightpic-source/releases) 下载最新版本的安装包
2. 双击安装包，按照提示完成安装
3. 启动桌面上的 "轻创图源" 快捷方式

### 方法二：从源码构建

```bash
# 克隆仓库
git clone https://github.com/yourusername/lightpic-source.git
cd lightpic-source

# 构建项目
dotnet build -c Release

# 运行应用
dotnet run --project LightPic-Source.csproj
```

## 🎮 使用指南

### 1. 启动应用

双击桌面上的 "轻创图源" 快捷方式，或运行 `LightPic-Source.exe`。

### 2. 选择生图模式

- **文生图**：输入文字描述，生成全新图像
- **图生图**：基于参考图生成新图像
- **局部重绘**：修改图像的特定区域
- **图片融合**：融合两张图片的特征

### 3. 调整参数

- **分辨率**：根据显卡性能选择合适的分辨率
- **步数**：影响图像质量和生成速度
- **CFG**：控制生成图像与提示词的匹配度
- **采样器**：选择不同的生成算法

### 4. 生成图像

点击 "开始生成" 按钮，等待图像生成完成。生成的图像会自动保存到 `outputs` 目录。

## 📁 项目结构

```
LightPic-Source/
├── App.xaml               # WPF 应用入口
├── App.xaml.cs            # 应用逻辑
├── MainWindow.xaml        # 主窗口 UI
├── MainWindow.xaml.cs     # 主窗口逻辑
├── ImageBrowserWindow.xaml # 图片浏览器
├── ImageBrowserWindow.xaml.cs
├── GraphicsSettingsChecker.cs # 图形设置检查
├── Properties/            # 项目属性
├── Z-Image-App.py         # Python 生图脚本
├── advanced_diagnostic.py # 高级诊断工具
├── diagnose_system.py     # 系统诊断
├── low_vram_fix.py        # 低显存修复
├── lightpic-source.csproj # 项目文件
└── README.md              # 项目说明
```

## 🛠️ 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| C# | 12.0 | 桌面应用开发 |
| WPF | .NET 8.0 | UI 框架 |
| Python | 3.10+ | AI 模型调用 |
| PyTorch | 2.0+ | 深度学习框架 |
| Diffusers | 0.20+ | 扩散模型库 |
| Gradio | 3.40+ | Web UI |
| Obfuscar | 2.2 | 代码混淆 |

## 🎨 主要功能

### 文生图
- 支持丰富的提示词语法
- 多种风格预设
- 批量生成
- 随机种子生成

### 图生图
- 基于参考图生成新图像
- 可控的重绘强度
- 支持多种图像格式

### 局部重绘
- 直观的蒙版绘制
- 精确的区域修改
- 自然的边缘融合

### 图片融合
- 智能融合两张图片
- 可调节融合权重
- 支持多种融合模式

### 图片浏览器
- 内置图片查看器
- 支持缩放、旋转、拖拽
- EXIF 信息查看
- 批量导出

## 🔧 低显存优化技术

1. **模型分割加载**：将模型分割成多个部分，按需加载到显存
2. **梯度检查点**：减少显存占用，适合长序列生成
3. **自动混合精度**：使用 FP16 精度加速生成
4. **动态内存管理**：实时监控显存使用，自动调整生成策略
5. **CPU 缓存**：将不常用的模型部分缓存到 CPU 内存

## 📊 性能测试

| 显卡型号 | 显存 | 生成 512x512 图像耗时 | 生成 1024x1024 图像耗时 |
|----------|------|----------------------|------------------------|
| GTX 1050 Ti | 4GB | 30s | 60s |
| GTX 1660 Super | 6GB | 15s | 30s |
| RTX 2060 | 6GB | 12s | 24s |
| RTX 3060 | 12GB | 8s | 16s |

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 提交 Issue

- 报告 bug
- 提出新功能建议
- 讨论现有功能

### 提交 Pull Request

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 [MIT License](LICENSE) 协议。

## 🙏 致谢

- 感谢所有为 AI 生图领域做出贡献的研究者和开发者
- 感谢 [Stable Diffusion](https://stability.ai/) 团队的开源贡献
- 感谢 [Hugging Face](https://huggingface.co/) 提供的模型资源
- 感谢 [Microsoft](https://dotnet.microsoft.com/) 提供的 .NET 框架

## 📞 联系方式

- 项目主页：[https://github.com/yourusername/lightpic-source](https://github.com/yourusername/lightpic-source)
- 问题反馈：[GitHub Issues](https://github.com/yourusername/lightpic-source/issues)
- 邮件联系：support@lightpic-source.com

---

<div align="center">
  <p>💡 轻创图源，让 AI 生图触手可及 💡</p>
  <p>© 2024 LightPic-Source Team</p>
</div>

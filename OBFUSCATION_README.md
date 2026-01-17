# Z-Image-Launcher 代码混淆指南

## 概述

本项目使用 **Obfuscar** 进行 .NET 代码混淆保护，可以有效防止反编译。

## 混淆功能

- ✅ **名称混淆**：变量、方法、类名重命名为无意义字符
- ✅ **字符串加密**：保护敏感字符串（如密钥）
- ✅ **控制流混淆**：使反编译代码难以理解
- ✅ **资源混淆**：隐藏嵌入资源
- ✅ **抑制 ILDASM**：阻止反汇编工具

## 使用方法

### 方式一：使用批处理脚本（推荐）

1. **构建 Release 版本**
   ```
   dotnet build -c Release
   ```

2. **运行混淆脚本**
   ```
   双击 Obfuscate.bat
   ```

3. **获取受保护的 EXE**
   ```
   bin\Release\net8.0-windows\Protected\Z-Image-Launcher.exe
   ```

### 方式二：手动使用 Obfuscar

```bash
# 安装 Obfuscar
dotnet tool install -g Obfuscar

# 执行混淆
obfuscar -c Release
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `obfuscar-config.xml` | 混淆配置文件 |
| `Obfuscate.bat` | 一键混淆脚本 |
| `Protected\` | 混淆后的输出目录 |
| `Mapping.txt` | 符号映射文件（用于调试） |

## 注意事项

### ⚠️ 重要提醒

1. **分发前混淆**：只分发 `Protected` 文件夹中的 `.exe`

2. **保留 Mapping.txt**：
   - 记录了混淆前后的符号映射
   - 用于崩溃调试时还原堆栈信息
   - **不要随程序分发**

3. **WPF 限制**：
   - 由于使用 WPF/XAML，部分类型无法混淆
   - 但核心业务逻辑仍然受到保护
   - 字符串加密有效（保护你的密钥）

4. **测试混淆后的程序**：
   - 每次混淆后请完整测试功能
   - 确保没有引入问题

## 混淆强度对比

| 原始代码 | 混淆后 |
|----------|--------|
| `DeployAndRunModel()` | `a.b()` |
| `_pythonProcess` | `c.d` |
| `"Z-Image-Secret-Key"` | 加密字符串 |

## 保护效果

- ❌ dnSpy/ILSpy：无法还原有意义的方法名
- ❌ Reflector：控制流混淆使代码难懂
- ❌ de4dot：字符串已加密
- ✅ 核心逻辑被有效保护

## 进一步保护建议

1. **加密敏感数据**：
   - 将密钥改为服务器验证
   - 不要在代码中硬编码重要信息

2. **代码虚拟化**（高级）：
   - 使用、商业混淆器如 Dotfuscator
   - 或 VMProtect（付费）

3. **服务器端验证**：
   - 关键逻辑放在服务器执行
   - 客户端只做 UI 和展示

## 故障排除

### 问题：混淆后程序无法启动

**原因**：WPF 类型被错误混淆

**解决**：检查 `obfuscar-config.xml` 中的 `<SkipMethod>` 部分

### 问题：某些功能失效

**原因**：被混淆的类型在 XAML 中使用

**解决**：在配置文件中添加跳过规则：
```xml
<SkipMethod type="你的命名空间.你的类" name="*" />
```

## 联系支持

如遇问题，请：
1. 查看 `bin\Release\net8.0-windows\Protected\Mapping.txt`
2. 对比原始代码定位问题
3. 调整 `obfuscar-config.xml` 配置

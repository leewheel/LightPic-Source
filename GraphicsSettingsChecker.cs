using System;
using System.IO;
using Microsoft.Win32;
using System.Management;

namespace ZImageLauncher
{
    /// <summary>
    /// Windows图形设置检查器
    /// 检查Python.exe是否在Windows图形设置中被设置为高性能模式
    /// </summary>
    public class GraphicsSettingsChecker
    {
        /// <summary>
        /// 获取系统中的GPU信息
        /// </summary>
        /// <returns>GPU信息字符串</returns>
        public static string GetGPUInfo()
        {
            try
            {
                var gpuInfo = new System.Text.StringBuilder();
                
                using (var searcher = new ManagementObjectSearcher("SELECT * FROM Win32_VideoController"))
                {
                    foreach (ManagementObject obj in searcher.Get())
                    {
                        string name = obj["Name"]?.ToString() ?? "未知";
                        
                        // 尝试获取显存信息
                        double ramGB = 0;
                        try
                        {
                            object adapterRAM = obj["AdapterRAM"];
                            if (adapterRAM != null)
                            {
                                long ramBytes = Convert.ToInt64(adapterRAM);
                                ramGB = ramBytes / (1024.0 * 1024.0 * 1024.0);
                            }
                        }
                        catch { }
                        
                        // 强制根据显卡型号覆盖显存值（WMI可能返回不准确的值）
                        if (name.Contains("RTX 3060") && !name.Contains("Ti"))
                        {
                            ramGB = 12.0; // RTX 3060通常是12GB
                        }
                        else if (name.Contains("RTX 3060 Ti"))
                        {
                            ramGB = 8.0; // RTX 3060 Ti通常是8GB
                        }
                        else if (name.Contains("RTX 3070"))
                        {
                            ramGB = 8.0; // RTX 3070通常是8GB
                        }
                        else if (name.Contains("RTX 3080"))
                        {
                            ramGB = 10.0; // RTX 3080通常是10GB
                        }
                        else if (name.Contains("RTX 3090"))
                        {
                            ramGB = 24.0; // RTX 3090通常是24GB
                        }
                        
                        if (ramGB > 0)
                        {
                            gpuInfo.AppendLine($"  - {name} ({ramGB:F1}GB)");
                        }
                        else
                        {
                            gpuInfo.AppendLine($"  - {name}");
                        }
                    }
                }
                
                return gpuInfo.ToString().Trim();
            }
            catch (Exception ex)
            {
                return $"检测失败: {ex.Message}";
            }
        }
        /// <summary>
        /// 检查指定可执行文件的图形设置
        /// </summary>
        /// <param name="exePath">可执行文件完整路径</param>
        /// <returns>图形设置信息</returns>
        public static GraphicsSettingInfo CheckGraphicsSetting(string exePath)
        {
            var info = new GraphicsSettingInfo
            {
                ExePath = exePath,
                Exists = File.Exists(exePath)
            };

            if (!info.Exists)
            {
                info.Status = "文件不存在";
                return info;
            }

            try
            {
                // Windows图形设置存储在注册表中
                // 路径1: HKEY_CURRENT_USER\Software\Microsoft\DirectX\UserGpuPreferences
                // 路径2: HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\DirectX\UserGpuPreferences
                
                var result = CheckRegistrySetting(RegistryHive.CurrentUser, exePath);
                if (result != null)
                {
                    info.IsConfigured = true;
                    info.GpuPreference = result.RawValue;
                    info.RegistryPath = @"HKEY_CURRENT_USER\Software\Microsoft\DirectX\UserGpuPreferences";
                    info.Status = ParseGpuPreference(result.RawValue);
                    info.DebugInfo = result.DebugInfo;
                    return info;
                }

                result = CheckRegistrySetting(RegistryHive.LocalMachine, exePath);
                if (result != null)
                {
                    info.IsConfigured = true;
                    info.GpuPreference = result.RawValue;
                    info.RegistryPath = @"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\DirectX\UserGpuPreferences";
                    info.Status = ParseGpuPreference(result.RawValue);
                    info.DebugInfo = result.DebugInfo;
                    return info;
                }

                info.IsConfigured = false;
                info.Status = "未配置";
            }
            catch (Exception ex)
            {
                info.Status = $"检查失败: {ex.Message}";
                info.DebugInfo = $"异常: {ex.GetType().Name} - {ex.Message}";
            }

            return info;
        }

        /// <summary>
        /// 解析GPU偏好值
        /// </summary>
        private static string ParseGpuPreference(string rawValue)
        {
            if (string.IsNullOrEmpty(rawValue))
                return "未知";

            // 尝试多种可能的格式
            // 格式1: GpuPreference := 2
            if (rawValue.Contains("GpuPreference"))
            {
                var match = System.Text.RegularExpressions.Regex.Match(rawValue, @"GpuPreference\s*:=\s*(\d+)");
                if (match.Success)
                {
                    int value = int.Parse(match.Groups[1].Value);
                    return value == 2 ? "高性能" : value == 1 ? "节能" : value == 0 ? "自动选择" : $"未知值({value})";
                }
            }

            // 格式2: 直接的数字值
            if (int.TryParse(rawValue, out int intValue))
            {
                return intValue == 2 ? "高性能" : intValue == 1 ? "节能" : intValue == 0 ? "自动选择" : $"未知值({intValue})";
            }

            // 格式3: 包含数字的字符串
            var numberMatch = System.Text.RegularExpressions.Regex.Match(rawValue, @"\d+");
            if (numberMatch.Success)
            {
                int numValue = int.Parse(numberMatch.Value);
                return numValue == 2 ? "高性能" : numValue == 1 ? "节能" : numValue == 0 ? "自动选择" : $"未知值({numValue})";
            }

            // 无法识别的格式
            return $"未知格式: {rawValue}";
        }

        /// <summary>
        /// 检查注册表中的图形设置
        /// </summary>
        private static RegistryCheckResult CheckRegistrySetting(RegistryHive hive, string exePath)
        {
            var result = new RegistryCheckResult();
            var debugInfo = new System.Text.StringBuilder();

            try
            {
                using (RegistryKey baseKey = RegistryKey.OpenBaseKey(hive, RegistryView.Registry64))
                using (RegistryKey gpuPrefsKey = baseKey.OpenSubKey(@"Software\Microsoft\DirectX\UserGpuPreferences"))
                {
                    if (gpuPrefsKey == null)
                    {
                        debugInfo.AppendLine($"注册表键不存在: {hive}\\Software\\Microsoft\\DirectX\\UserGpuPreferences");
                        result.DebugInfo = debugInfo.ToString();
                        return null;
                    }

                    debugInfo.AppendLine($"成功打开注册表键: {hive}\\Software\\Microsoft\\DirectX\\UserGpuPreferences");

                    // 列出所有键名用于调试
                    string[] allValueNames = gpuPrefsKey.GetValueNames();
                    debugInfo.AppendLine($"注册表中共有 {allValueNames.Length} 个值:");
                    foreach (string name in allValueNames)
                    {
                        object value = gpuPrefsKey.GetValue(name);
                        debugInfo.AppendLine($"  - {name} = {value}");
                    }

                    // Windows使用可执行文件的完整路径作为键名
                    // 需要尝试不同的路径格式
                    string[] possibleKeys = new string[]
                    {
                        exePath,
                        exePath.ToLower(),
                        exePath.ToUpper(),
                        exePath.Replace('\\', '/'),
                        exePath.Replace('/', '\\')
                    };

                    debugInfo.AppendLine($"尝试匹配路径: {exePath}");

                    foreach (string keyName in possibleKeys)
                    {
                        debugInfo.AppendLine($"  尝试键名: {keyName}");
                        object value = gpuPrefsKey.GetValue(keyName);
                        if (value != null)
                        {
                            debugInfo.AppendLine($"  ✓ 找到匹配: {keyName} = {value}");
                            result.RawValue = value.ToString();
                            result.DebugInfo = debugInfo.ToString();
                            return result;
                        }
                    }

                    // 也尝试使用文件名作为键
                    string fileName = Path.GetFileName(exePath);
                    debugInfo.AppendLine($"尝试文件名: {fileName}");
                    object fileNameValue = gpuPrefsKey.GetValue(fileName);
                    if (fileNameValue != null)
                    {
                        debugInfo.AppendLine($"  ✓ 找到匹配: {fileName} = {fileNameValue}");
                        result.RawValue = fileNameValue.ToString();
                        result.DebugInfo = debugInfo.ToString();
                        return result;
                    }

                    debugInfo.AppendLine("✗ 未找到匹配的注册表项");
                }
            }
            catch (Exception ex)
            {
                debugInfo.AppendLine($"异常: {ex.GetType().Name} - {ex.Message}");
                debugInfo.AppendLine($"堆栈跟踪: {ex.StackTrace}");
            }

            result.DebugInfo = debugInfo.ToString();
            return null;
        }

        /// <summary>
        /// 检查所有Python可执行文件的图形设置
        /// </summary>
        /// <param name="rootDir">程序根目录</param>
        /// <returns>所有Python的图形设置信息</returns>
        public static GraphicsSettingsReport CheckAllPythonSettings(string rootDir)
        {
            var report = new GraphicsSettingsReport();
            report.RootDirectory = rootDir;
            report.CheckTime = DateTime.Now;

            // 检查WebUI的Python
            string webuiPython = Path.Combine(rootDir, "python_env", "python.exe");
            report.WebUIPython = CheckGraphicsSetting(webuiPython);

            // 检查ComfyUI的Python
            string comfyPython = Path.Combine(rootDir, "ComfyUI_windows_portable", "python_embeded", "python.exe");
            report.ComfyUIPython = CheckGraphicsSetting(comfyPython);

            // 生成总体状态
            report.OverallStatus = GenerateOverallStatus(report);

            return report;
        }

        /// <summary>
        /// 生成总体状态
        /// </summary>
        private static string GenerateOverallStatus(GraphicsSettingsReport report)
        {
            bool webuiOk = report.WebUIPython.Exists && 
                          (report.WebUIPython.IsConfigured && report.WebUIPython.Status == "高性能");
            
            bool comfyOk = report.ComfyUIPython.Exists && 
                          (report.ComfyUIPython.IsConfigured && report.ComfyUIPython.Status == "高性能");

            if (webuiOk && comfyOk)
                return "✓ 所有Python已设置为高性能";
            else if (webuiOk || comfyOk)
                return "⚠ 部分Python未设置为高性能";
            else if (!report.WebUIPython.Exists && !report.ComfyUIPython.Exists)
                return "✗ 未找到Python可执行文件";
            else
                return "✗ 所有Python未设置为高性能";
        }

        /// <summary>
        /// 生成图形设置检查报告
        /// </summary>
        public static string GenerateReport(GraphicsSettingsReport report)
        {
            var sb = new System.Text.StringBuilder();
            
            sb.AppendLine("========================================");
            sb.AppendLine("Windows图形设置检查报告");
            sb.AppendLine("========================================");
            sb.AppendLine($"检查时间: {report.CheckTime:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"程序根目录: {report.RootDirectory}");
            sb.AppendLine();
            sb.AppendLine($"总体状态: {report.OverallStatus}");
            sb.AppendLine();
            sb.AppendLine("----------------------------------------");
            sb.AppendLine("WebUI Python (python_env\\python.exe)");
            sb.AppendLine("----------------------------------------");
            sb.AppendLine($"文件路径: {report.WebUIPython.ExePath}");
            sb.AppendLine($"文件存在: {(report.WebUIPython.Exists ? "是" : "否")}");
            sb.AppendLine($"已配置: {(report.WebUIPython.IsConfigured ? "是" : "否")}");
            sb.AppendLine($"GPU偏好: {report.WebUIPython.GpuPreference ?? "未设置"}");
            sb.AppendLine($"状态: {report.WebUIPython.Status}");
            if (!string.IsNullOrEmpty(report.WebUIPython.RegistryPath))
                sb.AppendLine($"注册表路径: {report.WebUIPython.RegistryPath}");
            if (report.WebUIPython.Status.Contains("未知") && !string.IsNullOrEmpty(report.WebUIPython.DebugInfo))
            {
                sb.AppendLine();
                sb.AppendLine("调试信息:");
                sb.AppendLine(report.WebUIPython.DebugInfo);
            }
            sb.AppendLine();
            sb.AppendLine("----------------------------------------");
            sb.AppendLine("ComfyUI Python (ComfyUI_windows_portable\\python_embeded\\python.exe)");
            sb.AppendLine("----------------------------------------");
            sb.AppendLine($"文件路径: {report.ComfyUIPython.ExePath}");
            sb.AppendLine($"文件存在: {(report.ComfyUIPython.Exists ? "是" : "否")}");
            sb.AppendLine($"已配置: {(report.ComfyUIPython.IsConfigured ? "是" : "否")}");
            sb.AppendLine($"GPU偏好: {report.ComfyUIPython.GpuPreference ?? "未设置"}");
            sb.AppendLine($"状态: {report.ComfyUIPython.Status}");
            if (!string.IsNullOrEmpty(report.ComfyUIPython.RegistryPath))
                sb.AppendLine($"注册表路径: {report.ComfyUIPython.RegistryPath}");
            if (report.ComfyUIPython.Status.Contains("未知") && !string.IsNullOrEmpty(report.ComfyUIPython.DebugInfo))
            {
                sb.AppendLine();
                sb.AppendLine("调试信息:");
                sb.AppendLine(report.ComfyUIPython.DebugInfo);
            }
            sb.AppendLine();
            sb.AppendLine("----------------------------------------");
            sb.AppendLine("配置建议");
            sb.AppendLine("----------------------------------------");
            
            if (!report.WebUIPython.Exists && !report.ComfyUIPython.Exists)
            {
                sb.AppendLine("✗ 未找到Python可执行文件，请检查安装是否完整");
            }
            else
            {
                if (report.WebUIPython.Exists && (!report.WebUIPython.IsConfigured || report.WebUIPython.Status != "高性能"))
                {
                    sb.AppendLine("⚠ WebUI Python未设置为高性能模式");
                    sb.AppendLine($"   路径: {report.WebUIPython.ExePath}");
                    sb.AppendLine("   建议操作:");
                    sb.AppendLine("   1. 打开 设置 -> 系统 -> 屏幕 -> 图形");
                    sb.AppendLine("   2. 点击 '添加'");
                    sb.AppendLine("   3. 浏览并选择上述Python.exe文件");
                    sb.AppendLine("   4. 在列表中找到该文件，点击 '选项'");
                    sb.AppendLine("   5. 选择 '高性能'");
                    sb.AppendLine("   6. 点击 '保存'");
                    sb.AppendLine();
                }

                if (report.ComfyUIPython.Exists && (!report.ComfyUIPython.IsConfigured || report.ComfyUIPython.Status != "高性能"))
                {
                    sb.AppendLine("⚠ ComfyUI Python未设置为高性能模式");
                    sb.AppendLine($"   路径: {report.ComfyUIPython.ExePath}");
                    sb.AppendLine("   建议操作:");
                    sb.AppendLine("   1. 打开 设置 -> 系统 -> 屏幕 -> 图形");
                    sb.AppendLine("   2. 点击 '添加'");
                    sb.AppendLine("   3. 浏览并选择上述Python.exe文件");
                    sb.AppendLine("   4. 在列表中找到该文件，点击 '选项'");
                    sb.AppendLine("   5. 选择 '高性能'");
                    sb.AppendLine("   6. 点击 '保存'");
                    sb.AppendLine();
                }

                if ((report.WebUIPython.Exists && report.WebUIPython.IsConfigured && report.WebUIPython.Status == "高性能") &&
                    (report.ComfyUIPython.Exists && report.ComfyUIPython.IsConfigured && report.ComfyUIPython.Status == "高性能"))
                {
                    sb.AppendLine("✓ 所有Python已正确设置为高性能模式");
                    sb.AppendLine("   这将确保AI图像生成使用独立GPU，获得最佳性能");
                }
            }

            sb.AppendLine("========================================");

            return sb.ToString();
        }
    }

    /// <summary>
    /// 注册表检查结果
    /// </summary>
    public class RegistryCheckResult
    {
        public string RawValue { get; set; }
        public string DebugInfo { get; set; }
    }

    /// <summary>
    /// 图形设置信息
    /// </summary>
    public class GraphicsSettingInfo
    {
        public string ExePath { get; set; }
        public bool Exists { get; set; }
        public bool IsConfigured { get; set; }
        public string GpuPreference { get; set; }
        public string Status { get; set; }
        public string RegistryPath { get; set; }
        public string DebugInfo { get; set; }
    }

    /// <summary>
    /// 图形设置检查报告
    /// </summary>
    public class GraphicsSettingsReport
    {
        public string RootDirectory { get; set; }
        public DateTime CheckTime { get; set; }
        public GraphicsSettingInfo WebUIPython { get; set; }
        public GraphicsSettingInfo ComfyUIPython { get; set; }
        public string OverallStatus { get; set; }
    }
}
using System.Configuration;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Threading;

namespace Z_Image_Launcher
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            // 添加全局异常处理
            this.DispatcherUnhandledException += App_DispatcherUnhandledException;
            AppDomain.CurrentDomain.UnhandledException += App_DomainUnhandledException;

            base.OnStartup(e);
        }

        private void App_DispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
        {
            WriteCrashLog("DispatcherUnhandledException", e.Exception);
            e.Handled = true;
            MessageBox.Show($"程序发生未处理异常: {e.Exception.Message}\n\n详细信息已写入 crash.log", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
        }

        private void App_DomainUnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            WriteCrashLog("UnhandledException", e.ExceptionObject as System.Exception);
            if (e.IsTerminating)
            {
                MessageBox.Show($"程序发生严重异常即将退出: {(e.ExceptionObject as System.Exception)?.Message}", "严重错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void WriteCrashLog(string exceptionType, System.Exception ex)
        {
            try
            {
                // 获取程序运行目录（Z-Image根目录）
                string baseDir = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
                string logsDir = System.IO.Path.Combine(baseDir, "logs");
                System.IO.Directory.CreateDirectory(logsDir);
                
                string timeStr = DateTime.Now.ToString("yyyyMMdd_HH_mm");
                string crashLogPath = System.IO.Path.Combine(logsDir, $"crash_{timeStr}.log");
                
                string logContent = $"=== Crash Report ===\n";
                logContent += $"Time: {DateTime.Now:yyyy-MM-dd HH:mm:ss}\n";
                logContent += $"Type: {exceptionType}\n";
                logContent += $"Message: {ex?.Message}\n";
                logContent += $"StackTrace:\n{ex?.StackTrace}\n";
                logContent += $"===================\n\n";
                System.IO.File.AppendAllText(crashLogPath, logContent);
            }
            catch { }
        }
    }

}

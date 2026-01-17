using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

public class ConvertToIco
{
    public static void Main(string[] args)
    {
        try
        {
            // 检查参数
            if (args.Length != 2)
            {
                Console.WriteLine("用法: ConvertToIco <输入PNG文件> <输出ICO文件>");
                return;
            }

            string inputPath = args[0];
            string outputPath = args[1];

            // 检查输入文件是否存在
            if (!File.Exists(inputPath))
            {
                Console.WriteLine($"错误: 输入文件 '{inputPath}' 不存在。");
                return;
            }

            // 读取PNG文件
            using (Image pngImage = Image.FromFile(inputPath))
            {
                // 创建ICO文件
                // 注意: ICO文件需要特定的尺寸，我们将使用多种尺寸以确保兼容性
                int[] sizes = { 16, 32, 48, 64, 128, 256 };
                
                // 保存为ICO格式
                pngImage.Save(outputPath, ImageFormat.Icon);
                
                Console.WriteLine($"成功将 '{inputPath}' 转换为 '{outputPath}'");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"转换过程中发生错误: {ex.Message}");
        }
    }
}
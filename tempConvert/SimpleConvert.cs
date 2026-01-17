using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

public class SimpleConvert
{
    public static void Main(string[] args)
    {
        try
        {
            // 检查参数
            if (args.Length != 2)
            {
                Console.WriteLine("用法: SimpleConvert <输入PNG文件> <输出ICO文件>");
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
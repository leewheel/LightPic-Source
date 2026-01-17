using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Microsoft.Win32;

namespace LightPic_Source
{
    public class ThumbnailItem
    {
        public string FilePath { get; set; }
        public string FileName { get; set; }
        public BitmapImage Thumbnail { get; set; }
    }

    public partial class ImageBrowserWindow : Window
    {
        private string _currentDirectory;
        private List<ThumbnailItem> _imageFiles = new List<ThumbnailItem>();
        private double _zoomLevel = 1.0;
        private const double ZoomStep = 0.25;

        // 拖放相关
        private bool _isDragging = false;
        private System.Windows.Point _startPoint;
        private double _translateX = 0;
        private double _translateY = 0;

        // 变换组（用于同时应用缩放和平移）
        private System.Windows.Media.TransformGroup _transformGroup;
        private System.Windows.Media.ScaleTransform _scaleTransform;
        private System.Windows.Media.TranslateTransform _translateTransform;

        private readonly string[] SupportedExtensions = {
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif",
            ".ico", ".svg", ".pcx", ".pnm", ".xbm", ".xpm", ".ppm", ".pgm", ".pbm"
        };

        public ImageBrowserWindow(string initialDirectory)
        {
            InitializeComponent();
            _currentDirectory = initialDirectory;
            Loaded += ImageBrowserWindow_Loaded;
        }

        private void ImageBrowserWindow_Loaded(object sender, RoutedEventArgs e)
        {
            // 初始化变换组
            _transformGroup = new System.Windows.Media.TransformGroup();
            _scaleTransform = new System.Windows.Media.ScaleTransform(1.0, 1.0);
            _translateTransform = new System.Windows.Media.TranslateTransform(0, 0);
            _transformGroup.Children.Add(_scaleTransform);
            _transformGroup.Children.Add(_translateTransform);

            // 应用变换到主图
            MainImage.RenderTransform = _transformGroup;

            CurrentPathText.Text = $"当前目录：{_currentDirectory}";
            LoadImages(_currentDirectory);
        }

        private void LoadImages(string directory)
        {
            _imageFiles.Clear();
            ThumbnailList.ItemsSource = null;

            if (!Directory.Exists(directory))
            {
                MessageBox.Show("目录不存在！", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            var files = Directory.GetFiles(directory)
                .Where(f => SupportedExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                .OrderByDescending(File.GetCreationTime)
                .ToList();

            int loadedCount = 0;
            foreach (var filePath in files)
            {
                try
                {
                    var thumbnail = LoadThumbnail(filePath);
                    if (thumbnail != null)
                    {
                        _imageFiles.Add(new ThumbnailItem
                        {
                            FilePath = filePath,
                            FileName = Path.GetFileName(filePath),
                            Thumbnail = thumbnail
                        });
                        loadedCount++;
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"加载缩略图失败: {filePath}, {ex.Message}");
                }
            }

            ThumbnailList.ItemsSource = _imageFiles;
            
            if (_imageFiles.Count > 0)
            {
                ThumbnailList.SelectedIndex = 0;
            }
        }

        private BitmapImage LoadThumbnail(string filePath)
        {
            try
            {
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.DecodePixelWidth = 180;
                bitmap.DecodePixelHeight = 120;
                bitmap.UriSource = new Uri(filePath);
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.EndInit();
                bitmap.Freeze();
                return bitmap;
            }
            catch
            {
                return null;
            }
        }

        private void ThumbnailList_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ThumbnailList.SelectedItem is ThumbnailItem selectedItem)
            {
                DisplayImage(selectedItem.FilePath);
            }
        }

        private void DisplayImage(string filePath)
        {
            try
            {
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.UriSource = new Uri(filePath);
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.EndInit();
                
                MainImage.Source = bitmap;
                
                // 重置缩放和拖放位置
                _zoomLevel = 1.0;
                _translateX = 0;
                _translateY = 0;
                ApplyTransform();
                
                // 加载图片信息
                LoadImageInfo(filePath, bitmap);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"无法加载图片: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void LoadImageInfo(string filePath, BitmapImage bitmap)
        {
            // 文件信息
            var fileInfo = new FileInfo(filePath);
            FileName.Text = fileInfo.Name;
            FileSize.Text = FormatFileSize(fileInfo.Length);
            FilePath.Text = filePath;
            FileCreated.Text = fileInfo.CreationTime.ToString("yyyy-MM-dd HH:mm:ss");
            FileModified.Text = fileInfo.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss");

            // 图像信息
            ImageDimensions.Text = $"{bitmap.PixelWidth} × {bitmap.PixelHeight} 像素";
            ImageResolution.Text = $"{bitmap.DpiX:F0} × {bitmap.DpiY:F0} DPI";
            ColorDepth.Text = GetColorDepth(bitmap);
            ImageFormat.Text = Path.GetExtension(filePath).ToUpperInvariant().TrimStart('.');
            AspectRatio.Text = CalculateAspectRatio(bitmap.PixelWidth, bitmap.PixelHeight);

            // 尝试加载EXIF信息
            LoadExifInfo(filePath);
        }

        private string FormatFileSize(long bytes)
        {
            if (bytes < 1024) return $"{bytes} B";
            if (bytes < 1024 * 1024) return $"{bytes / 1024.0:F1} KB";
            if (bytes < 1024 * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F1} MB";
            return $"{bytes / (1024.0 * 1024 * 1024):F2} GB";
        }

        private string GetColorDepth(BitmapImage bitmap)
        {
            if (bitmap.Format == PixelFormats.Bgra32) return "32 位 (ARGB)";
            if (bitmap.Format == PixelFormats.Rgb24) return "24 位 (RGB)";
            if (bitmap.Format == PixelFormats.Gray8) return "8 位 (灰度)";
            if (bitmap.Format == PixelFormats.Pbgra32) return "32 位 (预乘 Alpha)";
            if (bitmap.Format == PixelFormats.Rgba128Float) return "128 位 (RGBA 浮点)";
            return bitmap.Format.ToString();
        }

        private string CalculateAspectRatio(int width, int height)
        {
            int gcd = GCD(width, height);
            return $"{width / gcd}:{height / gcd} ({width}x{height})";
        }

        private int GCD(int a, int b)
        {
            while (b != 0)
            {
                int temp = b;
                b = a % b;
                a = temp;
            }
            return a;
        }

        private void LoadExifInfo(string filePath)
        {
            // 重置所有EXIF字段为"-"
            ResetExifFields();

            try
            {
                using (System.Drawing.Image image = System.Drawing.Image.FromFile(filePath))
                {
                    var properties = image.PropertyItems;

                    foreach (var prop in properties)
                    {
                        string value = GetExifValue(prop);
                        if (!string.IsNullOrEmpty(value))
                        {
                            SetExifProperty(prop.Id, value);
                        }
                    }

                    // 尝试获取GPS信息
                    LoadGPSInfo(image);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"读取EXIF信息失败: {ex.Message}");
            }
        }

        private void ResetExifFields()
        {
            CameraMake.Text = "-";
            CameraModel.Text = "-";
            LensModel.Text = "-";
            DateTime.Text = "-";
            ShutterSpeed.Text = "-";
            Aperture.Text = "-";
            ISOValue.Text = "-";
            FocalLength.Text = "-";
            ExposureComp.Text = "-";
            MeteringMode.Text = "-";
            Flash.Text = "-";
            WhiteBalance.Text = "-";
            ExposureMode.Text = "-";
            SceneMode.Text = "-";
            GPSLatitude.Text = "-";
            GPSLongitude.Text = "-";
            GPSAltitude.Text = "-";
        }

        private string GetExifValue(PropertyItem prop)
        {
            try
            {
                if (prop.Value == null || prop.Value.Length == 0)
                    return null;

                switch (prop.Id)
                {
                    case 0x010F: // Make (Camera Make)
                        return Encoding.ASCII.GetString(prop.Value).Trim('\0');
                    case 0x0110: // Model (Camera Model)
                        return Encoding.ASCII.GetString(prop.Value).Trim('\0');
                    case 0xA432: // Lens Model
                        return Encoding.ASCII.GetString(prop.Value).Trim('\0');
                    case 0x9003: // DateTimeOriginal
                    case 0x9004: // DateTimeDigitized
                        return Encoding.ASCII.GetString(prop.Value).Trim('\0');
                    case 0x829A: // ExposureTime
                        return FormatExposureTime(prop.Value);
                    case 0x829D: // FNumber
                        return FormatFNumber(prop.Value);
                    case 0x8827: // ISOSpeedRatings
                        return BitConverter.ToInt16(prop.Value, 0).ToString();
                    case 0x920A: // FocalLength
                        return FormatFocalLength(prop.Value);
                    case 0x9204: // ExposureBias
                        return FormatExposureBias(prop.Value);
                    case 0x9207: // MeteringMode
                        return FormatMeteringMode(prop.Value);
                    case 0x9209: // Flash
                        return FormatFlash(prop.Value);
                    case 0x9208: // LightSource (WhiteBalance)
                        return FormatLightSource(prop.Value);
                    case 0x9202: // ExposureProgram
                        return FormatExposureProgram(prop.Value);
                    case 0x9201: // ExposureMode
                        return FormatExposureMode(prop.Value);
                    case 0x9206: // SceneCaptureType
                        return FormatSceneMode(prop.Value);
                    default:
                        return null;
                }
            }
            catch
            {
                return null;
            }
        }

        private void SetExifProperty(int id, string value)
        {
            switch (id)
            {
                case 0x010F: CameraMake.Text = value; break;
                case 0x0110: CameraModel.Text = value; break;
                case 0xA432: LensModel.Text = value; break;
                case 0x9003:
                case 0x9004: DateTime.Text = value; break;
                case 0x829A: ShutterSpeed.Text = value; break;
                case 0x829D: Aperture.Text = value; break;
                case 0x8827: ISOValue.Text = $"ISO {value}"; break;
                case 0x920A: FocalLength.Text = value; break;
                case 0x9204: ExposureComp.Text = value; break;
                case 0x9207: MeteringMode.Text = value; break;
                case 0x9209: Flash.Text = value; break;
                case 0x9208: WhiteBalance.Text = value; break;
                case 0x9202: ExposureMode.Text = value; break;
                case 0x9206: SceneMode.Text = value; break;
            }
        }

        private string FormatExposureTime(byte[] value)
        {
            try
            {
                // 曝光时间通常存储为有理数
                int numerator = BitConverter.ToInt32(value, 0);
                int denominator = BitConverter.ToInt32(value, 4);
                
                if (denominator == 0) return "未知";
                
                double exposure = (double)numerator / denominator;
                if (exposure >= 1)
                    return $"{exposure:F1} 秒";
                else
                    return $"1/{(int)(1/exposure + 0.5)} 秒";
            }
            catch
            {
                return "未知";
            }
        }

        private string FormatFNumber(byte[] value)
        {
            try
            {
                int numerator = BitConverter.ToInt32(value, 0);
                int denominator = BitConverter.ToInt32(value, 4);
                
                if (denominator == 0) return "未知";
                
                double fNumber = (double)numerator / denominator;
                return $"f/{fNumber:F1}";
            }
            catch
            {
                return "未知";
            }
        }

        private string FormatFocalLength(byte[] value)
        {
            try
            {
                int numerator = BitConverter.ToInt32(value, 0);
                int denominator = BitConverter.ToInt32(value, 4);
                
                if (denominator == 0) return "未知";
                
                double focalLength = (double)numerator / denominator;
                return $"{focalLength:F0} mm";
            }
            catch
            {
                return "未知";
            }
        }

        private string FormatExposureBias(byte[] value)
        {
            try
            {
                int numerator = BitConverter.ToInt32(value, 0);
                int denominator = BitConverter.ToInt32(value, 4);
                
                if (denominator == 0) return "未知";
                
                double bias = (double)numerator / denominator;
                if (bias > 0)
                    return $"+{bias:F1} EV";
                else if (bias < 0)
                    return $"{bias:F1} EV";
                else
                    return "0 EV";
            }
            catch
            {
                return "未知";
            }
        }

        private string FormatMeteringMode(byte[] value)
        {
            try
            {
                int mode = BitConverter.ToInt16(value, 0);
                switch (mode)
                {
                    case 0: return "未知";
                    case 1: return "平均";
                    case 2: return "中心加权平均";
                    case 3: return "点测光";
                    case 4: return "多点测光";
                    case 5: return "评价测光";
                    case 6: return "部分测光";
                    default: return "未知";
                }
            }
            catch
            {
                return "未知";
            }
        }

        private string FormatFlash(byte[] value)
        {
            try
            {
                int flash = BitConverter.ToInt16(value, 0);
                bool fired = (flash & 0x1) != 0;
                return fired ? "闪光灯已触发" : "未使用闪光灯";
            }
            catch
            {
                return "未知";
            }
        }

        private string FormatLightSource(byte[] value)
        {
            try
            {
                int source = BitConverter.ToInt16(value, 0);
                switch (source)
                {
                    case 0: return "未知";
                    case 1: return "日光";
                    case 2: return "荧光灯";
                    case 3: return "钨丝灯";
                    case 4: return "闪光灯";
                    case 9: return "阴天";
                    case 10: return "白炽灯";
                    case 11: return "阴影";
                    default: return "自动";
                }
            }
            catch
            {
                return "未知";
            }
        }

        private string FormatExposureProgram(byte[] value)
        {
            try
            {
                int program = BitConverter.ToInt16(value, 0);
                switch (program)
                {
                    case 0: return "未知";
                    case 1: return "手动";
                    case 2: return "普通程序";
                    case 3: return "光圈优先";
                    case 4: return "快门优先";
                    case 5: return "创意程序";
                    case 6: return "动作程序";
                    case 7: return "人像模式";
                    case 8: return "风景模式";
                    default: return "未知";
                }
            }
            catch
            {
                return "未知";
            }
        }

        private string FormatExposureMode(byte[] value)
        {
            try
            {
                int mode = BitConverter.ToInt16(value, 0);
                switch (mode)
                {
                    case 0: return "自动";
                    case 1: return "手动";
                    case 2: return "自动包围曝光";
                    default: return "未知";
                }
            }
            catch
            {
                return "未知";
            }
        }

        private string FormatSceneMode(byte[] value)
        {
            try
            {
                int mode = BitConverter.ToInt16(value, 0);
                switch (mode)
                {
                    case 0: return "标准";
                    case 1: return "人像";
                    case 2: return "风景";
                    case 3: return "运动";
                    case 4: return "夜景";
                    case 5: return "逆光人像";
                    case 6: return "夜景人像";
                    case 7: return "微距";
                    default: return "未知";
                }
            }
            catch
            {
                return "未知";
            }
        }

        private void LoadGPSInfo(System.Drawing.Image image)
        {
            try
            {
                PropertyItem latRef = image.GetPropertyItem(0x0001);
                PropertyItem latVal = image.GetPropertyItem(0x0002);
                PropertyItem lonRef = image.GetPropertyItem(0x0003);
                PropertyItem lonVal = image.GetPropertyItem(0x0004);
                PropertyItem alt = image.GetPropertyItem(0x0005);

                if (latRef != null && latVal != null && lonRef != null && lonVal != null)
                {
                    double lat = ConvertGPSCoordinate(latRef.Value, latVal.Value);
                    double lon = ConvertGPSCoordinate(lonRef.Value, lonVal.Value);

                    GPSLatitude.Text = $"{lat:F6}°";
                    GPSLongitude.Text = $"{lon:F6}°";
                }

                if (alt != null)
                {
                    int altNumerator = BitConverter.ToInt32(alt.Value, 0);
                    int altDenominator = BitConverter.ToInt32(alt.Value, 4);
                    if (altDenominator != 0)
                    {
                        double altitude = (double)altNumerator / altDenominator;
                        GPSAltitude.Text = $"{altitude:F1} 米";
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"读取GPS信息失败: {ex.Message}");
            }
        }

        private double ConvertGPSCoordinate(byte[] refValue, byte[] coordValue)
        {
            try
            {
                double degrees = BitConverter.ToInt32(coordValue, 0) / (double)BitConverter.ToInt32(coordValue, 4);
                double minutes = BitConverter.ToInt32(coordValue, 8) / (double)BitConverter.ToInt32(coordValue, 12);
                double seconds = BitConverter.ToInt32(coordValue, 16) / (double)BitConverter.ToInt32(coordValue, 20);

                double result = degrees + minutes / 60.0 + seconds / 3600.0;

                // 南纬或西经需要取反
                string refStr = Encoding.ASCII.GetString(refValue).Trim('\0');
                if (refStr == "S" || refStr == "W")
                    result = -result;

                return result;
            }
            catch
            {
                return 0;
            }
        }

        private void ZoomIn_Click(object sender, RoutedEventArgs e)
        {
            _zoomLevel = Math.Min(_zoomLevel + ZoomStep, 5.0);
            ApplyZoom();
        }

        private void ZoomOut_Click(object sender, RoutedEventArgs e)
        {
            _zoomLevel = Math.Max(_zoomLevel - ZoomStep, 0.1);
            ApplyZoom();
        }

        private void ResetZoom_Click(object sender, RoutedEventArgs e)
        {
            ResetZoom();
        }

        private void ApplyZoom()
        {
            ZoomLevelText.Text = $"{(_zoomLevel * 100):F0}%";
            ApplyTransform();
        }

        private void ResetZoom()
        {
            _zoomLevel = 1.0;
            _translateX = 0;
            _translateY = 0;
            ApplyTransform();
            ZoomLevelText.Text = "100%";
        }

        private void ApplyTransform()
        {
            if (_scaleTransform != null)
            {
                _scaleTransform.ScaleX = _zoomLevel;
                _scaleTransform.ScaleY = _zoomLevel;
            }
            if (_translateTransform != null)
            {
                _translateTransform.X = _translateX;
                _translateTransform.Y = _translateY;
            }
        }

        private void BrowseFolder_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog();
            dialog.Description = "选择图片目录";
            // 设置默认选中目录为 .\outputs
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string defaultDir = System.IO.Path.Combine(baseDir, "outputs");

            if (System.IO.Directory.Exists(defaultDir))
            {
                dialog.SelectedPath = defaultDir;
            }
            else
            {
                dialog.SelectedPath = _currentDirectory; // 如果不存在则回退到当前目录
            }

            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                _currentDirectory = dialog.SelectedPath;
                CurrentPathText.Text = $"当前目录：{_currentDirectory}";
                LoadImages(_currentDirectory);
            }
        }

        private void BrowseComfyuiFolder_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog();
            dialog.Description = "选择图片目录";

            // 设置默认选中目录为 .\ComfyUI_windows_portable\ComfyUI\output
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string defaultDir = System.IO.Path.Combine(baseDir, "ComfyUI_windows_portable", "ComfyUI", "output");

            if (System.IO.Directory.Exists(defaultDir))
            {
                dialog.SelectedPath = defaultDir;
            }
            else
            {
                dialog.SelectedPath = _currentDirectory; // 如果不存在则回退到当前目录
            }

            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                _currentDirectory = dialog.SelectedPath;
                CurrentPathText.Text = $"当前目录：{_currentDirectory}";
                LoadImages(_currentDirectory);
            }
        }


        private void Refresh_Click(object sender, RoutedEventArgs e)
        {
            LoadImages(_currentDirectory);
        }

        private void Minimize_Click(object sender, RoutedEventArgs e)
        {
            this.WindowState = WindowState.Minimized;
        }

        private void Maximize_Click(object sender, RoutedEventArgs e)
        {
            if (this.WindowState == WindowState.Maximized)
                this.WindowState = WindowState.Normal;
            else
                this.WindowState = WindowState.Maximized;
        }

        private void Close_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }

        private void TopBar_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ButtonState == MouseButtonState.Pressed)
                this.DragMove();
        }

        private void ImageScrollViewer_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            // 如果按住Ctrl键，则缩放
            if (Keyboard.Modifiers == ModifierKeys.Control)
            {
                e.Handled = true;
                if (e.Delta > 0)
                    ZoomIn_Click(sender, e);
                else
                    ZoomOut_Click(sender, e);
            }
        }

        private void MainImage_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            // 滚轮缩放
            e.Handled = true;
            if (e.Delta > 0)
                ZoomIn_Click(sender, e);
            else
                ZoomOut_Click(sender, e);
        }

        private void MainImage_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ButtonState == MouseButtonState.Pressed)
            {
                _isDragging = true;
                _startPoint = e.GetPosition(ImageScrollViewer);
                MainImage.CaptureMouse();
                e.Handled = true;
            }
        }

        private void MainImage_MouseMove(object sender, MouseEventArgs e)
        {
            if (_isDragging)
            {
                var currentPoint = e.GetPosition(ImageScrollViewer);
                var deltaX = currentPoint.X - _startPoint.X;
                var deltaY = currentPoint.Y - _startPoint.Y;

                _translateX += deltaX;
                _translateY += deltaY;

                _startPoint = currentPoint;
                ApplyTransform();
            }
        }

        private void MainImage_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            _isDragging = false;
            MainImage.ReleaseMouseCapture();
        }
    }
}
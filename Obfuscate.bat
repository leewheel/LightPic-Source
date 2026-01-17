@echo off
chcp 65001 >nul
echo ============================================
echo   Z-Image-Launcher 代码混淆工具
echo ============================================
echo.

REM 检查是否已安装 Obfuscar
where obfuscar >nul 2>&1
if errorlevel 1 (
    echo [安装中] 正在安装 Obfuscar 混淆器...
    dotnet tool install -g Obfuscar
    echo.
    echo [完成] Obfuscar 已安装！
    echo.
)

REM 切换到脚本所在目录
cd /d "%~dp0"

echo [步骤1] 正在构建 Release 版本...
dotnet build -c Release
if errorlevel 1 (
    echo [错误] 构建失败！
    pause
    exit /b 1
)
echo.

echo [步骤2] 正在执行代码混淆...
echo.

REM 创建输出目录
if not exist "bin\Release\net8.0-windows\Protected" mkdir "bin\Release\net8.0-windows\Protected"

REM 执行混淆
obfuscar -c Release -p Z-Image-Launcher.csproj

if exist "bin\Release\net8.0-windows\Protected\Z-Image-Launcher.exe" (
    echo.
    echo ============================================
    echo   混淆完成！
    echo ============================================
    echo.
    echo 受保护的程序: bin\Release\net8.0-windows\Protected\Z-Image-Launcher.exe
    echo 混淆映射文件: bin\Release\net8.0-windows\Protected\Mapping.txt
    echo.
    echo [警告] 分发时请使用 Protected 文件夹中的 .exe 文件！
    echo [警告] 请妥善保存 Mapping.txt 用于调试！
    echo.
) else (
    echo [错误] 混淆失败，请检查配置！
    pause
    exit /b 1
)

pause

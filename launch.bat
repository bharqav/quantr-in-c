@echo off
title ⚡ Quantr Baremetal AI Engine
color 0B
cls

echo ==============================================================================
echo                 ⚡ QUANTR BAREMETAL AI VIRTUAL TERMINAL
echo ==============================================================================
echo.

cd /d "%~dp0"

:: Check if binaries exist, if not build with make or cmake if available
if not exist "inference.exe" (
    echo [INFO] Compiling Quantr Baremetal Engine...
    where gcc >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        make all
    ) else (
        echo [ERROR] GCC compiler not found in PATH. Please ensure GCC is installed.
        pause
        exit /b 1
    )
)

:: Launch Quantr Virtual Machine Terminal
if exist "quantr.exe" (
    quantr.exe %*
) else (
    inference.exe --vm %*
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo Engine exited with status %ERRORLEVEL%.
    pause
)

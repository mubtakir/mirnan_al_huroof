@echo off
chcp 65001 >nul
title مرنان - واجهة الأسئلة التفاعلية
set "JULIA_EXE=C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe"
set "PROJECT_DIR=%~dp0"
set "SCRIPT_PATH=%~dp0ask_mirnan.jl"

"%JULIA_EXE%" --project="%PROJECT_DIR%" "%SCRIPT_PATH%"
pause

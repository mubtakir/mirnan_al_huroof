@echo off
cd /d "%~dp0"
chcp 65001 >nul
title Mirnan V8 - Server
echo ==================================================
echo   Mirnan V8 - Starting Server
echo ==================================================
echo.
set "JULIA_EXE=C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe"
if "%JULIA_NUM_THREADS%"=="" set "JULIA_NUM_THREADS=auto"
echo Julia: "%JULIA_EXE%"
echo Threads: %JULIA_NUM_THREADS%
echo ==================================================
"%JULIA_EXE%" --threads=%JULIA_NUM_THREADS% --project=. api_server.jl
if %errorlevel% equ 0 goto success
echo.
echo ERROR: Server failed to start.
pause
exit /b 1
:success
pause

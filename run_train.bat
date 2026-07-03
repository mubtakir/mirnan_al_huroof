@echo off
setlocal EnableExtensions

chcp 65001 >nul

set "MIRNAN_DIR=%~dp0"
for %%I in ("%MIRNAN_DIR%..\..") do set "ROOT=%%~fI"
cd /d "%ROOT%"

set "JULIA_EXE=C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe"
if not exist "%JULIA_EXE%" (
    echo ERROR: Julia was not found at:
    echo   %JULIA_EXE%
    exit /b 1
)

set "MIRNAN_SEGMENT_LEVEL=paragraph"
set "MIRNAN_POST_SMOKE_HEAVY_GATES=0"
set "MIRNAN_BRIDGE_PROBE_LIMIT=200"
set "MIRNAN_BRIDGE_SCENE_ONLY_LIMIT=60"
set "MIRNAN_BRIDGE_PROBE_SECONDS=45"
set "MIRNAN_QUANTITY_TRAINED_PROBE_GATE_OFF=0"

echo ============================================================
echo Mirnan full training pipeline
echo Root: %ROOT%
echo Segment level: %MIRNAN_SEGMENT_LEVEL%
echo ============================================================

echo.
echo [1/6] Training Mirnan...
"%JULIA_EXE%" --project=models\mirnan models\mirnan\train.jl
if errorlevel 1 goto :failed

echo.
echo [2/6] Refreshing al_istinbat structured relation memory...
"%JULIA_EXE%" --project=models\mirnan models\mirnan\scripts\rebuild_istinbat_memory.jl
if errorlevel 1 goto :failed

echo.
echo [3/6] Running evolution...
"%JULIA_EXE%" --project=models\mirnan models\mirnan\run_evolution.jl
if errorlevel 1 goto :failed

echo.
echo [4/6] Rebuilding semantic scene memory...
"%JULIA_EXE%" --project=models\mirnan models\mirnan\scripts\build_semantic_scenes.jl
if errorlevel 1 goto :failed

echo.
echo [5/6] Rebuilding quantity memory...
"%JULIA_EXE%" --project=models\mirnan models\mirnan\scripts\rebuild_quantity_memory.jl
if errorlevel 1 goto :failed

echo.
echo [6/6] Running post-training smoke checks...
"%JULIA_EXE%" --project=models\mirnan models\mirnan\scripts\post_training_smoke.jl
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo Mirnan training pipeline completed successfully.
echo ============================================================
echo.
echo Press any key to close this window...
pause >nul
exit /b 0

:failed
echo.
echo ============================================================
echo ERROR: Mirnan training pipeline failed.
echo Check the last error above.
echo ============================================================
echo.
echo Press any key to close this window...
pause >nul
exit /b 1

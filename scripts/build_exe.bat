@echo off
setlocal
REM Build standalone Windows executables for GUI and pipeline using PyInstaller
REM Updated: fixed --paths to point to repository src, cleaned flags, removed stray '-' line.

set PYEXE=%~dp0\.venv\Scripts\python.exe
if not exist "%PYEXE%" (
  set PYEXE=python
)

echo [INFO] Using Python: %PYEXE%

REM Ensure build dependencies (PyInstaller) are available
%PYEXE% -m pip install -U pip
%PYEXE% -m pip install .[build]

REM Clean previous dist/build
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM ---------------- Pipeline CLI executable ----------------
REM Console application (keep stdout); --paths must include project src directory
%PYEXE% -m PyInstaller --noconfirm --onefile ^
  --name vt1-pipeline ^
  --paths "%~dp0..\src" ^
  --hidden-import vt1.pipeline.sam_general ^
  --hidden-import vt1.pipeline.sam_offline ^
  -s ^
  src\vt1\pipeline\sam_offline.py

if errorlevel 1 (
  echo [ERROR] Pipeline build failed
  exit /b 1
) else (
  echo [INFO] Pipeline exe built
)

REM ---------------- GUI executable ----------------
REM Windowed application (no console); include hidden imports for tabs
%PYEXE% -m PyInstaller --noconfirm --onefile ^
  --name vt1-gui ^
  --paths "%~dp0..\src" ^
  --hidden-import vt1.gui.pipeline_tab ^
  --hidden-import vt1.gui.clustering_tab ^
  --hidden-import vt1.gui.help_tab ^
  -w ^
  -s ^
  src\vt1\gui\main.py

if errorlevel 1 (
  echo [ERROR] GUI build failed
  exit /b 1
) else (
  echo [INFO] GUI exe built
)

echo.
echo Build finished. Find executables under .\dist\
endlocal

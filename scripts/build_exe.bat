@echo off
setlocal
REM Build standalone Windows executables for GUI and pipeline using PyInstaller
REM Updated: fixed --paths to point to repository src, cleaned flags, removed stray '-' line.
REM Added: set PYTORCH_JIT=0 to disable TorchScript JIT (avoids overload parse error under PyInstaller onefile)
REM Added: --collect-all torch to bundle full torch sources so inspect.getsource works if JIT toggled back.

set PYEXE=%~dp0\.venv\Scripts\python.exe
if not exist "%PYEXE%" (
  set PYEXE=python
)

REM Disable PyTorch JIT to prevent overload parser RuntimeError in frozen app
set PYTORCH_JIT=0

echo [INFO] Using Python: %PYEXE%

echo [INFO] PYTORCH_JIT=%PYTORCH_JIT%

REM Ensure build dependencies (PyInstaller) are available
%PYEXE% -m pip install -U pip
%PYEXE% -m pip install .[build]

REM Clean previous dist/build
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM ---------------- Pipeline CLI executable ----------------
REM Console application (keep stdout); --paths must include project src directory
REM Includes torch collection to embed full library sources.
echo.
echo [INFO] Building Pipeline executable (console mode)...
%PYEXE% -m PyInstaller --noconfirm --onefile ^
  --name vt1-pipeline ^
  --paths "%~dp0..\src" ^
  --additional-hooks-dir "%~dp0..\hooks" ^
  --runtime-hook "%~dp0..\hooks\rthook_suppress_torch_warnings.py" ^
  --hidden-import vt1.pipeline.sam_general ^
  --hidden-import vt1.pipeline.sam_offline ^
  --collect-all torch ^
  --noupx ^
  src\vt1\pipeline\sam_offline.py

if errorlevel 1 (
  echo [ERROR] Pipeline build failed with exit code %ERRORLEVEL%
  exit /b 1
) else (
  echo [SUCCESS] Pipeline exe built successfully
)

REM ---------------- GUI executable ----------------
REM Windowed application (no console); include hidden imports for tabs; collects torch.
echo.
echo [INFO] Building GUI executable (windowed mode)...
%PYEXE% -m PyInstaller --noconfirm --onefile ^
  --additional-hooks-dir "%~dp0..\hooks" ^
  --runtime-hook "%~dp0..\hooks\rthook_suppress_torch_warnings.py" ^
  --paths "%~dp0..\src" ^
  --hidden-import vt1.gui.pipeline_tab ^
  --hidden-import vt1.gui.clustering_tab ^
  --hidden-import vt1.gui.help_tab ^
  --hidden-import vt1.gui.startup_dialog ^
  --add-data "%~dp0..\GUI.md;." ^
  --add-data "%~dp0..\README.md;." ^
  --add-data "%~dp0..\config_defaults.toml;." ^
  --collect-all torch ^
  --noupx ^
  --windowed ^
  src\vt1\gui\main.py

if errorlevel 1 (
  echo [ERROR] GUI build failed with exit code %ERRORLEVEL%
  exit /b 1
) else (
  echo [SUCCESS] GUI exe built successfully
)

echo.
echo Build finished. Find executables under .\dist\
endlocal

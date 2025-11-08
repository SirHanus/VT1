@echo off
setlocal
REM Build standalone Windows executables for GUI and pipeline using PyInstaller

set PYEXE=%~dp0\.venv\Scripts\python.exe
if not exist "%PYEXE%" (
  set PYEXE=python
)

%PYEXE% -m pip install -U pip
%PYEXE% -m pip install .[build]

REM Clean previous dist/build
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM Build pipeline CLI exe
%PYEXE% -m PyInstaller --noconfirm --onefile ^
  --name vt1-pipeline ^
  --paths "%~dp0src" ^
  --hidden-import vt1.pipeline.sam_general ^
  --hidden-import vt1.pipeline.sam_offline ^
  -s ^
  -i NONE ^
  -y ^
  -q ^
  -F ^
  -w ^
  -c ^
  -r VERSION.txt ^
  -
  src\vt1\pipeline\sam_offline.py

REM Build GUI exe
%PYEXE% -m PyInstaller --noconfirm --onefile ^
  --name vt1-gui ^
  --paths "%~dp0src" ^
  --hidden-import vt1.gui.pipeline_tab ^
  --hidden-import vt1.gui.clustering_tab ^
  --hidden-import vt1.gui.help_tab ^
  -w ^
  src\vt1\gui\main.py

echo.
echo Build finished. Find executables under .\dist\
endlocal


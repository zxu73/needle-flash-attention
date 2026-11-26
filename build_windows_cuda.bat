@echo off
REM Windows build of the CUDA backend with nvcc + MSVC (stand-in for `make` on a GPU machine).
REM Run from anywhere: build_windows_cuda.bat   -- re-run after every change to src\ndarray_backend_cuda.cu
REM Needs the CUDA Toolkit (12.8+ for RTX 50-series GPUs) installed, which puts nvcc on PATH.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
if not defined PYTHON set PYTHON=python
where nvcc >nul 2>nul || (echo nvcc not found: install the CUDA Toolkit and open a new terminal & exit /b 1)
if not exist build mkdir build
for /f "delims=" %%i in ('call "%PYTHON%" -c "import sysconfig;print(sysconfig.get_paths()['include'])"') do set PY_INC=%%i
for /f "delims=" %%i in ('call "%PYTHON%" -c "import pybind11;print(pybind11.get_include())"') do set PYBIND_INC=%%i
for /f "delims=" %%i in ('call "%PYTHON%" -c "import sysconfig,os;print(os.path.join(sysconfig.get_config_var('installed_base'),'libs'))"') do set PY_LIBS=%%i
for /f "delims=" %%i in ('call "%PYTHON%" -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set EXT=%%i
REM -arch=native compiles for the GPU in this machine; cudart is linked statically so no DLL lookup is needed
nvcc -O2 -std=c++17 -arch=native --shared -cudart static ^
  -Xcompiler "/EHsc /utf-8" ^
  -I "%PY_INC%" -I "%PYBIND_INC%" ^
  src\ndarray_backend_cuda.cu ^
  -L "%PY_LIBS%" -Xlinker /IMPLIB:build\ndarray_backend_cuda.lib ^
  -o python\needle\backend_ndarray\ndarray_backend_cuda%EXT%

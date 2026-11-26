@echo off
REM Windows build of the CPU backend with MSVC (stand-in for `make`, which needs Linux/macOS).
REM Run from anywhere: build_windows.bat   -- re-run after every change to src\ndarray_backend_cpu.cc
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
if not defined PYTHON set PYTHON=python
if not exist build mkdir build
for /f "delims=" %%i in ('call "%PYTHON%" -c "import sysconfig;print(sysconfig.get_paths()['include'])"') do set PY_INC=%%i
for /f "delims=" %%i in ('call "%PYTHON%" -c "import pybind11;print(pybind11.get_include())"') do set PYBIND_INC=%%i
for /f "delims=" %%i in ('call "%PYTHON%" -c "import sysconfig,os;print(os.path.join(sysconfig.get_config_var('installed_base'),'libs'))"') do set PY_LIBS=%%i
for /f "delims=" %%i in ('call "%PYTHON%" -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set EXT=%%i
cl /nologo /LD /EHsc /O2 /std:c++17 /utf-8 /FI "%~dp0msvc_compat.h" ^
  /I "%PY_INC%" /I "%PYBIND_INC%" ^
  src\ndarray_backend_cpu.cc /Fobuild\ ^
  /link /LIBPATH:"%PY_LIBS%" /IMPLIB:build\ndarray_backend_cpu.lib /OUT:python\needle\backend_ndarray\ndarray_backend_cpu%EXT%

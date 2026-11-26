.PHONY: lib, pybind, clean, format, all

all: lib

ifeq ($(OS),Windows_NT)
# Windows: no cmake/gcc, so build with MSVC + nvcc via the batch scripts.
# Override the interpreter with: make PYTHON=C:\path\to\python.exe
SHELL := cmd.exe
PYTHON ?= python
export PYTHON

NVCC := $(shell where nvcc 2>nul)

lib:
	@.\build_windows.bat
ifneq ($(NVCC),)
	@.\build_windows_cuda.bat
else
	@echo nvcc not found, skipping CUDA backend
endif

format:
	$(PYTHON) -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	@if exist build rmdir /s /q build
	@del /q python\needle\backend_ndarray\ndarray_backend*.pyd 2>nul || ver >nul

else

lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so

endif

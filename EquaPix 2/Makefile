NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS ?= -O3 -std=c++17
LDLIBS ?=
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
HOST_SRC = src/image_io.cpp src/png_write.cpp src/cpu_reference.cpp src/metrics.cpp
CUDA_SRC = src/main.cu src/kernels.cu
HEADERS = $(wildcard src/*.h src/*.cuh)
.PHONY: all clean test gpu-test
all: equapix
equapix: $(CUDA_SRC) $(HOST_SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -Isrc $(CUDA_SRC) $(HOST_SRC) -o $@ $(LDLIBS)
build/test_host: tests/test_host.cpp $(HOST_SRC) $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Isrc tests/test_host.cpp $(HOST_SRC) -o $@ $(LDLIBS)
build/test_png_writer: tests/test_png_writer.cpp src/png_write.cpp $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Isrc tests/test_png_writer.cpp src/png_write.cpp -o $@ $(LDLIBS)
test: build/test_host build/test_png_writer
	./build/test_host
	./build/test_png_writer
	python3 tests/test_png_output.py
	python3 tests/test_dataset.py
gpu-test: equapix
	python3 tests/test_gpu.py
clean:
	rm -rf build equapix

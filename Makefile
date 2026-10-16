CXX ?= c++
CXXFLAGS = -std=c++17 -O3 -march=native -pthread
NVCC ?= nvcc

all: bin/play bin/train bin/test bin/exact

bin/%: src/%.cpp src/board.hpp src/ntuple.hpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $<

# GPU trainer (needs the CUDA toolkit)
gpu: bin/train_gpu
bin/train_gpu: src/train_gpu.cu src/board.hpp src/ntuple.hpp
	@mkdir -p bin
	$(NVCC) -std=c++17 -O3 -arch=native -Xcompiler -O3,-march=native -o $@ $<

clean:
	rm -rf bin

.PHONY: all gpu clean

CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= ../build
CXX = mpicxx

CXXFLAGS ?= -O2 -std=c++17
CPPFLAGS += -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include
LDFLAGS += -L$(NCCL_HOME)/lib -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(NCCL_HOME)/lib -Wl,-rpath,$(CUDA_HOME)/lib64
LDLIBS += -lnccl -lcudart

.PHONY: all clean

all: bodytime_nccl_test

bodytime_nccl_test: bodytime_nccl_test.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f bodytime_nccl_test

CC = gcc
CC_FLAGS = -O3

CUDA_CC = nvcc
CUDA_FLAGS = -arch=sm_86

ACPP=acpp 
ACPP_EXPLICIT=--acpp-targets="omp;cuda:sm_86"

ICPX=icpx
ICPX_FLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 

OMP_CC = nvc 
OMP_FLAGS = -fopenmp -mp=gpu -gpu=cc86 -Minfo=mp

LINK_FLAGS = -lcrypto

DEP_SRCS = md5sum.c

all: ma_seq ma_cuda ma_omp ma_acpp ma_acpp_explicit ma_icpx

ma_seq: seq.c ${DEP_SRCS}
	$(CC) $(CC_FLAGS) ${LINK_FLAGS} -o $@ $^

ma_cuda: cuda.cu ${DEP_SRCS}
	$(CUDA_CC) $(CC_FLAGS) $(CUDA_FLAGS) ${LINK_FLAGS} -o $@ $^

ma_omp: omp.c ${DEP_SRCS}
	$(OMP_CC) $(CC_FLAGS) $(OMP_FLAGS) ${LINK_FLAGS} -o $@ $^

ma_acpp: sycl.cpp ${DEP_SRCS}
	$(ACPP) $(CC_FLAGS) ${LINK_FLAGS} -o $@ $^

ma_acpp_explicit: sycl.cpp ${DEP_SRCS}
	$(ACPP) $(CC_FLAGS) $(ACPP_EXPLICIT) ${LINK_FLAGS} -o $@ $^

ma_icpx: sycl.cpp ${DEP_SRCS}
	$(ICPX) $(CC_FLAGS) $(ICPX_FLAGS) ${LINK_FLAGS} -o $@ $^

clean:
	rm -f ma_seq ma_cuda ma_omp ma_acpp ma_acpp_explicit ma_icpx
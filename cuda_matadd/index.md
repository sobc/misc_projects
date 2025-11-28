# Compiling and Running the CUDA Matrix Addition Example on the Turing Cluster

## Prerequisites

Set up the SSH connection to the Turing cluster as described in the [Labs Section](https://www.cs.uni-potsdam.de/bs/research/labs.html#login).
Make sure you have access. If not, please contact [me](mailto:mluebke@uni-potsdam.de).

## File description

There are four different implementations of matrix addition:

- [sequential](./seq.c)
- [cuda](./cuda.cu)
- [OpenMP w/ target offloading](./omp.c)
- [SYCL](./sycl.cpp)

In order to successfully compile the sources, you will need these additional
files:

- [matadd.h](./matadd.h)
- [md5sum.c](./md5sum.c)
- [Makefile](./Makefile)

## Setup your environment

### CUDA

You can compile and run the CUDA implementation using either one of the standalone CUDA modules or the NVIDIA HPC SDK.

```bash
module load nvhpc 
# or
module load cuda/11.6 # use for profiling, more on this in a separate file
#or 
module load cuda # latest
```

### OpenMP

I only tested the OpenMP implementation using the NVIDIA HPC SDK.

```bash
module load nvhpc
```

Unfortunately, if you plan to compile and optimize for one specific GPU
architecture, you have to manually set the `NVHPC_CUDA_HOME` environment
variable:

```bash
export NVHPC_CUDA_HOME=/mnt/beegfs/apps/nvhpc/24.5/Linux_x86_64/24.5/cuda/12.4/
```

### SYCL

You can choose between Intel's oneAPI and AdaptiveCpp. A big thank you to
Klemens for setting the latter up in such a short time!

#### oneAPI

In addition to the oneAPI compiler module, you also need to load the CUDA
module. Don't forget to use the `--auto` option with the `module load` command.

```bash
module load --auto compiler/latest cuda
```

#### AdaptiveCpp

Currently, the module file is missing a dependency to GCC. Therefore, you will
need to load it manually when loading the AdaptiveCpp module:

```bash
module load gcc/11.2.0 acpp 
``` 

## Compiling the code

Simply run `make` in the directory containing the `Makefile`. This will create
all executables, so make sure you have loaded all of the above-described
modules.

```bash
make 
```

You should now have the following binaries:

- `ma_seq` - Sequential implementation
- `ma_cuda` - CUDA implementation
- `ma_omp` - OpenMP implementation
- `ma_acpp` - SYCL implementation with AdaptiveCpp and generic target offloading (JIT compilation)
- `ma_explicit` - SYCL implementation with AdaptiveCpp and explicit CUDA target offloading (AOT compilation)
- `ma_icpx` - SYCL implementation with oneAPI

## Running the code

There is a reservation called `pram` on `hpc-node30` for the lecture exercises.
Feel free to use it! 

Additionally, be sure to request a GPU for your job:

```bash
srun --gres=gpu --reservation=pram ./<executable>
```

That's it! Enjoy developing and running your applications on the Turing cluster!
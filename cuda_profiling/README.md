# Profiling of GPU Code with NVIDIA Tools

For profiling your GPU kernels, you can use NVIDIA's `Nsight Systems` and
`Nsight Compute` tools. With these tools, you can analyze the performance of
your CUDA applications and identify bottlenecks. While `Nsight Systems` provides
a system-wide overview of your application's performance, `Nsight Compute`
offers detailed kernel-level analysis. 

Both of them provide a graphical user interface (GUI) as well as command-line
interfaces (CLI). If you want to use the GUI, you can use the GUI on the remote
system by setting up X11 forwarding with SSH. My suggestion is to
[download](https://developer.nvidia.com/tools-downloads/) and install `Nsight
Systems` and `Nsight Compute` on your local machine and then transfer the
profiling data from the remote system to your local machine for analysis.

## Setup environment

You can either use `cuda/11.6` or `nvhpc/24.5` modules to use the NVIDIA
profiling tools. 

**Do not use `cuda/12.0` as it will produce deadlocks when profiling.**

## Compiling your code for profiling 

For both tools, you don't need to add any special flags when compiling your CUDA
code. It is also possible to use the highest optimization level (e.g., `-O3` or
`-fast`).

```bash
nvcc -O3 -o my_cuda_app my_cuda_app.cu
```

The only exception is that if you want to examine the source code and the
utilization by the kernels in `Nsight Compute`, you should compile your code
with the `-lineinfo` flag. 

```bash
nvcc -o my_cuda_app my_cuda_app.cu -lineinfo
```

## Using Nsight Systems

To profile your CUDA application with `Nsight Systems`, you can use the
following command:

```bash
srun --gres=gpu nsys profile -o my_profile_report ./my_cuda_app
```

This will produce a report file named `my_profile_report.nsys-rep`. You can then
transfer this file to your local machine and open it with the `Nsight Systems`
GUI for analysis, or use `nsys stats` command to view a summary in the terminal:

```bash
nsys stats my_profile_report.nsys-rep
```

### Useful options for Nsight Systems

Useful options for `nsys stats` include:

- `--cuda-um-cpu-page-faults`: Show CUDA Unified Memory CPU page faults statistics.
- `--cuda-um-gpu-page-faults`: Show CUDA Unified Memory GPU page faults statistics.

See the [CLI Profile Command Switch Options](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profile-command-switch-options) for more details.

## Using Nsight Compute

### Basic usage

To profile a specific kernel in your CUDA application with `Nsight Compute`, you
can use the following command:

```bash
srun --gres=gpu ncu -o my_kernel_report ./my_cuda_app
```

This will produce a report file named `my_kernel_report.ncu-rep`. You can then
transfer this file to your local machine and open it with the `Nsight Compute`
GUI for detailed kernel analysis, or use `ncu --import` command to view a summary
in the terminal:

```bash
srun --gres=gpu ncu --import my_kernel_report.ncu-rep
```

### Profiling a specific kernel

As the `ncu` command profiles all kernels by default, which can take a long
time, you can specify a particular kernel to profile using the `--kernel-name`
option. For example:

```bash
srun --gres=gpu ncu --kernel-name my_kernel -o my_kernel_report ./my_cuda_app
```

Alternatively, if your currently using `Nsight Systems` to profile your
application, you can right click on the desired kernel in the timeline view and
select "Analyze the Selected Kernel with NVIDIA Nsight Compute" to generate a
CLI command for profiling that specific kernel.

### Profiling sets

Using `ncu --list-sets`, you can view the available profiling sets:

```bash
$ ncu --list-sets
---------- --------------------------------------------------------------------------- ------- -----------------
Identifier Sections                                                                    Enabled Estimated Metrics
---------- --------------------------------------------------------------------------- ------- -----------------
basic      LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution                  yes     144              
detailed   ComputeWorkloadAnalysis, LaunchStats, MemoryWorkloadAnalysis, MemoryWorkloa no      459              
           dAnalysis_Chart, Occupancy, SourceCounters, SpeedOfLight, SpeedOfLight_Roof                          
           lineChart, WorkloadDistribution                                                                      
full       ComputeWorkloadAnalysis, InstructionStats, LaunchStats, MemoryWorkloadAnaly no      613              
           sis, MemoryWorkloadAnalysis_Chart, MemoryWorkloadAnalysis_Tables, NumaAffin                          
           ity, Nvlink_Tables, Nvlink_Topology, Occupancy, PmSampling, SchedulerStats,                          
            SourceCounters, SpeedOfLight, SpeedOfLight_RooflineChart, WarpStateStats,                           
           WorkloadDistribution                                                                                 
nvlink     Nvlink, Nvlink_Tables, Nvlink_Topology                                      no      52               
pmsampling PmSampling, PmSampling_WarpStates                                           no      72               
roofline   SpeedOfLight, SpeedOfLight_HierarchicalDoubleRooflineChart, SpeedOfLight_Hi no      241              
           erarchicalHalfRooflineChart, SpeedOfLight_HierarchicalSingleRooflineChart,                           
           SpeedOfLight_HierarchicalTensorRooflineChart, SpeedOfLight_RooflineChart,                          
           WorkloadDistribution
```

The default profiling set is `basic`. I recommend using the `detailed` set for a more
comprehensive analysis. You can specify the profiling set using the `--set` option:

```bash
srun --gres=gpu ncu --set=detailed -o my_kernel_report ./my_cuda_app
```

### Analyzing source code

As mentioned earlier, if you want to analyze the source code and its
utilization, you should compile your code with the `-lineinfo` flag. Then, run `ncu` as
described above. In the `Nsight Compute` GUI, you will be able to see the source
annotations and their corresponding metrics in the `Source` tab of a kernel. If
you are running the GUI from your local machine, make sure to transfer the
source files along with the `.ncu-rep` file to your local machine to enable
source code analysis. You are then able to set the source file path mapping in
the settings.

## Compiling and running example code

Here you can find the `matadd` example from last week and the reproduction code
from Li et al., see
[08-Cuda-aware-MPI](https://www.cs.uni-potsdam.de/bs/teaching/docs/courses/ws2025/pr_am/folien/08-Cuda-aware-MPI.pdf#page=32).

Files:

- [matadd.cu](./matadd.cu)
- [managed_ubench.cu](./managed_ubench.cu)
- [file_writer.c](./file_writer.c)
- [file_writer.h](./file_writer.h)
- [CMakeLists.txt](./CMakeLists.txt)

As the example code uses CMake for building, you can create a build directory,
configure, and build the code as follows on the Turing Cluster:

```bash
module load cmake nvhpc # or cuda/11.6

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86"
make -j
```

Then, three executables will be created: `matadd`, `read16k`, and `write16k` in
the build directory. None of them require any command-line arguments. Feel free
to run profiling on any of these executables.

### Compiling with `-lineinfo` flag

If you want to compile the example code with the `-lineinfo` flag for source
code analysis in `Nsight Compute`, you can do so by:

1. Deleteing the current build directory
2. Creating a new build directory
3. Configuring CMake with the additional flag `-DCMAKE_CUDA_FLAGS="-lineinfo"`
   and leave out the `-DCMAKE_BUILD_TYPE=Release` flag. You might also set an
   optimization level, for example `-O2`. Furthermore, you can add
   `-DCMAKE_C_FLAGS="-O2"` to enable optimizations for the host code as well.
4. Building the code again

```bash
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_CUDA_FLAGS="-lineinfo -O2" -DCMAKE_C_FLAGS="-O2" -DCMAKE_CUDA_ARCHITECTURES="86"
make -j
```
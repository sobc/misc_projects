#!/bin/bash

#SBATCH --job-name=cuda_ubench
#SBATCH --output=cuda_ubench.out
#SBATCH --error=cuda_ubench.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --reservation=pram

module load nvhpc 

for exe in read*k; do
    echo "Running benchmark for $exe"
    nsys profile -o ${exe} \
        --force-overwrite=true \
        --cuda-um-gpu-page-faults=true \
        --cuda-um-cpu-page-faults=true \
        ./$exe
    nsys stats --force-overwrite=true ${exe}.nsys-rep > ${exe}_stats.txt
done

for exe in write*k; do
    echo "Running benchmark for $exe"
    nsys profile -o ${exe} \
        --force-overwrite=true \
        --cuda-um-gpu-page-faults=true \
        --cuda-um-cpu-page-faults=true \
        ./$exe
    nsys stats --force-overwrite=true ${exe}.nsys-rep > ${exe}_stats.txt
done
#!/bin/bash

#SBATCH --job-name=matmul_cuda
#SBATCH --output=matmul_cuda_out-%j.txt
#SBATCH --error=matmul_cuda_err-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --exclusive

module load gcc cuda 

exponent=14

echo "Running naive CUDA matrix multiplication..."
for i in {1..5}; do
    ./naive $exponent 
done

echo "Running naive CUDA matrix multiplication with transposing on GPU ..."
for i in {1..5}; do
    ./naive_gpu_transpose $exponent 
done

echo "Running naive CUDA matrix multiplication with transposing on Host ..."
for i in {1..5}; do
    ./naive_host_transpose $exponent 
done
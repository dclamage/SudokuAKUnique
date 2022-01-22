#!/bin/bash
nvcc -ptx -o kernel.ptx kernel.cu
cp kernel.ptx ../CPU/cuda/kernel.ptx

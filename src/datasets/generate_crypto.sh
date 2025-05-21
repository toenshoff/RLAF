#!/bin/bash

# Download and compile grainofsalt: https://github.com/msoos/grainofsalt
# We ran the following commands to generate out train, validation, and test instances.

# training
./grainofsalt --num 20000 --outputs 56 --base-shift 8 --crypto hitag2 --probBits 22 --karnaugh 8 --seed 0

# validation
./grainofsalt --num 200 --outputs 56 --base-shift 8 --crypto hitag2 --probBits 22 --karnaugh 8 --seed 1


# test
./grainofsalt --num 100 --outputs 56 --base-shift 8 --crypto hitag2 --probBits 20 --karnaugh 8 --seed 2
./grainofsalt --num 100 --outputs 56 --base-shift 8 --crypto hitag2 --probBits 15 --karnaugh 8 --seed 2
./grainofsalt --num 100 --outputs 56 --base-shift 8 --crypto hitag2 --probBits 10 --karnaugh 8 --seed 2


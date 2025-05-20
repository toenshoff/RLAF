#!/bin/bash

cd solvers/march
make clean
make
cd ../..


cd solvers/march_weighted
make clean
make
cd ../..


cd solvers/glucose/simp
make clean
make rs
cd ../../..

cd solvers/glucose_weighted/simp
make clean
make rs
cd ../../..

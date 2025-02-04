
cd ./examples/shear/
mpiexec -n 1   ../../cmake/build/lmp  -sf gpu -pk gpu 1 -in in.shear
cd ../../                                     
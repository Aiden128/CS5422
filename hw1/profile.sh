#!/usr/bin/env bash

cmake -DCMAKE_BUILD_TYPE=PROFILE ..
make -j
srun -n 4 -N 2 hw1 4 ../cases/01.in ./01.out
srun -n 15 -N 3 hw1 15 ../cases/02.in ./02.out
srun -n 28 -N 4 hw1 21 ../cases/03.in ./03.out
srun -n 1 -N 1 hw1 50 ../cases/04.in ./04.out
srun -n 12 -N 2 hw1 100 ../cases/05.in ./05.out
srun -n 10 -N 2 hw1 65536 ../cases/06.in ./06.out
srun -n 3 -N 3 hw1 12345 ../cases/07.in ./07.out
srun -n 36 -N 3 hw1 100000 ../cases/08.in ./08.out
srun -n 24 -N 2 hw1 99999 ../cases/09.in ./09.out
srun -n 36 -N 3 hw1 63942 ../cases/10.in ./10.out
srun -n 15 -N 3 hw1 15 ../cases/11.in ./11.out
srun -n 1 -N 1 hw1 1 ../cases/12.in ./12.out
srun -n 20 -N 2 hw1 20 ../cases/13.in ./13.out
srun -n 15 -N 3 hw1 12345 ../cases/14.in ./14.out
srun -n 21 -N 3 hw1 10059 ../cases/15.in ./15.out
srun -n 11 -N 1 hw1 54923 ../cases/16.in ./16.out
srun -n 20 -N 2 hw1 400000 ../cases/17.in ./17.out
srun -n 20 -N 2 hw1 400000 ../cases/18.in ./18.out
srun -n 20 -N 2 hw1 400000 ../cases/19.in ./19.out
srun -n 24 -N 2 hw1 11183 ../cases/20.in ./20.out
srun -n 24 -N 2 hw1 12347 ../cases/21.in ./21.out
srun -n 24 -N 2 hw1 54323 ../cases/22.in ./22.out
srun -n 24 -N 2 hw1 400009 ../cases/23.in ./23.out
srun -n 24 -N 2 hw1 400009 ../cases/24.in ./24.out
srun -n 24 -N 2 hw1 400009 ../cases/25.in ./25.out
srun -n 24 -N 2 hw1 400009 ../cases/26.in ./26.out
srun -n 24 -N 2 hw1 191121 ../cases/27.in ./27.out
srun -n 24 -N 2 hw1 1000003 ../cases/28.in ./28.out
srun -n 24 -N 2 hw1 1024783 ../cases/29.in ./29.out
srun -n 24 -N 2 hw1 64123483 ../cases/30.in ./30.out
srun -n 24 -N 2 hw1 64123487 ../cases/31.in ./31.out
srun -n 24 -N 2 hw1 64123513 ../cases/32.in ./32.out
srun -n 12 -N 1 hw1 536869888 ../cases/33.in ./33.out
srun -n 12 -N 1 hw1 536869888 ../cases/34.in ./34.out
srun -n 12 -N 1 hw1 536869888 ../cases/35.in ./35.out
srun -n 12 -N 1 hw1 536869888 ../cases/36.in ./36.out

rm *.out
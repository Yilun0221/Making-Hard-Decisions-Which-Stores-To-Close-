#!/bin/bash

#SBATCH --job-name=G_H
#SBATCH --output=G_H.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --mail-type=END
#SBATCH --mail-user=yilunx@uchicago.edu
#SBATCH --error=G_H.err

# Load Python and MPI modules
module load python/anaconda-2019.03
module unload intelmpi
module load intelmpi/2018.2.199+intel-18.0

mpirun -n 40 python3 4_PSO_2018_NYC_G_H.py
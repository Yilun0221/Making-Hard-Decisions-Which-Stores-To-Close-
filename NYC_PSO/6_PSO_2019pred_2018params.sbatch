#!/bin/bash

#SBATCH --job-name=6_PSO_2019pred_2018params
#SBATCH --output=6_PSO_2019pred_2018params.out
#SBATCH --ntasks=21
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --mail-type=END
#SBATCH --mail-user=yilunx@uchicago.edu
#SBATCH --error=6_PSO_2019pred_2018params.err

# Load Python and MPI modules
module load python/anaconda-2019.03
module unload intelmpi
module load intelmpi/2018.2.199+intel-18.0

mpirun -n 40 python3 6_PSO_2019pred_2018params.py
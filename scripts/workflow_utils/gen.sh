#!/bin/sh
#SBATCH --output=/b/home/pengfei_ji/person_logs/%A-%a-%x.out
#SBATCH --error=/b/home/pengfei_ji/person_logs/%A-%a-%x.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH -t 1-0
#SBATCH --array=0-8
#SBATCH --partition=cpu
srun -l python alpha_gen.py
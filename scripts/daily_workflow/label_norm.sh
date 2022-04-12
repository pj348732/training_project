#!/bin/sh
#SBATCH --output=/b/home/pengfei_ji/person_logs/%A-%a-%x.out
#SBATCH --error=/b/home/pengfei_ji/person_logs/%A-%a-%x.out
#SBATCH --mem-per-cpu=30G --ntasks=1
#SBATCH --array=0-99
#SBATCH -t 1-0
#SBATCH --partition=cpu
srun -l python /b/home/pengfei_ji/airflow/dags/sta_17001/scripts/daily_workflow/compute_norms.py $1

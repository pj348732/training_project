#!/bin/sh
#SBATCH --output=/b/home/pengfei_ji/person_logs/%A-%a-%x.out
#SBATCH --error=/b/home/pengfei_ji/person_logs/%A-%a-%x.out
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH -t 2-0
#SBATCH --gpus-per-task=1
#SBATCH --array=0-129
#SBATCH --partition=gpu
export PYTHONUNBUFFERED=1
srun -l python /b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/model_finetuner.py $1
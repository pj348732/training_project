#!/bin/sh
#SBATCH --output=/b/home/pengfei_ji/person_logs/%A-%a-%x.out
#SBATCH --error=/b/home/pengfei_ji/person_logs/%A-%a-%x.out
#SBATCH --mem=256G
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH -t 2-0
#SBATCH --gpus-per-task=1
#SBATCH --array=0-7
#SBATCH --partition=gpu
#SBATCH --constraint=rtx3090
export PYTHONUNBUFFERED=1
conda init bash
srun -l python /b/home/pengfei_ji/airflow/dags/sta_17001/scripts/weekly_workflow/model_pretrainer.py $1
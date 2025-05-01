#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --account=def-zhijing
#SBATCH --gres=gpu:1                 # request 1 GPU
#SBATCH --cpus-per-task=4           # optional: number of CPU cores
#SBATCH --mem=48G                # memory (RAM)
#SBATCH --time=00:30:00             # max runtime (hh:mm:ss)
#SBATCH --output=logs/%x-%j.out     # log file (%x = job name, %j = job ID)
#SBATCH --mail-user=1835928575qq@gmail.com
#SBATCH --mail-type=ALL

source venv/bin/activate
export $(cat .env | xargs)

python script/simulate.py

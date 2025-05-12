#!/bin/bash
#SBATCH --job-name=IPD_sim
#SBATCH --gres=gpu:rtx6000:2
#SBATCH --time=3:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --mail-user=xiaoo.zhang@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL

# Run
source venv/bin/activate
export PYTHONPATH=.

python script/simulate.py --log
#!/bin/bash
#SBATCH --job-name=IPD_sim
#SBATCH --gres=gpu:a40:2
#SBATCH --time=5:00:00
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --mail-user=xiaoo.zhang@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL

# Run
source venv/bin/activate
export PYTHONPATH=.

# python3 script/simulate.py --log --config base.yaml
python3 script/ipd.py --config prisoner_dilemma.yaml
# python3 script/simulate.py --log --config code_strat_toy.yaml
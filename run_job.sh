#!/bin/bash
#SBATCH --job-name=LLM_evolution_tournament
#SBATCH --account=aip-rgrosse
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --chdir=/project/aip-rgrosse/xiao215/agent-tournament
#SBATCH --mail-user=xiaoo.zhang@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL

# Run
source venv/bin/activate
export PYTHONPATH=.

python3 script/run_evolution.py --config prisoner_dilemma.yaml

# python3 script/run_evolution.py --config toy.yaml

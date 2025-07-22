srun --account=aip-rgrosse --time=1:00:00 --gres=gpu:2 --mem=48G -c 32 --chdir=/project/aip-rgrosse/xiao215/agent-tournament --pty bash
sinfo -N --Format=NodeHost,Partition,GresUsed,Gres,CPUsState
squeue -u $USER
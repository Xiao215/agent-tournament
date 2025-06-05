srun -c 16 --gres=gpu:a40:2 --partition=a40 --mem=48G --pty --time=1:00:00 bash
sinfo -N --Format=NodeHost,Partition,GresUsed,Gres,CPUsState
squeue -u $USER
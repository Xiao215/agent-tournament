srun -c 4 --gres=gpu:rtx6000:2 --partition=rtx6000 --mem=32G --pty --time=1:00:00 bash
export PYTHONPATH=.
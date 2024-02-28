#!/bin/bash
#SBATCH --chdir .
#SBATCH --account digital_humans
#SBATCH --time=48:00:00
#SBATCH -o /home/%u/slurm_output__%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=14G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dummy-env
# ...

# Run your experiment
python -c "import torch; print('Cuda available?', torch.cuda.is_available())"
python -c "import torch; torch.manual_seed(72); print(torch.randn((3,3)))"
# python my_script.py

echo "Done."
echo FINISHED at $(date)

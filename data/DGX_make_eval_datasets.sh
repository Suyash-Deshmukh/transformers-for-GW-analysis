#!/bin/bash
#SBATCH --job-name=generate_evaluation_datasets                 # Job name
#SBATCH --mail-user=suyash.deshmukh@vanderbilt.edu              # Email for updates
#SBATCH --mail-type=ALL                                         # Get all updates on email
#SBATCH --output=eval_generation_output.log                     # Standard output log file
#SBATCH --error=eval_generation_error.log                       # Standard error log file

#SBATCH --partition=interactive_gpu
#SBATCH --account=dsi_dgx_iacc
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1

#SBATCH --time=0-01:00:00                                       # Time limit (d-hh:mm:ss)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB

# Define the Singularity container path
CONTAINER_PATH="/data/p_dsi/ligo/gw_container.simg"

singularity exec --bind /nobackup/user/deshmus:/nobackup/user/deshmus --bind /data/p_dsi/ligo:/data/p_dsi/ligo  --nv  $CONTAINER_PATH python3 generate_eval_datasets.py
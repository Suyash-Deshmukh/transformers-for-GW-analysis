#!/bin/bash
#SBATCH --job-name=train_detection                          # Job name
#SBATCH --mail-user=suyash.deshmukh@vanderbilt.edu          # Email for updates
#SBATCH --mail-type=ALL                                     # Get all updates on email
#SBATCH --output=train_output.log                           # Standard output log file
#SBATCH --error=train_error.log                             # Standard error log file

#SBATCH --partition=interactive_gpu
#SBATCH --account=dsi_dgx_iacc
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1

#SBATCH --time=0-00:15:00                                   # Time limit (d-hh:mm:ss)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB

# Define the Singularity container path
CONTAINER_PATH="/data/p_dsi/ligo/gw_container.simg"

singularity exec --bind /nobackup/user/deshmus:/nobackup/user/deshmus --bind /data/p_dsi/ligo:/data/p_dsi/ligo  --nv  $CONTAINER_PATH python3 src/train.py --model_name "Mimi"
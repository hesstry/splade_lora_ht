#!/bin/bash
#SBATCH --job-name="basic_distil_r32_a32_two_epochs"
#SBATCH --output="./slurm_logs/basic_distil_r32_a32/basic_distil_r32_a32_two_epochs.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --account=csb185
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --mem=60G
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH -t 05:00:00

module purge
module load gpu
module load slurm
module load anaconda3/2021.05

# load conda env variables
__conda_setup="$('/cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/etc/profile.d/conda.sh" ]; then
        . "/cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/etc/profile.d/conda.sh"
    else
        export PATH="/cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-q4munrgvh7qp4o7r3nzcdkbuph4z7375/bin:$PATH"
    fi
fi
unset __conda_setup

# run our code
checkpoint_dir=experiments/lora_splade_ensemble_distil_monogpu_r32_a32_two_epochs/checkpoint
index_dir=experiments/lora_splade_ensemble_distil_monogpu_r32_a32_two_epochs/index
out_dir=experiments/lora_splade_ensemble_distil_monogpu_r32_a32_two_epochs/out 

conda activate splade_env

# torchrun \
#     --nproc_per_node number_of_gpu_you_have path_to_script.py \
# 	--all_arguments_of_the_script
# when loading from saved checkpoint, do full path
# SPLADE_CONFIG_FULLPATH=/expanse/lustre/projects/csb185/thess/splade/experiments/lora_splade_ensemble_distil_monogpu_r32_a32/checkpoint/config.yaml
# torchrun --nproc_per_node 2 -m splade.hf_train --config-name=config_lora_splade_r32_a32_max_distil.yaml
torchrun --nproc_per_node 2 -m splade.hf_train --config-name=config_lora_splade_r32_a32_two_epochs.yaml
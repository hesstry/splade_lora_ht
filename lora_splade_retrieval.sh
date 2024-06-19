#!/bin/bash
#SBATCH --job-name="retrieve_msmarco_lora_splade_max_distil"
#SBATCH --output="./slurm_logs/max_distil/retrieve_msmarco_lora_splade_max_distil.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --account=csb185
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH -t 40:00:00

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
# checkpoint_dir=experiments/lora_splade_ensemble_distil_monogpu_r8_a8/checkpoint
# index_dir=experiments/lora_splade_ensemble_distil_monogpu_r8_a8/index
# out_dir=experiments/lora_splade_ensemble_distil_monogpu_r8_a8/out
checkpoint_dir=experiments/lora_splade_r32_a32_bs16_16neg_distil_lr8e5_sl256/checkpoint
index_dir=experiments/lora_splade_r32_a32_bs16_16neg_distil_lr8e5_sl256/index
out_dir=experiments/lora_splade_r32_a32_bs16_16neg_distil_lr8e5_sl256/out
conda activate splade_env
python -m splade.retrieve --config-dir=$checkpoint_dir --config-name=config config.index_dir=$index_dir config.out_dir=$out_dir
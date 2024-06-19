#!/bin/bash
#SBATCH --job-name="beir_eval_lora_splade_r32_a32_bs16_16neg_distil_lr8e5_sl256"
#SBATCH --output="./slurm_logs/max_distil/beir_eval_lora_splade_r32_a32_bs16_16neg_distil_lr8e5_sl256.%j.%N.out"
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
# config_dir=/expanse/lustre/projects/csb185/thess/splade/experiments/lora_splade_ensemble_distil_monogpu_r64_a64/checkpoint/config.yaml
# index_dir=/expanse/lustre/projects/csb185/thess/splade/experiments/lora_splade_ensemble_distil_monogpu_r64_a64/index
# out_dir=/expanse/lustre/projects/csb185/thess/splade/experiments/lora_splade_ensemble_distil_monogpu_r64_a64/out
config_dir=/expanse/lustre/projects/csb185/thess/splade/experiments/lora_splade_r32_a32_bs16_16neg_distil_lr8e5_sl256/checkpoint/config.yaml
index_dir=/expanse/lustre/projects/csb185/thess/splade/experiments/lora_splade_r32_a32_bs16_16neg_distil_lr8e5_sl256/index
out_dir=/expanse/lustre/projects/csb185/thess/splade/experiments/lora_splade_r32_a32_bs16_16neg_distil_lr8e5_sl256/out
conda activate splade_env
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_FULLPATH=$config_dir
for dataset in arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq
do
    python3 -m splade.beir_eval \
      +beir.dataset=$dataset \
      +beir.dataset_path=data/beir \
      config.index_retrieve_batch_size=100
done
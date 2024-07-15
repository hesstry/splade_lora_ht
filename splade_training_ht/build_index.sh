#!/bin/bash
#SBATCH -A csb185
#SBATCH --job-name="build_index"
#SBATCH --output="./output/qd/build_index.%j.%N.out"
#SBATCH --error="./output/qd/build_index.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH -t 10:00:00

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

conda activate splade_env

# C_EMBS_OUTPUT from embed_docs.sh
# C_EMBS_OUTPUT=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/doc_embs
C_EMBS_OUTPUT=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/qd/d_embs

# OUTPUT_PREFIX
# OUTPUT_PREFIX=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/doc_embs/index
OUTPUT_PREFIX=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/qd/index

# NUM_FILES
NUM_FILES=89

/bin/bash -c "python buildIndex.py $C_EMBS_OUTPUT $OUTPUT_PREFIX $NUM_FILES"
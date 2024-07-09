#!/bin/bash
#SBATCH -A csb185
#SBATCH --job-name="finetuning"
#SBATCH --output="./output/plus_mean/finetuning-%j.out"
#SBATCH --error="./output/plus_mean/finetuning-%j.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 48:00:00

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
# /bin/bash -c "python -m train_splade --model_name naver/efficient-splade-VI-BT-large-doc --lambda_q 0.01 --lambda_d 0.008 --train_batch_size 32 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type marginmse --num_negs_per_system 20 --lr 1e-5 --continues --training_queries ../../msmarco_yingrui/train_queries_distill_splade_colbert_0.json"
# /bin/bash -c "python -m train_splade --model_name /expanse/lustre/projects/csb176/yryang/splade_cls/training_with_sentence_transformers/output_ckl/splade_distill_num1_kldiv_position_focal_mrr_diff_gamma5.0-alpha1.0_denoiseFalse_num20_kldiv_position_focal5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-24/25000/0_MLMTransformer --train_batch_size 32 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type marginmse --num_negs_per_system 20 --lr 1e-5 --continues --training_queries ../../msmarco_yingrui/train_queries_distill_splade_colbert_0.json"
# model_path=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/checkpoints/warmup_Splade_0_MLMTransformer
# model_path=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/55000/0_SpladeThresholding
model_path=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/60000/0_SpladeThresholding
train_queries=/expanse/lustre/projects/csb185/yifanq/msmarco/train_queries_distill_splade_colbert_0.json
# state_dict_path=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/state_dict.pt
# state_dict_path=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/55000/state_dict.pt
state_dict_path=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/testing_model_refactoring/60000/state_dict.pt
# save_path=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/qd
save_path=/expanse/lustre/projects/csb185/thess/splade/splade_training_ht/output/plus_mean
# /bin/bash -c "python -m train_splade --model_name  ../msmarco/warmup_Splade_0_MLMTransformer --train_batch_size 32 --accum_iter 4 --epochs 3000000 --warmup_steps 6000 --loss_type marginmse --continues --num_negs_per_system 20 --training_queries ../msmarco/train_queries_distill_splade_colbert_0.json"
/bin/bash -c "python -m train_splade --model_name $model_path --train_batch_size 32 --accum_iter 4 --epochs 40000 --warmup_steps 6000 --loss_type marginmse --continues --num_negs_per_system 20 --training_queries $train_queries --thresholding plus_mean --save_path $save_path --checkpoint --state_dict_path $state_dict_path"
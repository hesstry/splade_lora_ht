#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="treverse"
#SBATCH --output="treverse.%j.%N.out"
#SBATCH --error="treverse.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=4
#SBATCH --mem-per-gpu=32G
#SBATCH -t 48:00:00

/bin/bash -c "python -m train_distill_cls --model_name ../../../../csb175/yzound/re_splade/splade_param/150000/0_MLMTransformer --train_batch_size 64 --lambda_d 0.008 --lambda_q 0.01 --epochs 3000000 --warmup_steps 1000 --negs_to_use splade --loss_type marginmse --num_negs_per_system 10 --denoise --dense_weight 0.2"

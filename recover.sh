#!/bin/sh
#SBATCH -J recover0
#SBATCH -o recover0.out
#SBATCH -e recover0.err
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:3


python recovery.py \
--T_path target_model/target_ckp/VGG16_87.00_allclass.tar \
--G_path improvedGAN/improved_celeba_G.tar \
--D_path improvedGAN/improved_celeba_D.tar \
--save_dir not_pruned
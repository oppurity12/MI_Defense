#!/bin/sh
#SBATCH -J gan2
#SBATCH -o gan2.out
#SBATCH -e gan2.err
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:3


python k+1_gan.py --root imporved_GAN_VGG16_86.34_pruned_0.40 --path_T target_model/target_ckp/VGG16_86.34_pruned_0.40.tar



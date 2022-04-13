#!/bin/sh
#SBATCH -J fid
#SBATCH -o fid.out
#SBATCH -e fid.err
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:3

python -m pytorch_fid res_all data/trainset --device cuda:1
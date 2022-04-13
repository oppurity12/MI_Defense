#!/bin/sh
#SBATCH -J pruning
#SBATCH -o pruning.out
#SBATCH -e pruning.err
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:3


python pruning.py


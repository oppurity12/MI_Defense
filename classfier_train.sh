#!/bin/sh
#SBATCH -J train_classifier
#SBATCH -o train_classifier.out
#SBATCH -e train_classifier.err
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:3


python train_classifier.py


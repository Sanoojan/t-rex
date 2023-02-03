#!/bin/bash
#SBATCH --job-name=trex-food
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=5
#SBATCH --partition=default-long



python -m tasks.cls-train >checkfood.out
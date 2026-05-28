#!/bin/bash
#SBATCH --job-name=hsln
#SBATCH --output=/projects/F202600026AIVLABDEUCALION/helenice/word2vec-deucalion/slurm_hsln.log
#SBATCH --error=/projects/F202600026AIVLABDEUCALION/helenice/word2vec-deucalion/slurm_hsln.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --account=f202600026aivlabdeucalionx
#SBATCH --partition=normal-x86

source ~/.bashrc
conda activate /projects/F202600026AIVLABDEUCALION/helenice/py37

cd /projects/F202600026AIVLABDEUCALION/helenice/word2vec-deucalion/HSLN-Joint-Sentence-Classification

/projects/F202600026AIVLABDEUCALION/helenice/py37/bin/python train.py > train_hsln.log 2>&1

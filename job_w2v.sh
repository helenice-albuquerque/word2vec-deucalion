#!/bin/bash
#SBATCH --job-name=word2vec
#SBATCH --output=slurm_w2v.log
#SBATCH --error=slurm_w2v.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --account=f202600026aivlabdeucalionx
#SBATCH --partition=normal-x86

cd /projects/F202600026AIVLABDEUCALION/helenice/word2vec-deucalion

source /projects/F202600026AIVLABDEUCALION/helenice/word2vec-deucalion/w2v_env/bin/activate

python -u word2vec_med.py >> train_w2v.log 2>&1

#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# python train.py trainer.max_epochs=5
# python train.py trainer.max_epochs=10 logger=csv

stage=0

data_dir=/home/dasein/Projects/MMD/data/EATD-Corpus

if [ $stage -eq 0 ]; then
    # Extract audio feats
    python src/utils/feats/audio_feats.py -i $data_dir
    # Extract text feats
    python src/utils/feats/text_feats.py -i $data_dir
fi

#!/bin/bash
declare -a margins=("1" "0.1" "0.2")
declare -a batchSizes=("1" "10" "50" "100")
declare -a learningRates=("1e-3" "1e-5")
#TODO: modify datadir and data indices file
dataDir="./sanity_test_hard"
dataIndicesFile="./sanity_test_hard/indices.idx"
minOverlap=".4"
imgType="unnorm_intensity"

#TODO: modify model file
modelFile="./models/d2_tf.pth"
numEpochs="20"
safeRadius="4"
validationSize="0.05"

for lr in "${learningRates[@]}"; do
    for margin in "${margins[@]}"; do
        for batch in "${batchSizes[@]}"; do
            # Ignore score edges
            #TODO: modify logdir
            python train.py --data_dir $dataDir \
                --data_indices_file $dataIndicesFile \
                --img_type $imgType \
                --min_overlap $minOverlap \
                --model_file $modelFile \
                --num_epochs $numEpochs \
                --lr $lr \
                --safe_radius $safeRadius \
                --margin $margin \
                --ignore_score_edges \
                --batch_size $batch \
                --use_validation \
                --validation_size $validationSize \
                --log_dir sanity_test_hard_margin_"$margin"_batch_"$batch"_ignore_score_edges

            # Don't ignore score edges
            #TODO: modify logdir
            python train.py --data_dir $dataDir \
                --data_indices_file $dataIndicesFile \
                --img_type $imgType \
                --min_overlap $minOverlap \
                --model_file $modelFile \
                --num_epochs $numEpochs \
                --lr $lr \
                --safe_radius $safeRadius \
                --margin $margin \
                --batch_size $batch \
                --use_validation \
                --validation_size $validationSize \
                --log_dir sanity_test_hard_margin_"$margin"_batch_"$batch"
        done
    done
done

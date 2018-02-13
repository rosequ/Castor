#!/bin/sh
seedArray=()
while IFS= read -r line; do seedArray+="$line "; done < $1

for seed in $seedArray
do
	nohup python -u main.py --mode non-static --lr 0.9 --output_channel 130 --seed $seed --coarse --batch 32 --dataset clean_trec --trained_model saves/clean_trec/non-static_0.9_130_5_coarse_${seed}_32_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ling_head_pos --seed $seed --coarse --lr 0.93 --output_channel 110 \
	--batch 64 --dataset clean_trec --trained_model saves/clean_trec/ling_head_pos_0.93_110_5_coarse_${seed}_64_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ling_head_dep --seed $seed --lr 0.93 --output_channel 90 --coarse --batch 64 --dataset clean_trec --trained_model saves/clean_trec/ling_head_dep_0.93_90_5_coarse_${seed}_64_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ling_head_nonstatic --seed $seed --lr 0.93 --output_channel 130 --coarse --batch 64 --dataset clean_trec --trained_model saves/clean_trec/ling_head_nonstatic_0.93_130_5_coarse_${seed}_64_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode num_best --seed $seed --lr 0.98 --output_channel 120 --coarse --batch 64 --dataset clean_trec --trained_model saves/clean_trec/num_best_0.98_120_5_coarse_${seed}_64_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ner_best --seed $seed --lr 0.93 --output_channel 100 --coarse \
	--batch 64 --dataset clean_trec --batch 64 --trained_model saves/clean_trec/ner_best_0.93_100_5_coarse_${seed}_64_best_model.pt &
	sleep 3
done


for seed in $seedArray
do
	nohup python -u main.py --mode ner_num_best --seed $seed --lr 0.95 --output_channel 120 --coarse --batch 32 --dataset clean_trec \
	--trained_model saves/clean_trec/ner_num_best_0.95_120_5_coarse_${seed}_32_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ner_dep --lr 0.84 --output_channel 90 --seed $seed --coarse --batch 64 --dataset clean_trec \
	--trained_model saves/clean_trec/ner_dep_0.84_90_5_coarse_${seed}_64_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode num_pos_dep --lr 0.91 --output_channel 110 --seed $seed --coarse --batch 32 --dataset clean_trec \
 	--trained_model saves/clean_trec/num_pos_dep_0.91_110_5_coarse_${seed}_32_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode num_dep --seed $seed --lr 0.86 --output_channel 110 --coarse --batch 64 --dataset clean_trec \
 	--trained_model saves/clean_trec/num_dep_0.86_110_5_coarse_${seed}_64_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ner_num_dep --seed $seed --lr 0.95 --output_channel 100 --coarse --batch 64 --dataset clean_trec \
 	--trained_model saves/clean_trec/ner_num_dep_0.95_100_5_coarse_${seed}_64_best_model.pt &
	sleep 3
done


for seed in $seedArray
do
	nohup python -u main.py --mode ner_num_pos_dep --seed $seed --lr 0.95 --output_channel 110 --coarse --batch 32 --dataset clean_trec \
	 --trained_model saves/clean_trec/ner_num_pos_dep_0.95_110_5_coarse_${seed}_32_best_model.pt &
	sleep 3
done


for seed in $seedArray
do
	nohup python -u main.py --mode ner_pos_dep --seed $seed --lr 0.88 --output_channel 90 --coarse --batch 32 --dataset clean_trec \
	--trained_model saves/clean_trec/ner_pos_dep_0.88_90_5_coarse_${seed}_32_best_model.pt &
	sleep 3 
done

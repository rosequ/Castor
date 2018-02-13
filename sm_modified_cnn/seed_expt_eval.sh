#!/bin/sh
seedArray=()
while IFS= read -r line; do seedArray+="$line "; done < $1

for seed in $seedArray
do
	nohup python -u main.py --mode ner_pos_dep --seed $seed --lr 1.0 --output_channel 130 --coarse \
	--trained_model saves/TREC/ner_pos_dep_1.0_130_5_coarse_${seed}_64_best_model.pt&
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode non-static --trained_model saves/TREC/non-static_0.95_100_5_coarse_${seed}_64_best_model.pt \
	--lr 0.95 --output_channel 100 --seed $seed --coarse&
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ling_head_pos --trained_model saves/TREC/ling_head_pos_0.97_90_5_coarse_${seed}_64_best_model.pt \
	--lr 0.97 --output_channel 90 --seed $seed --coarse&
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ling_head_dep --seed $seed --lr 0.9 --output_channel 120 --coarse \
	--trained_model saves/TREC/ling_head_dep_0.9_120_5_coarse_${seed}_64_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ling_head_nonstatic --seed $seed --lr 1.0 --output_channel 110 --coarse \
	--trained_model saves/TREC/ling_head_nonstatic_1.0_110_5_coarse_${seed}_64_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode num_best --seed $seed --lr 0.95 --output_channel 130 --coarse \
	--trained_model saves/TREC/num_best_0.95_130_5_coarse_${seed}_64_best_model.pt&
	sleep 3
done


for seed in $seedArray
do
	nohup python -u main.py --mode ner_num_best --seed $seed --lr 0.93 --output_channel 120 --coarse \
	--trained_model saves/TREC/ner_num_best_0.93_120_5_coarse_${seed}_64_best_model.pt&
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py  --mode ner_dep --lr 0.91 --output_channel 110 --seed $seed --coarse \
	--trained_model saves/TREC/ner_dep_0.91_110_5_coarse_${seed}_64_best_model.pt&
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode num_pos_dep --lr 0.95 --output_channel 90 --seed $seed --coarse \
	--trained_model saves/TREC/num_pos_dep_0.95_90_5_coarse_${seed}_64_best_model.pt &
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode num_dep --seed $seed --lr 1.0 --output_channel 120 --coarse \
	--trained_model saves/TREC/num_dep_1.0_120_5_coarse_${seed}_64_best_model.pt&
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ner_num_dep --seed $seed --lr 0.97 --output_channel 120 --coarse \
	--trained_model saves/TREC/ner_num_dep_0.97_120_5_coarse_${seed}_64_best_model.pt&
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ner_num_pos_dep --seed $seed --lr 0.93 --output_channel 90 --coarse \
	--trained_model saves/TREC/ner_num_pos_dep_0.93_90_5_coarse_${seed}_64_best_model.pt&
	sleep 3
done

for seed in $seedArray
do
	nohup python -u main.py --mode ner_best --seed $seed --lr 0.97 --output_channel 90 --coarse \
	--trained_model saves/TREC/ner_best_0.97_90_5_coarse_${seed}_64_best_model.pt&
	sleep 3
done 

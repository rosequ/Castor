#!/bin/sh
seedArray=()
while IFS= read -r line; do seedArray+="$line "; done < $1

for seed in $seedArray
do
		nohup python -u main.py --mode num_best --seed $seed --lr 0.97 --output_channel 90 --coarse \
		--trained_model saves/TREC/ner_best_0.97_90_5_coarse_${seed}_64_best_model.pt&
		sleep 5
done

for seed in $seedArray
do
    	nohup python -u main.py --mode non-static --lr 0.93 --output_channel 100 --seed $seed --coarse --patience 30 --dataset wiki \
    	--trained_model saves/wiki/non-static_0.93_100_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done

for seed in $seedArray
do
		nohup python -u main.py --mode ling_head_pos --seed $seed --coarse --lr 1.0 --output_channel 90 \
		--patience 30 --dataset wiki --trained_model saves/wiki/ling_head_pos_1.0_90_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done

for seed in $seedArray
do
		nohup python -u main.py --mode ling_head_dep --seed $seed --lr 0.97 --output_channel 90 --coarse --patience 30 --dataset wiki \
		--trained_model saves/wiki/ling_head_dep_0.97_90_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done

for seed in $seedArray
do
		nohup python -u main.py --mode ling_head_nonstatic --seed $seed --lr 0.96 --output_channel 110 --coarse --patience 30 --dataset wiki \
		--trained_model saves/wiki/ling_head_nonstatic_0.96_110_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done

for seed in $seedArray
do
		nohup python -u main.py --mode num_best --seed $seed --lr 1.0 --output_channel 130 --coarse --patience 30 --dataset wiki \
		--trained_model saves/wiki/num_best_1.0_130_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done

for seed in $seedArray
do
		nohup python -u main.py --mode num_best --seed $seed --lr 0.91 --output_channel 90 --coarse \
		--patience 30 --dataset wiki --trained_model saves/wiki/ner_best_0.91_90_5_coarse_${seed}_32_best_model.pt --batch 32&
		sleep 5
done


for seed in $seedArray
do
		nohup python -u main.py --mode ner_num_best --seed $seed --lr 0.92 --output_channel 90 --coarse --patience 30 --dataset wiki \
		--trained_model saves/wiki/ner_num_best_0.92_90_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done

for seed in $seedArray
do
    	nohup python -u main.py --mode ner_dep --lr 0.91 --output_channel 110 --seed $seed --coarse --patience 30 --dataset wiki \
    	--trained_model saves/wiki/ner_dep_0.91_110_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done

for seed in $seedArray
do
		nohup python -u main.py --mode num_pos_dep --lr 0.98 --output_channel 120 --seed $seed --coarse --patience 30 --dataset wiki \
		--trained_model saves/wiki/num_pos_dep_0.98_120_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done

for seed in $seedArray
do
		nohup python -u main.py --mode num_dep --seed $seed --lr 0.9 --output_channel 130 --coarse --patience 30 --dataset wiki \
		--trained_model saves/wiki/num_dep_0.9_130_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done

for seed in $seedArray
do
		nohup python -u main.py --mode ner_num_dep --seed $seed --lr 0.98 --output_channel 90 --coarse --patience 30 --dataset wiki \
		--trained_model saves/wiki/ner_num_dep_0.98_90_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done


for seed in $seedArray
do
		nohup python -u main.py --mode ner_num_pos_dep --seed $seed --lr 0.9 --output_channel 100 --coarse --patience 30 --dataset wiki \
		--trained_model saves/wiki/ner_num_pos_dep_0.9_100_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done


for seed in $seedArray
do
		nohup python -u main.py --mode ner_pos_dep --seed $seed --lr 0.97 --output_channel 100 --coarse --patience 30 --dataset wiki \
		--trained_model saves/wiki/ner_pos_dep_0.97_100_5_coarse_${seed}_64_best_model.pt &
		sleep 5
done

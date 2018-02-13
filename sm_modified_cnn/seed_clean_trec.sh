#!/bin/sh
seedArray=()
while IFS= read -r line; do seedArray+="$line "; done < $1

for seed in $seedArray
do
	if [ ! -f saves/TREC/non-static_0.95_100_5_coarse_${seed}_64_best_model.pt ]; then
    	nohup python -u train.py --mode non-static --lr 0.95 --output_channel 100 --seed $seed --coarse&
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/TREC/ling_head_pos_0.97_90_5_coarse_${seed}_64_best_model.pt ]; then
		nohup python -u train.py --mode ling_head_pos --seed $seed --coarse &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/TREC/ling_head_dep_0.9_120_5_coarse_${seed}_64_best_model.pt ]; then
		nohup python -u train.py --mode ling_head_dep --seed $seed --lr 0.9 --output_channel 120 --coarse &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/TREC/ling_head_nonstatic_1.0_110_5_coarse_${seed}_64_best_model.pt ]; then
		nohup python -u train.py --mode ling_head_nonstatic --seed $seed --lr 1.0 --output_channel 110 --coarse &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/TREC/num_best_0.95_130_5_coarse_${seed}_64_best_model.pt ]; then
		nohup python -u train.py --mode num_best --seed $seed --lr 0.95 --output_channel 130 --coarse &
		sleep 30
	fi
done


for seed in $seedArray
do
	if [ ! -f saves/TREC/ner_num_best_0.93_120_5_coarse_${seed}_64_best_model.pt ]; then
		nohup python -u train.py --mode ner_num_best --seed $seed --lr 0.93 --output_channel 120 --coarse &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/TREC/ner_dep_0.91_110_5_coarse_${seed}_64_best_model.pt ]; then
    	nohup python -u train.py --mode ner_dep --lr 0.91 --output_channel 110 --seed $seed --coarse&
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/TREC/num_pos_dep_0.95_90_5_coarse_${seed}_64_best_model.pt ]; then
		nohup python -u train.py --mode num_pos_dep --lr 0.95 --output_channel 90 --seed $seed --coarse &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/TREC/num_dep_1.0_120_5_coarse_${seed}_64_best_model.pt ]; then
		nohup python -u train.py --mode num_dep --seed $seed --lr 1.0 --output_channel 120 --coarse &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/TREC/ner_num_dep_0.97_120_5_coarse_${seed}_64_best_model.pt ]; then
		nohup python -u train.py --mode ner_num_dep --seed $seed --lr 0.97 --output_channel 120 --coarse &
		sleep 30
	fi
done


for seed in $seedArray
do
	if [ ! -f saves/TREC/ner_num_pos_dep_0.93_90_5_coarse_${seed}_64_best_model.pt ]; then
		nohup python -u train.py --mode ner_num_pos_dep --seed $seed --lr 0.93 --output_channel 90 --coarse &
		sleep 30
	fi
done


for seed in $seedArray
do
	if [ ! -f saves/TREC/ner_pos_dep_best_1.0_130_5_coarse_${seed}_64_best_model.pt ]; then
		nohup python -u train.py --mode ner_pos_dep --seed $seed --lr 1.0 --output_channel 130 --coarse &
		sleep 30
	fi
done


for seed in $seedArray
do
	if [ ! -f saves/TREC/ner_best_0.97_90_5_coarse_${seed}_64_best_model.pt ]; then
		nohup python -u train.py --mode ner_best --seed $seed --lr 0.97 --output_channel 90 --coarse &
		sleep 30
	fi
done


for seed in $seedArray
do
	if [ ! -f saves/clean_trec/non-static_0.9_130_5_coarse_${seed}_32_best_model.pt ]; then
		echo "non-static" $seed
    	nohup python -u train.py --mode non-static --lr 0.9 --output_channel 130 --seed $seed --coarse --batch 32 --dataset clean_trec &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/clean_trec/ling_head_pos_0.93_110_5_coarse_${seed}_64_best_model.pt ]; then
		echo "ling_head_pos"
		nohup python -u train.py --mode ling_head_pos --seed $seed --coarse --lr 0.93 --output_channel 110 \
		--batch 64 --dataset clean_trec &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/clean_trec/ling_head_dep_0.93_90_5_coarse_${seed}_64_best_model.pt ]; then
		echo "ling_head_dep" $seed
		nohup python -u train.py --mode ling_head_dep --seed $seed --lr 0.93 --output_channel 90 --coarse --batch 64 --dataset clean_trec &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/clean_trec/ling_head_nonstatic_0.93_130_5_coarse_${seed}_64_best_model.pt ]; then
		echo "ling_head_nonstatic"
		nohup python -u train.py --mode ling_head_nonstatic --seed $seed --lr 0.93 --output_channel 130 --coarse --batch 64 --dataset clean_trec &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/clean_trec/num_best_0.98_120_5_coarse_${seed}_64_best_model.pt ]; then
		echo "num_best"
		nohup python -u train.py --mode num_best --seed $seed --lr 0.98 --output_channel 120 --coarse --batch 64 --dataset clean_trec &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/clean_trec/ner_best_0.93_100_5_coarse_${seed}_64_best_model.pt ]; then
		echo "ner_best"
		nohup python -u train.py --mode ner_best --seed $seed --lr 0.93 --output_channel 100 --coarse \
		--batch 64 --dataset clean_trec --batch 64&
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/clean_trec/ner_num_best_0.95_120_5_coarse_${seed}_32_best_model.pt ]; then
		echo "ner_num_best"
		nohup python -u train.py --mode ner_num_best --seed $seed --lr 0.95 --output_channel 120 --coarse --batch 32 --dataset clean_trec &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/clean_trec/ner_dep_0.84_90_5_coarse_${seed}_64_best_model.pt ]; then
		echo "ner_dep"
    	nohup python -u train.py --mode ner_dep --lr 0.84 --output_channel 90 --seed $seed --coarse --batch 64 --dataset clean_trec &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/clean_trec/num_pos_dep_0.91_110_5_coarse_${seed}_32_best_model.pt ]; then
		echo "num_pos_dep"
		nohup python -u train.py --mode num_pos_dep --lr 0.91 --output_channel 110 --seed $seed --coarse --batch 32 --dataset clean_trec &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/clean_trec/num_dep_0.86_110_5_coarse_${seed}_64_best_model.pt ]; then
		echo "num_dep"
		nohup python -u train.py --mode num_dep --seed $seed --lr 0.86 --output_channel 110 --coarse --batch 64 --dataset clean_trec &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/clean_trec/ner_num_dep_0.95_100_5_coarse_${seed}_64_best_model.pt ]; then
		echo "ner_num_dep"
		nohup python -u train.py --mode ner_num_dep --seed $seed --lr 0.95 --output_channel 100 --coarse --batch 64 --dataset clean_trec &
		sleep 30
	fi
done


for seed in $seedArray
do
	if [ ! -f saves/clean_trec/ner_num_pos_dep_0.95_110_5_coarse_${seed}_32_best_model.pt ]; then
		echo "ner_num_pos_dep"
		nohup python -u train.py --mode ner_num_pos_dep --seed $seed --lr 0.95 --output_channel 110 --coarse --batch 32 --dataset clean_trec &
		sleep 30
	fi
done

for seed in $seedArray
do
	if [ ! -f saves/clean_trec/ner_pos_dep_0.88_90_5_coarse_${seed}_32_best_model.pt ]; then
		echo "ner_pos_dep"
		nohup python -u train.py --mode ner_pos_dep --seed $seed --lr 0.88 --output_channel 90 --coarse --batch 32 --dataset clean_trec &
		sleep 30
	fi
done

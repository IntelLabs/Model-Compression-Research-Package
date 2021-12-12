
omp_num_threads=24

# best_bert
OMP_NUM_THREADS=$omp_num_threads python run_squad.py --model_type bert --model_name_or_path $OUTPUT_DIR/dynamic_tinybert/checkpoint-best --do_sigopt_search --do_lower_case --data_dir $SQUAD_DIR --train_file train-v1.1.json --predict_file dev-v1.1.json --per_gpu_eval_batch_size 16 --max_seq_length 384 --doc_stride 128 --output_dir $OUTPUT_DIR/sigopt_search

# convert to csv
python -c "import run_squad; run_squad.convert_fmt('./sigopt-pareto-front.log')"


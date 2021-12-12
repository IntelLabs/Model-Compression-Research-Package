
omp_num_threads=24
# best_bert
OMP_NUM_THREAS=$omp_num_threads python run_squad.py --model_type bert --model_name_or_path models/best_bert --do_lower_case --data_dir squad --train_file train-v1.1.json --predict_file dev-v1.1.json --per_gpu_eval_batch_size 16 --max_seq_length 384 --doc_stride 128 --output_dir eval --measure_rate "(0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2)"

# best_tinybert
OMP_NUM_THREAS=$omp_num_threads python run_squad.py --model_type bert --model_name_or_path models/best_tinybert --do_lower_case --data_dir squad --train_file train-v1.1.json --predict_file dev-v1.1.json --per_gpu_eval_batch_size 16 --max_seq_length 384 --doc_stride 128 --output_dir eval --measure_rate "(0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2)"

# best_lat_tinybert
OMP_NUM_THREAS=$omp_num_threads python run_squad.py --model_type bert --model_name_or_path models/best_lat_tinybert --do_lower_case --data_dir squad --train_file train-v1.1.json --predict_file dev-v1.1.json --per_gpu_eval_batch_size 16 --max_seq_length 384 --doc_stride 128 --output_dir eval --measure_rate "(0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2)"

# convert to csv
python -c "import run_squad; run_squad.convert_fmt('./metrics.log')"

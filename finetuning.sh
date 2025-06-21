nohup \
accelerate launch --main_process_port 0 finetuning.py \
--model_path "/data/download-model/DeepSeek-R1-0528-Qwen3-8B" \
--dataset_path "data/train.csv" \
--output_dir "finetuned_model" \
--checkpoint_dir "checkpoints" \
--checkpoint_interval 50 \
--num_epochs 50 \
--batch_size 3 \
> finetuning.log \
2>&1 \
&
disown

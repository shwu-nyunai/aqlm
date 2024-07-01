export MODEL_PATH={model_name_or_path}
export QUANTIZED_MODEL_PATH={quantized_model_path}
export TOKENIZED_DATASET_PATH={tokenized_dataset_path}
export CACHE_DIR={cache_dir}
export SAVE_PATH={finetuned_quantized_model_path}
export SEQLEN={model_seq_len}

export WANDB_PROJECT=PV_TUNE_LLAMA_2
export WANDB_NAME=llama-2-7b-1x16gs16-pv

torchrun --nproc-per-node=$NUM_GPUS finetune_fsdp.py \
    --base_model $MODEL_PATH --quantized_model $QUANTIZED_MODEL_PATH  --monkeypatch_old_pickle \
    --model_seqlen=$SEQLEN --block_type LlamaDecoderLayer --limit_parallel_inits 4 \
    --load_dtype bfloat16 --amp_dtype bfloat16 --code_dtype uint16 \
    --straight_through_buffer_dtype float32 \
    --dataset_name=$TOKENIZED_DATASET_PATH --split none --seed 1337 \
    --preprocessing_chunk_length 100000 --cache_dir=$CACHE_DIR --trust_remote_code \
    --update_codes --update_codebooks_and_scales --update_non_quantized_parameters \
    --lamb --debias --lr 3e-4 --adam_beta1 0.9 --adam_beta2 0.95 \
    --code_lr 3e-3 --code_beta1 0.0 --code_beta2 0.95 --beam_size 1 --delta_decay 0 \
    --max_code_change_per_step 1e-2 --code_trust_ratio 1e-2 --code_selection_temperature 0 \
    --batch_size=256 --microbatch_size=8 --max_epochs 10 --gradient_checkpointing \
    --print_every_steps=1 --verbose_optimizer --wandb  --eval_every_steps=10 --keep_best_model \
    --save $SAVE_PATH --save_every_steps 100
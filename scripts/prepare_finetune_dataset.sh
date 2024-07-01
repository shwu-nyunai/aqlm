TARGET_MODEL={model_name_or_path}                  # path or huggingface id of the base model. Used to access the tokenizer.
SEQLEN={model_seq_len}                             # model-specific maximal sequence length, 4096 for llama2, 8192 for mistral
DATASET={dataset_name_or_path}                     # name of the dataset
OUTPUT_PATH={tokenized_dataset_path}               # path to save the tokenized dataset

CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/LLM OMP_NUM_THREADS=16 torchrun \
    --master-port 3456 \
    --nproc-per-node=1 finetune_fsdp.py \
    --base_model $TARGET_MODEL \
    --quantized_model ./doesnt_matter \
    --dtype bfloat16 \
    --block_type LlamaDecoderLayer \
    --dataset_name=$DATASET \
    --split train \
    --cache_dir=./cache_dir \
    --trust_remote_code \
    --model_seqlen=$SEQLEN \
    --preprocessing_num_workers=64 \
    --preprocessing_chunk_length 100000 \
    --save_dataset_and_exit $OUTPUT_PATH

tar -cvf tokenized_data_llama2.tar $OUTPUT_PATH    # optionally pack for distribution
export MODEL_PATH={model_name_or_path}                                      # path or huggingface id of the base model
export DATASET_PATH={dataset}                                               # name of the dataset
export MODEL_SEQLEN={model_seq_len}                                         # model-specific maximal sequence length, 4096 for llama2, 8192 for mistral
export NBITS_PER_CODEBOOK={nbits_per_codebook}
export GROUP_SIZE={group_size}                                              # this corresponds to having a single 16-bit codebook for 16-dimensional vectors

export BLOCKWISE_FINETUNE_EPOCHS={blockwise_finetune_epochs}                # set to 0 to disable blockwise finetuning during calibration

export CUDA_VISIBLE_DEVICES={cuda_visible_devices}                          # or e.g. 0,1,2,3
export SAVE_PATH={quantized_model_path}                                      # path to save the quantized model
export WANDB_PROJECT={wandb_project}
export WANDB_NAME={wandb_name}

python main.py \
    $MODEL_PATH \
    $DATASET_PATH \
    --nsamples=2048 \
    --val_size=256 \
    --model_seqlen=4096 \
    --num_codebooks=1 \
    --nbits_per_codebook=$NBITS_PER_CODEBOOK \
    --out_group_size=1 \
    --in_group_size=$GROUP_SIZE \
    --beam_size=1 \
    --relative_mse_tolerance=0.01 \
    --max_epochs=100 \
    --finetune_lr=1e-4 \
    --finetune_adam_beta1=0.90 \
    --finetune_adam_beta2=0.999 \
    --finetune_keep_best \
    --finetune_batch_size=64 \
    --local_batch_size=4 \
    --finetune_max_epochs=$BLOCKWISE_FINETUNE_EPOCHS \
    --finetune_early_stop=3 \
    --offload_activations \
    --save $SAVE_PATH \
    --resume

export ORIG_MODEL_PATH={model_name_or_path}
export MODEL_PATH={finetuned_quantized_model_path}
export CONVERTED_CHECKPOINT_PATH={converted_checkpoint_path}

python convert_legacy_model_format.py\
    --base_model $ORIG_MODEL_PATH\
    --pv_fsdp_dir $MODEL_PATH\
    --code_dtype int32 --load_dtype auto --quantized_model=./doesnt_matter \
    --save $CONVERTED_CHECKPOINT_PATH

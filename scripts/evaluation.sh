export CUDA_VISIBLE_DEVICES={cuda_visible_devices}  
export QUANTIZED_MODEL={quantized_model_path}
export MODEL_PATH={model_name_or_path}
export DATASET={dataset}
export WANDB_PROJECT={wandb_project}
export WANDB_NAME={wandb_name}
export TASKS={tasks}

# for 0-shot evals
python lmeval.py \
    --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=float16,parallelize=True \
    --tasks $TASKS \
    --batch_size <EVAL_BATCH_SIZE> \
    --aqlm_checkpoint_path $QUANTIZED_MODEL # if evaluating quantized model

# for 5-shot MMLU
python lmeval.py \
    --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=float16,parallelize=True \
    --tasks $TASKS \
    --batch_size <EVAL_BATCH_SIZE> \
    --num_fewshot 5 \
    --aqlm_checkpoint_path $QUANTIZED_MODEL # if evaluating quantized model
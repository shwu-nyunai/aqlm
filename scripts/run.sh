set -e

this_file_path=$(realpath $0)
ROOT=$(dirname $this_file_path)

convert_pv_to_hf="$ROOT/convert_pv_to_hf.sh"
evaluation="$ROOT/evaluation.sh"
model_calibration="$ROOT/model_calibration.sh"
prepare_finetune_dataset="$ROOT/prepare_finetune_dataset.sh"
pv_tuning_quantized_model="$ROOT/pv_tuning_quantized_model.sh"

divider() {
    echo "----------------------------------------------------------------------------------------------------"
    echo " Running $1"
    echo "----------------------------------------------------------------------------------------------------"
}

divider "$model_calibration"
bash $model_calibration 2>&1 | tee $ROOT/model_calibration.log
divider "$prepare_finetune_dataset"
bash $prepare_finetune_dataset 2>&1 | tee $ROOT/prepare_finetune_dataset.log
divider "$pv_tuning_quantized_model"
bash $pv_tuning_quantized_model 2>&1 | tee $ROOT/pv_tuning_quantized_model.log
divider "$convert_pv_to_hf"
bash $convert_pv_to_hf 2>&1 | tee $ROOT/convert_pv_to_hf.log
divider "$evaluation"
bash $evaluation 2>&1 | tee $ROOT/evaluation.log

# paths

this_file_path=$(realpath $0)
ROOT=$(dirname $(dirname $this_file_path))

# aqlm
AQLM=$ROOT/AQLM
AQLM_REQUIREMENTS=$AQLM/requirements.txt




# checks
system_check() {
    echo "Checking system requirements"
    nvcc --version || exit 1
    nvidia-smi || exit 1
    echo "System requirements are met."
}

main() {
    system_check || bash scripts/install-cuda-toolkit.sh
    pip install -r $AQLM_REQUIREMENTS
}



main 2>&1 | tee $ROOT/setup.log

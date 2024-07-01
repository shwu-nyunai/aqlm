from pathlib import Path

## Paths
ROOT = Path(__file__).parent
AQLM_ROOT = ROOT / "AQLM"

## Experiments
EXPERIMENTS_DIR = ROOT / "experiments"                                                                  # take from sys.argv or default to experiments
CACHE_DIR = EXPERIMENTS_DIR / ".cache"                                                                  # take from sys.argv or default to experiments/.cache

## Scripts
RUN_FILES_DIR = ROOT / "scripts"

MODEL_CALIBRATION = RUN_FILES_DIR / "model_calibration.sh"
PREPARE_FINETUNE_DATASET = RUN_FILES_DIR / "prepare_finetune_dataset.sh"
PV_TUNING_QUANTIZED_MODEL = RUN_FILES_DIR / "pv_tuning_quantized_model.sh"
CONVERT_PV_TO_HF = RUN_FILES_DIR / "convert_pv_to_hf.sh"
EVALUATION = RUN_FILES_DIR / "evaluation.sh"


# ===========================================
#                Scripts to Keys
# ===========================================

SCRIPT_TO_KEYS = {
    MODEL_CALIBRATION: {
        "model_name_or_path":               "meta-llama/Meta-Llama-3-8B",                               # model name/path
        "dataset":                          "pajama",
        "model_seq_len":                    "4096",
        "nbits_per_codebook":               "16",
        "group_size":                       "16",
        "blockwise_finetune_epochs":        "25",
        "cuda_visible_devices":             "0",
        "quantized_model_path":              "::quantized_model_path",                                  # path to save quantized model

        # wandb
        "wandb_project":                    "nouseMY_EXPS",
        "wandb_name":                       "nouseYOUR_EXP_NAME"
    },
    PREPARE_FINETUNE_DATASET: {
        "model_name_or_path":               "meta-llama/Meta-Llama-3-8B",                               # model name/path
        "model_seq_len":                    "4096",                                                     
        "dataset_name_or_path":             "togethercomputer/RedPajama-Data-1T-Sample",                # dataset name/path
        "tokenized_dataset_path":           "::tokenized_dataset_path",                                 # path to save tokenized dataset
    },
    PV_TUNING_QUANTIZED_MODEL: {
        "model_name_or_path":               "meta-llama/Meta-Llama-3-8B",                               # model name/path
        "finetuned_quantized_model_path":   "::finetuned_quantized_model_path",                         # path to save quantized + finetuned model
        "cache_dir":                        str(CACHE_DIR),
        "quantized_model_path":             "::quantized_model_path",                                   # quantized model path
        "tokenized_dataset_path":           "::tokenized_dataset_path",                                 # to be updated for each model
        "model_seq_len":                    "4096",                                                     
    },
    CONVERT_PV_TO_HF: {
        "model_name_or_path":               "meta-llama/Meta-Llama-3-8B",                               # model name/path
        "finetuned_quantized_model_path":   "::finetuned_quantized_model_path",                         # quantized + finetuned model path
        "converted_checkpoint_path":        "::converted_checkpoint_path",                              # path to save quantized + finetuned model in HF format
    },

    # update as needed
    EVALUATION: {
        "model_name_or_path":               "meta-llama/Meta-Llama-3-8B",                               # model name/path
        "converted_checkpoint_path":        "::converted_checkpoint_path",                              # path to save quantized + finetuned model in HF format
        "source_code_path":                 str(AQLM_ROOT),
        "batch_size":                       "32",
        "cuda_visible_devices":             "0",
        "dataset":                          "pajama",
        "wandb_project":                    "nouseMY_EXPS",
        "wandb_name":                       "nouseYOUR_EXP_NAME",
        "tasks":                            "::tasks",                                                  # list of tasks to evaluate format - task:nshot,task:nshot
    }


}


# ===========================================
#                   Commands
# ===========================================

EXPORT_HF_DATASETS_CACHE = """
export HF_DATASETS_CACHE="/path/to/another/directory"
"""


## follow repo structure:
#   experiments
#       .cache
#       tokenized_datasets
#           <model name>_<dataset name>_<seq len>
#       outputs_<model name>
#           baseline_model
#           quantized_model
#           finetuned_model
#           evaluation_results



# ===========================================
#                   Experiments
# ===========================================

EVALS = [
    # task, nshot
    ("winogrande", 0),
    ("arc_challenge", 25),
    ("mmlu", 5),
    ("gsm8k", 5),
    ("hellaswag", 10),
    ("truthfulqa", 5)
]

EXPERIMENTS = {
    "meta-llama/Meta-Llama-3-8B": {
        "evals": EVALS,
        "dataset": "pajama",
        "dataset_name_or_path": "togethercomputer/RedPajama-Data-1T-Sample",
    },
    "NousResearch/Hermes-2-Pro-Llama-3-8B": {
        "evals": EVALS,
        "dataset": "pajama",
        "dataset_name_or_path": "togethercomputer/RedPajama-Data-1T-Sample",
    },
    # "path/to/llama-3*": {
    #     "evals": EVALS,
    #     "dataset": "pajama",
    #     "dataset_name_or_path": "togethercomputer/RedPajama-Data-1T-Sample",
    # },
}


# ===========================================
#                  Utils
# ===========================================

def load_hf_model(model_name, save_dir = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if save_dir:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    
    return model, tokenizer


class ExperimentSpec:
    _paths_to_create = []
    _hf_models_to_load = []

    ## available attributes:

    # model_name: str
    # root_path: Path
    # baseline_model: Path
    # quantized_model: Path
    # finetuned_model: Path
    # evaluation_results: Path

    def __init__(self, root_path: Path, model_name_or_path: str | Path, expt_dict: dict):
        self.root_path = root_path

        # create root path if it doesn't exist
        if not root_path.exists():
            root_path.mkdir(exist_ok=True, parents=True)

        # check if model_name_or_path is a local path
        if Path(model_name_or_path).exists():
            self.baseline_model = model_name_or_path
            self.model_name = model_name_or_path.name
        else:
            self.model_name = model_name_or_path

            self.baseline_model = root_path / "baseline_model"
            self._paths_to_create.append(self.baseline_model)

            self._hf_models_to_load.append({
                "model_name": model_name_or_path,
                "save_dir": self.baseline_model
            })

        self.quantized_model = root_path / "quantized_model"
        self._paths_to_create.append(self.quantized_model)

        self.finetuned_model = root_path / "finetuned_model"
        self._paths_to_create.append(self.finetuned_model)

        self.evaluation_results = root_path / "evaluation_results"
        self._paths_to_create.append(self.evaluation_results)



    @staticmethod
    def load_hf_models():
        from pqdm.processes import pqdm
        pqdm(ExperimentSpec._hf_models_to_load, load_hf_model, n_jobs=4, argument_type="kwargs")

    @staticmethod
    def create_paths():
        from pqdm.processes import pqdm

        @staticmethod
        def mkdir(path: Path): 
            return path.mkdir(exist_ok=True, parents=True)

        pqdm(ExperimentSpec._paths_to_create, mkdir, n_jobs=4)

    def __str__(self):
        return f"Model: {self.model_name},\nRoot: {self.root_path},\nBaseline: {self.baseline_model},\nQuantized: {self.quantized_model},\nFinetuned: {self.finetuned_model},\nEvaluation: {self.evaluation_results}"


class Experiment:
    _experiment_root = EXPERIMENTS_DIR
    _cache_dir = CACHE_DIR
    _tokenized_datasets_dir = _experiment_root / "tokenized_datasets"


    @staticmethod
    def get_tokenized_dataset_path(model_name, dataset_name, seq_len):
        return Experiment._tokenized_datasets_dir / f"{model_name}_{dataset_name}_{seq_len}"

    @staticmethod
    def get_output_dir(model_name):
        return Experiment._experiment_root / f"outputs_{model_name}"

    @staticmethod
    def get_cache_dir():
        return Experiment._cache_dir


    def __init__(self, experiments_dict: dict):
        self.init_folder_structure()

        self.experiment_specs = []
        for model_name, expt_dict in experiments_dict.items():
            exp_name = model_name.replace("/", "-") if not Path(model_name).exists() else model_name.name
            exp_root = Experiment.get_output_dir(exp_name)

            self.experiment_specs.append(ExperimentSpec(exp_root, model_name, expt_dict))

        ExperimentSpec.create_paths()
        ExperimentSpec.load_hf_models()
    

    def init_folder_structure(self):
        for path in [self._experiment_root, self._cache_dir, self._tokenized_datasets_dir]:
            path.mkdir(exist_ok=True, parents=True)
        
    def __str__(self):
        return f"Model: {self.model_name}, Evals: {self.evals}, Dataset: {self.dataset_name_or_path}"


if __name__ == "__main__":

    experiment = Experiment(EXPERIMENTS)
    for spec in experiment.experiment_specs:
        print(spec)

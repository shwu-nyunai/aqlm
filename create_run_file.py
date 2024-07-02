from pathlib import Path
from typing import List, Dict, DefaultDict
from collections import defaultdict
from tqdm import tqdm

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
RUN = RUN_FILES_DIR / "run.sh"

# Constants
SPACE = chr(32)
INDENT = SPACE * 4
NEW_LINE = "\n"


# ===========================================
#                Scripts to Keys
# ===========================================

SCRIPT_TO_KEYS = {
    MODEL_CALIBRATION: {
        "model_name_or_path":               "::meta-llama/Meta-Llama-3-8B",                             # model name/path
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
        "model_name_or_path":               "::meta-llama/Meta-Llama-3-8B",                             # model name/path
        "model_seq_len":                    "4096",                                                     
        "dataset_name_or_path":             "togethercomputer/RedPajama-Data-1T-Sample",                # dataset name/path
        "tokenized_dataset_path":           "::tokenized_dataset_path",                                 # path to save tokenized dataset
    },
    PV_TUNING_QUANTIZED_MODEL: {
        "model_name_or_path":               "::meta-llama/Meta-Llama-3-8B",                             # model name/path
        "finetuned_quantized_model_path":   "::finetuned_quantized_model_path",                         # path to save quantized + finetuned model
        "cache_dir":                        str(CACHE_DIR),
        "quantized_model_path":             "::quantized_model_path",                                   # quantized model path
        "tokenized_dataset_path":           "::tokenized_dataset_path",                                 # to be updated for each model
        "model_seq_len":                    "4096",                                                     
    },
    CONVERT_PV_TO_HF: {
        "model_name_or_path":               "::meta-llama/Meta-Llama-3-8B",                             # model name/path
        "finetuned_quantized_model_path":   "::finetuned_quantized_model_path",                         # quantized + finetuned model path
        "converted_checkpoint_path":        "::converted_checkpoint_path",                              # path to save quantized + finetuned model in HF format
    },

    # update as needed
    EVALUATION: {
        "model_name_or_path":               "::meta-llama/Meta-Llama-3-8B",                             # model name/path
        "quantized_model_path":             "::quantized_model_path",                                   # quantized model path
        "converted_checkpoint_path":        "::converted_checkpoint_path",                              # path to save quantized + finetuned model in HF format
        "source_code_path":                 str(AQLM_ROOT),
        "batch_size":                       "32",
        "cuda_visible_devices":             "0",
        "dataset":                          "pajama",
        "wandb_project":                    "nouseMY_EXPS",
        "wandb_name":                       "nouseYOUR_EXP_NAME",
        "tasks":                            "::tasks",                                                  # list of tasks to evaluate format - task:nshot,task:nshot
    },
    RUN: {}
}

def get_default_config():
    config = {}
    for keys in SCRIPT_TO_KEYS.values():
        config.update(keys)
    return config




# ===========================================
#                   Commands
# ===========================================

EXPORT_HF_DATASETS_CACHE = f"""
export HF_DATASETS_CACHE="{str(CACHE_DIR)}"
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
        "batch_size": "32",
    },
    "NousResearch/Hermes-2-Pro-Llama-3-8B": {
        "evals": EVALS,
        "dataset": "pajama",
        "dataset_name_or_path": "togethercomputer/RedPajama-Data-1T-Sample",
        "batch_size": "32",
    },
    # "path/to/llama-3*": {
    #     "evals": EVALS,
    #     "dataset": "pajama",
    #     "dataset_name_or_path": "togethercomputer/RedPajama-Data-1T-Sample",
    #     "batch_size": "16",
    # },
}


# ===========================================
#                  Utils
# ===========================================

def is_path_empty(path: Path):
    return not bool(list(path.glob("*")))

def load_hf_model(model_name, save_dir = None):
    if save_dir and not is_path_empty(save_dir):
        return
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

    if save_dir:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    
    return model, tokenizer

def dumps_config(cfg, path):
    import json
    with open(path, "w") as f:
        json.dump({
            k: str(v) for k,v in cfg.items()
        }, f, indent=4)


class ExperimentSpec:
    _paths_to_create = []
    _hf_models_to_load = []

    ## available attributes:

    # config: dict
    # model_name: str
    # root_path: Path
    # baseline_model: Path
    # quantized_model: Path
    # finetuned_model: Path
    # evaluation_results: Path
    # converted_checkpoint_path: Path

    def __init__(self, root_path: Path, model_name_or_path: str | Path, expt_dict: dict):
        self.root_path = root_path
        self._repr = None

        config = get_default_config()
        tasks = expt_dict.pop("evals")
        tasks = ",".join([f"{task}:{nshot}" for task, nshot in tasks])
        config.update(expt_dict)
        config["tasks"] = tasks
        self.config = config

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

        self.converted_checkpoint = root_path / "converted_checkpoint"
        self._paths_to_create.append(self.converted_checkpoint)

        self.scripts = root_path / "scripts"
        self._paths_to_create.append(self.scripts)

    @staticmethod
    def load_hf_models():
        from pqdm.processes import pqdm
        pqdm(ExperimentSpec._hf_models_to_load, load_hf_model, n_jobs=4, argument_type="kwargs")

    @staticmethod
    def create_paths():
        for path in ExperimentSpec._paths_to_create:
            if not path.exists():
                path.mkdir(exist_ok=True, parents=True)

    @property
    def model_type(self):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.baseline_model)
        return config.model_type

    @property
    def dataset_name_or_path(self):
        assert not Path(self.config["dataset_name_or_path"]).exists(), "Dataset path should be a HuggingFace dataset name"
        return self.config["dataset_name_or_path"].replace("/", "-")
    
    @property
    def model_seq_len(self):
        return str(self.config["model_seq_len"])

    @property
    def tokenized_dataset_path(self):
        # assumption: tokenizer is same for all models of the same type
        return str(Experiment.get_tokenized_dataset_path(self.model_type, self.dataset_name_or_path, self.model_seq_len))

    @property    
    def quantized_model_path(self):
        return str(self.quantized_model)

    @property
    def finetuned_quantized_model_path(self):
        return str(self.finetuned_model)

    @property
    def converted_checkpoint_path(self):
        return str(self.converted_checkpoint)

    @property
    def model_name_or_path(self):
        return self.baseline_model

    def update_config_paths(self):
        for k,v in self.config.items():
            if "::" in v:
                self.config[k] = getattr(self, v[2:]) if hasattr(self, v[2:]) else getattr(self, k)
        self._repr = None # reset repr
    
    def build_scripts(self):
        for script, keys in SCRIPT_TO_KEYS.items():
            script_path = self.scripts / f"{script.name}"
            with open(str(script), "r") as f:
                content = f.read()
                content = content.format(**{
                    k: f'"{v}"' for k, v in self.config.items()
                    if k in keys
                }) if keys else content

            with open(script_path, "w") as f:
                f.write(EXPORT_HF_DATASETS_CACHE)
                f.write(content)
        
        dumps_config(self.config, self.root_path / "experiment_config.json")


    def _repr_paths(self):
        repr = ""
        repr += INDENT + f"Root: {self.root_path}"                      + NEW_LINE
        repr += INDENT + f"Scripts: {self.scripts}"                     + NEW_LINE
        repr += INDENT + f"Baseline: {self.baseline_model}"             + NEW_LINE
        repr += INDENT + f"Quantized: {self.quantized_model}"           + NEW_LINE
        repr += INDENT + f"Finetuned: {self.finetuned_model}"           + NEW_LINE
        repr += INDENT + f"Evaluation: {self.evaluation_results}"       + NEW_LINE

        return repr
    
    def _repr_kwargs(self):
        repr = ""
        repr += f"Model: {self.model_name}" + NEW_LINE
        for k,v in self.config.items():
            repr += INDENT + f"{k.capitalize()}: {v}" + NEW_LINE
        repr += self._repr_paths()
        return repr

    def __repr__(self) -> str:
        if not self._repr:
            self._repr = self._repr_kwargs() # memoize
        return self._repr

    def __str__(self):
        return self.__repr__()


# class TokenizedDatasetSpec:

#     _registry: Dict[str, DefaultDict[str, List]] = {}

#     def __init__(self, key: str, root_path: Path, scripts: List[Path] = None):
#         self.key = key
#         self.root_path = root_path
#         self._registry[key] = self
    
#     @classmethod
#     def get(cls, key):
#         return cls._registry[key]


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

        self.experiment_specs: List[ExperimentSpec] = []
        for model_name, expt_dict in experiments_dict.items():
            exp_name = model_name.replace("/", "-") if not Path(model_name).exists() else model_name.name
            exp_root = Experiment.get_output_dir(exp_name)

            self.experiment_specs.append(ExperimentSpec(exp_root, model_name, expt_dict))

        ExperimentSpec.create_paths()
        ExperimentSpec.load_hf_models()
        self.build()

        self.create_tokenized_dataset_paths()

    def build(self):
        for spec in self.experiment_specs:
            spec.update_config_paths()
            spec.build_scripts()
    
    def create_tokenized_dataset_paths(self):
        for path in self.collect_unique_tokenized_datasets():
            Path(path).mkdir(exist_ok=True, parents=True)

    def collect_unique_tokenized_datasets(self):
        return set([spec.tokenized_dataset_path for spec in self.experiment_specs])
    

    def init_folder_structure(self):
        for path in [self._experiment_root, self._cache_dir, self._tokenized_datasets_dir]:
            path.mkdir(exist_ok=True, parents=True)
        
    def __str__(self):
        return f"Model: {self.model_name}, Evals: {self.evals}, Dataset: {self.dataset_name_or_path}"


if __name__ == "__main__":

    experiment = Experiment(EXPERIMENTS)


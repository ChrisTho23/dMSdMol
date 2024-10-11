from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SageMakerTrainingConfig:
    instance_type: str
    instance_count: int
    repo: str
    branch: str
    dataparallel: bool
    model_name: str
    hyperparameters: Dict[str, Any] = {
        "model_name_or_path": "",  # TODO: Add model path once uploaded
        "tokenizer_name": "",  # TODO: Add tokenizer path once uploaded
        "dataset_name": "",  # TODO: Add dataset name once uploaded
        "output_dir": "/opt/ml/model",
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "do_train": True,
        "do_predict": True,
        "predict_with_generate": True,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "seed": 42,
        "fp16": True,
    }

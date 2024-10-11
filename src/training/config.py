from dataclasses import dataclass

@dataclass
class SageMakerTrainingConfig:
    repo: str
    branch: str
    dataparallel: bool
    instance_type: str
    instance_count: int
    model_name: str
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    model_name_or_path: str = 'facebook/bart-large-cnn'
    dataset_name: str = 'samsum'
    do_train: bool = True
    do_predict: bool = True
    predict_with_generate: bool = True
    output_dir: str = '/opt/ml/model'
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    seed: int = 42
    fp16: bool = True

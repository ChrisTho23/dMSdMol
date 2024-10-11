from dataclasses import dataclass

@dataclass
class BartModelConfig:
    model_name: str = "facebook/bart-large"
    max_length: int = 128
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    output_dir: str = "./bart-chem-model"
    evaluation_strategy: str = "epoch"
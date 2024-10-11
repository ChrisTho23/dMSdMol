from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments,
)

from .config import BartModelConfig


class BartModel:
    def __init__(self, config: BartModelConfig = BartModelConfig()):
        self.config = config
        self.tokenizer = BartTokenizer.from_pretrained(
            self.config.model_name
        )  # TODO: @Aaron
        self.model = BartForConditionalGeneration.from_pretrained(
            self.config.model_name
        )

    def tokenize(self, smiles_string: str):
        """Tokenizes a SMILES string using the BART tokenizer."""
        pass

    def _get_trainer(self, train_dataset):
        """Creates and returns a Hugging Face Trainer with the provided training dataset."""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            evaluation_strategy=self.config.evaluation_strategy,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            weight_decay=self.config.weight_decay,
        )
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

    def train(self, train_dataset):
        """Trains the model using the provided training dataset."""
        trainer = self._get_trainer(train_dataset)
        trainer.train()

    def save(self, path: str):
        """Saves the model and tokenizer to the specified path."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model and tokenizer saved to {path}")

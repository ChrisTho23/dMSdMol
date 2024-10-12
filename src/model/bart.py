"""Wrapper of Hugging Face's BART model for MS/MS to SMILES prediction and vice versa."""

from transformers import BartForConditionalGeneration, BartTokenizer

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

    def save(self, path: str):
        """Saves the model and tokenizer to the specified path."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model and tokenizer saved to {path}")

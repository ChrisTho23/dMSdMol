import pytest
import torch
from datasets import Dataset

from src.data.data import Mol2MSDataset
from src.model.mol2ms import BartModelConfig, Mol2MSModel


@pytest.fixture
def config():
    return BartModelConfig(
        encoder_name="facebook/bart-base",
        max_length=128,
        max_ms_length=50,
        num_heads=8,
        num_layers=3,
    )


@pytest.fixture
def model(config):
    return Mol2MSModel(config)


def test_model_initialization(model, config):
    assert isinstance(model, Mol2MSModel)
    assert model.config.encoder_name == "facebook/bart-base"
    assert model.config.max_length == 128
    assert model.config.max_ms_length == 50


def test_forward_pass(model):
    batch_size = 4
    seq_length = 64
    ms_seq_length = 30

    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    index = torch.randint(0, ms_seq_length, (batch_size, ms_seq_length))

    mz_pred, intensity_pred, create_next_token_pred = model(
        input_ids, attention_mask, index
    )

    assert mz_pred.shape == (batch_size, ms_seq_length)
    assert intensity_pred.shape == (batch_size, ms_seq_length)
    assert create_next_token_pred.shape == (batch_size, ms_seq_length)


def test_tokenize(model, config):
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    tokenized = model.tokenize(smiles)

    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert tokenized.input_ids.shape[1] == config.max_length

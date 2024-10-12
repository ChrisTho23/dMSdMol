import pytest
from datasets import Dataset
from src.model.config import BartModelConfig
from src.data.data import Mol2MSDataset

@pytest.fixture
def dummy_data():
    return {
        'smiles': ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"],
        'mzs': [[100.0, 200.0, 300.0], [50.0, 150.0]],
        'intensities': [[0.5, 0.8, 0.3], [0.6, 0.9]]
    }

@pytest.fixture
def config():
    return BartModelConfig()

@pytest.fixture
def dataset(dummy_data, config):
    hf_dataset = Dataset.from_dict(dummy_data)
    return Mol2MSDataset(hf_dataset, config.encoder_name, config.max_ms_length, config.max_length)

def test_dataset_initialization(dataset, config):
    assert len(dataset) == 2
    assert isinstance(dataset.tokenizer, type(dataset.tokenizer))

def test_getitem(dataset, config):
    item = dataset[0]
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'mz' in item
    assert 'intensity' in item
    assert 'index' in item
    assert 'create_next_token' in item

    assert item['mz'].shape[0] == config.max_ms_length
    assert item['intensity'].shape[0] == config.max_ms_length
    assert item['index'].shape[0] == config.max_ms_length
    assert item['create_next_token'].shape[0] == config.max_ms_length
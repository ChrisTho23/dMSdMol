"""Create a dummy MS/MS dataset for testing."""

import os

import fire
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from dotenv import load_dotenv
from utils import dataset_to_hub

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
hf_username = os.getenv("HF_USERNAME")


def create_dummy_data():
    return [
        {
            "precursor_mz": 123.45,
            "precursor_charge": 1,
            "mzs": [
                100.5,
                150.75,
                200.85,
                250.95,
                300.15,
                350.25,
                400.35,
                450.45,
                500.55,
                550.65,
            ],
            "intensities": [0.1, 0.5, 0.9, 0.4, 0.6, 0.2, 0.8, 0.3, 0.7, 0.5],
            "collision_energy": 25.0,
            "instrument_type": "Orbitrap",
            "in_silico": False,
            "smiles": "CCO",
            "adduct": "[M+H]+",
            "compound_class": "alkaloid",
        },
        {
            "precursor_mz": 234.56,
            "precursor_charge": -1,
            "mzs": [
                110.1,
                160.25,
                210.95,
                260.35,
                310.45,
                360.55,
                410.65,
                460.75,
                510.85,
                560.95,
            ],
            "intensities": [0.2, 0.6, 0.8, 0.3, 0.7, 0.1, 0.9, 0.4, 0.5, 0.6],
            "collision_energy": 30.0,
            "instrument_type": "TOF",
            "in_silico": True,
            "smiles": "CCN",
            "adduct": "[M-H]-",
            "compound_class": "flavonoid",
        },
        {
            "precursor_mz": 345.67,
            "precursor_charge": 1,
            "mzs": [
                120.2,
                170.35,
                220.05,
                270.15,
                320.25,
                370.35,
                420.45,
                470.55,
                520.65,
                570.75,
            ],
            "intensities": [0.3, 0.7, 0.85, 0.2, 0.6, 0.4, 0.8, 0.1, 0.9, 0.5],
            "collision_energy": 35.0,
            "instrument_type": "Q-TOF",
            "in_silico": False,
            "smiles": "CCC",
            "adduct": "[M+H]+",
            "compound_class": "terpenoid",
        },
    ]


def create_dataset_dict():
    data = create_dummy_data()
    features = Features(
        {
            "precursor_mz": Value("float32"),
            "precursor_charge": Value("int32"),
            "mzs": Sequence(Value("float32")),
            "intensities": Sequence(Value("float32")),
            "collision_energy": Value("float32"),
            "instrument_type": Value("string"),
            "in_silico": Value("bool"),
            "smiles": Value("string"),
            "adduct": Value("string"),
            "compound_class": Value("string"),
        }
    )

    return DatasetDict(
        {
            "train": Dataset.from_list([data[0]], features=features),
            "validation": Dataset.from_list([data[1]], features=features),
            "test": Dataset.from_list([data[2]], features=features),
        }
    )


def main(repo_name):
    dataset = create_dataset_dict()
    dataset_to_hub(dataset, repo_name, hf_username, hf_token)
    print(f"Dataset '{repo_name}' uploaded successfully.")


if __name__ == "__main__":
    fire.Fire(main)

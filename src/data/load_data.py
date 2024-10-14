"""Loads data from S3, processes it, and loads it into a Hugging Face Dataset."""

import logging
import os

import boto3
import fire
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from src.data.utils import dataset_to_hub


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
hf_username = os.getenv("HF_USERNAME")


def load_from_s3(file_name):
    # Set up S3 client
    s3 = boto3.client("s3")
    bucket_name = "team4-bucket-hackathon"

    # Download file from S3
    local_file_path = f"/tmp/{file_name}"
    s3.download_file(bucket_name, file_name, local_file_path)

    # Load data into DataFrame
    df = pd.read_parquet(local_file_path)

    # Clean up temporary file
    os.remove(local_file_path)

    return df


def df_to_dataset(df):
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

    # Split data into train, test, and validation sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    # Convert DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df, features=features)
    test_dataset = Dataset.from_pandas(test_df, features=features)
    val_dataset = Dataset.from_pandas(val_df, features=features)

    # Create DatasetDict
    dataset_dict = DatasetDict(
        {"train": train_dataset, "test": test_dataset, "validation": val_dataset}
    )

    return dataset_dict
from datasets import load_dataset, DatasetDict

def load_and_split_parquet(file_path, test_size=0.2, validation_size=0.1, seed=42):
    """
    Loads a Parquet file and splits it into train, test, and validation sets.

    Parameters:
        file_path (str): Path to the Parquet file.
        test_size (float): Proportion of the dataset to include in the test split.
        validation_size (float): Proportion of the training set to include in the validation split.
        seed (int): Random seed for reproducibility.

    Returns:
        DatasetDict: A dictionary containing train, test, and validation splits.
    """
    # Load the dataset from the Parquet file
    dataset = load_dataset("parquet", data_files=file_path)["train"]

    # Split into train and test sets
    train_test_split = dataset.train_test_split(test_size=test_size, seed=seed)

    # Further split the training set to create a validation set
    train_validation_split = train_test_split["train"].train_test_split(test_size=validation_size, seed=seed)

    # Organize splits into a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_validation_split["train"],
        "test": train_test_split["test"],
        "validation": train_validation_split["test"]
    })

    return dataset_dict

def main(nist_file_name, enveda_file_name, repo_name):  # repo_name
    nist_df = load_from_s3(nist_file_name)
    enveda_df = load_from_s3(enveda_file_name)
    df = pd.concat([nist_df, enveda_df])
    dataset = df_to_dataset(df)
    dataset_to_hub(dataset, repo_name, hf_username, hf_token)
    logger.info(f"Dataset '{repo_name}' uploaded successfully.")


if __name__ == "__main__":
    fire.Fire(main)

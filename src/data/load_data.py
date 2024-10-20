"""Loads data from S3, processes it, and loads it into a Hugging Face Dataset."""

import logging
import os

import boto3
import fire
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from utils import dataset_to_hub

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
hf_username = os.getenv("HF_USERNAME")


def load_from_s3(bucket_path: str, file_name: str):
    # Set up S3 client
    s3 = boto3.client(
        "s3", region_name="us-east-1", endpoint_url="https://s3.amazonaws.com"
    )
    bucket_name = "team4-bucket-hackathon"

    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)

    local_file_path = os.path.join(data_dir, file_name)

    s3.download_file(bucket_name, bucket_path, local_file_path)

    df = pd.read_parquet(local_file_path)

    return df


def load_data(local_file_path: str = None):
    if local_file_path is None:
        df = load_from_s3(
            "team4-bucket-hackathon", "enveda-dataset/enveda_library_subset.parquet"
        )
    else:
        df = pd.read_parquet(local_file_path)
    return df


def df_to_dataset(df):
    columns_to_keep = [
        "precursor_mz",
        "precursor_charge",
        "mzs",
        "intensities",
        "collision_energy",
        "instrument_type",
        "in_silico",
        "smiles",
        "adduct",
        "compound_class",
    ]
    df = df[columns_to_keep].copy()

    # print number of NaN values in specified fields
    nan_counts = {
        "mzs": df["mzs"].isna().sum(),
        "intensities": df["intensities"].isna().sum(),
        "collision_energy": df["collision_energy"].isna().sum(),
        "instrument_type": df["instrument_type"].isna().sum(),
    }

    logger.info("Number of NaN values:")
    for field, count in nan_counts.items():
        logger.info(f"{field}: {count}")

    # Clean dataset
    df["collision_energy"] = df["collision_energy"].str.extract("(\d+)").astype(float)
    df = df.dropna(subset=["collision_energy", "instrument_type"])

    logger.info(f"Number of samples after dropping NaN values: {df.shape[0]}")

    # Convert data types to match our features
    df["precursor_mz"] = df["precursor_mz"].astype("float32")
    df["precursor_charge"] = df["precursor_charge"].astype("int32")
    df["mzs"] = df["mzs"].apply(lambda x: [np.float32(i) for i in x])
    df["intensities"] = df["intensities"].apply(lambda x: [np.float32(i) for i in x])
    df["compound_class"] = df["compound_class"].astype("string")

    # label encoding of collision energy and instrument type
    df["collision_energy"] = df["collision_energy"].astype("category").cat.codes
    df["instrument_type"] = df["instrument_type"].astype("category").cat.codes

    # logger.info number of distinct classes
    logger.info(
        f"Number of distinct collision energy classes: {df['collision_energy'].nunique()}"
    )
    logger.info(
        f"Number of distinct instrument type classes: {df['instrument_type'].nunique()}"
    )

    # Check if mzs and intensities have the same length in every row
    length_mismatch = df[df["mzs"].apply(len) != df["intensities"].apply(len)]
    if not length_mismatch.empty:
        logger.warning(
            f"Found {len(length_mismatch)} rows where mzs and intensities have different lengths."
        )
        logger.warning("Removing these rows from the dataset.")
        df = df[df["mzs"].apply(len) == df["intensities"].apply(len)]
        logger.info(f"Number of samples after removing mismatched rows: {df.shape[0]}")
    else:
        logger.info("All rows have matching lengths for mzs and intensities.")

    features = Features(
        {
            "precursor_mz": Value("float32"),
            "precursor_charge": Value("int32"),
            "mzs": Sequence(Value("float32")),
            "intensities": Sequence(Value("float32")),
            "collision_energy": Value("int32"),
            "instrument_type": Value("int32"),
            "in_silico": Value("bool"),
            "smiles": Value("string"),
            "adduct": Value("string"),
            "compound_class": Value("string"),
        }
    )

    train_indices, test_indices = train_test_split(
        np.unique(df.index), test_size=0.2, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.1, random_state=42
    )

    train_df = df.loc[train_indices]
    val_df = df.loc[val_indices]
    test_df = df.loc[test_indices]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # To hf dataset
    train_dataset = Dataset.from_pandas(train_df, features=features)
    test_dataset = Dataset.from_pandas(test_df, features=features)
    val_dataset = Dataset.from_pandas(val_df, features=features)

    dataset_dict = DatasetDict(
        {"train": train_dataset, "test": test_dataset, "validation": val_dataset}
    )

    return dataset_dict

def main(local_enveda_path: str, repo_name: str):  # local_nist_path
    logger.info("Loading enveda data...")
    enveda_df = load_data(local_enveda_path)
    logger.info(f"Enveda data loaded successfully. Data shape: {enveda_df.shape}")

    # logger.info("Loading enveda data...")
    # nist_df = load_data(local_nist_path)
    # logger.info(f"Enveda data loaded successfully. Data shape: {df.shape}")

    # df = pd.concat([enveda_df, nist_df])

    dataset = df_to_dataset(enveda_df)  # df

    logger.info("Uploading dataset to Hugging Face Hub...")
    dataset_to_hub(dataset, repo_name, hf_username, hf_token)
    logger.info(f"Dataset '{repo_name}' uploaded successfully.")


if __name__ == "__main__":
    fire.Fire(main)

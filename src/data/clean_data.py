import pandas as pd
from datasets import load_dataset

fileLoc = "/Users/aaronfanous/Downloads/enveda_library_subset.parquet"
# in this you need to clean the data frame from enved.
# dataset = load_dataset("parquet",data_files=fileLoc)
# print(dataset["train"].features)
# df=pd.read_parquet()

# print(df.columns)
# def cleanlists(x):
#     x.strip('\n')


# # 1. Check data types for each column
# print("Data types for each column in the DataFrame:")
# print(df.dtypes)

# # 2. Check sample values in `mzs` and `intensities` to confirm inner element types
# print("\nSample 'mzs' column data:")
# print(df["mzs"].head().apply(lambda x: [type(i) for i in x]))

# print("\nSample 'intensities' column data:")
# print(df["intensities"].head().apply(lambda x: [type(i) for i in x]))
# # Define the path to the original and subset Parquet files

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from sklearn.model_selection import train_test_split


def df_to_dataset_from_parquet(parquet_file):
    # Load the Parquet file directly with PyArrow
    table = pq.read_table(parquet_file)

    # Convert the PyArrow Table to a Pandas DataFrame without losing schema
    df = table.to_pandas(ignore_metadata=True)

    # Define the schema directly based on inferred features from load_dataset
    features = Features(
        {
            "precursor_mz": Value("float64"),
            "precursor_charge": Value("float64"),
            "mzs": Sequence(Value("float64")),
            "intensities": Sequence(Value("float64")),
            "in_silico": Value("bool"),
            "smiles": Value("string"),
            "adduct": Value("string"),
            "collision_energy": Value("string"),
            "instrument_type": Value("string"),
            "compound_class": Value("string"),
            "entropy": Value("float64"),
            "scaffold_smiles": Value("string"),
        }
    )

    # Split data into train, test, and validation sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    # Convert DataFrames to Hugging Face Datasets with the predefined features
    train_dataset = Dataset.from_pandas(train_df, features=features)
    test_dataset = Dataset.from_pandas(test_df, features=features)
    val_dataset = Dataset.from_pandas(val_df, features=features)

    # Create DatasetDict to organize train, test, and validation splits
    dataset_dict = DatasetDict(
        {"train": train_dataset, "test": test_dataset, "validation": val_dataset}
    )

    return dataset_dict


df_to_dataset_from_parquet(fileLoc)

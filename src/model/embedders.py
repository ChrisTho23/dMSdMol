# self.embed_tokens = BartScaledWordEmbedding(
#             config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
#         )
import sys

sys.path.append("/Users/aaronfanous/Documents/EnvedaChallenge/dMSdMol2")
import fire
import torch
import torch.nn as nn
from torch.nn import Module

from src.data.config import CollisionEnergyConfigWithClassification
from src.data.data import Mol2MSDataset
from src.data.load_data import load_and_split_parquet


class SingleSampleMzEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mz_embedding = nn.Embedding(1, embedding_dim)

        # Define a separate, learnable embedding for cases with no input
        self.default_embedding = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, mz_input=None):
        """
        Returns an embedding for the ms/mz sample if provided; otherwise, uses a learnable default embedding.

        Args:
            mz_input (torch.Tensor or None): ms/mz sample input, or None if it's the first step.

        Returns:
            torch.Tensor: Embedding of the ms/mz sample, or the learnable default embedding if mz_input is None.
        """
        if mz_input is None:
            # Return the learnable default embedding if mz_input is missing
            return self.default_embedding.unsqueeze(0)

        # Otherwise, return the embedding for the provided ms/mz sample
        return self.mz_embedding(mz_input)


class CumulativeMzEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()

        # Embedding network for cumulative sequence embedding
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, mz_sequence):
        """
        Args:
            mz_sequence (torch.Tensor): Tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Cumulative embedding of shape (batch_size, embedding_dim).
        """
        # Process each ms/mz value individually through the network
        embeddings = nn.relu(self.fc1(mz_sequence))
        embeddings = nn.relu(self.fc2(embeddings))

        # Sum embeddings across the sequence dimension to form a cumulative representation
        cumulative_embedding = embeddings.sum(
            dim=1
        )  # or average with embeddings.mean(dim=1)
        return cumulative_embedding


class Mol2MSModelEmbeddingWithOptional(nn.Module):
    def __init__(
        self,
        original_embedding_dim,
        category_dim,
        instrument_embed_dim,
        energy_type_embed_dim,
    ):
        super().__init__()

        # Embedding layers for instrument type and energy type
        self.instrument_embedding = nn.Embedding(category_dim, instrument_embed_dim)
        self.energy_type_embedding = nn.Embedding(category_dim, energy_type_embed_dim)

        # Projection layer for concatenated instrument and energy embeddings
        concatenated_dim = instrument_embed_dim + energy_type_embed_dim
        self.projection = nn.Linear(concatenated_dim, original_embedding_dim)

        # Use cumulative embedding module
        self.cumulative_mz_embedding = CumulativeMzEmbedding(
            input_dim=original_embedding_dim,
            hidden_dim=128,
            embedding_dim=original_embedding_dim,
        )

    def forward(
        self,
        input_ids,
        original_embeddings,
        instrument_type,
        energy_type,
        mz_sequence=None,
    ):
        """
        Combines original embeddings with instrument, energy-type, and cumulative mz embeddings.

        Args:
            input_ids (torch.Tensor): Original token IDs.
            original_embeddings (torch.Tensor): Original token embeddings.
            instrument_type (torch.Tensor): Instrument type indices.
            energy_type (torch.Tensor): Energy type indices.
            mz_sequence (torch.Tensor): Sequence of mz values.

        Returns:
            torch.Tensor: Combined embedding ready for encoder input.
        """
        # Get embeddings for instrument type and energy type
        OG_embeds = original_embeddings(input_ids)
        instrument_embed = self.instrument_embedding(instrument_type)
        energy_type_embed = self.energy_type_embedding(energy_type)

        # Concatenate and project to original embedding dimension
        concatenated_embeddings = torch.cat(
            [instrument_embed, energy_type_embed], dim=-1
        )
        projected_embedding = self.projection(concatenated_embeddings)

        # Compute cumulative mz embedding
        cumulative_mz_embed = self.cumulative_mz_embedding(mz_sequence)

        # Add original, projected, and cumulative embeddings
        combined_embedding = OG_embeds + projected_embedding + cumulative_mz_embed
        return combined_embedding


def main():  # repo_names
    df = load_and_split_parquet(
        "/Users/aaronfanous/Downloads/enveda_library_subset.parquet", 0.1, 0.1
    )
    unique_values = {"collision_energy": set(), "instrument_type": set()}

    for split in df:
        # Convert to Pandas and update the unique values for each specified column
        unique_values["collision_energy"].update(
            df[split].to_pandas()["collision_energy"].unique()
        )
        unique_values["instrument_type"].update(
            df[split].to_pandas()["instrument_type"].unique()
        )

    # Convert sets back to lists (optional, depending on your needs)
    unique_values = {key: list(value) for key, value in unique_values.items()}

    # Print the unique values
    print("Consolidated Unique Values:")
    print("Collision Energy:", unique_values["collision_energy"])
    print("Instrument Type:", unique_values["instrument_type"])
    config_with_classification = CollisionEnergyConfigWithClassification()
    selected_classification_cases = [x for x in unique_values["collision_energy"]]
    classification_results = [
        (
            energy_str,
            config_with_classification.prepare_for_embedding(
                energy_str, machine_type="HCD"
            ),
        )
        for energy_str in selected_classification_cases
    ]
    print(classification_results)


if __name__ == "__main__":
    fire.Fire(main)

import os
import sys

# Get the root directory (parent of 'src')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the root directory to the Python path
sys.path.append(project_root)

import logging

import fire

# import smdistributed.dataparallel.torch.torch_smddp
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data import Mol2MSDataset
from src.loss.lossFile import ragan_loss
from src.model import Mol2MSModel, Mol2MSModelConfig, MS2MolModel, MS2MolModelConfig
from src.model.cycleGanStructure import (  # ,Mol2MSWrapperGenerator,Mol2MSWrapperDiscriminator
    Mol2MSWrapperDiscriminator,
    MS2MolWrapperDiscriminator,
    MS2MolWrapperGenerator,
)
from src.training.config import Mol2MSTrainingConfig


def setup_gan_train(Mol2Ms, ms2mol):
    generator_A = Mol2Ms  # MS2MolWrapperGenerator(generator_model=model2).to(device)
    generator_B = ms2mol  # Mol2MSWrapperGenerator(generator_model=model1).to(device)

    discriminator_A = Mol2MSWrapperDiscriminator(
        base_model=ms2mol, selected_layer="decoder"
    ).to(device)
    discriminator_B = MS2MolWrapperDiscriminator(
        base_model=Mol2Ms, selected_layer="decoder"
    ).to(device)
    return {
        "generator_A": generator_A,
        "generator_B": generator_B,
        "discriminator_A": discriminator_A,
        "discriminator_B": discriminator_B,
    }


# CycleGAN training function with RaGAN loss
def train_cycle_gan_ragan(
    generator_A,
    generator_B,
    discriminator_A,
    discriminator_B,
    dataloader,
    num_epochs,
    device,
):

    # Optimizers
    optimizer_G_A = AdamW(generator_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_G_B = AdamW(generator_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = AdamW(discriminator_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = AdamW(discriminator_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function for cycle consistency (L1 loss)
    cycle_consistency_loss = nn.L1Loss()

    for epoch in range(num_epochs):
        total_G_loss = 0
        total_D_A_loss = 0
        total_D_B_loss = 0

        for i, batch in enumerate(tqdm(dataloader)):

            smiles = batch["smiles_ids"].to(device)  # SMILES input
            ms = batch["mz"].to(device)  # m/z values
            attention_mask = batch["attention_mask"].to(
                device
            )  # Attention mask for SMILES
            collision_energy = batch["collision_energy"].to(device)  # Collision energy
            instrument_type = batch["instrument_type"].to(device)  # Instrument type
            tgt_intensities = batch["tgt_intensity"].to(
                device
            )  # Target intensities for m/z
            tgt_mzs = batch["tgt_mz"].to(device)  # Target m/z values

            # 1. Train Generators A and B
            optimizer_G_A.zero_grad()
            optimizer_G_B.zero_grad()
            fake_ms, intensity = generator_A.generate(
                smiles, collision_energy, instrument_type
            )

            #         fake_ms,intensity = generator_A(
            #     smiles,
            #     attention_mask=attention_mask,
            #     collision_energy=collision_energy,
            #     instrument_type=instrument_type,
            #     tgt_intensities=tgt_intensities,
            #     tgt_mzs=tgt_mzs
            # )

            rec_smiles = generator_B(
                fake_ms, intensity=tgt_intensities, decoder_input_ids=smiles
            )

            (fake_smiles,) = generator_B(ms, tgt_intensities, smiles)
            rec_ms = generator_A(
                fake_smiles,
            )

            # Cycle consistency loss
            cycle_loss_A = cycle_consistency_loss(smiles, rec_smiles)
            cycle_loss_B = cycle_consistency_loss(ms, rec_ms)
            cycle_loss = cycle_loss_A + cycle_loss_B

            # Discriminator predictions
            D_real_A = discriminator_A(ms)
            D_fake_A = discriminator_A(fake_ms)
            D_real_B = discriminator_B(smiles)
            D_fake_B = discriminator_B(fake_smiles)

            # RaGAN loss for generators
            D_A_loss, G_A_loss = ragan_loss(D_real_A, D_fake_A)
            D_B_loss, G_B_loss = ragan_loss(D_real_B, D_fake_B)

            # Total generator loss
            G_loss = (
                G_A_loss + G_B_loss + cycle_loss * 10
            )  # Multiply cycle loss for emphasis
            G_loss.backward()
            optimizer_G_A.step()
            optimizer_G_B.step()

            total_G_loss += G_loss.item()

            # 2. Train Discriminators A and B
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            # RaGAN loss for discriminators
            D_A_loss, _ = ragan_loss(D_real_A, D_fake_A.detach())
            D_B_loss, _ = ragan_loss(D_real_B, D_fake_B.detach())

            D_A_loss.backward()
            D_B_loss.backward()
            optimizer_D_A.step()
            optimizer_D_B.step()

            total_D_A_loss += D_A_loss.item()
            total_D_B_loss += D_B_loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs} - G_loss: {total_G_loss:.4f}, D_A_loss: {total_D_A_loss:.4f}, D_B_loss: {total_D_B_loss:.4f}"
        )


from src.model import mol2ms, ms2mol

if __name__ == "__main__":
    # Initialize models (example)
    # generator_A = MS2MolWrapperGenerator(generator_model=CustomBARTModel(bart_model)).to(device)
    # generator_B = MS2MolWrapperGenerator(generator_model=Mol2MSModel(config)).to(device)

    # discriminator_A = MS2MolWrapperDiscriminator(base_model=Mol2MSModel(config), selected_layer='encoder').to(device)
    # discriminator_B = Mol2MSDiscriminator(base_model=CustomBARTModel(bart_model), selected_layer='decoder').to(device)
    hf_dataset = load_dataset("ChrisTho/dMSdMols")
    model_config = Mol2MSModelConfig()
    train_dataset = train_dataset = Mol2MSDataset(
        hf_dataset["train"],
        model_config.encoder_name,
        model_config.max_encoder_length,
        model_config.max_decoder_length,
    )
    # Load your data (assuming the DataLoader returns dictionaries with 'smiles' and 'ms')
    dataloader = DataLoader(
        train_dataset,
        batch_size=2,
    )  # Define your dataloader here

    # Set device (GPU if available)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    mol2ms = Mol2MSModel(Mol2MSModelConfig(), train_dataset.get_tokenizer()).to(device)

    configMs2mol = MS2MolModelConfig()
    # Train the CycleGAN with RaGAN

    ganDict = setup_gan_train(Mol2Ms=mol2ms, ms2mol=MS2MolModel(config=configMs2mol))

    num_epochs = 100  # Set your desired number of epochs

    # Assuming ganDict contains keys like 'generator_A', 'generator_B', 'discriminator_A', 'discriminator_B'
    train_cycle_gan_ragan(
        generator_A=ganDict["generator_A"],
        generator_B=ganDict["generator_B"],
        discriminator_A=ganDict["discriminator_A"],
        discriminator_B=ganDict["discriminator_B"],
        dataloader=dataloader,
        num_epochs=num_epochs,
        device=device,
    )

import logging

import fire
import smdistributed.dataparallel.torch.torch_smddp
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data import Mol2MSDataset
from src.model import Mol2MSModel, Mol2MSModelConfig
from src.training.config import Mol2MSTrainingConfig

from src.loss.lossFile import ragan_loss
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm






# CycleGAN training function with RaGAN loss
def train_cycle_gan_ragan(
    generator_A, generator_B, discriminator_A, discriminator_B,
    dataloader, num_epochs, device
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
            smiles, ms = batch['smiles'].to(device), batch['ms'].to(device)

            # 1. Train Generators A and B
            optimizer_G_A.zero_grad()
            optimizer_G_B.zero_grad()

            fake_ms = generator_A(smiles)
            rec_smiles = generator_B(fake_ms)
            
            fake_smiles = generator_B(ms)
            rec_ms = generator_A(fake_smiles)

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
            G_loss = G_A_loss + G_B_loss + cycle_loss * 10  # Multiply cycle loss for emphasis
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

        print(f"Epoch {epoch+1}/{num_epochs} - G_loss: {total_G_loss:.4f}, D_A_loss: {total_D_A_loss:.4f}, D_B_loss: {total_D_B_loss:.4f}")


if __name__ == "__main__":
    # Initialize models (example)
    generator_A = MS2MolWrapperGenerator(generator_model=CustomBARTModel(bart_model)).to(device)
    generator_B = MS2MolWrapperGenerator(generator_model=Mol2MSModel(config)).to(device)

    discriminator_A = MS2MolWrapperDiscriminator(base_model=Mol2MSModel(config), selected_layer='encoder').to(device)
    discriminator_B = Mol2MSDiscriminator(base_model=CustomBARTModel(bart_model), selected_layer='decoder').to(device)

    # Load your data (assuming the DataLoader returns dictionaries with 'smiles' and 'ms')
    dataloader = ...  # Define your dataloader here

    # Set device (GPU if available)
    device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

    # Train the CycleGAN with RaGAN
    num_epochs = 100  # Set your desired number of epochs
    train_cycle_gan_ragan(generator_A, generator_B, discriminator_A, discriminator_B, dataloader, num_epochs, device)

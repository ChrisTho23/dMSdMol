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

dist.init_process_group(backend="smddp")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
logger.info(f"Using device: {device}")


def collate_fn(batch):
    tokenized_smiles = pad_sequence(
        [item["tokenized_smiles"] for item in batch], batch_first=True, padding_value=0
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )

    mz = torch.tensor([item["mz"] for item in batch], dtype=torch.float).unsqueeze(1)
    intensity = torch.tensor(
        [item["intensity"] for item in batch], dtype=torch.float
    ).unsqueeze(1)
    index = torch.tensor([item["index"] for item in batch], dtype=torch.long).unsqueeze(
        1
    )
    collision_energy = torch.tensor(
        [item["collision_energy"] for item in batch], dtype=torch.long
    ).unsqueeze(1)
    instrument_type = torch.tensor(
        [item["instrument_type"] for item in batch], dtype=torch.long
    ).unsqueeze(1)
    stop_token = torch.tensor(
        [item["stop_token"] for item in batch], dtype=torch.float
    ).unsqueeze(1)

    return {
        "tokenized_smiles": tokenized_smiles,
        "attention_mask": attention_mask,
        "index": index,
        "collision_energy": collision_energy,
        "instrument_type": instrument_type,
        "mz": mz,
        "intensity": intensity,
        "stop_token": stop_token,
    }


def calculate_loss(mz_pred, mz, intensity_pred, intensity, stop_token_pred, stop_token):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([10.0]).to(stop_token.device)
    )

    loss_mz = mse_loss(mz_pred, mz)
    loss_intensity = mse_loss(intensity_pred, intensity)
    loss_stop_token = bce_loss(stop_token_pred, stop_token)

    # Scale losses to be of the same magnitude
    scaled_loss_mz = loss_mz / torch.mean(mz**2)
    scaled_loss_intensity = loss_intensity / torch.mean(intensity**2)

    total_loss = scaled_loss_mz + scaled_loss_intensity + loss_stop_token

    return total_loss, loss_mz, loss_intensity, loss_stop_token


def train(
    model_config: Mol2MSModelConfig = Mol2MSModelConfig(),
    training_config: Mol2MSTrainingConfig = Mol2MSTrainingConfig(),
    wandb_project: str = None,
    wandb_api_key: str = None,
):
    logger.info("Starting training process")

    if wandb_project and wandb_api_key and dist.get_rank() == 0:
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_project,
            config=dict(
                model_config=vars(model_config), training_config=vars(training_config)
            ),
        )
    else:
        logger.warning(
            "WANDB_PROJECT and WANDB_API_KEY are not set, skipping Weights & Biases logging"
        )

    logger.info(f"Loading dataset: {training_config.dataset_name}")
    hf_dataset = load_dataset(training_config.dataset_name)

    train_sampler = DistributedSampler(
        hf_dataset["train"], num_replicas=dist.get_world_size(), rank=dist.get_rank()
    )
    val_sampler = DistributedSampler(
        hf_dataset["test"], num_replicas=dist.get_world_size(), rank=dist.get_rank()
    )

    train_dataset = Mol2MSDataset(
        hf_dataset["train"],
        model_config.encoder_name,
        model_config.max_length,
    )
    val_dataset = Mol2MSDataset(
        hf_dataset["test"],
        model_config.encoder_name,
        model_config.max_length,
    )

    logger.info("Creating data loaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )

    logger.info("Initializing model")
    model = Mol2MSModel(model_config).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[dist.get_rank()]
    )
    if dist.get_rank() == 0:
        wandb.watch(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    total_steps = len(train_loader) * training_config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info("Starting training")
    for epoch in range(training_config.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{training_config.num_epochs}")
        model.train()
        total_loss = 0

        train_loader.sampler.set_epoch(epoch)
        progress_bar = (
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config.num_epochs}")
            if dist.get_rank() == 0
            else train_loader
        )
        for batch_idx, batch in enumerate(progress_bar):
            tokenized_smiles = batch["tokenized_smiles"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            index = batch["index"].to(device)
            collision_energy = batch["collision_energy"].to(device)
            instrument_type = batch["instrument_type"].to(device)
            mz = batch["mz"].to(device)
            intensity = batch["intensity"].to(device)
            stop_token = batch["stop_token"].to(device)

            optimizer.zero_grad()

            mz_pred, intensity_pred, stop_token_pred = model(
                tokenized_smiles,
                attention_mask,
                index,
                collision_energy,
                instrument_type,
            )

            loss, loss_mz, loss_intensity, loss_stop_token = calculate_loss(
                mz_pred, mz, intensity_pred, intensity, stop_token_pred, stop_token
            )
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if dist.get_rank() == 0:
                wandb.log(
                    {
                        "mz_loss": loss_mz.item(),
                        "intensity_loss": loss_intensity.item(),
                        "stop_token_loss": loss_stop_token.item(),
                        "train_loss": loss.item(),
                    }
                )
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1}/{training_config.num_epochs}, Average Loss: {avg_loss:.4f}"
        )

        logger.info("Starting validation")
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in (
                tqdm(val_loader, desc="Validation")
                if dist.get_rank() == 0
                else val_loader
            ):
                tokenized_smiles = batch["tokenized_smiles"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                index = batch["index"].to(device)
                collision_energy = batch["collision_energy"].to(device)
                instrument_type = batch["instrument_type"].to(device)
                mz = batch["mz"].to(device)
                intensity = batch["intensity"].to(device)
                stop_token = batch["stop_token"].to(device)

                mz_pred, intensity_pred, stop_token_pred = model(
                    tokenized_smiles,
                    attention_mask,
                    index,
                    collision_energy,
                    instrument_type,
                )

                loss, _, _, _ = calculate_loss(
                    mz_pred, mz, intensity_pred, intensity, stop_token_pred, stop_token
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        if dist.get_rank() == 0:
            wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})

    logger.info("Training completed")


if __name__ == "__main__":
    fire.Fire(train)

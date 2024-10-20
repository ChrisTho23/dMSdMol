import logging
import os

import fire
import smdistributed.dataparallel.torch.torch_smddp
import torch as t
import torch.distributed as dist
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data import Mol2MSDataset
from src.loss import Mol2MSLoss
from src.model import Mol2MSModel, Mol2MSModelConfig
from src.training import Mol2MSTrainingConfig

dist.init_process_group(backend="smddp")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


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
        model_config.max_encoder_length,
        model_config.max_decoder_length,
    )
    val_dataset = Mol2MSDataset(
        hf_dataset["test"],
        model_config.encoder_name,
        model_config.max_encoder_length,
        model_config.max_decoder_length,
    )

    logger.info("Creating data loaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        sampler=val_sampler,
    )

    logger.info("Initializing model")
    model = Mol2MSModel(model_config, train_dataset.get_tokenizer()).to(device)
    model = t.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
    if dist.get_rank() == 0:
        wandb.watch(model)

    optimizer = t.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    total_steps = len(train_loader) * training_config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = Mol2MSLoss()

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
        for _, batch in enumerate(progress_bar):
            smiles_ids = batch["smiles_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            collision_energy = batch["collision_energy"].to(device)
            instrument_type = batch["instrument_type"].to(device)
            tgt_intensity = batch["tgt_intensity"].to(device)
            tgt_mz = batch["tgt_mz"].to(device)
            intensity = batch["intensity"].to(device)
            mz = batch["mz"].to(device)

            optimizer.zero_grad()
            pred_mz, pred_intensity = model(
                smiles_ids,
                attention_mask,
                collision_energy,
                instrument_type,
                tgt_intensity,
                tgt_mz,
            )

            soft_jaccard_loss, loss_mz, loss_intensity, sign_penalty = loss_fn(
                pred_mz=pred_mz,
                mz=mz,
                intensity=intensity,
                pred_intensity=pred_intensity,
            )
            loss = soft_jaccard_loss + loss_mz + loss_intensity + sign_penalty
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if dist.get_rank() == 0:
                wandb.log(
                    {
                        "total_train_loss": loss.item(),
                        "soft_jaccard_loss": soft_jaccard_loss.item(),
                        "loss_mz": loss_mz.item(),
                        "loss_intensity": loss_intensity.item(),
                        "sign_penalty": sign_penalty.item(),
                    }
                )
                progress_bar.set_postfix({"total_train_loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1}/{training_config.num_epochs}, Average Loss: {avg_loss:.4f}"
        )

        logger.info("Starting validation")
        model.eval()
        val_loss = 0

        with t.no_grad():
            for batch in (
                tqdm(val_loader, desc="Validation")
                if dist.get_rank() == 0
                else val_loader
            ):
                smiles_ids = batch["smiles_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                collision_energy = batch["collision_energy"].to(device)
                instrument_type = batch["instrument_type"].to(device)
                tgt_intensity = batch["tgt_intensity"].to(device)
                tgt_mz = batch["tgt_mz"].to(device)
                intensity = batch["intensity"].to(device)
                mz = batch["mz"].to(device)

                pred_mz, pred_intensity = model(
                    smiles_ids,
                    attention_mask,
                    collision_energy,
                    instrument_type,
                    tgt_intensity,
                    tgt_mz,
                )

                soft_jaccard_loss, loss_mz, loss_intensity, sign_penalty = loss_fn(
                    pred_mz=pred_mz,
                    mz=mz,
                    intensity=intensity,
                    pred_intensity=pred_intensity,
                )
                loss = soft_jaccard_loss + loss_mz + loss_intensity + sign_penalty
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        if dist.get_rank() == 0:
            wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})

    logger.info("Training completed")

    if dist.get_rank() == 0:
        unwrapped_model = model.module if hasattr(model, "module") else model
        unwrapped_model.save(training_config.output_dir, "mol2ms")


if __name__ == "__main__":
    fire.Fire(train)

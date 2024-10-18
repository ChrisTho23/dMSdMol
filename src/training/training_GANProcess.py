import logging

import fire
import smdistributed.dataparallel.torch.torch_smddp
import torch
import torch.distributedtributed as dist
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



class trainer():
    def __init__(self,):
        pass

    def init_dist_backend(backend="smddp"):
        dist.init_process_group(backend=backend)


    def train(self, ):
        pass
    def cleanup():
        pass

    def collate_fn(batch):
        pass

    def setupOptimizers():
        pass
    def setUp():
        pass
    





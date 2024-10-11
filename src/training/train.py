import logging
from typing import Any, Dict

import fire
import sagemaker
from sagemaker.huggingface import HuggingFace

from ..model.bart import BartModel
from .config import SageMakerTrainingConfig
from .utils import upload_estimator_to_hf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initiate_sagemaker_session():
    """Initializes a SageMaker session and retrieves the execution role. Only works in SageMaker."""
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()

    logger.info(f"SageMaker session: {sess}")
    logger.info(f"SageMaker role: {role}")

    return sess, role


def get_sagemaker_estimator(
    role: Any,
    git_config: Dict[str, str],
    distribution: Dict[str, Dict[str, bool]],
    train_config: SageMakerTrainingConfig,
):
    """Creates a SageMaker HuggingFace estimator for distributed training."""
    huggingface_estimator = HuggingFace(
        entry_point="training_script.py",
        source_dir="./src/training",
        git_config=git_config,
        instance_type=train_config.instance_type,
        instance_count=train_config.instance_count,
        transformers_version="4.45",
        pytorch_version="2.4",
        py_version="py310",
        role=role,
        hyperparameters=train_config.hyperparameters,
        distribution=distribution,
    )
    return huggingface_estimator


def train(
    train_config: SageMakerTrainingConfig = SageMakerTrainingConfig(),
):
    sess, role = initiate_sagemaker_session()

    git_config = {"repo": train_config.repo, "branch": train_config.branch}
    distribution = {
        "smdistributed": {"dataparallel": {"enabled": train_config.dataparallel}}
    }

    logger.info("Creating SageMaker estimator...")
    estimator = get_sagemaker_estimator(
        role=role,
        git_config=git_config,
        distribution=distribution,
        train_config=train_config,
    )

    logger.info("Launched training job...")
    estimator.fit()

    logger.info("Training complete. Uploading model to Hugging Face...")
    upload_estimator_to_hf(estimator, train_config.model_name)

    logger.info("Model uploaded to Hugging Face.")


if __name__ == "__main__":
    fire.Fire(train)

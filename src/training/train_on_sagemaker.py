"""Script to launch distributed training job on SageMaker."""

import logging
from typing import Any

import boto3
import fire
import sagemaker
from sagemaker.huggingface import HuggingFace

from src.training.config import SageMakerTrainingConfig
from src.training.utils import upload_estimator_to_hf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initiate_sagemaker_session(train_config: SageMakerTrainingConfig):
    """Initializes a SageMaker session and retrieves the execution role. Only works in SageMaker."""
    boto_session = boto3.Session(region_name=train_config.aws_region)

    sess = sagemaker.Session(boto_session=boto_session)

    iam_client = boto_session.client("iam", region_name=train_config.aws_region)
    role = iam_client.get_role(
        RoleName="AmazonSageMaker-ExecutionRole-20241008T214280"
    )["Role"]["Arn"]

    logger.info(f"SageMaker session: {sess}")
    logger.info(f"SageMaker role: {role}")

    return sess, role


def get_sagemaker_estimator(
    role: Any,
    sess: Any,
    train_config: SageMakerTrainingConfig,
) -> HuggingFace:
    """Creates a SageMaker HuggingFace estimator for distributed training."""
    huggingface_estimator = HuggingFace(
        entry_point=train_config.entry_point,
        source_dir=train_config.source_dir,
        git_config=train_config.git_config,
        instance_type=train_config.instance_type,
        instance_count=train_config.instance_count,
        transformers_version=train_config.transformers_version,
        pytorch_version=train_config.pytorch_version,
        py_version=train_config.py_version,
        role=role,
        dependencies=train_config.dependencies,
        hyperparameters=train_config.hyperparameters,
        distribution=train_config.distribution,
        sagemaker_session=sess,
        image_uri=train_config.image_uri,
    )
    return huggingface_estimator


def train(
    train_config: SageMakerTrainingConfig = SageMakerTrainingConfig(),
):
    sess, role = initiate_sagemaker_session(train_config)

    logger.info("Creating SageMaker estimator...")
    estimator = get_sagemaker_estimator(
        role=role,
        sess=sess,
        train_config=train_config,
    )

    logger.info("Launched training job...")
    estimator.fit()

    # logger.info("Training complete. Uploading model to Hugging Face...")
    # upload_estimator_to_hf(estimator, train_config.model_name)

    # logger.info("Model uploaded to Hugging Face.")


if __name__ == "__main__":
    fire.Fire(train)

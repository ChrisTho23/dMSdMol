import logging

import fire
import sagemaker
from sagemaker.huggingface import HuggingFace

from ..model.bart import BartModel
from .config import SageMakerTrainingConfig
from .utils import upload_estimator_to_hf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initiate_sagemaker_session():
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()

    logger.info(f"SageMaker session: {sess}")
    logger.info(f"SageMaker role: {role}")

    return sess, role

def get_sagemaker_estimator(role, git_config, distribution, train_config):
    """Creates a SageMaker HuggingFace estimator for distributed training."""
    huggingface_estimator = HuggingFace(
        entry_point='training_script.py', 
        source_dir='.',
        role=role,
        git_config=git_config,
        distribution=distribution,
        instance_type=train_config.instance_type,
        instance_count=train_config.instance_count,
        transformers_version='4.45',  
        pytorch_version='2.4',  
        py_version='py310',
        hyperparameters={
            'model_name': train_config.model_name,
            'output_dir': '/opt/ml/model',
            'train_batch_size': train_config.train_batch_size,
            'learning_rate': train_config.learning_rate,
            'num_train_epochs': train_config.num_train_epochs,
            'weight_decay': train_config.weight_decay,
            'evaluation_strategy': train_config.evaluation_strategy,
        }
    )
    return huggingface_estimator


def train(
    train_config: SageMakerTrainingConfig = SageMakerTrainingConfig(),
):
    sess, role = initiate_sagemaker_session()

    git_config = {'repo': train_config.repo, 'branch': train_config.branch}
    distribution = {'smdistributed':{'dataparallel':{ 'enabled': train_config.dataparallel }}}

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

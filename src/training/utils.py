import json
import os
import tarfile

from huggingface_hub import HfApi, Repository
from sagemaker.huggingface import HuggingFace
from sagemaker.s3 import S3Downloader
from dotenv import load_dotenv

from .config import SageMakerTrainingConfig

load_dotenv()


def upload_estimator_to_hf(estimator: HuggingFace, training_config: SageMakerTrainingConfig, local_path: str, model_card_path: str, repo_name: str):
    S3Downloader.download(
        s3_uri=estimator.model_data, # s3 uri where the trained model is located
        local_path=local_path, # local path where *.tar.gz will be saved
        sagemaker_session=estimator.sagemaker_session # sagemaker session used for training the model
    )

    # untar the model
    tar = tarfile.open(f"{local_path}/model.tar.gz", "r:gz")
    tar.extractall(path=local_path)
    tar.close()
    os.remove(f"{local_path}/model.tar.gz")

    # create readme for model card
    with open(f"{local_path}/eval_results.json") as f:
        eval_results_raw = json.load(f)
        eval_results={}
        # eval_results["eval_rouge1"] = eval_results_raw["eval_rouge1"] TODO: Update with actual evaluation metrics

    with open(f"{local_path}/test_results.json") as f:
        test_results_raw = json.load(f)
        test_results={}
        # test_results["test_rouge1"] = test_results_raw["test_rouge1"] TODO: Update with actual evaluation metrics

    # generate model card (todo: add more data from Trainer)
    with open(model_card_path, "r") as f:
        model_card_template = f.read()
    
    model_card = model_card_template.format(
        model_name=f"{training_config.model_name_or_path.split('/')[1]}-{training_config.dataset_name}",
        hyperparameters=json.dumps(training_config.__dict__, indent=4, sort_keys=True),
        eval_table="\n".join(f"| {k} | {v} |" for k, v in eval_results.items()),
        test_table="\n".join(f"| {k} | {v} |" for k, v in test_results.items()),
    )

    with open(f"{local_path}/README.md", "w") as f:
        f.write(model_card)

    if not os.getenv("HF_TOKEN") or not os.getenv("HF_USERNAME"):
        raise ValueError("HF_TOKEN, HF_USERNAME, and HF_REPO_NAME must be set in .env file")
    
    # upload model to hugging face
    api = HfApi(token=os.getenv("HF_TOKEN"))
    repo_url = api.create_repo(token=os.getenv("HF_TOKEN"), name=repo_name, exist_ok=True)
    model_repo = Repository(
        use_auth_token=os.getenv("HF_TOKEN"),
        clone_from=repo_url,
        local_dir=local_path,
        git_user=os.getenv("HF_USERNAME")
    )
    model_repo.push_to_hub()
    
    
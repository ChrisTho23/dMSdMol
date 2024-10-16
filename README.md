# dMSdMol

To run training on Sagemaker:
- Configure AWS CLI [reference](https://docs.aws.amazon.com/cli/v1/userguide/cli-authentication-user.html)
- Set wandb api key in .env
- Change training run configuration if necessary (./src/training/config.py)
- Run training script (./src/training/train_on_sagemaker.py)

This will create a docker container on AWS Sagemaker which will orchestrate the training run. Logs can be seen in the terminal.

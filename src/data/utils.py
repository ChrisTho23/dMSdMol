"""Utility functions for data scripts."""


def dataset_to_hub(dataset, repo_name, hf_username, hf_token):
    dataset.save_to_disk(f"./data/{repo_name}")
    dataset.push_to_hub(f"{hf_username}/{repo_name}", token=hf_token)

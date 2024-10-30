# Team 4: de-MS-tifying dark Mols - Evolved 2024 Hackathon Submission

This repository represents the submission of Team 4 (de-MS-tifying dark Mols) for the Evolved 2024 hackathon (organized by Evolutionary Scale and Lux Capital). Upon review of the over 200 proposals submitted, this project got selected as one of the 20 finalists participating in the 10-day hackathon period from October 10 to Octover 20 in 2024. dMSdMol was eventually ranked Top 3 in the Enveda challenge. For more information on the hackathon, visit [here](https://hackathon.bio/), and on the Enveda challenge, [here](https://hackathon.bio/challenge/enveda-2024). The hackathon also got mentioned in Nature Biotech, see [here](https://www.nature.com/articles/d41586-024-03335-z).

## The Challenge

Mass spectrometry (MS) identifies and quantifies molecules by measuring the mass-to-charge ratio (m/z) of ions, with tandem mass spectrometry (MS/MS) fragmenting molecules into smaller ions to deduce their structure. A key challenge is the limited spectral libraries, which restrict the identification of unknown compounds. To address this, **in silico fragmentation** predicts fragmentation patterns from molecular structures, expanding identification possibilities. 

Modern approaches include:

- **CFM-ID**: A probabilistic model predicting fragmentation based on learned rules and fragment stability.
- **GrAFF-MS by Enveda**: A graph neural network (GNN) approach treating mass spectrum prediction as a graph classification problem, offering high accuracy and efficiency.

The Enveda challenge aims to further advance in silico fragmentation, focusing on gathering more data for spectrum-to-structure translation.

## Our Approach

We identified the main limitation in solving the forward problem to be the restricted domain mappings between molecule structures and mass spectra. Drawing inspiration from other fields such as x-ray classification, self-driving cars, and satellite image-to-map translations, we decided to utilize an adversarial training schemes with two models—each mapping from one domain to the other—so we would not be constrained by the limited availability of existing mappings. 

Since models for both the forward (e.g., CFM-ID and GrAFF-MS) and backward problem (e.g., MS2Mol) have already been developed, we found this two-way approach to be very promising. We pursued a [**cycle contrastive GAN architecture**](https://arxiv.org/html/2407.11750v1), which allows us to create mappings between mass spectra and molecular structures, learning directly from the data in both directions.

However, since none of the models solving the forward and backward problems are open-source, we had to develop our models from scratch.

### Model Architecture

*This section will be detailed later (placeholder for model architectures).*

### Dataset and Training

We trained our forward and backward models on a **self-curated dataset**, which included the:

- **Enveda dataset**: Approximately 800,000 MS/MS spectra after cleaning.
- **Proprietary NIST dataset**: 800,000 high-resolution samples and 600,000 low-resolution samples after cleaning.

In addition to including the core domain data (mass-to-charge ratio and intensity for MS, SMILES for molecular structure), we also conditioned our joint probability distribution on the **machine configuration** (notably collision energy, machine type, and atmospheric pressure chemical ionization (APCI) pressure). Since these machine configurations can have a significant impact on fragmentation patterns, we found this to be an important consideration.

Unfortunately, due to time constraints during the 10-day hackathon, we were able to train the two one-directional models but did not get the chance to train the cycle GAN itself. However, based on prior training runs and precedents for cycle GANs, we are optimistic that this approach could be valuable in pushing the boundaries of molecular discovery.

We are currently evaluating our forward and backward models with experts and plan to begin training the cycle GAN soon.

## Training Setup

We used an **ml.p4d.24xlarge (8xA100 GPU) endpoint on AWS SageMaker** for training, utilizing Torch's data parallelism to optimize efficiency. Training for both the forward and backward directions took approximately 5 hours.

### How to Set Up and Train

1. Clone the repository:
    ```bash
    git clone https://github.com/ChrisTho23/dMSdMol.git
    cd dMSdMol
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. If you're using **Weights & Biases (wandb)** for logging, create a `.env` file from the provided template and add your wandb credentials.

4. Set up your **AWS credentials** using the AWS CLI:
    ```bash
    aws configure
    ```

5. Customize the `src/training/config.py` file as needed to specify your training parameters.

6. Launch training on SageMaker by running the following command:
    ```bash
    python src/training/train_on_sagemaker.py
    ```

Note: Since the **NIST dataset** is proprietary, the provided HuggingFace dataset is not publicly available. Users will need to supply their own dataset via the config file to train the models.

---

This project marks our first step toward addressing the limited domain mappings between mass spectra and molecular structures by exploring the potential of a cycle contrastive GAN for bidirectional MS spectrum and molecular structure mapping.

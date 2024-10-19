from dataclasses import dataclass


@dataclass
class Mol2MSLossConfig:
    mse_mz_weight: float = 0.5
    mse_intensity_weight: float = 0.5
    soft_jaccard_weight: float = 1.0
    sign_penalty_weight: float = 5.0

    soft_match_threshold: float = 100
    epsilon: float = 1e-6

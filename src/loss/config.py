from dataclasses import dataclass

@dataclass
class Mol2MSLossConfig:
    soft_match_threshold: float = 0.01
    length_penalty_weight: float = 0.0
    epsilon: float = 1e-6
    
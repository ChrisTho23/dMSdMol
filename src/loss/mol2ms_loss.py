import torch as t
import torch.nn as nn

from jaxtyping import Float

from .config import Mol2MSLossConfig

class Mol2MSLoss(nn.Module):
    def __init__(self, config: Mol2MSLossConfig = Mol2MSLossConfig()):
        super(Mol2MSLoss, self).__init__()
        self.config = config

    def forward(self, pred_mz: Float[t.Tensor, "batch seq-1"], mz: Float[t.Tensor, "batch seq-1"], pred_intensity: Float[t.Tensor, "batch seq-1"]) -> Float[t.Tensor, "1"]:
        pred_mz = pred_mz.unsqueeze(-1) # batch seq-1 1
        mz = mz.unsqueeze(1) # batch 1 seq-1

        # Calculate pairwise absolute differences in m/z
        mz_diff_matrix = t.abs(pred_mz - mz) # batch seq-1 seq-1

        # Soft match based on m/z proximity, weighted by true intensity
        # Find the closest match for each predicted m/z in true m/z and weight by true intensity
        soft_match = t.exp(-mz_diff_matrix / self.config.soft_match_threshold).max(2).values # batch seq-1

        # Weight the closest matches by true intensities
        weighted_soft_match = soft_match * pred_intensity # batch seq-1

        # Calculate intensity-weighted intersection and union
        soft_intersect = t.sum(weighted_soft_match, dim=1) # batch
        soft_union = pred_mz.shape[1] + mz.shape[2]

        # Calculate soft Jaccard
        soft_jaccard = (soft_intersect + self.config.epsilon) / (soft_union + self.config.epsilon) # batch
        jaccard_dist = 1 - soft_jaccard # batch

        # length penalty
        pred_length = t.sum(t.where(pred_mz >= 0.0, 1, 0), dim=1).squeeze(-1) # batch
        true_length = t.sum(t.where(mz >= 0.0, 1, 0), dim=2).squeeze(-1) # batch
        length_penalty = t.abs(pred_length - true_length) * self.config.length_penalty_weight

        total_loss = jaccard_dist + length_penalty # batch

        return total_loss.mean()
    
def calculate_loss(
    pred_mz: Float[t.Tensor, "batch seq-1"],
    mz: Float[t.Tensor, "batch seq-1"],
    pred_intensity: Float[t.Tensor, "batch seq-1"],
    intensity: Float[t.Tensor, "batch seq-1"],
    sign_penalty_weight: float = 5.0,
):
    mse_loss = nn.MSELoss()

    loss_mz = mse_loss(pred_mz, mz)
    loss_intensity = mse_loss(pred_intensity, intensity)

    sign_penalty_mz = t.mean(t.abs(t.sign(pred_mz) - t.sign(mz)))
    sign_penalty_intensity = t.mean(t.abs(t.sign(pred_intensity) - t.sign(intensity)))
    sign_penalty = sign_penalty_weight * (sign_penalty_mz + sign_penalty_intensity)

    # Scale losses to be of the same magnitude
    scaled_loss_mz = loss_mz / t.mean(mz**2)
    scaled_loss_intensity = loss_intensity / t.mean(intensity**2)

    total_loss = scaled_loss_mz + scaled_loss_intensity + sign_penalty

    return total_loss, loss_mz, loss_intensity, sign_penalty


        

import torch as t
import torch.nn as nn

from jaxtyping import Float

from .config import Mol2MSLossConfig

class Mol2MSLoss(nn.Module):
    def __init__(self, config: Mol2MSLossConfig = Mol2MSLossConfig()):
        super(Mol2MSLoss, self).__init__()
        self.config = config

    def _soft_jaccard_loss(self, pred_mz: Float[t.Tensor, "batch seq-1"], mz: Float[t.Tensor, "batch seq-1"], pred_intensity: Float[t.Tensor, "batch seq-1"]) -> Float[t.Tensor, "1"]:
        # Create masks for valid values
        pred_mask = pred_mz >= 0
        true_mask = mz >= 0

        pred_mz = t.where(pred_mask, pred_mz, t.zeros_like(pred_mz))
        pred_intensity = t.where(pred_mask, pred_intensity, t.zeros_like(pred_intensity))
        mz = t.where(true_mask, mz, t.zeros_like(mz))

        pred_mz = pred_mz.unsqueeze(2)  # batch pred_seq 1
        mz = mz.unsqueeze(1)  # batch 1 true_seq

        # Calculate pairwise absolute differences in m/z
        mz_diff_matrix = t.abs(pred_mz - mz)  # batch pred_seq true_seq

        # Soft match based on m/z proximity
        soft_match = t.exp(-mz_diff_matrix / self.config.soft_match_threshold).max(2).values  # batch pred_seq

        # Weight the closest matches by predicted intensities
        weighted_soft_match = soft_match * pred_intensity  # batch pred_seq

        # Calculate intensity-weighted intersection and union
        soft_intersect = t.sum(weighted_soft_match, dim=1)  # batch
        soft_union = pred_mask.sum(1) + true_mask.sum(1)  # batch

        # Calculate soft Jaccard
        soft_jaccard = (soft_intersect + self.config.epsilon) / (soft_union + self.config.epsilon)  # batch
        jaccard_dist = 1 - soft_jaccard  # batch

        return jaccard_dist.mean()
    
    def _mse_loss(self, pred_mz: Float[t.Tensor, "batch seq-1"], mz: Float[t.Tensor, "batch seq-1"], pred_intensity: Float[t.Tensor, "batch seq-1"], intensity: Float[t.Tensor, "batch seq-1"]) -> tuple[Float[t.Tensor, "1"], Float[t.Tensor, "1"]]:
        mse_loss = nn.MSELoss()
        loss_mz = mse_loss(pred_mz, mz)
        loss_intensity = mse_loss(pred_intensity, intensity)

        # Normalize MSE losses to [0, 1] range
        max_mz = t.max(t.abs(mz))
        max_intensity = t.max(t.abs(intensity))
        
        loss_mz_normalized = loss_mz / (max_mz ** 2)
        loss_intensity_normalized = loss_intensity / (max_intensity ** 2)

        return loss_mz_normalized, loss_intensity_normalized

    def _sign_penalty(self, pred_mz: Float[t.Tensor, "batch seq-1"], mz: Float[t.Tensor, "batch seq-1"], pred_intensity: Float[t.Tensor, "batch seq-1"], intensity: Float[t.Tensor, "batch seq-1"]) -> Float[t.Tensor, "1"]:
        sign_penalty_mz = t.mean(t.abs(t.sign(pred_mz) - t.sign(mz)))
        sign_penalty_intensity = t.mean(t.abs(t.sign(pred_intensity) - t.sign(intensity)))
        
        # The sign penalties are already between 0 and 2, so we divide by 2 to get [0, 1]
        sign_penalty = (sign_penalty_mz + sign_penalty_intensity) / 2

        return sign_penalty

    def forward(
        self, 
        pred_mz: Float[t.Tensor, "batch seq-1"], 
        mz: Float[t.Tensor, "batch seq-1"], 
        intensity: Float[t.Tensor, "batch seq-1"], 
        pred_intensity: Float[t.Tensor, "batch seq-1"]
    ) -> tuple[Float[t.Tensor, "1"], Float[t.Tensor, "1"], Float[t.Tensor, "1"], Float[t.Tensor, "1"]]:
        soft_jaccard_loss = self._soft_jaccard_loss(
            pred_mz=pred_mz,
            mz=mz,
            pred_intensity=pred_intensity,
        )

        loss_mz, loss_intensity = self._mse_loss(
            pred_mz=pred_mz,
            mz=mz,
            pred_intensity=pred_intensity,
            intensity=intensity,
        )
        sign_penalty = self._sign_penalty(pred_mz, mz, pred_intensity, intensity)

        soft_jaccard_loss = self.config.soft_jaccard_weight * soft_jaccard_loss 
        loss_mz = self.config.mse_mz_weight * loss_mz 
        loss_intensity = self.config.mse_intensity_weight * loss_intensity 
        sign_penalty = self.config.sign_penalty_weight * sign_penalty

        return soft_jaccard_loss, loss_mz, loss_intensity, sign_penalty


        

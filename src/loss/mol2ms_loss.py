import torch as t
import torch.nn as nn

from jaxtyping import Float

from .config import Mol2MSLossConfig

class Mol2MSLoss(nn.Module):
    def __init__(self, config: Mol2MSLossConfig = Mol2MSLossConfig()):
        super(Mol2MSLoss, self).__init__()
        self.config = config

    def _filter_ms_data(self, tensor: Float[t.Tensor, "batch seq-1"]) -> Float[t.Tensor, "batch seq-1"]:
        mask = tensor < 0
        first_negative = mask.cumsum(dim=-1).argmax(dim=-1)
        tensor = t.stack([row[:idx] for row, idx in zip(tensor, first_negative)])
        return tensor

    def _soft_jaccard_loss(self, pred_mz: Float[t.Tensor, "batch seq-1"], mz: Float[t.Tensor, "batch seq-1"], pred_intensity: Float[t.Tensor, "batch seq-1"]) -> Float[t.Tensor, "1"]:
        pred_mz = self._filter_ms_data(pred_mz) # batch ms_seq
        pred_mz = pred_mz.unsqueeze(-1) # batch ms_seq 1
        mz = self._filter_ms_data(mz) # batch ms_seq
        mz = mz.unsqueeze(1) # batch 1 ms_seq

        # Calculate pairwise absolute differences in m/z
        mz_diff_matrix = t.abs(pred_mz - mz) # batch ms_seq ms_seq

        # Soft match based on m/z proximity, weighted by true intensity
        # Find the closest match for each predicted m/z in true m/z and weight by true intensity
        soft_match = t.exp(-mz_diff_matrix / self.config.soft_match_threshold).max(2).values # batch ms_seq

        # Weight the closest matches by true intensities
        weighted_soft_match = soft_match * pred_intensity # batch ms_seq

        # Calculate intensity-weighted intersection and union
        soft_intersect = t.sum(weighted_soft_match, dim=1) # batch
        soft_union = pred_mz.shape[1] + mz.shape[2]

        # Calculate soft Jaccard
        soft_jaccard = (soft_intersect + self.config.epsilon) / (soft_union + self.config.epsilon) # batch
        jaccard_dist = 1 - soft_jaccard # batch

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


        

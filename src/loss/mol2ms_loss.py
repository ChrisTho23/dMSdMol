import torch
import torch.nn as nn

class SoftJaccardLossWithTrueIntensity(nn.Module):
    def __init__(self, threshold=0.1, length_penalty_weight=0.5, smooth=1e-6):
        super(SoftJaccardLossWithTrueIntensity, self).__init__()
        self.threshold = threshold  # Interval threshold for soft match
        self.length_penalty_weight = length_penalty_weight  # Weight for the length penalty
        self.smooth = smooth  # Smoothing term to avoid division by zero

    def forward(self, y_pred, y_true):
        """
        y_pred and y_true are lists of tensors of shape [(sequence_length_pred, 2), ...]
        where each tensor row represents [m/z, intensity].
        """
        batch_size = len(y_pred)
        losses = []

        for i in range(batch_size):
            # Separate m/z and intensity for y_pred and y_true
            mz_pred = y_pred[i][:, 0].unsqueeze(1)  # (sequence_length_pred, 1)
            mz_true = y_true[i][:, 0].unsqueeze(0)  # (1, sequence_length_true)
            intensity_true = y_true[i][:, 1]  # Intensities of true values (used for weighting)

            # Calculate pairwise absolute differences in m/z
            diff_matrix = torch.abs(mz_pred - mz_true)

            # Soft match based on m/z proximity, weighted by true intensity
            # Find the closest match for each predicted m/z in true m/z and weight by true intensity

            soft_matches = torch.exp(-diff_matrix / self.threshold).max(dim=0).values  # Closest pred m/z for each true m/z

            # Step 2: Weight the closest matches by true intensities
            weighted_soft_matches = soft_matches * intensity_true

            # Step 3: Calculate intensity-weighted intersection
            soft_intersection = torch.sum(weighted_soft_matches)

            # Step 4: Calculate union based on m/z count (no subtraction)
            soft_union = len(y_pred[i]) + len(y_true[i])


            # Soft Jaccard index for the sequence pair
            soft_jaccard = (soft_intersection + self.smooth) / (soft_union + self.smooth)
            jaccard_loss = 1 - soft_jaccard  # Loss is 1 - Jaccard index for this sequence pair

            # Length penalty for mismatched sequence lengths
            length_penalty = abs(len(y_pred[i]) - len(y_true[i])) * self.length_penalty_weight

            # Combined loss for this sequence pair
            total_loss = jaccard_loss + length_penalty
            losses.append(total_loss)

        # Mean loss over the batch
        return torch.stack(losses).mean()
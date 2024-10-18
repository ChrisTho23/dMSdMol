import torch
import torch.nn as nn


torch.nn.function.F

class contrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature=temperature

    def forward(self, embedding1, embedding2,normalize=False):
        """
        y_pred and y_true are lists of tensors of shape [(sequence_length_pred, 2), ...]
        where each tensor row represents [m/z, intensity].
        


        """

        #
        if normalize:
            embedding1= nn.function.normalize(embedding1,dim=1)
            embedding2= nn.functional.normalize(embedding2,dim=2)
        combined=torch.cat([embedding1,embedding2],dim=0)
        sim_matrix=torch.matmul(combined,combined.T)
        batch_size = embedding1.shape[0]
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(embedding1.device)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(embedding1.device)
        sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)

        # Scale similarity scores by temperature
        sim_matrix /= self.temperature
        
        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(sim_matrix, labels)
        # Mean loss over the batch
        return loss
    def adjust_tempWithFunction(func):
        



class interContrastiveLoss(nn.Module):
    """basically the intercontrastive loss, so between, the following that should be done 
    so between the 
    Args:
        nn (_type_): _description_
    """

    pass



class WganLoss(nn.Module):
    pass


class 



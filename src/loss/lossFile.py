import torch
import torch.nn as nn


import torch.nn.functional as F

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
        pass


class interContrastiveLoss(nn.Module):
    """basically the intercontrastive loss, so between, the following that should be done 
    so between the 
    Args:
        nn (_type_): _description_
    """
    pass




def wgan_loss(D, real, fake, device, lambda_gp=10):
    real_validity = D(real)
    fake_validity = D(fake)

    w_loss = torch.mean(fake_validity) - torch.mean(real_validity)

    # Gradient penalty
    alpha = torch.rand((real.size(0), 1, 1), device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    interpolated_validity = D(interpolated)

    grad_outputs = torch.ones(interpolated_validity.size(), device=device)
    gradients = torch.autograd.grad(
        outputs=interpolated_validity, inputs=interpolated,
        grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return w_loss + gradient_penalty
def ragan_loss(D_real, D_fake):
    """Relativistic Average GAN Loss."""
    # Discriminator loss
    D_loss_real = torch.mean(torch.nn.functional.softplus(-(D_real - torch.mean(D_fake))))
    D_loss_fake = torch.mean(torch.nn.functional.softplus(D_fake - torch.mean(D_real)))
    D_loss = (D_loss_real + D_loss_fake) / 2

    # Generator loss (relativistic)
    G_loss_real = torch.mean(torch.nn.functional.softplus(D_real - torch.mean(D_fake)))
    G_loss_fake = torch.mean(torch.nn.functional.softplus(-(D_fake - torch.mean(D_real))))
    G_loss = (G_loss_real + G_loss_fake) / 2

    return D_loss, G_loss




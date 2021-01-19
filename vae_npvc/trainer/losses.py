import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad as torch_grad

def gradient_penalty_loss( x_real, x_fake, discriminator):
    assert x_real.shape == x_fake.shape
    batch_size = x_real.size(0)
    device = x_real.device

    alpha_size = [1 for i in range(x_real.ndim)]
    alpha_size[0] = batch_size
    alpha = torch.rand(alpha_size, device=device)

    interpolated = alpha * x_real.data + (1 - alpha) * x_fake.data
    interpolated.requires_grad = True

    inte_logit = discriminator(interpolated)

    gradients = torch_grad(outputs=inte_logit, inputs=interpolated,
                           grad_outputs=torch.ones_like(inte_logit, device=device),
                           create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    grad_l2 = torch.sqrt(torch.sum(gradients ** 2, dim=-1) + 1e-12)
    gradient_penalty = ((grad_l2 - 1) ** 2).mean()

    return gradient_penalty
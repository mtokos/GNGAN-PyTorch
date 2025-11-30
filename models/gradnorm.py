import torch
import torch.nn as nn

def normalize_gradient(net_D, x, **kwargs):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    x.requires_grad_(True)
    f = net_D(x, **kwargs)
    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(grad, start_dim=1).cpu(), p=2, dim=1)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
    f_hat = (f / (grad_norm.cuda() + torch.abs(f)))
    return f_hat

def get_gradient(net_D, x, **kwargs):
    f = net_D(x, **kwargs)
    return f

class GradNorm(nn.Module):
    def __init__(self, net_D, out_channels = None):
        super().__init__()
        self.net_D = net_D
        self.out_channels = out_channels

    def forward(self, input):
        """
                        f
        f_hat = --------------------
                || grad_f || + | f |
        """
        input.requires_grad_(True)
        #try:
        f = self.net_D(input)
        # except:
        #    print(input, self.out_channels)
        #    f = self.net_D(in_channels=self.out_channels, out_channels=self.out_channels)
        grad = torch.autograd.grad(
            f, [input], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
        grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
        grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
        f_hat = (f / (grad_norm + torch.abs(f)))
        return f_hat

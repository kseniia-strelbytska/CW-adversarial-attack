import torch
from torch.optim import Adam, AdamW
import torch_dct
from image_support import load_image_url, get_normalized_image, get_unnormalized_image, display_image

class AdversarialNoise(torch.nn.Module):
    def __init__(self, image_x):
        super().__init__()
        # define delta as 0.5(tahn(w) + 1) - x
        # (where delta is the added noise)
        self.w = torch.nn.Parameter(torch.atanh(2 * image_x - 1))
        self.register_buffer("x_orig", image_x) 

    def forward(self, model, x, target, c):
        model.eval()

        x_pred = 0.5 * (torch.tanh(self.w) + 1)
        logits = model(get_normalized_image(x_pred))[0]

        # loss for distance to target class
        f = torch.maximum(torch.tensor(0), torch.masked_select(logits, torch.arange(0, logits.size(0)) != target).max() - logits[target])

        delta = 0.5 * (torch.tanh(self.w) + 1) - x

        # L2 distance 
        d = torch.pow(delta, 2).sum()

        loss = d + c * f
        return self.w, loss

class LowFreqAdversarialNoise(torch.nn.Module):
    def __init__(self, x, ratio):
        super().__init__()
        # define delta as 0.5(tahn(w) + 1) - x
        # (where delta is the added noise)

        self.side = round(ratio * x.shape[2])
        self.v = torch.nn.Parameter(torch.rand_like(x, dtype=torch.float32) * 1e-3)
        self.register_buffer("x_orig", x)

        mask = torch.zeros_like(x, dtype=torch.float)
        mask[:, :, :self.side, :self.side] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, model, x, target, c, alpha):
        model.eval()

        delta = self.mask * torch_dct.dct_2d(self.v, norm='ortho')
        delta = 10* torch_dct.idct_2d(delta, norm='ortho')
        
        x_pred = (x + delta)
        logits = model(get_normalized_image(x_pred))[0]
        
        # loss for distance to target class
        f = torch.clamp(torch.masked_select(logits, torch.arange(0, logits.size(0), device=logits.device) != target).max() - logits[target], min=0.0)

        # L2 distance
        d = torch.pow(delta, 2).sum()

        # penalty for high-frequency noide

        loss = d + c * f
        return delta, loss

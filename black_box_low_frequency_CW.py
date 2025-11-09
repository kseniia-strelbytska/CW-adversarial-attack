import torch
from torch.optim import Adam, AdamW
import torch_dct
from image_support import load_image_url, get_normalized_image, get_unnormalized_image, display_image

class LowFreqAdversarialNoise(torch.nn.Module):
    def __init__(self, image_x, ratio):
        super().__init__()
        # define delta as 0.5(tahn(w) + 1) - x
        # (where delta is the added noise)

        self.side = round(ratio * image_x.shape[2])

        self.v = torch.nn.Parameter(torch.rand((image_x.shape[0], image_x.shape[1], self.side, self.side), dtype=torch.float32))
        
        self.register_buffer("x_orig", image_x)

    def forward(self, model, x, target, c, alpha):
        model.eval()

        V = torch.zeros_like(x, dtype = torch.float32)
        V[:, :, :self.side, :self.side] = self.v

        delta = torch_dct.idct_2d(V)

        x_pred = (x + delta)

        logits = model(get_normalized_image(x_pred))[0]

        # print(x.shape, x_pred.shape, x[0][0][0], x_pred[0][0][0])
        # print(unmasked.min(), unmasked.max())
        # print(torch_dct.idct_2d(unmasked, norm='ortho'))
        # print(self.w)

        # loss for distance to target class
        f = torch.maximum(torch.tensor(0), torch.masked_select(logits, torch.arange(0, logits.size(0)) != target).max() - logits[target])

        # L2 distance
        d = torch.pow(delta, 2).sum()

        # penalty for high-frequency noide

        loss = d + c * f
        return self.v, loss
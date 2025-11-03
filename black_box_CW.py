import torch
from torch.optim import Adam

class AdversarialNoise(torch.nn.Module):
    def __init__(self, image_x):
        # define delta as 0.5(tahn(w) + 1) - x
        # (where delta is the added noise)
        self.w = torch.nn.Parameter(torch.atan(2 * image_x - 1))

    def forward(self):
        return self.w 

def get_optimized_noise(model, x, epochs=10, lr=0.01, target=1):
    C, H, W = x.shape
    # initialise for delta = 0
    adv_model = AdversarialNoise(x) 
    optimizer = Adam(adv_model.parameters, lr=0.01)

    c = 0.01

    for _ in range(epochs):
        w = adv_model()
        logits = model(w)

        # loss for distance to target class
        f = max(torch.tensor(0), logits[torch.arange(0, logits.size(1)) != target].max() - logits[target])
        x_pred = 0.5 * torch.tan(w + 1) - x 
        # L2 distance 
        d = torch.pow(x_pred, 2).sum()

        loss = d + c * f

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w = adv_model() 

    return 0.5 * (torch.tan(w) + 1) - x
    
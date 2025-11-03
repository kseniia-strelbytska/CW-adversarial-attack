import torch
from torch.optim import Adam
from model import top_5_classes
from image_support import load_image_url, get_normalized_image, get_unnormalized_image, display_image
from torchvision import transforms

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

def get_optimized_noise(model, x, epochs=10, lr=0.01, target=1):
    B, C, H, W = x.shape
    # initialise for delta = 0
    adv_model = AdversarialNoise(x) 
    optimizer = Adam(adv_model.parameters(), lr=0.01)

    # constant that defines the confidence in adversality
    c = 1

    for epoch in range(epochs):
        w, loss = adv_model(model, x, target, c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Loss after {epoch} epochs is {loss:.4f}') 
            x_pred = 0.5 * (torch.tanh(w) + 1)

            display_image(x_pred)

            y_pred = top_5_classes(model(get_normalized_image(x_pred)))[0]
            print(f'Best prediction: \'{y_pred[0]}\' with probability {y_pred[1]:.4f}') 

    w, loss = adv_model(model, x, target, c) 

    return 0.5 * (torch.tanh(w) + 1)
    
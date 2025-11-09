import torch
from torch.optim import Adam, AdamW
from model import top_5_classes
from image_support import load_image_url, get_normalized_image, get_unnormalized_image, display_image
from torchvision import transforms, utils
from model import get_imagenet_class_label
import torch_dct

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

def get_optimized_noise(model, x, adv_model, low_freq=False, epochs=10, lr=0.01, target=1, confidence=0.9):
    # class_label = get_imagenet_class_label(target)

    B, C, H, W = x.shape
    # initialise for delta = 0
    optimizer = AdamW(adv_model.parameters(), lr=0.01)

    # constant that defines the confidence in adversality
    c = 1
    alpha = 0.1

    for epoch in range(epochs):
        v, loss = adv_model(model, x, target, c, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Loss after {epoch} epochs is {loss:.4f}') 

            if low_freq == False:
                x_pred = 0.5 * (torch.tanh(v) + 1)
                delta = (0.5 * (torch.tanh(v) + 1) - x)
            else:
                V = torch.zeros_like(x, dtype = torch.float32)
                V[:, :, :adv_model.side, :adv_model.side] = v

                delta = torch_dct.idct_2d(V)
                x_pred = (x + delta).clamp(0.0, 1.0)

                noise = x_pred - x # some might become negative. how do we display negative? as black

                mx = delta.max()
                # print(mx)

                # if mx >= torch.tensor(0.0041):
                #     for i in range(delta.shape[2]):
                #         for j in range(delta.shape[3]):
                #             # if torch.isclose(noise[0, :, i, j].sum(), torch.tensor(0.0)) == False:
                #             if delta[0, :, i, j].max() >= mx:
                #                 single_colour = torch.tile(delta[0, :, i, j], (1, 3, 100, 100))
                #                 print(single_colour)

                #                 display_image(single_colour)

                #                 break

            print((x_pred-x).max())
            display_image(delta)

            utils.save_image(x_pred, './cat_image.png')

            u_pred = model(get_normalized_image(x_pred)).softmax(dim=-1)
            prediction = u_pred.argmax(1)[0]
            print(f'Preduction is class {prediction} with probability {u_pred[0][prediction]}; class 951 probability is {u_pred[0][951]}')

            # y_pred = top_5_classes(model(get_normalized_image(x_pred)))[0]
            # print(f'Best prediction: \'{y_pred[0]}\' with probability {y_pred[1]:.4f}') 

            # terminate if confidence treshold is reached
            # if epoch >= 500 and y_pred[0] == class_label and y_pred[1] >= confidence:
            #     break 

    v, loss = adv_model(model, x, target, c, alpha)
    
    if low_freq == False:
        x_pred = 0.5 * (torch.tanh(w) + 1)
        delta = (0.5 * (torch.tanh(v) + 1) - x)
    else:
        V = torch.zeros_like(x, dtype = torch.float32)
        V[:, :, :adv_model.side, :adv_model.side] = v

        delta = torch_dct.idct_2d(V)

        x_pred = (x + delta).clamp(0.0, 1.0)

    return x_pred
    
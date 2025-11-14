import torch
from torchvision import utils
from image_support import get_normalized_image

# https://arxiv.org/pdf/2005.14137

# x doesn't have a batch dimension. 

def phi(model, x, y_mal):
    # returns sign(S) = sign(p[y_mal] - p[j != y_mal])

    logits = model(get_normalized_image(x).unsqueeze(0))[0]
    L = logits.shape[0] # number of classes

    pred = torch.argmax(logits)

    # if s(x) = 0, phi(x) = 1

    return 1 if pred == y_mal else -1

def estimate_gradient(model, x, y_mal, delta=0.01, B=100):
    # returns monte carlo gradient estimation of S at point x

    # need to sample B adversarial noise perturbations
    # shape : (B, C, H, W)
    C, H, W = x.shape
    u = torch.randn((B, C, H, W))

    # normalise (L2 norm), so that SUM(ub) = 1 (unit length vector). 
    # effect: removes the effect of magnitude from noise
    L2 = torch.sqrt((u**2).sum(dim=-1).sum(dim=-1).sum(dim=-1))
    u /= L2[:, None, None, None]

    # create B perturbed iamges 
    x_pert = x.unsqueeze(0) + delta * u

    gradient = torch.zeros(C, H, W)
    for idx in range(B):
        p = phi(model, x_pert[idx], y_mal) * u[idx]
        gradient += p 

    gradient *= 1/B

    return gradient

def project_to_boundary(model, x_hat, x_target, y_mal):
    # run binary search to find the greatest alpha, 
    # s.t. phi(x_proj) = 1, phi(x_proj+e) = -1, where e is degree of accuracy (e.g. 0.0001)
    # i.e. get x_proj as close as possible to target (higher alpha), while still misclassifying x_proj as y_mal
    # returns x_proj = alpha * x_target + (1 - alpha) * x_hat 

    l_b, r_b = 0.0, 1

    # degree of accuracy
    ACC = 0.0001

    while r_b - l_b > ACC:
        # print(l_b, r_b)
        alpha = (r_b + l_b) / 2.0

        # use vector interpolation to along x_target -> x_hat vector
        x_proj = alpha * x_target + (1 - alpha) * x_hat 

        if phi(model, x_proj, y_mal) == 1: # still y_mal
            l_b = alpha 
        else:
            r_b = alpha

    alpha = (r_b + l_b) / 2.0
    return alpha * x_target + (1 - alpha) * x_hat 

def boundary_attack(model, x, x_target, y_mal, eps=0.001, epochs=2*10**6, delta=0.001, B=100):
    # returns adversarial image apearing to be x_target with label y_mal
    # x is an example from malicious class
    # we want to keep label y_mal (malicious), 
    # but optimize x to be as close to x_target as possible
    # (i.e. L2 norm of (x - x_target) is as small as possible)

    x = project_to_boundary(model, x, x_target, y_mal)

    for epoch in range(epochs):
        gradient = estimate_gradient(model, x, y_mal, delta, B)
        L2 = torch.sqrt(torch.sum(gradient**2))
        gradient /= L2

        # take a step to increase prob. of y_mal class
        x_hat = x + eps * gradient

        # project to decision boundary 
        x_proj = project_to_boundary(model, x_hat, x_target, y_mal)
        L2 = torch.sqrt(torch.sum((x - x_proj)**2))
        x = x_proj

        L2_target = torch.sqrt(torch.sum((x - x_target)**2))

        print(f"Step {epoch} done. L2 from last step: {L2:.10f}. L2 to target {L2_target}")
        utils.save_image(x, './images/intermidiate_BA_image.png')

        if (epoch + 1) % 100 == 0:
            utils.save_image(x, f'./images/intermidiate_BA_image_epoch{epoch}.png')
    
    return x


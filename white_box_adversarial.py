# Performs one backprop iteration on image 
# Optimises for != original class if targeted=False 
#   and for target_class_label otherwise
def get_adversarial(x, alpha=0.001, targeted = False, target_class_label=0):
  x.requires_grad = True
  y = resnet18(x)

  loss = nn.functional.cross_entropy(y, torch.tensor([target_class_label], dtype = torch.long))

  loss.backward()

  with torch.no_grad():
    x_adv = x + alpha * torch.sign(x.grad) * (1 if targeted == False else -1)
    x_adv.grad = None

  return x_adv

# Fast Gradient Sign method
def iterative_FGS(x, epochs=10, alpha=0.001, targeted = False, target_class_label=0):
  for _ in range(epochs):
    x = get_adversarial(x, alpha, targeted, target_class_label)
    predictions = top_5_classes(resnet18(x))
    # print(predictions)
    # display_image(get_unnormalized_image(x))

  return x
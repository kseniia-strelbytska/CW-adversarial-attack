import torch
from torchvision import models
from image_support import load_image, get_normalized_image, get_unnormalized_image, display_image
from black_box_CW import get_optimized_noise

if __name__ == "__main__":
    resnet18 = models.resnet18(pretrained=True)

    # cat image
    img = load_image('https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg?cs=srgb&dl=pexels-pixabay-104827.jpg&fm=jpg')
    img = get_normalized_image(img)

    print(img.shape)

    exit(0)

    # 951 class = 'lemon'
    noise = get_optimized_noise(model=resnet18, x=img, epochs=10, lr=0.01, target=951)
    adv_image = img + noise 
    display_image(get_normalized_image(adv_image))
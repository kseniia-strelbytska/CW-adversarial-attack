import torch
from torchvision import models, transforms
from image_support import load_image, get_normalized_image, get_unnormalized_image, display_image
from black_box_CW import get_optimized_noise
from model import top_5_classes

if __name__ == "__main__":
    resnet18 = models.resnet18(pretrained=True)
    resnet18.eval()

    # Siamese cat image
    img = load_image('https://www.catster.com/wp-content/uploads/2023/11/Siamese-Cat_Andreas-LischkaPixabay.jpg')

    transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                                ])

    img = transform(img).unsqueeze(0)

    # norm_img = get_normalized_image(img)

    # 951 class = 'lemon'
    adv_img = get_optimized_noise(model=resnet18, x=img, epochs=10, lr=0.01, target=951)

    predictions = top_5_classes(resnet18(get_normalized_image(adv_img)))
    print(predictions)
    
    display_image(adv_img)
import torch
from torchvision import models, transforms, utils
from image_support import load_image_url, load_image_file, get_normalized_image, get_unnormalized_image, display_image
from noise_optimization import get_optimized_noise
from noise_models import LowFreqAdversarialNoise
from boundary_attack import boundary_attack
from model import top_5_classes, get_imagenet_class_idx
import argparse
import json
import requests

# receives img in pil format
def process_query(model, img, class_idx, output_file=None, epochs=100000, lr=0.01, confidence=0.9):
    model.eval() 

    transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                                ])

    img = transform(img).unsqueeze(0)
    adv_model = LowFreqAdversarialNoise(img, 1/4)
    adv_img = get_optimized_noise(model=model, x=img, adv_model=adv_model, low_freq=True, epochs=epochs, lr=lr, target=class_idx, confidence=confidence)

    noise = adv_img - img
    torch.save(noise, './noise_tensor')
    utils.save_image(noise, './noise_image.png')
    # prediction = top_5_classes(model(get_normalized_image(adv_img)))[0]

    # if output_file == None:
    #     output_file = f'./final_image_{prediction[0]}.png'
    
    # utils.save_image(adv_img, output_file)
    
    return output_file

def process_boundary_attack_query(output_file, model, x, x_target, y_mal, eps=0.01, epochs=1000, delta=0.01, B=100): 
    transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                                ])

    x, x_target = transform(x), transform(x_target)

    x_adv = boundary_attack(model, x, x_target, y_mal, eps, epochs, delta, B)

    if output_file == None:
        output_file = './images/BA_adversarial_image.png'
    utils.save_image(noise, output_file)

    return output_file

def get_image_prediction(model, img):
    model.eval() 

    transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                                ])
                                
    img = transform(img).unsqueeze(0)
    prediction = top_5_classes(model(get_normalized_image(img)))[0]

    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates an adversarial image for a given black-box model, image and target class')
    parser.add_argument('--model', type=str, required=True, help='Black-box model for attack')
    parser.add_argument('--image_url', type=str, required=False, help='URL of the image file') 
    parser.add_argument('--image_file', type=str, required=False, help='Path to the image file')
    parser.add_argument('--output_file', type=str, required=False, help='Path for the adverserial image output')
    parser.add_argument('--target_class', type=str, required=True, help='Target class of the adversarial image for the chosen model')
    parser.add_argument('--epochs', type=str, required=False, help='Number of epochs to train (with automatic termination if confidence treshold is reached)')
    parser.add_argument('--lr', type=str, required=False, help='Learning rate')
    parser.add_argument('--confidence', type=str, required=False, help='Confidence treshold for termination (searching for model_prediction >= confidence)')
    
    args = parser.parse_args()

    model_name = args.model 
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
    model.eval()

    target_class = args.target_class 
    class_idx = get_imagenet_class_idx(target_class)

    # 1: 'goldfish, Carassius auratus' # 0.99
    # 283: 'Persian cat' # 0.70

    x = load_image_file('./images/goldfish.JPEG')
    x_target = load_image_file('./images/Persian_cat.JPEG')
    y_mal = 1

    if args.image_url is not None:
        img = load_image_url(args.image_url)
    elif args.image_file is not None:
        img = load_image_file(args.image_file)

    output_file = args.output_file

    output_file = process_boundary_attack_query(None, model, x, x_target, y_mal)
    # output_file = process_query(model=model, img=img, class_idx=951, output_file=output_file, epochs=100000, lr=0.001, confidence=0.8)
    
    label, prob = get_image_prediction(model, load_image_file(output_file))
    print(f'Best prediction: class \'{label}\' with probability {prob:.4f}') 

import torch
from torchvision import models, transforms, utils
from image_support import load_image_url, load_image_file, get_normalized_image, get_unnormalized_image, display_image
from black_box_CW import get_optimized_noise
from model import top_5_classes
import argparse
import json
import requests

# receives img in pil format
def process_query(model, img, target_class, output_file=None, epochs=1000, lr=0.01):
    model.eval() 

    transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                                ])

    img = transform(img).unsqueeze(0)
    adv_img = get_optimized_noise(model=model, x=img, epochs=epochs, lr=lr, target=target_class)
    prediction = top_5_classes(model(get_normalized_image(adv_img)))[0]

    if output_file == None:
        output_file = f'./final_image_{prediction[0]}.png'
    
    utils.save_image(adv_img, output_file)
    
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

def get_imagenet_class_idx(class_label):
    imagenet_labels_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    response = requests.get(imagenet_labels_url)
    imagenet_class_names = json.loads(response.text)

    return imagenet_class_names.index(class_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates an adversarial image for a given black-box model, image and target class')
    parser.add_argument('--model', type=str, required=True, help='Black-box model for attack')
    parser.add_argument('--image_url', type=str, required=False, help='URL of the image file') 
    parser.add_argument('--image_file', type=str, required=False, help='Path to the image file')
    parser.add_argument('--output_file', type=str, required=False, help='Path for the adverserial image output')
    parser.add_argument('--target_class', type=str, required=True, help='Target class of the adversarial image for the chosen model')
    args = parser.parse_args()

    model_name = args.model 
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resne34':
        model = models.resnet34(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
    model.eval()

    target_class = args.target_class 
    class_idx = get_imagenet_class_idx(target_class)

    if args.image_url is not None:
        img = load_image_url(args.image_url)
    elif args.image_file is not None:
        img = load_image_file(args.image_file)

    output_file = args.output_file

    output_file = process_query(model=model, img=img, target_class=class_idx, output_file=output_file, epochs=10000, lr=0.001)
    
    label, prob = get_image_prediction(model, load_image_file(output_file))
    print(f'Best prediction: class \'{label}\' with probability {prob:.4f}') 

import requests
import json
from torch.nn.functional import softmax

def top_5_classes(y, class_names = None):
  imagenet_labels_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
  response = requests.get(imagenet_labels_url)
  imagenet_class_names = json.loads(response.text)

  if class_names==None:
    imagenet_labels_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    response = requests.get(imagenet_labels_url)
    class_names = json.loads(response.text)

  p = softmax(y[0,:], dim=0)
  values, indices = p.topk(5)
  return [(class_names[index], value) for index, value in zip(indices.detach().cpu().numpy(), values.detach().cpu().numpy())]

def get_imagenet_class_idx(class_label):
    imagenet_labels_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    response = requests.get(imagenet_labels_url)
    imagenet_class_names = json.loads(response.text)

    return imagenet_class_names.index(class_label)

def get_imagenet_class_label(class_idx):
    imagenet_labels_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    response = requests.get(imagenet_labels_url)
    imagenet_class_names = json.loads(response.text)

    return imagenet_class_names[class_idx]
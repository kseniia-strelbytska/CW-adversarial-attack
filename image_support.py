import urllib.request
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

# url = 'https://upload.wikimedia.org/wikipedia/commons/7/7e/Oebb_1216_050-5_at_wegberg_wildenrath.jpg'
# url = 'https://www.auran.com/trainz/database/images/taurus/1016_006.jpg'

def load_image(url):
  try:
      # Add a User-Agent header to the request
      req = urllib.request.Request(
          url,
          data=None,
          headers={
              'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
          }
      )
      with urllib.request.urlopen(req) as url_response:
        img = Image.open(url_response)
        print("Image loaded successfully!")

        # You can now work with the 'img' object, e.g., display it:
        return torch.tensor(np.array(img)).unsqueeze(0) # return with batch dimension
  except Exception as e:
      print(f"Error loading image: {e}")

# receives tensor
def get_normalized_image(x):
    print("Compose")
    normalize = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                )
                                ])

    print("Composed")

    return normalize(x).clip(0.0, 1.0)

# receives tensor
def get_unnormalized_image(x):
  unnormalize = transforms.Compose([
        transforms.Normalize(mean = [0.0, 0.0, 0.0], std = [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [1.0, 1.0, 1.0])
    ])

  return unnormalize(x).clip(0.0, 1.0)

# receives tensor with batch dimension
def display_image(x):
  img = transforms.ToPILImage()(x[0])
  plt.imshow(img)
  plt.show()
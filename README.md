# Black-Box Carlini–Wagner Attack (Low-Frequency Variant)

This repository implements a black-box variant of the Carlini & Wagner (CW) adversarial attack, extended to the low-frequency (LF-DCT) domain. The implementation supports both:

- Standard CW attacks in pixel space (white-box).
- Low-frequency CW attacks using a truncated 2D Discrete Cosine Transform (DCT) basis (black-box variant).

Carlini-Wagner:
https://arxiv.org/pdf/1608.04644
Low-frequncy:
https://arxiv.org/pdf/1809.08758

## Overview

This code trains a small per-image adversarial variable `w`, optimized so that the adversarial image `x'` is bounded in `[0, 1]` by a tanh-based transform:

$$
x' = 0.5 \cdot (\tanh(w) + 1)
$$

The perturbation is `δ = x' - x`. The optimization minimizes the objective

$$
L(x') = \|x' - x\|_2^2 + c \cdot f(x')
$$

where `c` is a scalar weight and `f(x')` is the CW margin loss defined as

$$
f(x') = \max\big(\max_{i \ne t} Z(x')_i - Z(x')_t,\, 0\big)
$$

Here `Z(x')` are the logits produced by the model and `t` is the target class.

## Features

- Pixel-space CW attack (white-box).
- Low-frequency DCT-space CW attack (black-box variant).
- Per-image optimization using PyTorch autograd.
- Configurable loss weighting (`c`), step size, and iteration count.
- Compatible with any pre-trained model (for example, `resnet18` from `torchvision`).

## Usage
Generates adversarial images for a given black-box model, image, and target class.

### Installation
Install the required dependencies:
bashpip install -r requirements.txt
Usage
bashpython main.py [-h] --model MODEL [--image_url IMAGE_URL] [--image_file IMAGE_FILE] 
               [--output_file OUTPUT_FILE] --target_class TARGET_CLASS
### Arguments

-h, --help - Show help message and exit
--model MODEL - Black-box model for attack (options: 'resnet18', 'resnet34', 'resnet50')
--image_url IMAGE_URL - URL of the image file
--image_file IMAGE_FILE - Path to the image file
--output_file OUTPUT_FILE - Path for the adversarial image output (default: './final_image.png')
--target_class TARGET_CLASS - Target class of the adversarial image for the chosen model

Example
bashpython main.py --model resnet18 --image_file ./cat.jpg \
               --output_file final_image_cat_to_lemon.png --target_class lemon --confidence 0.9

### Requirements
Create a requirements.txt file with your dependencies:
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.21.0
requests>=2.28.0
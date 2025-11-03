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

## To run:
Usage: main.py [-h] --model MODEL [--image_url IMAGE_URL] [--image_file IMAGE_FILE] [--output_file OUTPUT_FILE] --target_class TARGET_CLASS

Generates an adversarial image for a given black-box model, image and target class

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Black-box model for attack ('resnet18', 'resnet34', 'resnet50')
  --image_url IMAGE_URL
                        URL of the image file
  --image_file IMAGE_FILE
                        Path to the image file
  --output_file OUTPUT_FILE
                        Path for the adverserial image output (default './final_image.png')
  --target_class TARGET_CLASS
                        Target class of the adversarial image for the chosen model

Example:
python3 main.py --model 'resnet18' --image_file './cat.jpg' --output_file 'final_image_cat_to_lemon.png' --target_class 'lemon'
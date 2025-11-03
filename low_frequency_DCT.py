import numpy as np
from scipy.fftpack import dct, idct, dctn, idctn

def sample_noise(C, H, W, r):
  noise = np.random.normal(scale=1, size=(C, H, W))

  adv_noise = np.zeros((C, H, W))

  adv_noise[0:C, 0:int(r*H), 0:int(r*W)] = noise[0:C, 0:int(r*H), 0:int(r*W)]

  print(adv_noise[0])

  return adv_noise

def get_dct(image):
  dct_image = np.zeros_like(image)

  # for ch in range(3):
  #   dct_image[ch] = dct(dct(image[ch], axis=0, norm='ortho'), axis=1, norm='ortho')

  dct_image = dctn(image, axes=(1, 2), norm='ortho')

  return dct_image

def get_idct(dct_image):
  image = np.zeros_like(dct_image)

  # for ch in range(3):
  #   image[ch] = idct(idct(dct_image[ch], axis=1, norm='ortho'), axis=0, norm='ortho')

  image = idctn(dct_image, axes=(2, 1), norm='ortho')

  return image
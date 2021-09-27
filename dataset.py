"""The data set class that prepares the images to be inputted into the model."""

import numpy as np
import torch
from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
from pretrainedmodels import utils
from model import inception

ENCODER_SIZE = 224
INCEPTION_SIZE = 299

load_img = utils.LoadImage()
tf_img = utils.TransformImage(inception) 
encoder_transform = transforms.Compose([transforms.CenterCrop(ENCODER_SIZE)])
inception_transform = transforms.Compose([transforms.CenterCrop(INCEPTION_SIZE)])


class ImageDataset(datasets.ImageFolder):
  """
  Subclass of ImageFolder that separates LAB channels into L and AB channels.
  It also transforms the image into the correctly formatted input for Inception.
  """
  def __getitem__(self, index):
    img_path, _ = self.imgs[index]

    img_inception = tf_img(inception_transform(load_img(img_path)))
    img = self.loader(img_path)

    img_original = encoder_transform(img)
    img_original = np.asarray(img_original)

    img_lab = rgb2lab(img_original)
    img_lab = (img_lab + 128) / 255
    
    img_ab = img_lab[:, :, 1:3]
    img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

    img_gray = rgb2gray(img_original)
    img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

    return img_gray, img_ab, img_inception
    
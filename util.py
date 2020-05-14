import numpy as np
import torch
from skimage.color import lab2rgb

def stack_lab_channels(grayscale_input, ab_input):
  """
  Stacks the L and AB channels together to create the LAB image. Then, the LAB
  image is converted to RGB.

  Parameters:
    grayscale_input: The L channel as a tensor.
    ab_input: The AB channels as a tensor.
  
  Returns:
    The RGB channels as a numpy array.
  """
  color_image = torch.cat((grayscale_input, ab_input), axis=0).numpy()
  color_image = color_image.transpose((1, 2, 0)) 

  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   

  color_image = lab2rgb(color_image.astype(np.float64))

  return color_image

def convert_to_rgb(grayscale_input, ab_input, ab_ground_truth):
  """
  Converts the grayscale, predicted, and ground truth tensors into
  image-displayable numpy arrays. 

  Parameters:
    grayscale_input: The L channel as a tensor.
    ab_input: The AB channels as a tensor.
  
  Returns:
    The grayscale, predicted, and ground truth images as RGB numpy arrays.
  """
  predicted_image = stack_lab_channels(grayscale_input, ab_input)
  ground_truth_image = stack_lab_channels(grayscale_input, ab_ground_truth)
  grayscale_input = grayscale_input.squeeze().numpy()

  return grayscale_input, predicted_image, ground_truth_image

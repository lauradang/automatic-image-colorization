import os
import glob
from typing import List

import numpy as np
import torch
from skimage.color import lab2rgb
import streamlit as st


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


def upload_st_files_local(
    uploaded_files: List[st.uploaded_file_manager.UploadedFile],
    gray_images_dir: str,
    color_images_dir: str,
    image_dir_root: str,
    image_dir_sub: str
):
    # Empty input directory before uploading new images to it
    blobs_to_remove = [
        f"{gray_images_dir}/*", 
        f"{color_images_dir}/*", 
        f"{image_dir_root}/{image_dir_sub}/*"
    ]
    globs = [glob.glob(blob_to_remove) for blob_to_remove in blobs_to_remove]
    for item in globs:
        for _file in item:
            os.remove(_file)

    for uploaded_file in uploaded_files:
        with open(os.path.join(image_dir_root, image_dir_sub, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer()) 

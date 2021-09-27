"""The main entrypoint for prediction."""

import os

import streamlit as st
import torch
import matplotlib.pyplot as plt

from model import Network, Encoder, Decoder, device
from dataset import ImageDataset
from util import convert_to_rgb


def predict(image_dir_root, image_dir_sub, checkpoint_path, coloured_images_dir, gray_images_dir):
    test_data = ImageDataset(image_dir_root)
    num_images = len(os.listdir(f"{image_dir_root}/{image_dir_sub}"))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=num_images)

    model = Network()
    model = model.to(device)

    if device == torch.device("cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    elif device == torch.device("cuda"):
        checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    img_gray, img_ab, img_inception = iter(test_dataloader).next()
    img_gray, img_ab, img_inception = img_gray.to(device), img_ab.to(device), img_inception.to(device)

    with torch.no_grad():
        output = model(img_gray, img_inception)

    for idx in range(num_images):
        try:
            _, predicted_image, gray_image = convert_to_rgb(img_gray[idx].cpu(), output[idx].cpu(), img_ab[idx].cpu())
            plt.imsave(arr=predicted_image, fname=f"{coloured_images_dir}/colourized_{idx}.jpg")
            plt.imsave(arr=gray_image, fname=f"{gray_images_dir}/gray_{idx}.jpg")
        except IndexError:
            break

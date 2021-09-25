"""The Streamlit frontend entrypoint."""

import os
import argparse

import streamlit as st
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import Network, Encoder, Decoder, device
from dataset import ImageDataset
from util import convert_to_rgb

IMAGE_DIR = "test_images"
CHECKPOINT_PATH = "models/colorization_model (0.002478).pt"
COLOURED_IMAGES_DIR = "image_results"


def predict(image_dir, checkpoint_path, coloured_images_dir):
    test_data = ImageDataset(image_dir)
    num_images = len(os.listdir(f"{image_dir}/test"))
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
            bw_image, predicted_image, _ = convert_to_rgb(img_gray[idx].cpu(), output[idx].cpu(), img_ab[idx].cpu())
            plt.imsave(arr=predicted_image, fname=f"{coloured_images_dir}/colourized_{idx}.jpg")
        except IndexError:
            break


st.title("Automatic Image Colorizer")
uploaded_files = st.file_uploader(
    "Choose a B&W picture", accept_multiple_files=True
)

for uploaded_file in uploaded_files:
    with open(os.path.join(IMAGE_DIR, "test", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

if uploaded_files:
    predict(IMAGE_DIR, CHECKPOINT_PATH, COLOURED_IMAGES_DIR)
    i = 0
    coloured_photos = [pic for pic in os.listdir(COLOURED_IMAGES_DIR) if "colourized" in pic]
    print(uploaded_files)
    print(coloured_photos)
    if len(coloured_photos) >= 2:
        coloured_photos[0], coloured_photos[1] = coloured_photos[1], coloured_photos[0]
        
    for coloured_photo in coloured_photos:
        if "colourized" in coloured_photo:
            images = [
                os.path.join(IMAGE_DIR, "test", uploaded_files[i].name),
                os.path.join(COLOURED_IMAGES_DIR, coloured_photo)
            ]
            captions = [uploaded_files[i].name, "Coloured"]
            st.image(images, caption=captions)
            i += 1

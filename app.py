"""The Streamlit frontend entrypoint."""

import os
import argparse
import glob
from typing import List

import streamlit as st

from model import Network, Encoder, Decoder, device
from dataset import ImageDataset
from predict import predict
from util import upload_st_files_local, convert_to_rgb

IMAGE_DIR_ROOT = "input_images"
IMAGE_DIR_SUB = "images"
CHECKPOINT_PATH = "models/colorization_model (0.002478).pt"
OUTPUT_IMAGES_DIR = "output_images"
COLOR_IMAGES_DIR = os.path.join(OUTPUT_IMAGES_DIR, "color")
GRAY_IMAGES_DIR = os.path.join(OUTPUT_IMAGES_DIR, "gray")


st.title("Automatic Image Colorizer")
uploaded_files = st.file_uploader("Choose a B&W picture", accept_multiple_files=True)
upload_st_files_local(uploaded_files, GRAY_IMAGES_DIR, COLOR_IMAGES_DIR, IMAGE_DIR_ROOT, IMAGE_DIR_SUB)

if uploaded_files:
    predict(IMAGE_DIR_ROOT, IMAGE_DIR_SUB, CHECKPOINT_PATH, COLOR_IMAGES_DIR, GRAY_IMAGES_DIR)

    for i in range(len(os.listdir(GRAY_IMAGES_DIR))):
        images = [
            os.path.join(GRAY_IMAGES_DIR, f"gray_{i}.jpg"),
            os.path.join(COLOR_IMAGES_DIR, f"colourized_{i}.jpg")
        ]
        st.image(images, caption=["B&W", "Color"])

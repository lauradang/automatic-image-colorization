"""The main entrypoint that colourizes a given image."""

import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Network, Encoder, Decoder, device
from dataset import ImageDataset
from util import convert_to_rgb

def main(image_dir, checkpoint_path, coloured_images_dir):
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
            _, predicted_image, _ = convert_to_rgb(img_gray[idx].cpu(), output[idx].cpu(), img_ab[idx].cpu())
            plt.imsave(arr=predicted_image, fname=f"{coloured_images_dir}/colourized_{idx}.jpg")
        except IndexError:
            break

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="A script that defines and trains the convolutional neural network.") 
    arg_parser.add_argument("--image_dir", type=str, default="test_images", help="The directory where the images you want to colourize are located.")    
    arg_parser.add_argument("--checkpoint_path", type=str, default="models/colorization_model (0.002478) (1).pt", help="The path to the colourization model.")  
    arg_parser.add_argument("--coloured_images_dir", type=str, default="image_results", help="The folder to output the coloured images.")      
    args = arg_parser.parse_args()
# models/colorization_model (0.002478) (1).pt
    main(args.image_dir, args.checkpoint_path, args.coloured_images_dir)

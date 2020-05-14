# Automatic Image Colorization using CNNs and InceptionResnetV2
## Overview
This project showcases my implementation of Baldassarre et al.'s [Deep Koalarization: Image Colorization using CNNs and Inception-ResNet-v2](https://arxiv.org/abs/1712.03400) paper from 2017 using PyTorch. The network is trained using 60 000 images from ImageNet. You can find more information in the [Jupyter notebook](https://github.com/lauradang/automatic-image-colorization/blob/master/notebooks/inception_resnet.ipynb).

## Results
More results can be found in the notebook and the image results folder.

![](image_results/input_10.jpg) ![](image_results/result_10.jpg)

![](image_results/input_7.jpg) ![](image_results/result_7.jpg)

![](image_results/input_18.jpg) ![](image_results/result_18.jpg)

![](image_results/input_12.jpg) ![](image_results/result_12.jpg)

![](image_results/input_13.jpg) ![](image_results/result_13.jpg)

![](image_results/input_15.jpg) ![](image_results/result_15.jpg)

![](image_results/input_6.jpg) ![](image_results/result_6.jpg)

## Steps to Run
1. Run `pip install -r requirements.txt` to install the necessary dependencies for both training and predicting.
2. To retrain the model using your own dataset, run the [notebook](https://github.com/lauradang/automatic-image-colorization/blob/master/notebooks/inception_resnet.ipynb) and replace the file paths with your own.
3. To colourize your own images using the model provided in `models`, run `python3 predict.py`. Run `python3 predict.py -h` for instructions on how to run the prediction script.

**Note**: The model's image size output is 224x224. If the grayscale image is larger than this, it will be centre cropped to fit these dimensions. To prevent the image from being cut off, resize the image to fit these dimensions before running the prediction script.
   
## Built With
- [PyTorch](https://pytorch.org/)
- [Pretrained-Models.PyTorch](https://github.com/Cadene/pretrained-models.pytorch) - Leveraged InceptionResnetV2
- [ImageNet Downloader](https://github.com/mf1024/ImageNet-Datasets-Downloader) - Downloading the dataset

## Credits
Here are the implementations that gave me inspiration for this project:
- [deep-koalarization - baldassarreFe](https://github.com/baldassarreFe/deep-koalarization)
- [Automatic-Image-Colorization - lukemelas](https://github.com/lukemelas/Automatic-Image-Colorization/)
- [hands-on-transfer-learning-with-python - dipanjanS](https://github.com/dipanjanS/hands-on-transfer-learning-with-python)

## Author
Laura Dang

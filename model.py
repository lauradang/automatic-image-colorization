"""The model architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

torch.nn.Module.dump_patches = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception = pretrainedmodels.__dict__["inceptionresnetv2"](
    num_classes=1001, 
    pretrained="imagenet+background"
)

inception = inception.to(device)
inception.eval()

class Encoder(nn.Module):
  """
  The encoder for the neural network. 
  The input shape is a 224x224x1 image, which is the L channel.
  """
  def __init__(self):
    super(Encoder, self).__init__()    

    self.input_ = nn.Conv2d(1, 64, 3, padding=1, stride=2)
    self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
    self.conv2 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
    self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
    self.conv4 = nn.Conv2d(256, 256, 3, padding=1, stride=2)
    self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
    self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
    self.conv7 = nn.Conv2d(512, 256, 3, padding=1)
  
  def forward(self, x):
    x = F.relu(self.input_(x))
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
    x = F.relu(self.conv7(x))

    return x

class Decoder(nn.Module):
  """
  The decoder for the neural network. 
  The input shape is the fusion layer indicated in the paper.
  """
  def __init__(self):
    super(Decoder, self).__init__()

    self.input_1 = nn.Conv2d(1257, 256, 1)
    self.input_ = nn.Conv2d(256, 128, 3, padding=1)
    self.conv1 = nn.Conv2d(128, 64, 3, padding=1)
    self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
    self.conv4 = nn.Conv2d(32, 2, 3, padding=1)

  def forward(self, x):
    x = F.relu(self.input_1(x))
    x = F.relu(self.input_(x))
    x = F.interpolate(x, scale_factor=2)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.interpolate(x, scale_factor=2)
    x = F.relu(self.conv3(x))
    x = torch.tanh(self.conv4(x))
    x = F.interpolate(x, scale_factor=2)

    return x

class Network(nn.Module):
  """
  Combines the outputs of the encoder and InceptionResNetV2 model and feeds this
  fused output into the decoder to output a predicted 224x224x2 AB channel.
  """
  def __init__(self):
    super(Network, self).__init__()

    self.encoder = Encoder()
    self.encoder = self.encoder.to(device)
    
    self.decoder = Decoder()
    self.decoder = self.decoder.to(device)

  def forward(self, encoder_input, feature_input):
    encoded_img = self.encoder(encoder_input)

    with torch.no_grad():
      embedding = inception(feature_input)

    embedding = embedding.view(-1, 1001, 1, 1)

    rows = torch.cat([embedding] * 28, dim=3)
    embedding_block = torch.cat([rows] * 28, dim=2)
    fusion_block = torch.cat([encoded_img, embedding_block], dim=1)

    return self.decoder(fusion_block)

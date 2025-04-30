import numpy as np
import pandas as pd
import torch
from torch import nn
import PIL
import torchvision
from torchvision import transforms

with open('algos.csv', 'r') as f:
    matrix1 = list(map(float, f.readline().split(',')))
    matrix2 = list(map(float, f.readline().split(',')))

mat1 = np.array(matrix1).reshape(3, 3)
mat2 = np.array(matrix2).reshape(3, 3)

dirpath = 'best_pictures/*.png'

transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

for filepath in glob.glob(dirpath):
    img = Image.open(filepath)
    print(transform_train(img).shape)
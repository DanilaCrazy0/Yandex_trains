from PIL import Image
import numpy as np
import glob
import torch
import torchvision
from numba.np.arrayobj import np_array
from torchvision import transforms


arr2 = np.array([[0.12495883, 0.25008738, 0.12495108],
                [0.25005478, 0.4998652, 0.25008452],
                [0.12498412, 0.25005153, 0.12496214]]).flatten().tolist()

arr1 = np.array([[0.0625, 0.0625, 0.0625],
                [0.0625, 0.0625, 0.0625],
                [0.0625, 0.0625, 0.0625]]).flatten().tolist()

arr3 = np.array([[-1.,  -0.5,  0., ],
                [-0.5,  0.5,  0.5],
                [ 0.,   0.5,  1. ]]).flatten().tolist()

st1 = ','.join(map(str, arr1))
st2 = ','.join(map(str, arr2))
st3 = ','.join(map(str, arr3))

with open('reconstructed_algos.csv', 'w') as f:
    f.write(st1 + '\n')  # Первая строка - arr1
    f.write(st2 + '\n')  # Вторая строка - arr2
    f.write(st3 + '\n')
# import torch
# import torchvision
# import  numpy as np
#
# batch_size_train =32
#
# train_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('data', train=True, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])),
#   batch_size=batch_size_train, shuffle=True)
#
# ii,(example_data, example_targets) = next(enumerate(train_loader))
#
#
# Gs = []
# shapes = torch.shape(example_data)
# N = len(shapes)
# Is = np.range()
# for jj in range(N+1):
#     Gs = [torch.randn((shapes[jj]%N, shapes[jj], shapes[jj+1]))]


import os

import matplotlib.pyplot as plt
import numpy as np
import imageio


class El(object):
    def __init__(s, o):
        s.o = o
        s.n = o.split('_')[0]

    def __hash__(s): return s.n.__hash__()

    def __eq__(s, o): return s.n == o.n


def get_data(dataset_name=''):
    """
    :param dataet_name: data/tabby_cat.mp4, ("data/coil-100", '100')
    :param video:
    :return:
    """
    video = type(dataset_name) is str and dataset_name.endswith('.mp4')
    if video:
        vid = imageio.get_reader(dataset_name, 'ffmpeg')
    else:
        files_object = sorted([El(o) for o in os.listdir(dataset_name[0])
                               if o.endswith('png') and dataset_name[1] in o], key=lambda x: x.o)

    frame_id = 0

    tensor_4d = None

    while True:
        try:
            if video:
                image = vid.get_data(frame_id)
            else:
                image = plt.imread(f'{dataset_name[0]}/{files_object[frame_id].o}')

            if tensor_4d is None:
                tensor_4d = image[np.newaxis, :]
            else:
                tensor_4d = np.concatenate([tensor_4d, image[np.newaxis, :]], 0)
        except IndexError as e:
            break
        frame_id += 1
    return tensor_4d



def get_object():
    l = [El(o) for o in os.listdir('coil-100') if o.endswith('png')]

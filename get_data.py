import os

import matplotlib.pyplot as plt
import numpy as np
import imageio


def get_coil_dataset(dataset_name, obj='100'):
    if obj is None:
        return get_coil_dataset(dataset_name, list(range(1,101)))
    elif type(obj) is list:
        return np.concatenate([get_coil_dataset(dataset_name, o) for o in obj],0)
    else:
        tensor_4d = None

        for frame_id in range(0,360,5):
            image = plt.imread(f'{dataset_name}/obj{obj}__{frame_id}.png')[::6,::6,:]
            image=(image*255).astype('uint8')

            if tensor_4d is None:
                tensor_4d = image[np.newaxis, :]
            else:
                tensor_4d = np.concatenate([tensor_4d, image[np.newaxis, :]], 0)

        return tensor_4d

def get_data(dataset_name='', obj=None):
    """
    :param dataet_name: data/tabby_cat.mp4, ("data/coil-100", '100')
    :param video:
    :return:
    """
    video = type(dataset_name) is str and dataset_name.endswith('.mp4')
    if video:
        vid = imageio.get_reader(dataset_name, 'ffmpeg')

        frame_id = 0

        tensor = None

        while True:
            try:
                if video:
                    image = vid.get_data(frame_id)

                if tensor is None:
                    tensor = image[np.newaxis, :]
                else:
                    tensor_4d = np.concatenate([tensor, image[np.newaxis, :]], 0)
            except IndexError as e:
                break
            frame_id += 1
    else:
        tensor = get_coil_dataset(dataset_name, obj)
    return tensor

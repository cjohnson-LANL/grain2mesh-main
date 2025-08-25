"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

utils.py
"""
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
import cubit
import os
from tqdm import tqdm


def create_turbo_with_black():
    # Load the turbo colormap
    turbo = cm.get_cmap('turbo')
    
    # Create a new colormap from 'turbo' with black color for values <= 0
    colors = turbo(np.linspace(0, 1, 256))
    # Set the first portion of the colormap to black
    colors[:1] = [0, 0, 0, 1]  # Black for the first half (values ≤ 0)
    
    # Create a new colormap from the modified colors
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_turbo_with_black', colors)
    
    return custom_cmap

def init_cubit():
    cubit.init(['cubit','-nojournal'])
    cubit.cmd("reset")

def save_cubit(exp_path, name):
    cubit.cmd(f"Save as '{exp_path}/{name}.cub' overwrite")

def png_to_npz(filepath):
    name = os.path.splitext(filepath)[0]
    image = plt.imread(filepath)
    np.savez(f'{name}.npz', array1=image)
    return f'{name}.npz'

def make_single_channel(image):
    avg_channels = np.zeros((image.shape[0], image.shape[1]))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            avg_channels[i, j] = np.dot(image[i,j], [0.299, 0.587, 0.114])

    return avg_channels

def make_tqdm_callback(title):
    pbar = tqdm(total=1.0, desc=title, ncols=70)

    def update_progress(current, total):
        pbar.n = current / total
        pbar.refresh()
        if current == total:
            pbar.close()

    return update_progress
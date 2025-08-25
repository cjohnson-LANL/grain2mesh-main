"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Class definition for ImageProcessor

Functionality:
- Load base image
- Filter image to remove noise
- Segment image using watershed algorithm

"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage import morphology, measure, segmentation
from skimage.filters import gaussian
from skimage.feature import peak_local_max



class ImageProcessor:
    def __init__(self):
        pass

    def load_image(self, image_basename):
        image_filename = os.path.join(os.getcwd(), image_basename)
        data = np.load(image_filename)
        image = data['array1']
        image_name = os.path.splitext(os.path.basename(image_basename))[0]

        return image, image_name

    def apply_gaussian(self, image, sigma):
        filtered = gaussian_filter(image.astype(float), sigma=sigma) # Gaussian smoothing
        binary_filtered_image = (filtered > 0.5).astype(np.uint8)    # binary mask

        return binary_filtered_image

    # OTHER FILTERS
    # def closing(self, image, radius):
    #     return morphology.closing(image, morphology.disk(radius))

    # def opening(self, image, radius):
    #     return morphology.opening(image, morphology.disk(radius))

    # def morphological_gradient(self, image, radius):
    #     return morphology.morphological_gradient(image, morphology.disk(radius))

    def watershed(self, image, sigma, min_dist, threshold):
        """
        Performs watershed segmentation on an image array

        Args:
        - image (np.ndarray): input image
        - sigma (float): blur amount for distance map
        - min_dist (int): minimum distance separating peaks
        - threshold (float): minimum intensity of peaks

        Returns:
        - labels (np.ndarray): image array labled by segmented region
        """
        # Distance transform and smoothing
        distance = distance_transform_edt(image)
        smooth_distance = gaussian(distance, sigma=sigma)

        # Find local maxima
        local_maxi = peak_local_max(smooth_distance, min_distance=min_dist, threshold_abs=threshold)
        local_maxi_binary = np.zeros_like(smooth_distance, dtype=bool)
        local_maxi_binary[tuple(local_maxi.T)] = True

        # Label markers
        markers = measure.label(local_maxi_binary)

        # Perform watershed segmentation
        labels = segmentation.watershed(-smooth_distance, markers, mask=image)
        return labels

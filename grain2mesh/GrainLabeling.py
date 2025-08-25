"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Class definition for GrainLabeling

Functionality:
- Relabel regions
- Label grains based on area
- Remove floating grain spaces
- Merge small regions
"""
import numpy as np
from skimage import morphology, measure
from skimage.morphology import dilation, footprint_rectangle



class GrainLabeling:
    def __init__(self, labels):
        self.labels = labels    # section IDs
        self.grain_sizes = None # section areas

    def label_zero_spaces(self):
        """
        Assign unique negative label to each discrete 0 region
        """
        # Identify regions of zeros (ie. pore space)
        zero_regions = self.labels == 0 

        # Uniquely lable each zero space
        zero_labels = measure.label(zero_regions, connectivity=1)

        # Give each pore space a unique negative label, grain spaces all labeled 0
        negative_labels = np.where(zero_labels > 0, -zero_labels, 0)

        # Update labels with negative values for pore spaces
        self.labels[zero_regions] = negative_labels[zero_regions]

    def assign_grain_sizes(self):
        """ Measure area of each discrete region """
        properties = measure.regionprops(self.labels)
        self.grain_sizes = np.zeros_like(self.labels, dtype=np.float32)

        # For each label, assign grain size based on area
        for prop in properties:
            if prop.label > 0:
                self.grain_sizes[self.labels == prop.label] = prop.area

    def remove_floaters(self):
        """ Remove regions adjacent only to pore space (floaters) """
        adjacencies = self._find_adjacent_regions()
        for key, value_set in adjacencies.items():
            if all(x < 0 for x in value_set):
                print('Found a floater. Removing now.')
                self.labels[self.labels == key] = list(value_set)[0]

    def _find_adjacent_regions(self):
        """ Create a dictionary of adjacent regions """
        unique_labels = np.unique(self.labels)
        adjacencies = {label: set() for label in unique_labels} # map of adjacent regions

        for label in unique_labels:
            if label == 0:
                continue
            
            mask = self.labels == label # region with current label
            dilated = dilation(mask, footprint_rectangle((3, 3))) # region expanded by 1 pixel in all directions

            neighbors = np.unique(self.labels[dilated & ~mask]) # inverse of mask; just neighbors
            for n in neighbors:
                if n != label:
                    adjacencies[label].add(n) # track adjacent pixels not in label/region

        return adjacencies

    def remap_labels(self, start_from_zero):
        """
        Shift labels to be consecutive starting at either 0 or 1

        Args:
        - start_from_zero (bool): flag for whether to start at 0 or 1
        """
        unique_values = np.unique(self.labels)
        new_values = None
        if start_from_zero:
            new_values = np.arange(len(unique_values)) # shift labels so now starting at 0
        else:
            new_values = np.arange(1, len(unique_values) + 1) # shift labels starting at 1
        
        self.value_mapping = {old_val: new_val for old_val, new_val in zip(unique_values, new_values)}
        # reorder lables based on mapping: old value = key, new value = value
        self.labels = np.vectorize(self.value_mapping.get)(self.labels)

    def merge_small_regions(self, threshold):
        """
        Merges small regions into adjacent regions

        Args:
        - threshold (float): minimum area for a region
        """
        self.remap_labels(False)

        properties = measure.regionprops(self.labels)
        # adjacencies = self._find_adjacent_regions()
        for region in properties:
            if region.area < threshold:
                print(f'Merging small region: {region.label}')

                # Find the adjacent regions
                adjacencies = self._find_adjacent_regions()
                adj_regions = list(adjacencies[region.label])
  
                small_region_mask = (self.labels == region.label)
                small_centroid = properties[region.label - 1].centroid
                small_region_coords = np.argwhere(small_region_mask)

                #Initialize variable to store the new label
                new_label = None

                #special case (single pixel..)
                if region.area == 1.0:
                    self.labels[self.labels == region.label] = adj_regions[0]

                
                # Iterate over each adjacent region to find if the centroid is within its bounding box
                for adj_label in adj_regions:
                    # Get the properties of the adjacent region
                    adj_region_props = properties[adj_label - 1]
                    adj_bbox = adj_region_props.bbox  # Bounding box of the adjacent region
                    
                    # Check if the centroid is within the bounding box of the adjacent region
                    if self._is_point_within_bbox(small_centroid, adj_bbox):
                        new_label = adj_label
                        break  # Exit loop once the matching adjacent region is found
                
                if new_label is not None:
                    # If centroid is within bounding box, update the label
                    self.labels[self.labels == region.label] = new_label
                else:
                    # adjacencies = self._find_adjacent_regions()
                    # adj_regions = list(adjacencies[region.label])
                    # Create a distance map to find the closest adjacent region for each pixel
                    distance_map = np.full(small_region_mask.shape, np.inf)
                    for adj_label in adj_regions:
                        # Get the binary mask of the adjacent region
                        adj_region_mask = (self.labels == adj_label)
                        adj_region_coords = np.argwhere(adj_region_mask)

                        # Calculate distance from each pixel in the small region to the current adjacent region
                        for pixel in small_region_coords:
                            distances = np.sqrt(np.sum((adj_region_coords - pixel) ** 2, axis=1))
                            min_distance = np.min(distances)
                            if min_distance < distance_map[tuple(pixel)]:
                                distance_map[tuple(pixel)] = min_distance
                                self.labels[tuple(pixel)] = adj_label

    def _is_point_within_bbox(self, point, bbox):
        """
        Determine if a point is within a given bounding box

        Args:
        - point (tuple): 2D point coordinates
        - bbox (tuple): bounding box

        Returns:
        - bool:
        """
        x, y = point
        min_row, min_col, max_row, max_col = bbox
        return min_row <= x < max_row and min_col <= y < max_col

    def set_negative_labels_to_zero(self):
        """ Combine negative labels into a single region """
        for key, value in self.value_mapping.items():
            if key >= 0:
                break
        pp_thresh = value - 1

        # Label all negative spaces 0 (pore spaces)
        self.labels[self.labels <= pp_thresh] = 0

    # def label_by_phase(self, labels, original):
    #     """"""
    #     mapped_labels = np.zeros_like(labels)

    #     for label in np.unique(labels):
    #         if label == 0:
    #             continue

    #         mask = labels == label
    #         original_labels = original[mask]

    #         if original_labels.size > 0:
    #             majority_label = np.bincount(original_labels.astype(np.uint8)).argmax()
    #             mapped_labels[mask] = majority_label

    #     return mapped_labels

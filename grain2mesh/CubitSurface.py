"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Class definition for CubitSurface

Functionality:
- Creates a Cubit surface for each phase
"""
import numpy as np
import pickle
import cubit
from skimage.measure import label
from collections import defaultdict 
from .utils import init_cubit


class CubitSurface:
    def __init__(self):
        self.points = None
        self.facets = None
        self.regions = None
        self.phase_numbers = None
        self.phases = None
        self.height = None
        self.width = None
        self.phase_to_pixels = {}

        init_cubit()


    def load_pmesh(self, filename):
        """
        Reads polygon mesh file and updates element data

        Args:
        - filename (string): polygon mesh PKL file name
        """
        with open(filename, 'rb') as f:
            self.points, self.facets, self.regions, self.phase_numbers, self.height, self.width = pickle.load(f)

        # Give each spore space a unique negative label
        self._label_zero_spaces()
        
        # Map phase to pixels indices
        self.phases = np.unique(self.phase_numbers)
        self.phase_to_pixels = {phase: [] for phase in self.phases}
        for i, phase_num in enumerate(self.phase_numbers):
            self.phase_to_pixels[phase_num].append(i)


    def _label_zero_spaces(self):
        """
        Assign a unique negative label to each discrete pore space
        """
        zero_regions = self.phase_numbers == 0
        zero_regions = zero_regions.reshape(self.height,self.width)

        zero_labels = label(zero_regions, connectivity=1) # label each connected zero region
        zero_regions = zero_regions.flatten()
        zero_labels = zero_labels.flatten()

        negative_labels = np.where(zero_labels > 0, -zero_labels, 0) # unique negative label for each discrete pore region
        self.phase_numbers[zero_regions] = negative_labels[zero_regions] # combine with phase numbers

    

    def create_surfaces(self):
        """
        Create a Cubit surface for each phase region
        """
        for phase in self.phases:
            top_borders = []
            bottom_borders = []
            right_borders = []
            left_borders = []

            # Find borders of phase
            for pixel in self.phase_to_pixels[phase]:
                index = pixel - self.width
                self._add_border(phase, pixel, top_borders, index, 0) # top

                index = pixel + self.width
                self._add_border(phase, pixel, bottom_borders, index, 2) # bottom

                index = pixel - 1
                self._add_border(phase, pixel, left_borders, index, 1) # left

                index = pixel + 1
                self._add_border(phase, pixel, right_borders, index, 3) # right


            # Combine colinear points into a single edge
            top_borders = self._combine_colinear_edges(top_borders, 1)
            bottom_borders = self._combine_colinear_edges(bottom_borders, 1)
            left_borders = self._combine_colinear_edges(left_borders, 0)
            right_borders = self._combine_colinear_edges(right_borders, 0)
            

            # Create surface for phase
            curves = []
            for curve in top_borders:
                c = self._create_curve(curve)
                curves.append(c)
            for curve in bottom_borders:
                c = self._create_curve(curve)
                curves.append(c)
            for curve in left_borders:
                c = self._create_curve(curve)
                curves.append(c)
            for curve in right_borders:
                c = self._create_curve(curve)
                curves.append(c)

            cubit.create_surface(curves)


            # Unite pore space
            if phase == -1:
                bodies = cubit.parse_cubit_list("body", "all")
                unite_string = "Unite Body "
                for body in bodies:
                    unite_string += f"{body} "
                cubit.cmd(unite_string)

                bodies = cubit.parse_cubit_list("body", "all")
                cubit.cmd(f'body 1 rename "C_phase0"')

            else:
                bodies = cubit.parse_cubit_list("body", "all")
                cubit.cmd(f'body {bodies[-1]} name "C_phase{phase}"')


                    


    def _add_border(self, phase, pixel, borders, index, side):
        """
        Store border facet in corresponding side list for adjacent pixels of different phases

        Args:
        - phase (int): phase id
        - pixel (int): index of pixel
        - borders (list): list of facets for a given side of a phase region
        - index (int): adjacent pixel index

        Returns:
        - NONE - updates border list
        """
        if 0 <= index < (self.width * self.height): # border between two regions
            if self.phase_numbers[index] != phase:
                borders.append(self.regions[pixel][side])
        else:
            borders.append(self.regions[pixel][side]) # boundary edge


    def _combine_colinear_edges(self, borders, col):
        """
        Find furthest endpoints of colinear edges to form single line segment

        Args:
        - borders (list): list of facets ids
        - col (int): colinear axis (0 - x, 1 - y)

        Returns:
        - segements (np.ndarray): list of segment endpoints
        """
        colinear_points = defaultdict(list)
        segments = []

        # Group points by shared coordinate
        for border in borders:
            facet = self.facets[border]
            p1, p2 = self.points[facet]
            key = p1[col]

            colinear_points[key].append(p1)
            colinear_points[key].append(p2)

        # Determine end points for border segments
        for key in colinear_points.keys():
            colinear_points[key] = np.array([p[1-col] for p in colinear_points[key]]) # get unique coordinates

            unique_vals, counts = np.unique(colinear_points[key], return_counts=True)
            endpoints = unique_vals[counts == 1] # endpoints that have a count of 1 form new segment endpoints

            # Construct segments
            for i in range(0, len(endpoints), 2):
                start = np.zeros(2)
                end = np.zeros(2)
                start[col] = key
                start[1-col] = endpoints[i]
                end[col] = key
                end[1-col] = endpoints[i + 1]

                segments.append(np.array([start, end]))

        return np.array(segments)


    def _create_curve(self, curve_points):
        """
        Creates a curve between two Cubit vertices

        Args:
        - curve_points (np.ndarray): 2D array of line segment endpoints

        Returns:
        - Cubit Curve object
        """
        x1 = curve_points[0,0]
        x2 = curve_points[1,0]

        y1 = curve_points[0,1]
        y2 = curve_points[1,1]

        V1 = cubit.create_vertex(x1, y1, 0.0)
        V2 = cubit.create_vertex(x2, y2, 0.0)

        C1 = cubit.create_curve(V1, V2)
        return C1

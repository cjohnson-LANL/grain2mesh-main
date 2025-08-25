"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Class definition for PolygonMeshBuilder

Functionality:
- Make facets and regions for each pixel
- Generate polygon mesh
- Color each phase uniquely
- Plot the mesh
"""
import numpy as np
from matplotlib import pyplot as plt
from .utils import make_tqdm_callback




class PolygonMesh:
    def __init__(self, labels):
        np.random.seed(0)
        self.bin_nums = labels
        self.m, self.n = self.bin_nums.shape
        self.unique_cololrs_rgb = None
        self.pmesh = None
        self.phases = None

        # Generate mesh grid & points
        x = np.arange(self.n + 1).astype('float')
        y = self.m + 1 - np.arange(self.m + 1).astype('float')
        xx, yy = np.meshgrid(x,y)

        self.pts = np.array([xx.flatten(), yy.flatten()]).T   # list mesh points
        self.kps = np.arange(len(self.pts)).reshape(xx.shape) # key for each point

        # Mesh elements
        n_facets = 2 * (self.m + self.m * self.n + self.n)
        self.n_regions = self.m * self.n
        self.facets = np.full((n_facets, 2), -1)
        self.regions = np.full((self.n_regions, 4), 0)
        self.region_phases = np.full(self.n_regions, 0)

        # Index trackers
        self.facet_top = np.full((self.m, self.n), -1, dtype='int') 
        self.facet_bottom = np.full((self.m, self.n), -1, dtype='int')
        self.facet_left = np.full((self.m, self.n), -1, dtype='int')
        self.facet_right = np.full((self.m, self.n), -1, dtype='int')

        # Counters
        self.k_facets = 0
        self.k_regions = 0


    def make_regions(self):
        """ Create a region for each pixel """
        for i in range(self.m):
            for j in range(self.n):
                # Get vertices of current pixel
                kp_top_left = self.kps[i, j]
                kp_bottom_left = self.kps[i + 1, j]
                kp_top_right = self.kps[i, j + 1]
                kp_bottom_right = self.kps[i + 1, j + 1]


                # Get facets of current pixel
                left_facet = self._get_or_create_facet(self.facet_left, i, j, kp_top_left,
                                                    kp_bottom_left, self.facet_right,
                                                    opp_i=i, opp_j=j-1 if j > 0 else None)


                right_facet = self._get_or_create_facet(self.facet_right, i, j, kp_top_right,
                                                    kp_bottom_right, self.facet_left,
                                                    opp_i=i, opp_j=j+1 if j+1 < self.n else None)


                top_facet = self._get_or_create_facet(self.facet_top, i, j, kp_top_left,
                                                    kp_top_right, self.facet_bottom,
                                                    opp_i=i-1 if i > 0 else None, opp_j=j)


                bottom_facet = self._get_or_create_facet(self.facet_bottom, i, j, kp_bottom_left,
                                                    kp_bottom_right, self.facet_top,
                                                    opp_i=i+1 if i+1 < self.m else None, opp_j=j)

                # Create region for current pixel
                self._update_region(top_facet, left_facet, bottom_facet, right_facet, i, j)


    def _get_or_create_facet(self, facet_tracker, i, j, kp1, kp2, opp_tracker, opp_i, opp_j):
        """
        Gets facet for a pixel edge or creates a new one if it doesn't exist

        Args:
        - facet_tracker (np.ndarray): 2D array that tracks indices of facets for a given side
        - i, j (ints): pixel index
        - kp1, kp2 (ints): endpoints of facet
        - opp_tracker (np.ndarray): facet indices of opposite side
        - opp_i, opp_j (ints): index of adjacent pixel

        Returns:
        - facet_index (int): index of facet
        """
        # Create facet if it does not exit
        if facet_tracker[i, j] < 0:
            facet_index = self.k_facets              # determine new facet index
            self.facets[facet_index] = (kp1, kp2)    # create facet
            self.k_facets += 1                       # inc facet count

            # Add facet to corresponding tracker (left/right, top/bottom)
            if opp_i is not None and opp_j is not None:
                opp_tracker[opp_i, opp_j] = facet_index
        
        else:
            facet_index = facet_tracker[i, j] # return facet it it exist
        
        return facet_index


    def _update_region(self, top, left, bottom, right, i, j):
        """
        Store what facts make up each pixel

        Args:
        - top, left, bottom, right (ints): facet indices
        - i, j (ints): pixel index
        """
        region = (top, left, bottom, right)                      # create region
        self.regions[self.k_regions] = region                    # store region
        self.region_phases[self.k_regions] = self.bin_nums[i, j] # store phase of region
        self.k_regions += 1                                      # inc region count


    def map_phase_colors(self):
        """ Determine what color to make each phase """
        num_colors = len(np.unique(self.region_phases))

        self.unique_colors_rgb = self._generate_unique_colors(num_colors)
        self.unique_colors_rgb[0] = [0,0,0]

        self.phases = [{'color': c, 'material_type': 'amorphous'} for c in self.unique_colors_rgb]

        return self.phases

    def _generate_unique_colors(self, num_colors):
        # Ensure num_colors is feasible
        if num_colors > 256 ** 3:
            raise ValueError("Too many colors requested. Maximum is 256^3.")
        
        unique_colors = set()
        
        while len(unique_colors) <= num_colors:
            # Generate random RGB values
            color = tuple(np.random.randint(0, 256, size=3)/255)
            unique_colors.add(color)
        
        return np.array(list(unique_colors))



    def plot_mesh(self, export_path, image_name):
        """ Make a pyplot of polygon mesh """
        # Configure plot
        fig = plt.figure(figsize=(8,8))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.add_axes(ax)

        # Plot each region
        self.pts = np.array(self.pts)
        progress_callback = make_tqdm_callback('Pmesh Progress')
        i = 0
        for region, phase_num in zip(self.regions, self.region_phases):
            color = self.phases[phase_num]['color']

            facets = [self.facets[f] for f in region]
            kps = self._ordered_kps(facets)

            y, x = zip(*[self.pts[kp] for kp in kps])
            plt.fill(y, x, color=color, alpha=0.8, edgecolor='none')

            progress_callback(i, len(self.region_phases))
            i += 1

        plt.axis('square')
        plt.axis('off')
        plt.savefig(f'{export_path}/B1_{image_name}_pmesh.png', bbox_inches='tight', pad_inches=0)


    def _ordered_kps(self, pairs):
        t_pairs = [tuple(p) for p in pairs]
        kps = list(t_pairs.pop())
        while t_pairs:
            for i, pair in enumerate(t_pairs):
                if kps[-1] in pair:
                    break
            assert kps[-1] in pair, pairs
            kps += [kp for kp in t_pairs.pop(i) if kp != kps[-1]]
        return kps[:-1]

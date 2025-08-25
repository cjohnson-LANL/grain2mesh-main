"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""
from .PetroSeg import convolution_segmentation, smooth, label_segments
from .ImageProcessor import ImageProcessor
from .GrainLabeling import GrainLabeling
from .PolygonMesh import PolygonMesh
from .CubitSurface import CubitSurface
from .SplinedSurface import SplinedSurface
from .FinalMesh import FinalMesh
import utils

import numpy as np
from matplotlib import pyplot as plt
import pickle
import cubit
import os
import json
import time


class Grain2Mesh:
    def read_config(self, file):
        with open(file, "r") as f:
            config = json.load(f)

        self.export_path = config.get("export_path", "./results")
        os.makedirs(self.export_path, exist_ok=True)
        self.image_basename = config.get("image_basename", None)
        self.gaussian_sigma = config.get("gaussian_sigma", 1)
        self.area_threshold = config.get("area_threshold", 100)
        self.min_spline_length = config.get("min_spline_length", 0)
        self.inner_mesh_size = config.get("inner_mesh_size", 8.0)
        self.boundary_mesh_size = config.get("boundary_mesh_size", 8.0)
        self.watershed_sigma = config.get("watershed_sigma", 1)
        self.peak_min_distance = config.get("peak_min_distance", 6)
        self.watershed_peak_threshold = config.get("watershed_peak_threshold", 0.5)
        self.verbose = config.get("verbose", False)

        if self.min_spline_length > 4.0:
            print(f'ERROR: min_spline_length is too large! current={self.min_spline_length}')
            print('Please provide a value <= 4.0.')
            exit()


    def run(self):
        print("\n------------ Starting Image Pre-Processing ------------")
        ip = ImageProcessor()

        _, ext = os.path.splitext(self.image_basename)
        if ext != ".npz":
            npz_file = utils.png_to_npz(self.image_basename)
        else:
            npz_file = self.image_basename

        image, image_name = ip.load_image(npz_file)

        # Preview Original Image
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.show()

        # Confirm image is suitable
        confirm = input("Is this image suitable for processing? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Exiting. Please provide a better image.")
            exit()

        # Save A1
        plt.figure(figsize=(8,8))
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(f'{self.export_path}/A1_{image_name}_RAW_nostitch.png',dpi=300,bbox_inches='tight')
        plt.close()

        segmentation_type = int(input("Would you like watershed (0) or CNN (1) segmentation? "))


        # Watershed Segmentation
        if segmentation_type == 0:
            while True:
                # Gaussian Filter
                filtered_image = ip.apply_gaussian(image, self.gaussian_sigma)

                # Watershed Segmentation
                segmented_image = ip.watershed(filtered_image, self.watershed_sigma, self.peak_min_distance, self.watershed_peak_threshold)

                # Show images
                plt.subplot(1,3,1)
                plt.imshow(image, cmap='gray')
                plt.title("Original")

                plt.subplot(1,3,2)
                plt.imshow(filtered_image)
                plt.title("Gaussian Filter")

                plt.subplot(1,3,3)
                plt.imshow(segmented_image)
                plt.title("Watershed Segmentation")

                plt.tight_layout()
                plt.show()

                # Confirm segmentation is satisfactory
                satisfied = input("Are you satisfied with the segmentation? (y to save / n to adjust / q to quit): ").strip().lower()

                if satisfied == 'y':
                    # Save A2
                    plt.figure(figsize=(8,8))
                    plt.imshow(filtered_image)
                    plt.savefig(f'{self.export_path}/A2_{image_name}_Gaussian_filter.png')
                    plt.close()
                    
                    plt.figure(figsize=(8,8))
                    plt.imshow(segmented_image)
                    plt.savefig(f'{self.export_path}/A3_{image_name}_watershed_segmentation.png')
                    plt.close()
                    print("Saved final Gaussian filter and watershed segmentation. Moving to clean up.")
                    break

                elif satisfied == 'q':
                    print("Exiting without saving.")
                    exit()
                    break

                elif satisfied == 'n':
                    try: 
                        user_input = input(f"Enter new Pre-Watershed sigma (current={self.gaussian_sigma}): ").strip()
                        if user_input:
                            self.gaussian_sigma = float(user_input)

                        user_input = input(f"Enter new Watershed sigma (current={self.watershed_sigma}): ").strip()
                        if user_input:
                            self.watershed_sigma = float(user_input)

                        user_input = input(f"Enter new Watershed min_dist (current={self.peak_min_distance}): ").strip()
                        if user_input:
                            self.peak_min_distance = int(user_input)

                        user_input = input(f"Enter new Watershed thres_abs (current={self.watershed_peak_threshold}): ").strip()
                        if user_input:
                            self.watershed_peak_threshold = float(user_input)

                    except ValueError:
                        print("Invalid input. Keeping previous values.")

        # Convolution Segmentation
        elif segmentation_type == 1:
            segmented_npz = convolution_segmentation(self.image_basename, self.export_path)

            image, _ = ip.load_image(segmented_npz)
            single_channel = utils.make_single_channel(image)

            while True:
                filtered_image = smooth(single_channel, self.gaussian_sigma)

                plt.subplot(1,3,1)
                plt.imshow(image)
                plt.title("Original")

                plt.subplot(1,3,2)
                plt.imshow(single_channel)
                plt.title("Single Channel")

                plt.subplot(1,3,3)
                filtered_plot = plt.imshow(filtered_image)
                unique_labels = np.unique(filtered_image)
                cbar = plt.colorbar(filtered_plot, ticks=unique_labels, shrink=0.5)
                cbar.set_label('Phase Labels')
                plt.title("Gaussian Filter")

                plt.tight_layout()
                plt.show()


                satisfied = input("Are you satisfied with filtering? (y to save / n to adjust / q to quit): ").strip().lower()
                if satisfied == 'y':
                    filtered_plot = plt.imshow(filtered_image)
                    unique_labels = np.unique(filtered_image)
                    cbar = plt.colorbar(filtered_plot, ticks=unique_labels, shrink=0.5)
                    cbar.set_label('Phase Labels')
                    plt.title("Gaussian Filter")
                    plt.savefig(f'{self.export_path}/A2_{image_name}_Gaussian_filter.png')
                    plt.close()
                    break

                elif satisfied == 'q':
                    print('Exiting without saving.')
                    exit()

                else:
                    user_input = input(f"Enter new Gaussian sigma (current={self.gaussian_sigma}): ").strip()
                    if user_input:
                        self.gaussian_sigma = float(user_input)


            pore = input("Enter pore space label: ").strip()
            if pore:
                pore = int(pore)
                filtered_image[filtered_image == pore] = 0
                filtered_image[filtered_image >= pore] -= 1


            segmented_image = label_segments(filtered_image)
            plt.figure(figsize=(10,10))
            plt.imshow(segmented_image)
            #plt.title("Convolution Segmentation")
            plt.axis('off')
            plt.savefig(f'{self.export_path}/A3_{image_name}_CNN_segmentation.png',dpi=300,bbox_inches='tight')
            plt.show()


        print("\n------------ Labeling Grain Spaces ------------")
        gl = GrainLabeling(segmented_image)
        custom_cmap = utils.create_turbo_with_black()

        # Label zero spaces
        gl.label_zero_spaces()
        if self.verbose:
            plt.imshow(gl.labels)
            plt.title("Labeled Pore Space")
            plt.show()

        # Label grains
        gl.assign_grain_sizes()
        plt.figure(figsize=(10,10))
        plt.imshow(gl.grain_sizes, cmap=custom_cmap, interpolation='nearest')
        plt.title("Grains Colored by Size")
        plt.colorbar(label='Grain Size (Number of Pixels)')
        plt.axis('off')
        plt.savefig(f'{self.export_path}/A4_{image_name}_grainSize.png')
        plt.show()

        # Remove floaters
        gl.remove_floaters()
        gl.assign_grain_sizes()
        plt.figure(figsize=(10,10))
        plt.imshow(gl.grain_sizes, cmap=custom_cmap, interpolation='nearest')
        plt.title("Floaters Removed")
        plt.colorbar(label='Grain Size (Number of Pixels)')
        plt.axis('off')
        plt.savefig(f'{self.export_path}/A5_{image_name}_grainSize_no_floaters.png')
        plt.show()


        # Merge small regions
        while True:
            gl.merge_small_regions(self.area_threshold)
            plt.imshow(gl.labels)
            plt.title("Merged Small Regions")
            plt.show()

            satisfied = input("Are you satisfied with filtering? (y/n): ").strip().lower()
            if satisfied == 'y':
                break

            else:
                user_input = input(f"Enter new area threshold (current={self.area_threshold}): ").strip()
                if user_input:
                    self.area_threshold = float(user_input)


        # Combine negative regions
        gl.set_negative_labels_to_zero()
        if self.verbose:
            plt.imshow(gl.labels)
            plt.title("Connected Pore Space")
            plt.show()

        # Relabel grains
        gl.assign_grain_sizes()
        plt.figure(figsize=(10,10))
        plt.imshow(gl.grain_sizes, cmap=custom_cmap, interpolation='nearest')
        plt.title("Merged Small Grains")
        plt.colorbar(label='Grain Size (Number of Pixels)')
        plt.axis('off')
        plt.savefig(f'{self.export_path}/A6_{image_name}_grainSize_cleaned.png')
        plt.show()

        # Map labels
        if segmentation_type == 0:
            gl.remap_labels(True)

        elif segmentation_type == 1:
            if pore:
                gl.remap_labels(True)
            else:
                gl.remap_labels(False)
                # gl.labels = gl.label_by_phase(gl.labels, filtered_image)


        print("\n------------ Starting Pmesh Routine ------------")
        start = time.time()
        pmesh = PolygonMesh(gl.labels)
        pmesh.make_regions()
        pmesh.map_phase_colors()
        pmesh.plot_mesh(self.export_path, image_name)
        end = time.time()
        print(f"Pmesh duration: {end - start}")

        pmesh_file = f'{self.export_path}/{image_name}_pmesh_{pmesh.unique_colors_rgb.shape[0]}phases.pkl'
        with open(pmesh_file,'wb') as f: 
            pickle.dump([pmesh.pts, pmesh.facets, pmesh.regions, pmesh.region_phases, pmesh.m, pmesh.n], f)


        print("\n------------ Creating Cubit Surfaces ------------")
        cs = CubitSurface()
        cs.load_pmesh(pmesh_file)
        cs.create_surfaces()
        utils.save_cubit(self.export_path, 'baseCub')


        print("\n------------ Splining Surfaces ------------")
        spliner = SplinedSurface()
        spliner.load_cubit_file(self.export_path, True)
        spliner.smooth_edges(self.verbose)
        spliner.clean_small_curves(self.min_spline_length)
        spliner.create_splined_surfaces()
        has_pore = True
        if segmentation_type == 1:
            has_pore = True if pore else False
        spliner.unite_by_phase(has_pore)
        utils.save_cubit(self.export_path, 'baseSpline')


        print("\n------------ Generating Final Mesh ------------")
        fm = FinalMesh(self.export_path)
        fm.remove_phase()

        # Add boundary conditions
        boundaries = input("Do you want boundary plates? (y/n) ")
        if boundaries == 'y':
            fm.create_boundary_plates()

        fm.create_mesh(self.inner_mesh_size, self.boundary_mesh_size)

        # Create nodesets
        top = input("Do you want a TOP nodeset? (y/n) ")
        right = input("Do you want a RIGHT nodeset? (y/n) ")
        bottom = input("Do you want a BOTTOM nodeset? (y/n) ")
        left = input("Do you want a LEFT nodeset? (y/n) ")
        if top == 'y':
            fm.create_nodeset('top', True if boundaries == 'y' else False)
        if right == 'y':
            fm.create_nodeset('right', True if boundaries == 'y' else False)
        if bottom == 'y':
            fm.create_nodeset('bottom', True if boundaries == 'y' else False)
        if left == 'y':
            fm.create_nodeset('left', True if boundaries == 'y' else False)


        # fm.fixed_nodeset()
        # fm.symmetry_nodeset('xsymm')
        # fm.symmetry_nodeset('ysymm')

        fm.group_blocks()
        fm.export_abaqus(self.export_path, 16)
        utils.save_cubit(self.export_path, 'finalMesh')

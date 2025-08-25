"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

import numpy as np
import pickle
import tempfile

from grain2mesh.ImageProcessor import ImageProcessor
from grain2mesh.GrainLabeling import GrainLabeling
from grain2mesh.PolygonMesh import PolygonMesh
from grain2mesh.CubitSurface import CubitSurface
from grain2mesh.SplinedSurface import SplinedSurface
from grain2mesh.FinalMesh import FinalMesh
from grain2mesh.PetroSeg import *
from grain2mesh.utils import *


def sample_image():
    img = np.array([
        [1, 1, 2, 2, 0, 0, 3, 3],
        [1, 1, 2, 0, 0, 3, 3, 3],
        [0, 0, 0, 4, 4, 4, 3, 3],
        [0, 0, 0, 4, 4, 0, 0, 3],
        [5, 5, 5, 6, 6, 6, 6, 3],
        [5, 5, 5, 5, 6, 0, 0, 0],
        [0, 5, 5, 7, 7, 0, 8, 0],
        [0, 0, 0, 7, 7, 0, 0, 0]
    ])

    return img


def test_ImageProcessor():
    ip = ImageProcessor()
    image, image_name = ip.load_image('./tests/data/test_binary.npz')

    assert isinstance(image, np.ndarray)
    assert image.shape == (300, 300)
    assert isinstance(image_name, str)
    assert image_name == 'test_binary'


    # Gaussian filter
    filtered = ip.apply_gaussian(image, 3)
    assert isinstance(filtered, np.ndarray)
    assert np.all((filtered == 0) | (filtered == 1))
    assert image.shape == filtered.shape
    assert filtered.dtype == np.uint8


    # Watershed segmentation
    segmented = ip.watershed(filtered, 1, 6, 0.5)
    assert isinstance(segmented, np.ndarray)
    assert image.shape == segmented.shape
    assert len(np.unique(segmented)) == 30
    assert np.all(segmented >= 0)


def test_PetroSeg_resize():
    params = {"target_size": (512, 512)}

    # Call without image
    output = load_and_resize_image("", params)
    assert output == None

    # Resize RGB image
    original, resized = load_and_resize_image('./tests/data/test_RGB.png', params)
    assert isinstance(original, np.ndarray)
    assert original.shape == (342, 341, 3)
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (512, 512, 3)


def test_CNN_segmentation():
    image = np.ones((32,32,3), dtype=np.uint8)

    params = {
        'train_epoch': 1,
        'mod_dim1': 8,
        'mod_dim2': 4,
        'min_label_num': 1,
        'max_label_num': 10,
        'target_size': (32, 32)
    }

    segmented = perform_custom_segmentation(image, params)
    assert isinstance(segmented, np.ndarray)
    assert segmented.shape == image.shape


def test_GrainLabeling():
    img = sample_image()
    values = np.unique(img)[1:]
    correct_adjacencies = {
        0: set(), 
        1: {0, 2},
        2: {0, 1, 4},
        3: {0, 4, 6},
        4: {0, 2, 3, 5, 6},
        5: {0, 4, 6, 7},
        6: {0, 3, 4, 5, 7},
        7: {0, 5, 6},
        8: {0}
    }

    # Constructor
    gl = GrainLabeling(img)
    assert isinstance(gl.labels, np.ndarray)
    assert gl.labels.shape == img.shape
    assert np.array_equal(img, gl.labels)
    assert gl.grain_sizes is None

    # Adjacency map helper function
    adjacencies = gl._find_adjacent_regions()
    assert adjacencies == correct_adjacencies

    # Assign zero spaces negative labels
    gl.label_zero_spaces()
    assert gl.labels.shape == img.shape
    assert np.all(gl.labels != 0)
    assert np.all(np.isin(values, gl.labels))


    # Assign grain sizes
    gl.assign_grain_sizes()
    assert gl.grain_sizes is not None
    assert isinstance(gl.grain_sizes, np.ndarray)
    assert gl.grain_sizes.shape == img.shape
    assert np.all(gl.grain_sizes >= 0)


    # Remove floaters
    gl.remove_floaters()
    gl.assign_grain_sizes()
    assert gl.labels.shape == img.shape
    assert np.all(gl.labels != 8)

    # Remap labels starting from 1
    gl.remap_labels(False)
    assert gl.labels.shape == img.shape
    assert np.all(gl.labels >= 1)


    # Merge small regions
    gl.merge_small_regions(4.0)
    assert gl.grain_sizes.shape == img.shape
    assert gl.labels.shape == img.shape
    assert np.all(gl.labels != 3)
    assert np.all(gl.labels != 7)


    # All negaive spaces labeld 0
    gl.set_negative_labels_to_zero()
    assert gl.labels.shape == img.shape
    assert np.all(gl.labels >= 0)


    # Remapp labels starting from 0
    gl.remap_labels(True)
    assert gl.labels.shape == img.shape
    assert np.all(gl.labels >= 0)
    unique_values = np.unique(gl.labels)
    assert np.all(np.diff(unique_values) == 1)


def test_PolygonMesh():
    img = sample_image()
    length = img.shape[0] * img.shape[1]

    # Constructor
    pmesh = PolygonMesh(img)
    assert pmesh.n_regions == (length)
    assert pmesh.pts.shape == ((img.shape[0] + 1) * (img.shape[1] + 1), 2)
    assert pmesh.kps.shape == (img.shape[0] + 1, img.shape[1] + 1)
    assert np.all(pmesh.facets == -1)
    assert np.all(pmesh.regions == 0)
    assert np.all(pmesh.region_phases == 0)
    assert np.all(pmesh.facet_top == -1)
    assert np.all(pmesh.facet_bottom == -1)
    assert np.all(pmesh.facet_left == -1)
    assert np.all(pmesh.facet_right == -1)


    # Make region for each pixel
    pmesh.make_regions()
    assert pmesh.regions.shape == (length, 4)
    assert pmesh.k_regions == length
    num_facets = (img.shape[0] + 1) * img.shape[1] * 2
    assert np.all(np.isin(np.arange(num_facets), pmesh.regions))
    assert pmesh.k_facets == num_facets
    assert len(pmesh.regions) == length
    assert np.all(np.isin([0,1,2,3], pmesh.regions[0]))


    # Create phase colors
    pmesh.map_phase_colors()
    assert pmesh.phases is not None
    assert len(pmesh.phases) == len(np.unique(pmesh.region_phases)) + 1


def test_Cubit_pipeline():

    with tempfile.TemporaryDirectory() as tmpdir:
        # Creating Cubit Surfaces
        test_pmesh = './tests/data/test_pmesh.pkl'
        cs = CubitSurface()
        cs.load_pmesh(test_pmesh)
        cs.create_surfaces()
        save_cubit(tmpdir, 'baseCub')

        assert os.path.exists(f"{tmpdir}/baseCub.cub")

        # Splining Surfaces
        spliner = SplinedSurface()
        spliner.load_cubit_file(tmpdir, True)
        assert os.path.exists(f"{tmpdir}/baseSpline_ImprintOriginal.cub")
        spliner.smooth_edges()
        spliner.clean_small_curves(4.0)
        spliner.create_splined_surfaces()
        spliner.unite_by_phase()
        save_cubit(tmpdir, 'baseSpline')

        assert os.path.exists(f"{tmpdir}/baseSpline.cub")

        # Final Mesh
        fm = FinalMesh(tmpdir)
        fm.remove_phase()
        fm.create_boundary_plates()
        fm.create_mesh(8.0, 8.0)
        fm.fixed_nodeset()
        fm.symmetry_nodeset('xsymm')
        fm.symmetry_nodeset('ysymm')
        fm.group_blocks()
        fm.export_abaqus(tmpdir, 16)
        save_cubit(tmpdir, 'finalMesh')

        assert os.path.exists(f"{tmpdir}/finalMesh.cub")
        assert os.path.exists(f"{tmpdir}/finalMesh.inp")




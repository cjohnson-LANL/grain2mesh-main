# Grain2Mesh - Image-based Grain Partitioning & Cubit Mesh Generation

Grain2Mesh is a Python and Cubit mesh generator for unprocessed mesoscale images. Features of Grain2Mesh include:

- Binary and RGB image processing
- Integrated convolutional neural network for RGB segmentation
- Grain segmentation
- 2D Cubit mesh generation

Grain2Mesh is a component of "AI for Earthquake Physics" (O4943) and has been acknowledged by NNSA for open-source release.

## Requirements
- **Cubit** -  (v14.16 or later)

### Adding Cubit to PYTHONPATH (MacOS)
Add the following line to `~/.bashrc`:  
```bash
export PYTHONPATH="$PYTHONPATH:/<your_path_to_Cubit>/Cubit.app/Contents/MacOS
```  
Then run `source ~/.bashrc`.  
Verify it worked using `echo $PYTHONPATH`.  

## Installation

### 1. Recommended Pip Installation

Create and activate a new conda environment:

```bash
conda create --name <your-env-name>
conda activate <your-env-name>
```

(Optional) If you’re on macOS and need to force `osx-64` packages:

```bash
conda config --env --set subdir osx-64
```

Install Python:

```bash
conda install python=3.10
```

Finally, install `grain2mesh` directly from the repository:

```bash
pip install git+ssh://git@git.lanl.gov/rghill/grain2mesh.git
```


### 2. Installation (via local clone and environment file)

Clone the repository:

```bash
git clone git@git.lanl.gov:rghill/grain2mesh.git
cd grain2mesh
```

Create and activate the conda environment from `environment.yml`:

```bash
CONDA_SUBDIR=osx-64 conda env create -f environment.yml
conda activate grain2mesh-env
```


### 3. Run Example

The `example/` folder contains a sample configuration file and script for importing and running grain2mesh. Running this script is a great way to ensure everything is installed and working properly before you start using the package. The example takes ~5 minutes to run.

From the root directory, run:
```bash
python example/example.py
```

When propmted to, enter `y` for statsifactory image processing, and `0` for watershed segmentation. Running the script should populate `example/results/` with the following:
- A1_binary_example_RAW_nostitch.png
- A2_binary_example_Gaussian_filter.png
- A3_binary_example_watershed_segmentation.png
- A4_binary_example_grainSize.png
- A6_binary_example_grainSize_no_floaters.png
- A6_binary_example_grainSize_cleaned.png
- B1_binary_example_pmesh.png
- binary_example_pmesh_32phases.pkl
- baseCub.cub
- baseSpline_ImprintOriginal.cub
- baseSpline.cub
- finalMes.cub
- finalMesh.inp

### CNN Example

To run an example of the CNN segmentation, change the `example/example.py` file to load `RGB_config.json`. When prompted, enter `1` for CNN segmentation and use the following parameters:
- number of epochs: `100`
- mod_dim1: `20`
- mod_dim2: `5`
- min_label_num: `1`
- max_label_num: `20`
- width: `600`
- height: `600`
- random color: `0`

Press `enter` when prompted for the pore space label for no pore space.


## Configuration Files
A JSON configuration file is required to run grain2mesh. The table below explains each configuration value.

| Key                | Type   | Required    | Description                              | Default |
|--------------------|--------|-------------|------------------------------------------|---------|
| export_path        | string | Yes         | output directory                         | None    |
| image_basename     | string | Yes         | path to input image.                     | None    |
| gaussian_sigma     | float  | Yes         | sigma for Gaussian filter                | 1       |
| area_threshold     | float  | Yes         | minimum area of grain                    | 100     |
| min_spline_length  | float  | Yes         | minimum length of grain edge (0.0 - 4.0) | 0.0     |
| inner_mesh_size    | float  | Yes         | dimension of material mesh element       | 8.0     |
| boundary_mesh_size | float  | Yes         | dimension of boundary mesh element       | 8.0     |
| watershed_sigma    | float  | Binary Only | sigma for distance transform smoothing   | 1       |
| peak_min_distance  | int    | Binary Only | minimum distance separating peaks        | 6       |
| watershed_peak_threshold | float | Binary Only | minimum intensity of peaks          | 0.5     |
| verbose            | bool   | No          | shows extra plots                        | False   |

NOTE: 
- Cubit cannot create surfaces if two closed loops intersect. If Cubit fails to create surfaces, consider increasing `gaussian_sigma` value.
- `min_spline_length` is used to prevent small mesh elements. However, Cubit cannot resolve splines >4.0 pixels in length. Use a value between 0.0 and 4.0.



## Package Structure
```text
grain2mesh/
├── src/
│   ├── __init__.py
│   ├── CubitSurface.py
│   ├── FinalMesh.py
│   ├── GrainLabeling.py
│   ├── ImageProcessor.py
│   ├── main.py
│   ├── PetroSeg.py
│   ├── PolygonMesh.py
│   ├── SplinedSurface.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── data/
│   └── test_main.y
├── example/
│   ├── example.py
│   ├── data/
│   ├── results/
│   └── config.json
├── environment.yml
├── pyproject.toml
└── README.md
```

## Publications

## License
O# O4943 
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

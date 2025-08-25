"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Class defintion for FinalMesh

Functionality:
- Remove pore space
- Create boundary plate
- Create nodesets
- Group phases/blocks
- Generate Cubit mesh
"""
import numpy as np
import pickle
import cubit
from .utils import init_cubit


class FinalMesh:
    def __init__(self, exp_path):
        self.phase_names = []
        self.boundary_ids = {}
        self.nodeset_count = 0
        self.plate_defs = {}

        init_cubit()
        cubit.cmd(f'Open "{exp_path}/baseSpline.cub"')

        surface_ids = cubit.parse_cubit_list("surface", "all")
        self.bbox = cubit.get_total_bounding_box("surface", surface_ids)


    def remove_phase(self):
        phase_to_delete = 'phase0'

        all_bodies = cubit.parse_cubit_list("body", "all")

        bodies_to_delete = [
            b_id for b_id in all_bodies 
            if list(cubit.get_entity_names("body", b_id)) == ['C_' + phase_to_delete]
        ]

        for body in bodies_to_delete:
            print(f"Deleting Fulid Phase from Body {body}")
            cubit.cmd(f'Delete Body {body}')

        self.phase_names = [f'phase{i}' for i in range(1, len(all_bodies))]


    def create_boundary_plates(self):
        # Bounding box dimensions
        delta_x = abs(self.bbox[1] - self.bbox[0])
        thickness = delta_x / 20

        x_min, x_max = self.bbox[0], self.bbox[1]
        y_min, y_max = self.bbox[3], self.bbox[4]

        # Define plate boundaries
        self.plate_defs = {
            'top': [(x_min, y_max + thickness), (x_max, y_max + thickness), (x_max, y_max), (x_min, y_max)],
            'right': [(x_max, y_max), (x_max + thickness, y_max), (x_max + thickness, y_min), (x_max, y_min)],
            'bottom': [(x_min, y_min), (x_max, y_min), (x_max, y_min - thickness), (x_min, y_min - thickness)],
            'left': [(x_min - thickness, y_max), (x_min, y_max), (x_min, y_min), (x_min - thickness, y_min)]
        }

        # Create boundary surface for each side
        for side, coords in self.plate_defs.items():
            vertices = [cubit.create_vertex(x, y, 0.0) for x, y in coords]
            curves = [cubit.create_curve(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]

            boundary = cubit.create_surface(curves)
            # self.boundary_ids.append(boundary.surfaces()[0].id())
            self.boundary_ids[side] = boundary.surfaces()[0].id()


    def create_mesh(self, h, h2):
        boundary_plate_str = ' '.join(map(str, self.boundary_ids.values()))

        # Set mesh configuration
        cubit.cmd('surface all scheme TriDelaunay')# TriDelaunay meshing 
        cubit.cmd('Set Tridelaunay point placement gq')
        cubit.cmd(f"surface all except {boundary_plate_str} size {h}") # mesh size for speicimen
        cubit.cmd(f"surface {boundary_plate_str} size {h2}") # mesh size for plates
        cubit.cmd("mesh surface all")
        cubit.cmd("surface all smooth scheme centroid area pull")
        cubit.cmd("smooth surface all")

    def create_nodeset(self, side, boundary: bool):
        self.nodeset_count += 1

        if boundary:
            plate = self.boundary_ids[side]
            surface = cubit.surface(plate)
            curves = surface.curves()

            if side == 'top':
                curve = curves[1]
            elif side == 'right':
                curve = curves[2]
            elif side == 'bottom':
                curve = curves[3]
            elif side == 'left':
                curve = curves[0]

        else:
            x_min, x_max = self.bbox[0], self.bbox[1]
            y_min, y_max = self.bbox[3], self.bbox[4]

            side_def = {
                "top": [(x_min, y_max), (x_max, y_max)],
                "right": [(x_max, y_min), (x_max, y_max)],
                "bottom": [(x_min, y_min), (x_max, y_min)],
                "left": [(x_min, y_min), (x_min, y_max)],
            }

            vertices = [cubit.create_vertex(x, y, 0.0) for x, y in side_def[side]]
            curve = cubit.create_curve(vertices[0], vertices[1])

        cubit.cmd(f'nodeset {self.nodeset_count} curve {curve.id()}')
        cubit.cmd(f'nodeset {self.nodeset_count} name "{side}"')

    def group_blocks(self):
        plate_names = ['plate_top','plate_right','plate_bottom','plate_left']

        # Group phases
        for phase in self.phase_names:
            cubit.cmd(f"group '{phase}' add tri in C_{phase}")

        # Group bounding surfaces
        for plate, surfID in zip(plate_names, self.boundary_ids):
            cubit.cmd(f"group '{plate}' ad tri in surface {surfID}")

        # Assign and name blocks for each phase
        for i, phase in enumerate(self.phase_names, start=1):
            cubit.cmd(f"block {i} group {phase}")
            cubit.cmd(f'block {i} name "C_{phase}"')

        # Assign and name block for each bounding surface
        for i, plate in enumerate(plate_names, start=len(self.phase_names) + 2):
            cubit.cmd(f"block {i} group {plate}")
            cubit.cmd(f'block {i} name "N_{plate}"')

    def export_abaqus(self, exp_path, precision):
        cubit.cmd(f'Set Abaqus Precision {precision}')
        cubit.cmd(f'export abaqus "{exp_path}/finalMesh.inp" overwrite')

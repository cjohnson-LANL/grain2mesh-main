"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Class defintion for SplinedSurface
"""
import numpy as np
import cubit
import math
from collections import defaultdict
from .utils import init_cubit


class SplinedSurface:
    def __init__(self):
        self.bodies_to_delete = None    # old and imprinted bodies
        self.surfaces_to_check = None   # original and imprinted surfaces
        self.surfaces = None            # imprinted surfaces to be splined

        self.spline_to_points = {}      # map spline id to corresponding points
        self.spline_set = set()         # list of spline ids
        self.edge_dict = {}             # ensures unique splines (points to curve id)

        self.imprint_to_splines = {}    # map imprinted surface to corresponding splines
        self.curves_to_delete = []      # original splines not used to make surfaces

        self.new_cubit_surfaces = None  # splined surfaces
        self.new_areas = None
        self.new_centers = None
        self.new_bodies = None
        self.new_phase_map = defaultdict(list)

        self.verbose = False

    def load_cubit_file(self, exp_path, imprint: bool):
        init_cubit()

        if imprint:
            cubit.cmd(f"Open '{exp_path}/baseCub.cub'")
            self._imprint_and_save(exp_path)
        else:
            cubit.cmd(f"Open '{exp_path}/baseSpline_ImprintOriginal.cub'")
            self.bodies_to_delete = cubit.parse_cubit_list("body", "all")
            self.surfaces_to_check = cubit.parse_cubit_list("surface","all")
            self.surfaces = cubit.body(self.bodies_to_delete[-1]).surfaces()

    def _imprint_and_save(self, exp_path):
        """ Imprint bodies to ensure consitent geometry """
        # Make bouding surface
        surface_list = cubit.parse_cubit_list("surface", "all")
        bbox = cubit.get_total_bounding_box("surface", surface_list)
        self._make_bounding_surface(bbox)

        # Imprint bodies
        self.bodies_to_delete = cubit.parse_cubit_list("body", "all")
        bbox_body = cubit.body(self.bodies_to_delete[-1])
        for body in self.bodies_to_delete[:-1]:
            cubit.cmd(f"imprint Body {bbox_body.id()} with Body {body}")

        self.surfaces_to_check = cubit.parse_cubit_list("surface","all")
        self.surfaces = bbox_body.surfaces()

        cubit.cmd(f"Save as '{exp_path}/baseSpline_ImprintOriginal.cub' overwrite")

    def _make_bounding_surface(self, bbox):
        """ Create surface for boundaries """
        # Get corner points
        b1 = cubit.create_vertex(bbox[0],bbox[3],0.0)
        b2 = cubit.create_vertex(bbox[0],bbox[4],0.0)
        b3 = cubit.create_vertex(bbox[1],bbox[4],0.0)
        b4 = cubit.create_vertex(bbox[1],bbox[3],0.0)

        # Create edges
        bb_curves = []
        bb_curves.append(cubit.create_curve(b1,b2))
        bb_curves.append(cubit.create_curve(b2,b3))
        bb_curves.append(cubit.create_curve(b3,b4))
        bb_curves.append(cubit.create_curve(b4,b1))

        cubit.create_surface(bb_curves)

    def smooth_edges(self, verbose):
        self.verbose = verbose
        """ Create splines for borders between surfaces """
        for surface in self.surfaces:
            adjacent_surfaces = self._get_adjacent_surfaces(surface)
            vertices = self._get_vertices(surface)

            # Get all vertices of all adjacent surfaces
            adjacent_surface_vertices = []
            for i in adjacent_surfaces:
                surf = cubit.surface(int(i))
                adjacent_surface_vertices.append(self._get_vertices(surf))

            # Make splines
            splines = self._make_splines(adjacent_surface_vertices, vertices, surface)

            # Create remaning curves not shared with an adjacent surface
            flat_adjacent_vertices = [pair for sub_list in adjacent_surface_vertices for pair in sub_list]
            not_shared_points = [pair for pair in vertices if pair not in flat_adjacent_vertices]
            new_curves = []
            for pair in not_shared_points:
                v1 = cubit.create_vertex(pair[0][0], pair[0][1], pair[0][2])
                v2 = cubit.create_vertex(pair[1][0], pair[1][1], pair[1][2])
                cur = cubit.create_curve(v1,v2)
                new_curves.append(cur.id())
                self.spline_set.add(cur.id())

                key = (v1.coordinates(), v2.coordinates())
                if key not in self.edge_dict:
                    self.edge_dict[key] = cur.id()
                    self.spline_to_points[cur.id()] = [v1.coordinates(), v2.coordinates()]
                    # self.surface_dict[surface.id()].append(cur.id())
                else:
                    sid = self.edge_dict[key]
                    # self.surface_dict[surface.id()].append(sid)

            # Combine splines and unshared curves
            for spline in splines:
                new_curves.append(spline.id())
                self.spline_set.add(spline.id())

            # Map imprinted surface to corresponding splines
            self.imprint_to_splines[surface.id()] = new_curves

    def _get_adjacent_surfaces(self, surface):
        """ Return a list of ll the surfaces adjacent to given surface """
        adjacencies = cubit.get_adjacent_surfaces("surface", surface.id())
        adjacencies = np.unique(adjacencies)
        adjacencies = adjacencies[adjacencies != surface.id()]

        return adjacencies

    def _get_vertices(self, surface):
        """ Return a list of all the vertices of a given surface """
        vertices = []
        for curve in surface.curves():
            vertices.append([curve.vertices()[0].coordinates(), curve.vertices()[1].coordinates()])

        return vertices

    def _make_splines(self, adj_vertices, vertices, surface):
        """
        Create spline between each adjacent surface

        Args:
        - adj_vertices (list): lists of vertices for each adjacent surface
        - vertices (list): list of vertices of current surface
        - surface (Cubit surface): current surface

        Returns:
        - spline_list (list): list of Cubit Curve objects
        """
        spline_list = []
        for points in adj_vertices:
            shared_points = [pair for pair in vertices if pair in points]

            pairs_list, start_point_list, end_point_list= self._organizePointsList(shared_points)
            for i, pairs in enumerate(pairs_list):
                points_to_spline = self._generate_spline_list(pairs, start_point_list[i], end_point_list[i]) # get points to spline

                # Ensure spline is unique
                key = tuple(sorted(points_to_spline))
                if key not in self.edge_dict:
                    spline = cubit.create_spline(points_to_spline, surface.id()) # create spline
                    self.edge_dict[key] = spline.id()                            # track spline
                    self.spline_to_points[spline.id()] = points_to_spline        # map spline to points that make it
                    spline_list.append(spline)
                else:
                    sid = self.edge_dict[key] # use existing spline
                    spline_list.append(cubit.curve(sid))


        return spline_list

    def _generate_spline_list(self, pairs, start_point, end_point):
        """ Put points in correct spline order"""
        splineList = []

        splineList.append(start_point) # put the final anchor point as the spline anchor..

        for pair in pairs:
            splineList.append(self._midpoint(pair[0],pair[1]))

        splineList.append(end_point) # put the final anchor point as the spline anchor..

        return splineList

    def _midpoint(self, point1, point2):
        """
        Calculate the midpoint between two 3D points.

        Args:
        point1 (tuple): A tuple containing the coordinates (x, y, z) of the first point.
        point2 (tuple): A tuple containing the coordinates (x, y, z) of the second point.

        Returns:
        tuple: A tuple containing the coordinates (x_mid, y_mid, z_mid) of the midpoint.
        """
        x_mid = (point1[0] + point2[0]) / 2
        y_mid = (point1[1] + point2[1]) / 2
        z_mid = (point1[2] + point2[2]) / 2
        return (x_mid, y_mid, z_mid)

    def _organizePointsList(self, shared_points):
        # This new function accounts for the pesky situations where there are multiple curves touch a boundary 
  
        bugged_points = sorted(shared_points, key=lambda point: (point[1], point[0]))
        all_points = [point for pair in bugged_points for point in pair]  # Flatten the list of pairs into a single list of points

        # Count occurrences of each point
        point_counts = {}
        for point in all_points:
            if point in point_counts:
                point_counts[point] += 1
            else:
                point_counts[point] = 1

        # Find points that occur only once
        unique_points = [point for point, count in point_counts.items() if count == 1]

        #unique_points  = sorted(unique_points, key=lambda point: (point[1], -point[0]))# very importnat to include so that the surfaces are the same in the situation where there are multple..

        number_of_curves = len(unique_points)//2

        if number_of_curves%2 != 0 and number_of_curves > 1:
            print('NOT MODULAR CURVE ENDS.. BIG PROBLEM')
            #raise ValueError("Not modular curve ends.. big problem ..")
        organized_List = []
        start_point_List = []
        end_point_List = []

        if not unique_points:
            start_point = bugged_points[0][0]

            organized = []
            current_point = []
            while len(organized) < len(bugged_points):
                # first point when organized has no length..
                if not organized:
                    for pair in bugged_points:
                        if start_point in pair:
                            current_pair = pair
                            current_point = pair[0] if pair[0] != start_point else pair[1]
                            organized.append(pair)

                # subsequent points
                # Find the next pair where the other point occurs
                for pair in bugged_points:
                    if current_point in pair and pair != current_pair:
                        if self.verbose:
                            print("Next pair found:", pair)

                        current_pair = pair
                        current_point = pair[0] if pair[0] != current_point else pair[1]
                        organized.append(pair)
                        break
            #now fix the 'start point' to be same as the end point..
            start_point = end_point = current_point
            organized_List.append(organized)
            start_point_List.append(start_point)
            end_point_List.append(end_point)
            print('CLOSED LOOP Point List')
            return organized_List, start_point_List, end_point_List



        for ii in range(0,number_of_curves):

            start_point = unique_points.pop(0)
            #end_point = unique_points[(ii*2)+1]
            
            organized = []
            current_point = []
            while current_point not in unique_points: # whilee the current potential end point is NOT in the uniq
                
                # first point when organized has no length..
                if not organized:
                    for pair in bugged_points:
                        if start_point in pair:
                            current_pair = pair
                            current_point = pair[0] if pair[0] != start_point else pair[1]
                            organized.append(pair)

                # subsequent points
                # Find the next pair where the other point occurs
                for pair in bugged_points:
                    if current_point in pair and pair != current_pair:
                        #print("Next pair found:", pair)
                        #time.sleep(5)
                        current_pair = pair
                        current_point = pair[0] if pair[0] != current_point else pair[1]
                        organized.append(pair)
                        break

            # which we know it is from above.. now remove from the unique list
            if current_point in unique_points:
                index_to_remove = unique_points.index(current_point)
                end_point = unique_points.pop(index_to_remove)

            sum1 = start_point[0]+start_point[1]
            sum2 = end_point[0]+end_point[1]

            if not sum1 < sum2: # this will make sure that each time we look at the unique shared points whether from the adject side or the vertex side.. we begin at the same start point (the lower sum)
                dummy = start_point
                dummy2 = end_point
                start_point = dummy2
                end_point = dummy

            start_point_List.append(start_point)
            end_point_List.append(end_point)
            organized = []
            current_point = start_point
            while current_point != end_point: # while our organized list is not generated.. generate it

                # first point when organized has no length..
                if not organized:
                    for pair in bugged_points:
                        if start_point in pair:
                            current_pair = pair
                            current_point = pair[0] if pair[0] != start_point else pair[1]
                            organized.append(pair)

                # subsequent points
                # Find the next pair where the other point occurs
                for pair in bugged_points:
                    if current_point in pair and pair != current_pair:
                        print("Next pair found:", pair)
                        #time.sleep(5)
                        current_pair = pair
                        current_point = pair[0] if pair[0] != current_point else pair[1]
                        organized.append(pair)
                        break
            
            organized_List.append(organized)
            
        return organized_List, start_point_List, end_point_List

    def clean_small_curves(self, min_length):
        """ Merges small curves with adjacent splines to prevent bad mesh generation """
        surface_list = cubit.parse_cubit_list("surface", "all")
        bbox = cubit.get_total_bounding_box("surface", surface_list)
        short_adj_dict = self._find_small_adjacent_curves(min_length)

        for spline in self.spline_set:
            curve = cubit.curve(spline)

            # Process small curves
            if curve.length() < min_length:
                self.curves_to_delete.append(spline)

                border = self._on_boundary(curve, bbox)

                # Short curve is on a boundary
                if border:
                    startpoint = curve.vertices()[0].coordinates()
                    endpoint = curve.vertices()[1].coordinates()

                    # Extend neighboring curves
                    for points in self.spline_to_points.values():
                        if startpoint in points:
                            if points.index(startpoint) == 0:
                                points[0] = border
                            else:
                                points[-1] = border

                        elif endpoint in points:
                            if points.index(endpoint) == 0:
                                points[0] = border
                            else:
                                points[-1] = border

                # Short curve is has short neighbor
                elif spline in short_adj_dict.keys():
                    midpoint = short_adj_dict[spline][1]
                    v1 = short_adj_dict[spline][2]
                    v2 = short_adj_dict[spline][3]

                    # Extend neighboring curves
                    for edge, points in self.spline_to_points.items():
                        if v1 in points:
                            if points.index(v1) == 0:
                                points[0] = midpoint
                            else:
                                points[-1] = midpoint

                        elif v2 in points:
                            if points.index(v2) == 0:
                                points[0] = midpoint
                            else:
                                points[-1] = midpoint

                # Regular short curve
                else:
                    startpoint = curve.vertices()[0].coordinates()
                    endpoint = curve.vertices()[1].coordinates()

                    # Extened neighboring curves
                    for line, points in self.spline_to_points.items():
                        if startpoint in points:
                            if points.index(startpoint) == 0:
                                points[0] = endpoint
                            else:
                                points[-1] = endpoint


    def _find_small_adjacent_curves(self, min_length):
        """
        Finds adjacent pairs of of short edges that need to be merged at shared point

        Args:
        - min_length (float): minimum length of a spline

        Returns:
        - short_adjacent_dict (dictionary): map of pairs of adjacent short curves
        """
        short_adj_dict = {}

        # Create a dicitonary of short adjacent curves
        for spline in self.spline_set:
            curve = cubit.curve(spline)

            # Find small curves
            if curve.length() < min_length:
                v1 = curve.vertices()[0].coordinates()
                v2 = curve.vertices()[1].coordinates()

                adjs = []
                # Find all adjacent curves
                for spln, pts in self.spline_to_points.items():
                    if v1 in pts and spln != spline:
                        other = pts[-1] if pts.index(v1) == 0 else pts[0]
                        adjs.append((spln, v1, v2, other))

                    if v2 in pts and spln != spline:
                        other = pts[-1] if pts.index(v2) == 0 else pts[0]
                        adjs.append((spln, v2, v1, other))

                # Identify adajacent curves that are also small
                for side in adjs:
                    check = cubit.curve(side[0])
                    if check.length() < min_length:
                        short_adj_dict[spline] = side
                        short_adj_dict[side[0]] = (spline, side[1], side[3], side[2])

        return short_adj_dict

    def _on_boundary(self, curve, bbox):
        """
        Determine if a curve lies on, or has an endpoint on, a boundary edge

        Args:
        - curve (Cubit Curve object): curve to check
        - bbox (tuple): boundary coodinates

        Returns:
        - vertex on boundary if there is one, else None
        """


        v1 = curve.vertices()[0].coordinates()
        v2 = curve.vertices()[1].coordinates()

        v1_side = False
        v2_side = False
        v1_tb = False
        v2_tb = False
        border_points = 0
        if v1[0] == bbox[0] or v1[0] == bbox[1]: # v1 on side border
            border_points += 1
            v1_side= True
        if v2[0] == bbox[0] or v2[0] == bbox[1]: # v2 on side border
            border_points += 1
            v2_side = True
        if v1[1] == bbox[3] or v1[1] == bbox[4]: # v1 on top/bottom border
            border_points += 1
            v1_tb = True
        if v2[1] == bbox[3] or v2[1] == bbox[4]: # v2 on top/bottom border
            border_points += 1
            v2_tb = True


        if border_points == 1:
            if v1_side or v1_tb:
                return v1
            elif v2_side or v2_tb:
                return v2

        elif border_points == 2:
            return self._midpoint(v1, v2)

        elif border_points == 3:
            if v1_side and v1_tb:
                return v1
            elif v2_side and v2_tb:
                return v2


        return None

    def create_splined_surfaces(self):
        """ Creates a surfaces from corresponding splines """
        for surface, splines in self.imprint_to_splines.items():
            curves = []

            for spline in splines:
                valid = True
                if spline not in self.curves_to_delete:
                    points = self.spline_to_points[spline]

                    if points[0] == points[-1]: # Loop curves
                        for spln in splines:
                            if spln != spline and spln not in self.curves_to_delete:
                                pts = self.spline_to_points[spln]
                                if points[0] in pts: # Ignore loops that itersect larger surface at a single point
                                    valid = False
                    if valid:
                        curves.append(cubit.create_spline(points, surface))

            # Create smooth surface
            if self.verbose:
                print(surface, [c.id() for c in curves])
            if len(curves) > 0:
                cubit.create_surface(curves)


    def unite_by_phase(self, has_pore):
        """ Unify and label bodies of each phase """
        all_surfaces = cubit.parse_cubit_list("surface", "all")
        all_surfaces = set(all_surfaces)
        self.surfaces_to_check = set(self.surfaces_to_check)
        new_surfaces = list(all_surfaces.difference(self.surfaces_to_check))

        # Get surface attributes of new splined surfaces
        (self.new_cubit_surfaces, self.new_areas, 
         self.new_centers, self.new_bodies) = self._get_surface_info(new_surfaces)

        # Match original surface to splined surface
        for i, body in enumerate(self.bodies_to_delete[:-1]):
            phase_body = cubit.body(body)
            phase_surfaces = phase_body.surfaces()

            for surface in phase_surfaces:
                self._match_surface(surface, i)


        # Delete old curves
        delete_string = 'Delete Curve ' + ' '.join(str(curve) for curve in self.spline_to_points.keys())
        cubit.cmd(delete_string)
        
        # Delete old bodies
        delete_string = 'Delete Body ' + ' '.join(map(str, list(self.bodies_to_delete)))
        cubit.cmd(delete_string)

        # Unite bodies and label phases
        unite_string_list = []
        for i, bodies in self.new_phase_map.items():
            bodies = list(set(bodies))

            if len(bodies) > 1:
                unite_string = "Unite Body " + ' '.join(str(body) for body in bodies)
                unite_string_list.append(unite_string)
                cubit.cmd(unite_string)

                group_id = int(self._extract_second_number(unite_string_list[0]))
            else:
                group_id = bodies[0]

            pid = (i + 1) if not has_pore else i
            print(has_pore, pid)
            cubit.cmd(f'body {group_id} name \"C_phase{pid}\"')

    def _get_surface_info(self, surface_list):
        new_cubit_surfaces = []
        new_areas = []
        new_centers = []
        new_bodies = []

        for surface in surface_list:
            cubit_surface = cubit.surface(surface)
            new_cubit_surfaces.append(cubit_surface)
            new_areas.append(cubit_surface.area())
            new_centers.append(cubit_surface.center_point())
            new_bodies.append(cubit.get_owning_body("surface", cubit_surface.id()))

        return new_cubit_surfaces, new_areas, new_centers, new_bodies

    def _match_surface(self, surface, phase):
        """ Match original surface to splined surface """
        cp = surface.center_point() # center of old surface
        idx = self._find_closest(cp) # index of matching new surface

        new_surface = self.new_cubit_surfaces[idx] # matching new surface
        new_body = self.new_bodies[idx]            # owning body of new surface

        self.new_phase_map[phase].append(new_body) # track phases

    def _find_closest(self, cp):
        """ Finds closest center point of splined surfaces """
        closest = None
        min_dist = 100

        for i, point in enumerate(self.new_centers):
            dist = math.sqrt(
                (point[0] - cp[0])**2 +
                (point[1] - cp[1])**2 +
                (point[2] - cp[2])**2
            )

            if dist < min_dist:
                min_dist = dist
                closest = i

        return closest

    def _extract_second_number(self, s):
        # Split the string by spaces
        parts = s.split()
        
        # Check if there are at least three parts (one word and at least two numbers)
        if len(parts) >= 3:
            # Extract the second number from the list
            second_number = parts[2]
            return second_number
        else:
            return None  # or raise an exception if the format is not as expected

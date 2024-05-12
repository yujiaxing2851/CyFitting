import numpy as np
import open3d as o3d


class Cylinder:
    def __init__(self, top_center: 'tuple[float, float, float]', bottom_center: 'tuple[float, float, float]',
                 radius: float):
        self.top_center = top_center
        self.bottom_center = bottom_center
        self.radius = radius
        self.points = None

    def get(self) -> 'tuple[tuple[float, float, float], tuple[float, float, float], float]':
        return self.top_center, self.bottom_center, self.radius

    def split_get(self) -> 'tuple[float, float, float, float, float, float, float]':
        return self.top_center[0], self.top_center[1], self.top_center[2], self.bottom_center[0], self.bottom_center[1], \
        self.bottom_center[2], self.radius

    def numpy_get(self) -> 'np.array[float, float, float, float, float, float, float]':
        return np.array(
            [self.top_center[0], self.top_center[1], self.top_center[2], self.bottom_center[0], self.bottom_center[1],
             self.bottom_center[2], self.radius])

    def get_direction(self) -> 'np.array[float, float, float]':
        delta = np.array(self.top_center) - np.array(self.bottom_center)
        return delta / np.linalg.norm(delta)

    def get_height(self) -> float:
        return np.linalg.norm(np.array(self.top_center) - np.array(self.bottom_center))

    def is_point_inside(self, point: 'tuple(float, float, float)') -> bool:
        direction = self.get_direction()
        delta = np.array(point) - np.array(self.bottom_center)
        if np.linalg.norm(np.cross(delta, direction)) > self.radius:
            return False
        if np.dot(delta, direction) < 0 or np.dot(delta, direction) > np.linalg.norm(
                np.array(self.bottom_center) - np.array(self.top_center)):
            return False
        return True

    def is_point_near_the_surface(self, point: 'tuple(float, float, float)',
                                  threshold: float) -> bool:
        direction = self.get_direction()
        delta = np.array(point) - np.array(self.bottom_center)
        if np.linalg.norm(np.cross(delta, direction)) > self.radius + threshold or \
                np.linalg.norm(np.cross(delta, direction)) < self.radius - threshold:
            return False
        if np.dot(delta, direction) < 0 - threshold or \
                np.dot(delta, direction) > np.linalg.norm(
            np.array(self.bottom_center) - np.array(self.top_center)) + threshold:
            return False
        return True

    def is_point_near_the_surface_batch(self, points: 'np.array[[float, float, float]]',
                                        threshold: float) -> 'np.array[bool]':
        direction = self.get_direction()
        delta = np.array(points) - np.array(self.bottom_center)
        delta_cross = np.linalg.norm(np.cross(delta, direction), axis=1)
        delta_dot = np.dot(delta, direction)
        condition1 = np.logical_and(delta_cross > self.radius - threshold,
                                    delta_cross < self.radius + threshold)
        condition2 = np.logical_and(delta_dot > -threshold,
                                    delta_dot < self.get_height() + threshold)
        return np.logical_and(condition1, condition2)

    def get_rotation_matrix(self) -> 'np.array[[float, float, float], [float, float, float], [float, float, float]]':
        z = self.get_direction()
        y = np.cross(z, np.array([0, 0, 1]))
        y = y / np.linalg.norm(y) if np.abs(np.linalg.norm(y)) > 1e-6 else np.array([0, 1, 0])
        x = np.cross(y, z)
        assert np.abs(np.linalg.norm(x)) > 1e-6
        x = x / np.linalg.norm(x)
        rotation_matrix = np.array([x, y, z])
        return rotation_matrix

    def to_o3d_mesh(self) -> 'o3d.geometry.TriangleMesh':
        cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(self.radius, self.get_height())
        rotation_matrix = self.get_rotation_matrix()
        cylinder_mesh.rotate(rotation_matrix.T, center=(0, 0, 0))
        cylinder_mesh.compute_vertex_normals()
        center = (np.array(self.top_center) + np.array(self.bottom_center)) / 2
        cylinder_mesh.translate(center)
        return cylinder_mesh

    def get_minimum_bounding_box(self) -> np.ndarray:  # (8, 3)
        # TODO: proceed quickly
        return np.asarray(self.to_o3d_mesh().get_minimal_bounding_box().get_box_points())
        pass

    def set_points(self, points: 'np.array[[float, float, float]]') -> None:
        self.points = points

    def reverse(self) -> 'Cylinder':
        new = Cylinder(self.bottom_center, self.top_center, self.radius)
        new.set_points(self.points)
        return new

    def __str__(self) -> str:
        return f"Cylinder({self.top_center}, {self.bottom_center}, {self.radius})"


    @classmethod
    def dis_from_two_points(cls, point_x: 'tuple[float, float, float]', point_y: 'tuple[float, float, float]') -> float:
        return np.linalg.norm(np.array(point_x) - np.array(point_y))

    @classmethod
    def save_cylinders(cls, cylinders: 'list[Cylinder]', save_prefix: str) -> None:
        cylinder_list, points_list = [], []
        for i, cylinder in enumerate(cylinders):
            _data = cylinder.split_get()
            _data = [*_data, i]
            cylinder_list.append(_data)
            _points = np.hstack((cylinder.points, np.ones((cylinder.points.shape[0], 1)) * i))
            points_list.append(_points)
        cylinder_list = np.vstack(cylinder_list)
        points_list = np.vstack(points_list)
        np.save(f'{save_prefix}_cylinders.npy', cylinder_list)
        np.save(f'{save_prefix}_points.npy', points_list)

    @classmethod
    def load_cylinders(cls, save_prefix: str) -> 'list[Cylinder]':
        cylinder_list = np.load(f'{save_prefix}_cylinders.npy')

        import os, sys
        if os.path.exists(f'{save_prefix}_points.npy'):
            points_list = np.load(f'{save_prefix}_points.npy')
        else:
            print(f'Warning: {save_prefix}_points.npy not found, points loading skipped.', file=sys.stderr)

        cylinders = []
        for i in range(cylinder_list.shape[0]):
            cylinder = Cylinder(cylinder_list[i, :3], cylinder_list[i, 3:6], cylinder_list[i, 6])
            if points_list is not None:
                cylinder.set_points(points_list[points_list[:, -1] == i, :-1])
            cylinders.append(cylinder)
        return cylinders

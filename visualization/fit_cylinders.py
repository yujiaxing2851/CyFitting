import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
from Cylinder import Cylinder

def get_center(p1, p2, p3):
    '''三点求圆，返回圆心和半径'''
    x, y, z = p1[0]+p1[1]*1j, p2[0]+p2[1]*1j, p3[0]+p3[1]*1j
    w = z-x
    w /= y-x
    c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
    return (-c.real,-c.imag,0),abs(c+x)

def fit_radius(points, thresh, interation, min_fit_rate, size):
    center = np.mean(points, axis=0)
    thresh = np.linalg.norm(size) / 40 if thresh is None else thresh # 5% of radius
    radius = 0
    rate = -1
    inliners = []
    for _ in range(interation):
        a, b, c = points[np.random.choice(points.shape[0], 3, replace=False)]
        _center, _radius = get_center(a, b, c)
        _inliners = np.where(np.abs(np.linalg.norm(points - _center, axis=1) - _radius) < thresh)[0]
        if len(_inliners) / points.shape[0] > rate and _radius < 0.2:
            rate = len(_inliners) / points.shape[0]
            # print(rate)
            center, radius, inliners = _center, _radius, _inliners
    # print(center, radius, len(inliners), len(inliners) / points.shape[0])
    if len(inliners) / points.shape[0] < min_fit_rate:
        return None, None, None
    return center, radius, inliners

def try_fitting_by_axises(points, axises, thresh=None, interation=1000, min_fit_rate=0.5):
    axises = [np.asarray(axis) for axis in axises]
    size = np.max(points, axis=0) - np.min(points, axis=0)
    axises = [axis / np.linalg.norm(axis) for axis in axises]
    points = np.asarray(points)
    data = np.dot(points, np.asarray(axises).T)
    height = np.max(data[:, 2]) - np.min(data[:, 2])
    data[:, 2] = 0
    center, radius, inliners = fit_radius(data, thresh, interation, min_fit_rate, size)
    return center, radius, inliners, height

def fit_cylinder_from_points(points):
    raw = np.copy(points)
    pos = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    mobb = pcd.get_minimal_oriented_bounding_box()
    mobb.color = (1, 0, 0)
    mobb_center = mobb.get_center()
    pcd.translate(-mobb_center)
    points = np.asarray(pcd.points)
    box_point = np.asarray(mobb.get_box_points())
    axises = [box_point[0] - box_point[1], box_point[0] - box_point[2], box_point[0] - box_point[3]]
    eye = np.eye(3)
    exchange = np.array([eye[1], eye[2], eye[0]])
    rate = -1
    center, radius, inliners, height = None, None, None, None
    _axises = np.copy(axises)
    for __ in range(3):
        _axises = exchange @ _axises
        # print(_axises, axises)
        _center, _radius, _inliners, _height = try_fitting_by_axises(points, _axises)
        if _inliners is not None and len(_inliners) / points.shape[0] > rate:
            # print(len(_inliners) / points.shape[0])
            rate = len(_inliners) / points.shape[0]
            center, radius, inliners, height, axises = _center, _radius, _inliners, _height, _axises
    # print(axises)
    if center is None:
        return None, None, None, None
    # cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius, height)
    # cylinder_mesh.translate(center)
    # axises = np.array([axis / np.linalg.norm(axis) for axis in axises])
    # cylinder_mesh.rotate(axises.T, center=(0, 0, 0))
    # cylinder_mesh.translate(mobb_center)
    # cylinder_mesh.compute_vertex_normals()
    axises = np.array([axis / np.linalg.norm(axis) for axis in axises])
    pcd.translate(mobb_center)
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, -0.5 * height], [0, 0, 0.5 * height]]))
    _pcd.translate(center)
    _pcd.rotate(axises.T, center=(0, 0, 0))
    _pcd.translate(mobb_center)
    myCylinder = Cylinder(tuple(np.asarray(_pcd.points[0])), tuple(np.asarray(_pcd.points[1])), radius)
    myCylinder.set_points(raw)
    _myCylinderMesh = myCylinder.to_o3d_mesh()
    _myCylinderMesh.paint_uniform_color([0, 1, 0])
    # cylinder_mesh.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([pcd, _myCylinderMesh])
    return _myCylinderMesh, mobb, inliners, myCylinder
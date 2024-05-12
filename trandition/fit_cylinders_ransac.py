import os
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
import argparse
from visualization.Cylinder import Cylinder
from util.cyfitting_dataset_for_test import CyFitting
from util.cyfitting_dataset_for_test import generator_iter
from torch.utils.data import DataLoader
from src.CyLoss import CyPCoverage
from tools.ransac_circle import CircleLeastSquareModel
from tools.ransac_circle import ransac
import math


def get_center(p1, p2, p3):
    '''三点求圆，返回圆心和半径'''
    x, y, z = p1[0] + p1[1] * 1j, p2[0] + p2[1] * 1j, p3[0] + p3[1] * 1j
    w = z - x
    w /= y - x
    c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
    return (-c.real, -c.imag, 0), abs(c + x)


def fit_radius(points, thresh, interation, min_fit_rate, size):
    center = np.mean(points, axis=0)
    thresh = np.linalg.norm(size) / 40 if thresh is None else thresh  # 5% of radius
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


def get_mobb_axises_and_center(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mobb = pcd.get_minimal_oriented_bounding_box()
    mobb_center = mobb.get_center()
    pcd.translate(-mobb_center)
    points = np.asarray(pcd.points)
    box_point = np.asarray(mobb.get_box_points())
    axises = [box_point[0] - box_point[1], box_point[0] - box_point[2], box_point[0] - box_point[3]]
    return axises, mobb_center


def get_rotated_axises(axises):
    eye = np.eye(3)
    axises = np.asarray(axises)
    exchange = np.array([eye[1], eye[2], eye[0]])
    return [axises, exchange @ axises, exchange @ exchange @ axises]


def visual_point_cloud(point_cloud, other_entities=[]):
    if isinstance(point_cloud, np.ndarray):
        assert point_cloud.shape[1] >= 3
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        if point_cloud.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(
                point_cloud[:, 3:6] / 255 if np.max(point_cloud[:, 3:6]) > 1 else point_cloud[:, 3:6])
        o3d.visualization.draw_geometries([pcd, *other_entities])
    elif isinstance(point_cloud, o3d.geometry.PointCloud):
        o3d.visualization.draw_geometries([point_cloud, *other_entities])
    elif isinstance(point_cloud, o3d.utility.Vector3dVector):
        pcd = o3d.geometry.PointCloud()
        pcd.points = point_cloud
        o3d.visualization.draw_geometries([pcd, *other_entities])
    else:
        raise ValueError('point_cloud should be numpy array or o3d.geometry.PointCloud')


def fit_cylinder_from_points(points):
    raw = np.copy(points)
    pos = points[:, :3]
    mobb_axises, mobb_center = get_mobb_axises_and_center(pos)
    pca_axises = get_pca_axises(pos)
    natrual_axises = np.eye(3)
    try_axises = [mobb_axises, pca_axises, natrual_axises]
    try_axises = np.array([get_rotated_axises(axises) for axises in try_axises]).reshape(-1, 3)
    center, radius, inliners, height, axises, rate = None, None, None, None, None, -1
    for _axises in try_axises:
        _center, _radius, _inliners, _height = try_fitting_by_axises(points, _axises)
        if _inliners is not None and len(_inliners) / points.shape[0] > rate:
            rate = len(_inliners) / points.shape[0]
            center, radius, inliners, height, axises = _center, _radius, _inliners, _height, _axises
    if center is None:
        return None, None, None, None
    axises = np.array([axis / np.linalg.norm(axis) for axis in axises])
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, -0.5 * height], [0, 0, 0.5 * height]]))
    _pcd.translate(center)
    _pcd.rotate(axises.T, center=(0, 0, 0))
    _pcd.translate(mobb_center)
    myCylinder = Cylinder(tuple(np.asarray(_pcd.points[0])), tuple(np.asarray(_pcd.points[1])), radius)
    myCylinder.set_points(raw)
    return inliners, myCylinder, rate

def fit_cylinders(args):
    split_dict = {"test": args.num_test}
    dataset=CyFitting(args.dataset_path,args.batch_size,splits=split_dict)
    get_test_data = dataset.load_test_data()
    loader = generator_iter(get_test_data, int(1e10))
    get_test_data = iter(
        DataLoader(
            loader,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
            pin_memory=False,
        )
    )
    test_p_coverage = 0
    for train_id in range(args.num_test // args.batch_size - 1):
        points_, paras_ = next(get_test_data)[0]
        # 在数据加载中设置与x轴同向，所以圆面在yz平面
        model=CircleLeastSquareModel()
        Output=np.empty((0,7))
        for i in range(args.batch_size):
            data=points_[i]
            circle_points=data[:,1:3]
            x=data[:,0]
            ransac_fit, ransac_data = ransac(circle_points, model, 20, 1000, 20, 50, debug=False, return_all=True)
            y0=ransac_fit.a*-0.5
            z0=ransac_fit.b*-0.5
            r=0.5*math.sqrt(ransac_fit.a**2+ransac_fit.b**2-4*ransac_fit.c)
            height=np.max(x)-np.min(x)
            cylinder_top=np.array([height/2,y0,z0])
            cylinder_bottom=np.array([-height/2,y0,z0])
            output=np.append(np.concatenate((cylinder_top,cylinder_bottom)),r)
            Output=np.vstack((Output,output))
        points = torch.from_numpy(points_).cuda()
        points = points.permute(0, 2, 1)
        Output=torch.from_numpy(Output).cuda()
        p_coverage = CyPCoverage(Output, points, args.threshold)
        test_p_coverage += p_coverage
    test_p_coverage /= (args.num_test // args.batch_size - 1)
    with open("experiments/logs/p-coverage_ransac.txt", "a") as f:
        f.write("p-coverage under threshold {}:{}".format(args.threshold, test_p_coverage))
        f.write("\n")

EPS = np.finfo(np.float32).eps


def pca_numpy(X):
    S, U = np.linalg.eig(X.T @ X)
    return S, U


def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0],
                  [sin, cos, 0],
                  [0, 0, 1]])

    # B = R @ A
    try:
        R = F @ G @ np.linalg.inv(F) # 不理解
    except:
        R = np.eye(3, dtype=np.float32)
    return R



def fit_cylinder_ransac(points):
        S, U = pca_numpy(points)
        max_ev = U[:, np.argmax(S)]
        R = rotation_matrix_a_to_b(max_ev, np.array([1, 0, 0]))
        # rotate input points such that the minor principal
        # axis aligns with x axis.
        points = R @ points.T
        points = points.T
        model=CircleLeastSquareModel()
        Output=np.empty((0,7))
        circle_points=points[:,1:3]
        x=points[:,0]
        ransac_fit, ransac_data = ransac(circle_points, model, 20, 1000, 20, 50, debug=False, return_all=True)
        y0=ransac_fit.a*-0.5
        z0=ransac_fit.b*-0.5
        r=0.5*math.sqrt(ransac_fit.a**2+ransac_fit.b**2-4*ransac_fit.c)
        cylinder_top=np.array([np.max(x),y0,z0])
        cylinder_bottom=np.array([np.min(x),y0,z0])
        return Cylinder(tuple(cylinder_top),tuple(cylinder_bottom), r)

# test 2024.04.29
# data = np.load("D:\desktop\\reconstruction project\\algorithm\cyfitting0412\data\process_inst\\1.npy")[:, :3]
# pcd=o3d.geometry.PointCloud()
# pcd.points=o3d.utility.Vector3dVector(data)
# t = fit_cylinder_ransac(data)
# o3d.visualization.draw_geometries([pcd,t.to_o3d_mesh()])

def get_parser():
    parser = argparse.ArgumentParser(description='infer of cylinders')
    parser.add_argument('--config', type=str, default='config/model.yaml')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--epoch',type=int,default=200)
    parser.add_argument('--num_test',type=int,default=288)
    parser.add_argument('--dataset_path',type=str,default="data/process_inst_test")
    parser.add_argument('--lr',type=int,default=0.001)
    parser.add_argument('--mode',type=int,default=0)
    parser.add_argument('--threshold',type=float,default=0.15)
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args=get_parser()
    fit_cylinders(args)
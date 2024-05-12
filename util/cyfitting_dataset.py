import numpy as np
import os
import open3d as o3d
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from visualization.Cylinder import Cylinder

def vis(points: 'numpy',cylinder):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd, coordinate_frame,cylinder])

EPS = np.finfo(np.float32).eps
class generator_iter(Dataset):
    """This is a helper function to be used in the parallel data loading using Pytorch
    DataLoader class"""

    def __init__(self, generator, train_size):
        self.generator = generator
        self.train_size = train_size

    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        return next(self.generator)

class CyFitting:
    def __init__(self,path,batch_size,splits={}):
        self.path=path
        self.transform=None
        self.batch_size=batch_size
        self.train_size=splits["train"]
        self.test_size=splits["test"]
        self.train_points=[]
        self.test_points=[]
        for i in range(1,self.train_size+1):
            self.train_points.append(np.load(os.path.join(self.path,"{}.npy".format(i))))
        for i in range(self.train_size+1,self.train_size+self.test_size+1):
            self.test_points.append(np.load(os.path.join(self.path,"{}.npy".format(i))))

    #   =================== 数据格式说明 =======================
    # process_inst中的数据格式：(x,y,z,inst_label,top_center,bottom_center,radius)
    def load_train_data(self,if_regular_points=False, align_canonical=True, anisotropic=False, if_augment=False):
        while True:
            for batch_id in range(self.train_size//self.batch_size-1):
                Points=[]
                Paras=[]
                RS=[]
                for i in range(self.batch_size):
                    data=self.train_points[batch_id*self.batch_size+i]
                    points=data[:,:3]
                    labels=data[:,3]
                    cylinder_points=np.vstack((data[0,4:7],data[0,7:10])) # 圆柱顶面和底面中心
                    cylinder_radius=data[0,10]
                    mean=np.mean(points,0)
                    points=points-mean
                    cylinder_points=cylinder_points-mean
                    # vis(points)
                    if align_canonical:
                        S,U=self.pca_numpy(points)
                        max_ev = U[:, np.argmax(S)]
                        R = self.rotation_matrix_a_to_b(max_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points = R @ points.T
                        points = points.T
                        cylinder_points=R@cylinder_points.T
                        cylinder_points=cylinder_points.T
                        RS.append(R)
                    # 要变换尺度
                    if anisotropic:
                        std = np.abs(np.max(points, 0) - np.min(points, 0))
                        std = std.reshape((1, 3))
                        points = points / (std + EPS)
                        cylinder_points=cylinder_points/(std+EPS)
                        # 半径无法按三个方向缩放
                    else:
                        std = np.max(np.max(points, 0) - np.min(points, 0))
                        points = points / std
                        cylinder_points = cylinder_points/std
                        cylinder_radius=cylinder_radius/std
                    test_cylinder=Cylinder(cylinder_points[0],cylinder_points[1],cylinder_radius)
                    # vis(points,test_cylinder.to_o3d_mesh())
                    Points.append(points)
                    Paras.append(test_cylinder.numpy_get())
                    # Labels.append(labels)
                Points = np.stack(Points, 0)
                Paras=np.stack(Paras,0)
                # Labels=np.stack(Labels,0)
                yield [Points,np.float32(Paras)]

    def load_val_data(self,if_regular_points=False, align_canonical=True, anisotropic=False, if_augment=False):
        while True:
            for batch_id in range(self.test_size//self.batch_size-1):
                Points=[]
                Paras=[]
                RS=[]
                for i in range(self.batch_size):
                    data=self.test_points[batch_id*self.batch_size+i]
                    points=data[:,:3]
                    labels=data[:,3]
                    cylinder_points=np.vstack((data[0,4:7],data[0,7:10])) # 圆柱顶面和底面中心
                    cylinder_radius=data[0,10]
                    mean=np.mean(points,0)
                    points=points-mean
                    cylinder_points=cylinder_points-mean
                    # vis(points)
                    if align_canonical:
                        S,U=self.pca_numpy(points)
                        max_ev = U[:, np.argmax(S)]
                        R = self.rotation_matrix_a_to_b(max_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points = R @ points.T
                        points = points.T
                        cylinder_points=R@cylinder_points.T
                        cylinder_points=cylinder_points.T
                        RS.append(R)
                    # 要变换尺度
                    if anisotropic:
                        std = np.abs(np.max(points, 0) - np.min(points, 0))
                        std = std.reshape((1, 3))
                        points = points / (std + EPS)
                        cylinder_points=cylinder_points/(std+EPS)
                        # 半径无法按三个方向缩放
                    else:
                        std = np.max(np.max(points, 0) - np.min(points, 0))
                        points = points / std
                        cylinder_points = cylinder_points/std
                        cylinder_radius=cylinder_radius/std
                    test_cylinder=Cylinder(cylinder_points[0],cylinder_points[1],cylinder_radius)
                    # vis(points,test_cylinder.to_o3d_mesh())
                    Points.append(points)
                    Paras.append(test_cylinder.numpy_get())
                    # Labels.append(labels)
                Points = np.stack(Points, 0)
                Paras=np.stack(Paras,0)
                # Labels=np.stack(Labels,0)
                yield [Points,np.float32(Paras)]


    def rotation_matrix_a_to_b(self, A, B):
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
            R = F @ G @ np.linalg.inv(F)
        except:
            R = np.eye(3, dtype=np.float32)
        return R

    def pca_torch(self, X):
        covariance = torch.transpose(X, 1, 0) @ X
        S, U = torch.eig(covariance, eigenvectors=True)
        return S, U

    def pca_numpy(self, X):
        S, U = np.linalg.eig(X.T @ X)
        return S, U
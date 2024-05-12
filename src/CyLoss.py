import numpy as np
import torch
from visualization.Cylinder import Cylinder
# from tools.chamfer_distance import ChamferDistance
from pytorch3d.ops import knn_points

import open3d as o3d

def vis(points: 'numpy',cylinder):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd, coordinate_frame,cylinder])

def find_nearest_neighbors(pc1, pc2):
    # Convert point clouds to PyTorch tensors

    N = pc1.shape[0]

    if isinstance(pc1, np.ndarray):
        pc1_tensor = torch.from_numpy(pc1).cuda()
    else:
        pc1_tensor = pc1.cuda()
    if isinstance(pc2, np.ndarray):
        pc2_tensor = torch.from_numpy(pc2).cuda()
    else:
        pc2_tensor = pc2.cuda()

    # Compute pairwise Euclidean distances between points
    # pc1_norms = (pc1_tensor ** 2).sum(dim=1, keepdim=True)
    # pc2_norms = (pc2_tensor ** 2).sum(dim=1, keepdim=True)
    # distances = pc1_norms - 2 * torch.mm(pc1_tensor, pc2_tensor.t()) + pc2_norms.t()
    # distances = distances.sqrt()

    # # Find the nearest neighbor indices
    # _, indices = torch.min(distances, dim=1)

    # return indices.cpu().numpy()

    idx = knn_points(pc1_tensor.unsqueeze(0), pc2_tensor.unsqueeze(0))[1].reshape((N,))

    return idx



def CyLoss(output,target,weights=[0.3,0.3,0.4]):
    output_normal=(output[:,:3]-output[:,3:6])/torch.norm(output[:,:3]-output[:,3:6],2,1).view(-1,1)
    target_normal=target[:,:3]-target[:,3:6]/torch.norm(target[:,:3]-target[:,3:6],2,1).view(-1,1)

    # 法线损失，夹角正弦值
    l1=torch.norm(torch.cross(output_normal,target_normal),2,1)
    l1=l1.mean()
    # 位置损失
    output_center=(output[:,:3]+output[:,3:6])/2
    target_center=(target[:,:3]+target[:,3:6])/2
    l2=torch.norm(output_center-target_center,2,1)
    l2=l2.mean()
    # 半径损失
    scale=2
    l3=scale*torch.abs(output[:,6]-target[:,6])
    l3=l3.mean()
    loss=l1*weights[0]+l2*weights[1]+l3*weights[2]
    return loss

def CyPCoverage(output,points,threshold=0.01):
    p_coverage=0
    points=points.permute(0,2,1)
    output=output.detach().cpu().numpy()
    cylinder_top=output[:,0:3]
    cylinder_bottom=output[:,3:6]
    cylinder_radius=output[:,6]
    my_cylinders = [Cylinder(cylinder_top[i],cylinder_bottom[i], cylinder_radius[i]) for i in range(output.shape[0])]
    meshes=[my_cylinder.to_o3d_mesh() for my_cylinder in my_cylinders]
    cylinder_points=np.asarray([mesh.sample_points_uniformly(number_of_points=10000).points for mesh in meshes])
    cylinder_points=torch.from_numpy(cylinder_points).cuda()
    indexes=[find_nearest_neighbors(points[i],cylinder_points[i]) for i in range(points.shape[0])]
    points=points.cpu().numpy()
    cylinder_points=cylinder_points.cpu().numpy()
    for i,index in enumerate(indexes):
        index=index.cpu().numpy()
        dis=(np.linalg.norm(points[i]-cylinder_points[i][index],axis=1))
        count=np.sum(dis<threshold)
        p_coverage+=count/points[i].shape[0]
    p_coverage/=points.shape[0]
    return p_coverage




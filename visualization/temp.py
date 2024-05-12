import os.path

from fit_cylinders import fit_cylinder_from_points
import numpy as np
import open3d as o3d
num=20
data=np.load("D:\desktop\\reconstruction project\\algorithm\cyfitting0412\data\process_inst\\{}.npy".format(num))
_myCylinderMesh, mobb, inliners, myCylinder=fit_cylinder_from_points(data[:,:3])

dir="D:\desktop\\reconstruction project\\algorithm\cyfitting0412\experiments\origin"
np.savetxt(os.path.join(dir,"{}.txt".format(num)), data[:,:3], fmt='%f')
o3d.io.write_triangle_mesh(os.path.join(dir,"{}.obj".format(num)),_myCylinderMesh)



import numpy as np
import os
import open3d as o3d

# partition of the data to get single cylinder file
path="data/segments_inst_test"

save_dir="data/process_inst_test"
file_nums=20

count=1
for file_num in range(file_nums):
    data_bin=np.load(os.path.join(path,"{}.bin".format(file_num)))[1,:]
    data_npy=np.load(os.path.join(path,"{}.npy".format(file_num)))
    for index in set(data_bin):
        label=np.where(data_bin==index)
        cylinders=data_npy[label][:,:3]
        cylinders=np.hstack((cylinders,np.full((label[0].shape[0],1),count)))
        cylinders=np.hstack((cylinders,data_npy[label][:,6:]))
        np.save(os.path.join(save_dir,"{}.npy".format(count)),cylinders)
        count+=1

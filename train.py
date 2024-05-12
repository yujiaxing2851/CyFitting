import argparse

import numpy as np
import torch.utils.data
import yaml
import logging
from tensorboardX import SummaryWriter
from util.cyfitting_dataset import CyFitting
from util.cyfitting_dataset import generator_iter
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR

from visualization.Cylinder import Cylinder
from src.CyLoss import CyLoss
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from tqdm import tqdm
import open3d as o3d
from src.model import DGCNNCyParameters


def vis(cylinder: 'pcd'):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([cylinder, coordinate_frame])

def main(args):
    # path='config/model.yaml'
    # with open(path,encoding='utf-8') as f:
    #     data=yaml.load(f,Loader=yaml.FullLoader)
    #     print(data)

    # model=Model()
    # train_transform = t.Compose([t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(), t.ChromaticJitter(), t.HueSaturationTranslation()])
    split_dict = {"train": args.num_train,"test": args.num_test}
    dataset=CyFitting(args.dataset_path,args.batch_size,splits=split_dict)
    get_train_data = dataset.load_train_data()
    loader = generator_iter(get_train_data, int(1e10))
    get_train_data = iter(
        DataLoader(
            loader,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
            pin_memory=False,
        )
    )

    get_val_data = dataset.load_val_data()
    loader = generator_iter(get_val_data, int(1e10))
    get_val_data = iter(
        DataLoader(
            loader,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
            pin_memory=False,
        )
    )

    cy_decoder = DGCNNCyParameters( num_points=10, mode=args.mode).cuda()
    pre_train="experiments/models/0.0500.047.pth"
    checkpoint = torch.load(pre_train)
    cy_decoder.load_state_dict(checkpoint)

    optimizer = optim.Adam(cy_decoder.parameters(), lr=args.lr)
    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=10, verbose=True, min_lr=3e-5
    # )
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=args.lr / 20, verbose=True)
    writer = SummaryWriter(os.path.join("experiments/writer"))
    best_loss=1000.
    with open("experiments/logs/0427_{}.txt".format(args.lr),"w") as f:
        f.write("train begin\n")
        for e in range(args.epoch):
            print("epoch",e)
            cy_decoder.train()
            train_loss=0
            for train_id in range(args.num_train // args.batch_size-1):
                torch.cuda.empty_cache()
                points_, paras_ = next(get_train_data)[0]
                points=torch.from_numpy(points_).cuda()
                paras=torch.from_numpy(paras_).cuda()
                points=points.permute(0,2,1)
                output = cy_decoder(points)
                # 计算损失
                loss=CyLoss(output,paras)
                loss.backward()
                optimizer.step()
                train_loss+=loss.detach().cpu().numpy()
            train_loss/=(args.num_train // args.batch_size)
            writer.add_scalars("loss3", {"train": train_loss}, e)
            test_loss=0
            with torch.no_grad():
                cy_decoder.eval()
                for train_id in range(args.num_test // args.batch_size-1):
                    optimizer.zero_grad()
                    points_, paras_ = next(get_val_data)[0]
                    points=torch.from_numpy(points_).cuda()
                    paras=torch.from_numpy(paras_).cuda()
                    points=points.permute(0,2,1)
                    output = cy_decoder(points)
                    # 计算损失
                    loss=CyLoss(output,paras)
                    test_loss+=loss.detach().cpu().numpy()
                test_loss/=(args.num_test // args.batch_size-1)
                writer.add_scalars("loss3", {"dev": test_loss}, e)
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(
                        cy_decoder.state_dict(),
                        "experiments/models/{:.3f}{:.3f}.pth".format(train_loss,test_loss)
                    )
                print("e {}: train {} test{}\n".format(e, train_loss, test_loss))
                f.write("e {}: train {} test{}\n".format(e, train_loss, test_loss))
            scheduler.step()
        writer.close()


def get_parser():
    parser = argparse.ArgumentParser(description='primitive fitting of cylinders')
    parser.add_argument('--config', type=str, default='config/model.yaml')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--epoch',type=int,default=200)
    parser.add_argument('--num_train',type=int,default=1000)
    parser.add_argument('--num_test',type=int,default=314)
    parser.add_argument('--dataset_path',type=str,default="data/process_inst")
    parser.add_argument('--lr',type=int,default=0.00001)
    parser.add_argument('--mode',type=int,default=0)
    args = parser.parse_args()
    return args


# firstly set the dirname cyfitting0412 as the working path
if __name__=='__main__':
    args=get_parser()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        # filename='logs/my.log',
                        filemode='w')

    main(args)

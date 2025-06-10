import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import sys
import argparse
from models.PointPAD import PointPAD, PointPAD_o
from cfgs.upsampling import parse_pu1k_args, parse_pugan_o_args, parse_pugan_args
from cfgs.utils import *
from dataset.dataset import PUDataset
import torch.optim as optim
from glob import glob
import open3d as o3d
from einops import repeat
from models.utils import *
import time
from datetime import datetime

def eval(out_path, gt_path, save_dir):
    test_input_path = glob(os.path.join(out_path, '*.xyz'))
    total_cd = 0
    total_dcd = 0
    total_hd = 0
    counter = 0
    txt_result = []
    for i, path in enumerate(test_input_path):
        pcd = o3d.io.read_point_cloud(path)
        pcd_name = path.split('/')[-1]
        gt = torch.Tensor(np.asarray(o3d.io.read_point_cloud(os.path.join(gt_path, pcd_name)).points)).unsqueeze(0).cuda()
        input_pcd = np.array(pcd.points)
        input_pcd = torch.from_numpy(input_pcd).float().cuda()
        input_pcd = rearrange(input_pcd, 'n c -> c n').contiguous()
        input_pcd = input_pcd.unsqueeze(0)
        #cd = chamfer_sqrt(input_pcd.permute(0,2,1).contiguous(), gt).cpu().item()  
        dcd = get_dcd_loss(input_pcd, gt, train=False)
        #txt_result.append(f'{pcd_name}_cd: {cd * 1e3}') 
        txt_result.append(f'{pcd_name}_dcd: {dcd}') 
        #total_cd += cd
        total_dcd += dcd
        counter += 1.0

        #txt_result.append(f'overall_cd: {total_cd/counter*1e3}')
    txt_result.append(f'overall_dcd: {total_dcd/counter}')
        #txt_result.append(f'overall_hd: {total_hd/counter}')

    with open(os.path.join(save_dir,'cd.txt'), "w") as f:
        for ll in txt_result:
            f.writelines(ll+'\n')
    return True 

out_path = "path to output data"
gt_path = "path to ground truth"
save_dir = "path to save directory"
eval(out_path,gt_path,save_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from dataset import Vimeo90K
from net.model import FlowEstimator
from options import Options


def main():
    opt = Options()
    opt.save()

    test_set = Vimeo90K(opt.data_root, 'test', crop_size=opt.train['crop_size'])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = FlowEstimator(opt.train['channels']).cuda()
    ckpt = torch.load(opt.test['ckpt_dir'])
    model.load_state_dict(ckpt)
    # model = nn.DataParallel(model).cuda()

    model.eval()
    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        for (img0, img1, gt) in test_loader:
            img0_0 = img0.float().cuda()
            img1_0 = img1.float().cuda()
            gt_0 = gt.float().cuda()

            img0_1 = F.interpolate(img0_0, scale_factor=0.5, mode='bilinear', align_corners=False)
            img0_2 = F.interpolate(img0_0, scale_factor=0.25, mode='bilinear', align_corners=False)
            img1_1 = F.interpolate(img1_0, scale_factor=0.5, mode='bilinear', align_corners=False)
            img1_2 = F.interpolate(img1_0, scale_factor=0.25, mode='bilinear', align_corners=False)

            img0 = (img0_0, img0_1, img0_2)
            img1 = (img1_0, img1_1, img1_2)

            result = model(img0, img1, return_all=False)
            frame_0, _, _ = result['frame']

            pred = (frame_0[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8) 
            gt = (gt[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

            psnr_list.append(psnr(pred, gt))
            ssim_list.append(ssim(pred, gt))

    psnr_avg = np.array(psnr_list).mean()
    ssim_avg = np.array(ssim_list).mean()

    print(psnr_avg, ssim_avg)
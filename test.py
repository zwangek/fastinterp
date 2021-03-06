import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Vimeo90K
from net.model import FlowEstimator
from options import Options
import imageio

def main():
    opt = Options()
    opt.save()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpus)

    test_set = Vimeo90K(opt.data_root, 'test', crop_size=opt.train['crop_size'])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = FlowEstimator(opt.train['channels'], opt.train['lateral_num'])
    ckpt = torch.load(opt.test['ckpt_dir'], map_location='cpu')
    model.load_state_dict(ckpt)
    model.cuda()
    print(f'Checkpoint loaded from {opt.test["ckpt_dir"]}')
    # model = nn.DataParallel(model).cuda()
    
    model.eval()
    psnr_list = []
    ssim_list = []
    time_list = []
    with torch.no_grad():
        for (img0, img1, gt) in tqdm(test_loader):
            img0_0 = img0.float().cuda() # 448 * 256
            img1_0 = img1.float().cuda()
            gt_0 = gt.float().cuda()

            start = time.time()
            result = model(img0_0, img1_0, return_all=False)
            end = time.time()
            frame_0, _, _ = result['frame']

            pred = (frame_0[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8) 
            gt = (gt_0[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

            psnr_list.append(psnr(pred, gt))
            ssim_list.append(ssim(pred, gt, multichannel=True))
            time_list.append(end-start)

    psnr_avg = np.array(psnr_list).mean()
    ssim_avg = np.array(ssim_list).mean()
    time_avg = np.array(time_list).mean()

    print(psnr_avg, ssim_avg, time_avg)


if __name__ == '__main__':
    main()


def perceptual_loss(pred, gt):
    pass


# 2 
# 4 33.10648394832165 0.9462927205260064 0.011724015192529122, with 1 another process running
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Vimeo90K
from net.model import FlowEstimator
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader
from options import Options
from tqdm import tqdm
import numpy as np
import math

def main():
    opt = Options()
    opt.save()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpus)

    model = FlowEstimator(opt.train['features'])
    model = nn.DataParallel(model).cuda()
    
    optimizer = AdamW(model.parameters(), lr=opt.train['base_lr'], weight_decay=opt.train['weight_decay'])

    train_set = Vimeo90K(opt.data_root, 'train', crop_size=opt.train['crop_size'])
    val_set = Vimeo90K(opt.data_root, 'val', crop_size=opt.train['crop_size'])
    train_loader = DataLoader(train_set, batch_size=opt.train['batch_size'], shuffle=True, num_workers=opt.train['num_workers'])
    val_loader = DataLoader(val_set, batch_size=opt.train['batch_size'], shuffle=False, num_workers=opt.train['num_workers'])

    log_dir = opt.train['log_dir']
    ckpt_dir = opt.train['ckpt_dir']

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0
    total_steps = len(train_loader) * opt.train['epochs']

    writer = SummaryWriter(log_dir)

    for e in range(1, opt.train['epochs']+1):
        model.train()
        for (img0, img1, gt) in tqdm(train_loader):
            global_step += 1 
            img0_0 = img0.float().cuda()
            img1_0 = img1.float().cuda()
            gt_0 = gt.float().cuda()

            img0_1 = F.interpolate(img0_0, scale_factor=0.5, mode='bilinear', align_corners=False)
            img0_2 = F.interpolate(img0_0, scale_factor=0.25, mode='bilinear', align_corners=False)
            img1_1 = F.interpolate(img1_0, scale_factor=0.5, mode='bilinear', align_corners=False)
            img1_2 = F.interpolate(img1_0, scale_factor=0.25, mode='bilinear', align_corners=False)
            gt_1 = F.interpolate(gt_0, scale_factor=0.5, mode='bilinear', align_corners=False)
            gt_2 = F.interpolate(gt_0, scale_factor=0.25, mode='bilinear', align_corners=False)

            img0 = (img0_0, img0_1, img0_2)
            img1 = (img1_0, img1_1, img1_2)

            frame_0, frame0_0, frame1_0, frame_1, frame0_1, frame1_1, frame_2, frame0_2, frame1_2 = model(img0, img1)

            loss_cons = (frame0_0-frame1_0).abs().mean() + (frame0_1-frame1_1).abs().mean() + (frame0_2-frame1_2).abs().mean()
            loss_rec = (frame_0-gt_0).abs().mean() + (frame_1-gt_1).abs().mean() + (frame_2-gt_2).abs().mean()
            loss = loss_rec*opt.train['loss_weight']['rec'] + loss_cons*opt.train['loss_weight']['cons']

            lr = get_learning_rate(global_step, total_steps, opt.train['base_lr'], opt.train['min_lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if global_step % opt.train['print_every'] == 0:
                tqdm.write(f'[Epoch {e} Iter {global_step}/{total_steps}] Rec: {loss_rec.item():.4f}\t Cons: {loss_cons.item():.4f}\t LR: {lr:.2e}')

            writer.add_scalar('lr', lr, global_step)
            writer.add_scalar('train/loss/overall', loss.item(), global_step)
            writer.add_scalar('train/loss/reconstruction', loss_rec.item(), global_step)
            writer.add_scalar('train/loss/consistency', loss_cons.item(), global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # val
        model.eval()
        with torch.no_grad():
            loss_cons = 0
            loss_rec = 0
            psnr = 0
            best_psnr = 200
            for (img0, img1, gt) in tqdm(val_loader):
                img0_0 = img0.float().cuda()
                img1_0 = img1.float().cuda()
                gt_0 = gt.float().cuda()
                img0_1 = F.interpolate(img0_0, scale_factor=0.5, mode='bilinear', align_corners=False)
                img0_2 = F.interpolate(img0_0, scale_factor=0.25, mode='bilinear', align_corners=False)
                img1_1 = F.interpolate(img1_0, scale_factor=0.5, mode='bilinear', align_corners=False)
                img1_2 = F.interpolate(img1_0, scale_factor=0.25, mode='bilinear', align_corners=False)
                gt_1 = F.interpolate(gt_0, scale_factor=0.5, mode='bilinear', align_corners=False)
                gt_2 = F.interpolate(gt_0, scale_factor=0.25, mode='bilinear', align_corners=False)

                img0 = (img0_0, img0_1, img0_2)
                img1 = (img1_0, img1_1, img1_2)

                frame_0, frame0_0, frame1_0, flow0_0, flow1_0 = model(img0, img1, return_flow=True)
                loss_cons += (frame0_0-frame1_0).abs().mean()
                loss_rec += (frame_0-gt_0).abs().mean()

                mse = ((gt_0 - frame_0)**2).mean(dim=(1,2,3))
                psnr += -10 * torch.log10(mse).mean()

            loss_rec /= len(val_loader)
            loss_cons /= len(val_loader)
            loss = loss_rec*opt.train['loss_weight']['rec'] + loss_cons*opt.train['loss_weight']['cons']
            psnr /= len(val_loader)
            writer.add_scalar('val/loss/overall', loss.item(), e)
            writer.add_scalar('val/loss/reconstruction', loss_rec.item(), e)
            writer.add_scalar('val/loss/consistency', loss_cons.item(), e)
            writer.add_scalar('val/psnr', psnr.item(), e)
            print(f'[Validation] Rec: {loss_rec.item():.4f}\t Cons: {loss_cons.item():.4f}\t PSNR: {psnr.item()}')
            if psnr < best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), f"{opt.train['ckpt_dir']}/ckpt_best.pth")
            # visualization
            if global_step % opt.train['viz_every'] == 0:
                for i in range(2):
                    img0 = (img0_0[i].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                    img1 = (img1_0[i].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                    gt = (gt_0[i].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                    pred = (frame_0[i].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                    flow0 = flow0_0[i].permute(1,2,0).cpu().numpy()
                    flow1 = flow1_0[i].permute(1,2,0).cpu().numpy()
                    writer.add_image(f'vis_{i}/img0', img0, e, dataformats='HWC')
                    writer.add_image(f'vis_{i}/img1', img1, e, dataformats='HWC')
                    writer.add_image(f'vis_{i}/gt', gt, e, dataformats='HWC')
                    writer.add_image(f'vis_{i}/pred', pred, e, dataformats='HWC')
                    writer.add_image(f'vis_{i}/flow', np.concatenate((flow2rgb(flow0), flow2rgb(flow1)), 1), e, dataformats='HWC')
        if e % opt.train['save_every'] == 0:
            torch.save(model.state_dict(), f"{opt.train['ckpt_dir']}/ckpt_{e}.pth")


def get_learning_rate(step, total_steps, base_lr=3e-4, min_lr=3e-5, warmup=2000.):
    # warm up
    if step < warmup:
        mul = step / warmup
        return base_lr * mul
    # cosine
    else:
        mul = np.cos((step - warmup) / (total_steps - warmup) * math.pi) * 0.5 + 0.5
        return (base_lr - min_lr) * mul + min_lr

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

if __name__ == '__main__':
    main()
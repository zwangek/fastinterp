
from net.model import FlowEstimator
import torch
import torch.nn.functional as F
import imageio
import argparse
import numpy as np

# parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# parser.add_argument('--img', dest='img', nargs=2, required=True)
# parser.add_argument('--exp', default=4, type=int)
# parser.add_argument('--model', dest='model_dir', type=str, default='train_log', help='directory with trained model files')

# args = parser.parse_args()

class Args():
    def __init__(self):
        self.img1_dir = '/home/wzy/workspace/fastinterp/demo/im1.png'
        self.img2_dir = '/home/wzy/workspace/fastinterp/demo/im3.png'
        self.model_dir = './output_100/checkpoints/ckpt_best.pth'
        self.iteration = 3
        self.save_dir = './temp/'

args = Args()

def infer(model, img1, img2, iteration=1):
    middle = model(img1, img2, return_all=False)['frame'][0]
    if iteration == 1:
        return [middle]
    else:
        left = infer(model, img1, middle, iteration-1)
        right = infer(model, middle, img2, iteration-1)

        res = []
        for i in left:
            res.append(i)
        res.append(middle)
        for i in right:
            res.append(i)

        return res

if __name__ == '__main__':
    model = FlowEstimator(32, 4)
    model.load_state_dict(torch.load(args.model_dir))
    model.cuda()

    img1 = imageio.imread(args.img1_dir)
    img2 = imageio.imread(args.img2_dir)

    h,w,_ = img1.shape
    sh = h // 2 - 112 
    sw = w // 2 - 112 - 50

    img1 = img1[sh:sh+224, sw:sw+224, :]
    img2 = img2[sh:sh+224, sw:sw+224, :]

    img1 = torch.tensor(img1/255).float().cuda().permute(2,0,1).unsqueeze(0)
    img2 = torch.tensor(img2/255).float().cuda().permute(2,0,1).unsqueeze(0)

    with torch.no_grad():
        result = infer(model, img1, img2, args.iteration)

    final_result = [img1]
    for frame in result:
        final_result.append(frame)
    final_result.append(img2)

    for i,frame in enumerate(final_result):
        imageio.imsave(f'{args.save_dir}/frame_{i}.png', (frame.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))

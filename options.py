
class Options():
    def __init__(self):
        self.data_root = '/home/wzy/workspace/data/vimeo_triplet'
        self.output_root = './output_100/'
        self.gpus = [0]

        self.train = {
            'crop_size': (224, 224),
            'channels': 32,
            'lateral_num': 4,
            'batch_size': 32,
            'num_workers': 4,
            'epochs': 100,
            'loss_weight': {'rec': 1.0, 'cons': 0.1, 'perc': 0.1},
            'base_lr': 1e-3,
            'min_lr': 1e-5,
            'weight_decay': 1e-4,
            'print_every': 25, # step
            'viz_every': 1, # epoch
            'save_every': 5, # epoch
            'log_dir': f'{self.output_root}/logs/',
            'ckpt_dir': f'{self.output_root}/checkpoints/'
        }

        self.test = {
            'ckpt_dir': f'{self.output_root}/checkpoints/ckpt_best.pth'
        }

    def save(self):
        pass
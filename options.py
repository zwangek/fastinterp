
class Options():
    def __init__(self):
        self.data_root = '/home/wzy/workspace/data/vimeo_triplet'
        self.output_root = './output/'
        self.gpus = [0,3]

        self.train = {
            'planes': 64,
            'batch_size': 32,
            'num_workers': 4,
            'epochs': 300,
            'loss_weight': {'rec': 1.0, 'cons': 0.1, 'perc': 0.1}
            'base_lr': 3e-4,
            'min_lr': 1e-5,
            'weight_decay': 1e-4,
            'print_every': 50, # step
            'viz_every': 1000
            'save_every': 30, # epoch
            'log_dir': f'{self.output_root}/logs/',
            'ckpt_dir': f'{self.output_root}/checkpoints/'
        }

    def save(self):
        pass
import sys

from deepsem.DeepSEM_cell_type_non_specific_GRN_model import non_celltype_GRN_model

class deepsemopt:
    def __init__(self, dt, bm):
        self.n_epochs = 120
        self.task="non_celltype_GRN"
        self.setting="default"
        self.batch_size = 64
        self.net_file = f'data/BEELINE/{bm}_{dt}/label.csv'
        self.data_file = f'data/BEELINE/{bm}_{dt}/data.csv'
        self.beta = 1
        self.alpha = 100
        self.K = 1
        self.K1 = 1
        self.K2 = 2
        self.n_hidden = 128
        self.gamma = 0.95
        self.lr = 1e-4
        self.lr_step_size = 0.99
        self.n_epochs=120
        self.save_name = f'results/deepsem/{bm}_{dt}'

if __name__ == '__main__':
    bm = sys.argv[1]
    dt = sys.argv[2]

    opt = deepsemopt(dt, bm)
    for _ in range(100):
        trainer = non_celltype_GRN_model(opt)
        trainer.train_model()

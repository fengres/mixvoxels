import torch
import numpy as np
import copy

class Dynamics:
    def __init__(self, args, device, use_volumetric_render=False):
        self.args = args
        self.use_volumetric_render = use_volumetric_render
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([95/5]).to(device)) #r/p 92/94
        self.dynamics_ray_wise_prediction  = None
        self.dynamics_supervision = None

        self.tpr = [[a, None, None] for a in np.linspace(0.0, 1.0, 11)]
        self.tprs = []
        self.pos_ratios = []

    def calculate_loss(self, dynamics, dynamics_supervision, max_dynamics):
        # dynamics: Nr Ns
        # dynamics_supervision Nr
        if dynamics is None:
            self.dynamics_ray_wise_prediction  = None
            self.dynamics_supervision = None
            return 0
        if self.use_volumetric_render:
            dynamics_ray_wise_prediction = dynamics
            loss = 0.5*self.criterion(dynamics_ray_wise_prediction, dynamics_supervision.float())
            loss += 1.0*self.criterion(max_dynamics, dynamics_supervision.float())
        else:
            dynamics_ray_wise_prediction = max_dynamics
            loss = self.criterion(dynamics_ray_wise_prediction, dynamics_supervision.float())
        # dynamics_ray_wise_prediction = dynamics.mean(dim=1)
        self.dynamics_ray_wise_prediction = dynamics_ray_wise_prediction.cpu().detach().numpy()
        self.max_dynamics = max_dynamics.cpu().detach().numpy()
        self.dynamics_supervision = dynamics_supervision.cpu().detach().numpy()
        return loss

    def compute_metrics(self):
        if self.dynamics_ray_wise_prediction  is None:
            print('No Dynamic Metrics')
            return
        for i in range(len(self.tpr)):
            threshold = self.tpr[i][0]
            predictions = torch.sigmoid(torch.from_numpy(self.dynamics_ray_wise_prediction).float()).numpy() > threshold
            # predictions = torch.sigmoid(torch.from_numpy(self.max_dynamics).float()).numpy() > threshold
            tp = (predictions * self.dynamics_supervision).sum()
            fp = (predictions * (1-self.dynamics_supervision)).sum()
            fn = ((1-predictions) * self.dynamics_supervision).sum()
            precision = tp/(fp+tp+0.00001)
            recall = tp/(tp+fn)
            self.tpr[i][1] = precision
            self.tpr[i][2] = recall
        self.tprs.append(copy.deepcopy(self.tpr))
        self.pos_ratios.append(self.dynamics_supervision.sum() / self.dynamics_supervision.shape[0])

    def print_metrics(self, last=100):
        print('='*40)
        pos_ratio = np.array(self.pos_ratios[-last:]).mean()
        print('  positive ratio: {}'.format(pos_ratio))
        print('|   thresh    |     prec    |    recall  |')
        print('-'*40)
        tprs = np.array(self.tprs[-last:])
        for i in range(len(self.tpr)):
            print("|    {:.3f}    |".format(self.tpr[i][0]), end='')
            print("    {:.3f}    |".format(tprs[:, i, 1].mean()), end='')
            print("    {:.3f}   |".format(tprs[:, i, 2].mean()))
        print('='*40)




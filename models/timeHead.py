import torch
import torch.nn
import math
import numpy as np
from .utils_model import positional_encoding


class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class DirectDyRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, using_view=False, n_time_embedding=6,
                 total_time=300, featureD=128, time_embedding_type='abs', net_spec='i-d-d-o', gain=1.0):
        super(DirectDyRender, self).__init__()

        self.in_mlpC = inChanel + using_view * (3 + 2*viewpe*3)
        self.viewpe = viewpe
        self.n_time_embedding = n_time_embedding
        self.time_embedding_type = time_embedding_type
        self.using_view = using_view
        self.time_pos_encoding = torch.nn.Parameter(
            0.1 * torch.randn(total_time, n_time_embedding)
        )
        self.total_time = total_time

        self.out_dim = 3*total_time if using_view else total_time
        self.gain = gain
        layers = []
        _net_spec = net_spec.split('-')
        for i_mk, mk in enumerate(_net_spec):
            if mk == 'i':
                continue
            if mk == 'd' and _net_spec[i_mk-1] == 'i':
                layer = torch.nn.Linear(self.in_mlpC, featureD)
            if mk == 'd' and _net_spec[i_mk-1] == 'd':
                layer = torch.nn.Linear(featureD, featureD)
            if mk == 'o' and _net_spec[i_mk-1] == 'i':
                layer = torch.nn.Linear(self.in_mlpC, self.out_dim)
            if mk == 'o' and _net_spec[i_mk-1] == 'd':
                layer = torch.nn.Linear(featureD, self.out_dim)
            torch.nn.init.constant_(layer.bias, 0)
            # torch.nn.init.xavier_uniform_(layer.weight, gain=(0.5 if not using_view else 1))
            torch.nn.init.xavier_uniform_(layer.weight, gain=(self.gain if not using_view else 1))
            # torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

            layers.append(layer)
            if mk != 'o':
                layers.append(torch.nn.ReLU(inplace=True))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, features, time=None, viewdirs=None, temporal_indices=None):
        # spatio_temporal_sigma_mask: for rgb branch prunning
        # temporal_mask is for re-sampling temporal sequence by variance of training pixels.
        Ns = features.shape[0]
        num_frames = self.total_time
        indata = [features, ]
        if self.using_view:
            indata += [viewdirs,]
            indata += [positional_encoding(viewdirs, self.viewpe),]
        mlp_in = torch.cat(indata, dim=-1)
        output = self.mlp(mlp_in)
        if temporal_indices is not None and len(temporal_indices.shape) == 1:
            output = output.reshape(Ns, self.total_time, -1)
            output = output[:, temporal_indices, :]
        else:
            output = output.reshape(Ns, num_frames, -1)

        if self.using_view:
            output = torch.sigmoid(output)

        output = output.squeeze(dim=-1)
        return output


class ForrierDyRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, using_view=False, n_time_embedding=60,
                 total_time=300, featureD=128, time_embedding_type='abs'):
        super(ForrierDyRender, self).__init__()

        self.in_mlpC = inChanel + using_view * (3 + 2*viewpe*3)
        self.viewpe = viewpe
        self.n_time_embedding = n_time_embedding
        self.time_embedding_type = time_embedding_type
        self.using_view = using_view
        self.total_time = total_time

        layer1 = torch.nn.Linear(self.in_mlpC, featureD)
        layer2 = torch.nn.Linear(featureD, featureD)
        self.out_dim = 3*(2*n_time_embedding+1) if using_view else 1*(2*n_time_embedding+1)
        layer3 = torch.nn.Linear(featureD, self.out_dim)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)

        def forrier_basis(t, n_basis):
            ret = [1, ]
            for n in range(1, n_basis + 1):
                ret.append(math.cos(n * 2*math.pi * t/self.total_time))
                ret.append(math.sin(n * 2*math.pi * t/self.total_time))
            return ret

        # norm_time = lambda T: (T - self.total_time//2)/(self.total_time//2)
        self.forrier_basis = np.stack([forrier_basis(T, self.n_time_embedding) for T in range(self.total_time)], axis=1)
        self.forrier_basis = torch.from_numpy(self.forrier_basis).to(torch.float16).cuda().detach()

        if using_view:
            torch.nn.init.constant_(self.mlp[-1].bias, 0)
        else:
            torch.nn.init.constant_(self.mlp[0].bias, 0)
            torch.nn.init.constant_(self.mlp[2].bias, 0)
            torch.nn.init.constant_(self.mlp[4].bias, 0)
            torch.nn.init.xavier_uniform(self.mlp[0].weight)
            torch.nn.init.xavier_uniform(self.mlp[2].weight)
            torch.nn.init.xavier_uniform(self.mlp[4].weight)

    def forward(self, features, time=None, viewdirs=None, spatio_temporal_sigma_mask=None, temporal_mask=None, temporal_indices=None):
        # spatio_temporal_sigma_mask: for rgb branch prunning
        # temporal_mask is for re-sampling temporal sequence by variance of training pixels.
        Ns = features.shape[0]
        num_frames = self.total_time
        indata = [features, ]
        if self.using_view:
            indata += [viewdirs,]
            indata += [positional_encoding(viewdirs, self.viewpe),]
        mlp_in = torch.cat(indata, dim=-1)

        output = self.mlp(mlp_in)
        frequency_output = output.reshape(Ns, 3 if self.using_view else 1, 2*self.n_time_embedding+1).transpose(1,2)
        output = output.reshape(-1, 2*self.n_time_embedding+1)
        basis = self.forrier_basis
        if temporal_indices is not None and len(temporal_indices.shape) == 1:
            basis = self.forrier_basis[:, temporal_indices]
            num_frames = temporal_indices.shape[0]
            output = output @ basis
            output = output.reshape(Ns, -1, num_frames).transpose(1,2)

        if temporal_indices is not None and len(temporal_indices.shape) == 2:
            output = output @ basis
            output = output.reshape(Ns, -1, self.total_time).transpose(1,2)
            output = torch.gather(output, dim=1, index=temporal_indices.unsqueeze(dim=-1).expand(-1, -1, output.shape[-1]))
            # output Ns,

        if temporal_indices is None:
            output = output @ basis
            output = output.reshape(Ns, -1, self.total_time).transpose(1,2)

        if self.using_view:
            output = torch.sigmoid(output)

        output = output.squeeze(dim=-1)
        return output, 0# frequency_output.squeeze(dim=-1)

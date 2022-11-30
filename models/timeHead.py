import torch
import torch.nn
import math
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from functools import reduce

def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def generate_temporal_mask(temporal_mask, n_frames=300, n_frame_for_static=2):
    """
    temporal_mask: Ns
        true for select all frames to train
        false for random select one (or fixed small numbers) frame to train
    """
    Ns = temporal_mask.shape[0]
    keep = torch.ones(Ns, n_frames, device=temporal_mask.device)
    drop = torch.zeros(Ns, n_frames, device=temporal_mask.device)
    drop[:, 0] = 1
    # for i_choice in range(n_frame_for_static):
    #     drop[np.arange(Ns), np.random.choice(n_frames, Ns, replace=True)] = 1
    detail_temporal_mask = torch.where(temporal_mask.unsqueeze(dim=1).expand(-1, n_frames), keep, drop).bool()
    return detail_temporal_mask


class DyRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, using_view=False, n_time_embedding=6,
                 total_time=300, featureD=128, time_embedding_type='abs'):
        super(DyRender, self).__init__()

        self.in_mlpC = n_time_embedding + inChanel + using_view * (3 + 2*viewpe*3)
        self.viewpe = viewpe
        self.n_time_embedding = n_time_embedding
        self.time_embedding_type = time_embedding_type
        self.using_view = using_view
        self.time_pos_encoding = torch.nn.Parameter(
            0.1 * torch.randn(total_time, n_time_embedding)
        )
        self.total_time = total_time

        layer1 = torch.nn.Linear(self.in_mlpC, featureD)
        layer2 = torch.nn.Linear(featureD, featureD)
        self.out_dim = 3 if using_view else 1
        layer3 = torch.nn.Linear(featureD, self.out_dim)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)

        if using_view:
            torch.nn.init.constant_(self.mlp[-1].bias, 0)
        else:
            torch.nn.init.constant_(self.mlp[0].bias, 0)
            torch.nn.init.constant_(self.mlp[2].bias, 0)
            torch.nn.init.constant_(self.mlp[4].bias, 0)
            torch.nn.init.xavier_uniform(self.mlp[0].weight)
            torch.nn.init.xavier_uniform(self.mlp[2].weight)
            torch.nn.init.xavier_uniform(self.mlp[4].weight)

    def forward_with_time(self, features, time=None, viewdirs=None):
        Ns = features.shape[0]
        time_embedding = self.time_pos_encoding[time].unsqueeze(0).expand(Ns, -1)
        indata = [features, time_embedding]
        if self.using_view:
            indata += [viewdirs]
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        # mlp_in Ns x (ds + dt)
        output = self.mlp(mlp_in)
        if self.using_view:
            output = torch.sigmoid(output)
        output = output.squeeze(dim=-1)
        return output

    def forward(self, features, time=None, viewdirs=None, spatio_temporal_sigma_mask=None, temporal_mask=None, temporal_indices=None):
        # spatio_temporal_sigma_mask: for rgb branch prunning
        # temporal_mask is for re-sampling temporal sequence by variance of training pixels.
        Ns = features.shape[0]
        if temporal_indices is None:
            num_frames = self.total_time
            time_embedding = self.time_pos_encoding.unsqueeze(0).expand(Ns, -1, -1)
        elif len(temporal_indices.shape) == 1:
            num_frames = self.total_time if temporal_indices is None else len(temporal_indices)
            time_embedding = self.time_pos_encoding[temporal_indices].unsqueeze(0).expand(Ns, -1, -1)
        else:
            # temporal_indices Ns x T_train
            num_frames = temporal_indices.shape[1]
            time_embedding = (self.time_pos_encoding[temporal_indices.reshape(-1)]).reshape(Ns, num_frames, -1)
        features = features.unsqueeze(1).expand(-1, num_frames, -1)
        assert len(features.shape) == 3
        indata = [features, time_embedding]
        if self.using_view:
            indata += [viewdirs.unsqueeze(dim=1).expand(-1, num_frames, -1)]
            indata += [positional_encoding(viewdirs, self.viewpe).unsqueeze(dim=1).expand(-1, num_frames, -1)]
        mlp_in = torch.cat(indata, dim=-1)

        origin_output = torch.zeros(Ns, num_frames, self.out_dim).to(features)
        st_mask = torch.ones(Ns, num_frames).to(features).bool()
        if spatio_temporal_sigma_mask is not None:
            # mlp_in Ns x T x (ds + dt)
            # spatio_temporal_sigma_mask: Ns x T
            st_mask = st_mask & spatio_temporal_sigma_mask
        if temporal_mask is not None:
            st_mask = st_mask & temporal_mask
        mlp_in = mlp_in[st_mask]
        output = self.mlp(mlp_in)
        if self.using_view:
            output = torch.sigmoid(output)

        # mlp_in Ns x T x (ds + dt)
        # TODO Wrong Bellow
        origin_output[st_mask] = output
        output = origin_output
        output = output.squeeze(dim=-1)
        return output

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

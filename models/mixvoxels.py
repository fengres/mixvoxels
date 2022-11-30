import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
from functools import reduce
from .timeHead import DyRender, DirectDyRender, ForrierDyRender


def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def static_raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]

def sigma2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples, n_frame]
    n_frames = sigma.shape[2]
    dist = dist.unsqueeze(dim=-1).expand(-1, -1, n_frames)
    alpha = 1. - torch.exp(-sigma*dist)
    return alpha

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples, n_frame]
    n_frames = sigma.shape[2]
    dist = dist.unsqueeze(dim=-1).expand(-1, -1, n_frames)
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1, n_frames).to(alpha.device), 1. - alpha + 1e-10], dim=1), dim=1)

    weights = alpha * T[:, :-1, :]  # [N_rays, N_samples, n_frame]
    return alpha, weights, T[:,-1:,:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


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

class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()
        print('Using MLPRender_PE')
        self.in_mlpC = (2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPDepthRender(torch.nn.Module):
    def __init__(self, inChanel, relpospe=6, featureD=128):
        super(MLPDepthRender, self).__init__()

        self.in_mlpC = 3 + 2 * relpospe * 3 + inChanel
        self.relpospe = relpospe

        layer1 = torch.nn.Linear(self.in_mlpC, featureD)
        layer2 = torch.nn.Linear(featureD, featureD)
        layer3 = torch.nn.Linear(featureD, 1)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[0].bias, 0)
        torch.nn.init.constant_(self.mlp[2].bias, 0)
        torch.nn.init.constant_(self.mlp[4].bias, 0)
        torch.nn.init.xavier_uniform(self.mlp[0].weight)
        torch.nn.init.xavier_uniform(self.mlp[2].weight)
        torch.nn.init.xavier_uniform(self.mlp[4].weight)

    def forward(self, pts, features):
        indata = [features, pts]
        if self.relpospe > 0:
            indata += [positional_encoding(pts, self.relpospe)]
        mlp_in = torch.cat(indata, dim=-1)
        sigma = self.mlp(mlp_in)

        return sigma


class MixVoxels(torch.nn.Module):
    def __init__(self, args, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    rayMarch_weight_thres_static=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus', den_dim=None, densityMode=None, featureD=128, rel_pos_pe=6,
                    n_frames=300, amp=False, temporal_variance_threshold=0.1, n_frame_for_static=2,
                    dynamic_threshold=0.9, n_time_embedding=24, static_dynamic_seperate=0, dynamic_use_volumetric_render=0,
                    zero_dynamic_sigma=0, zero_dynamic_sigma_thresh=0.001, sigma_static_thresh=1., n_train_frames=0,
                    net_layer_add=0, density_n_comp_dynamic=0, app_n_comp_dynamic=0, interpolation='bilinear',
                    dynamic_granularity=None, point_wise_dynamic_threshold=None, static_point_detach=1,
                    dynamic_pool_kernel_size=1, time_head='dyrender',
                    # frequency parameters
                    frequency_threshold=0, filter_thresh=1.0,
                    static_featureC=128,
                 ):
        super(MixVoxels, self).__init__()
        self.args = args
        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.density_n_comp_dynamic = density_n_comp_dynamic or density_n_comp
        self.app_n_comp_dynamic = app_n_comp_dynamic or appearance_n_comp
        print(self.density_n_comp_dynamic, self.app_n_comp_dynamic)
        self.app_dim = app_dim
        self.den_dim = den_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device=device
        self.amp = amp
        self.n_frames = n_frames
        self.n_frame_for_static = n_frame_for_static
        self.temporal_variance_threshold = temporal_variance_threshold
        self.dynamic_threshold = dynamic_threshold
        self.point_wise_dynamic_threshold = point_wise_dynamic_threshold
        self.n_time_embedding = n_time_embedding
        self.static_dynamic_seperate = static_dynamic_seperate
        self.dynamic_use_volumetric_render = dynamic_use_volumetric_render
        self.zero_dynamic_sigma = zero_dynamic_sigma
        self.zero_dynamic_sigma_thresh = zero_dynamic_sigma_thresh
        self.sigma_static_thresh = sigma_static_thresh
        self.n_train_frames = n_train_frames
        self.net_layer_add = net_layer_add
        self.interpolation = interpolation
        self.dynamic_granularity = dynamic_granularity
        self.static_point_detach = static_point_detach
        self.frequency_threshold = frequency_threshold
        self.filter_thresh = filter_thresh
        self.time_head = time_head
        self.static_featureC = static_featureC
        self.TimeHead = {'dyrender': DyRender, 'directdyrender': DirectDyRender, 'forrier': ForrierDyRender}[time_head]

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.rayMarch_weight_thres_static = rayMarch_weight_thres_static
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio


        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]


        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.densityMode = None if densityMode == 'None' else densityMode
        self.featureD = featureD

        self.init_svd_volume(gridSize[0], device)

        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)
        self.init_render_den_func(self.densityMode, rel_pos_pe, featureD, device)
        self.init_static_render_func(device)

        self.maxpool1d = torch.nn.MaxPool1d(kernel_size=dynamic_pool_kernel_size, stride=1,
                                            padding=(dynamic_pool_kernel_size-1)//2)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        self.renderModule = self.TimeHead(self.app_dim, viewpe=view_pe, using_view=True, n_time_embedding=self.n_time_embedding, featureD=featureC,
                                        total_time=self.n_frames, net_spec=self.args.netspec_dy_color).to(device)
        print(self.renderModule)

    def init_render_den_func(self, densityMode, pos_pe, featureD, device):
        self.renderDenModule = self.TimeHead(self.den_dim, viewpe=0, using_view=False, n_time_embedding=self.n_time_embedding, featureD=featureD,
                                        total_time=self.n_frames, net_spec=self.args.netspec_dy_density).to(device)
        print(self.renderDenModule)

    def init_static_render_func(self, device):
        self.renderStaticModule = MLPRender_Fea(27, viewpe=0, feape=0, featureC=self.static_featureC).to(device)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def update_stepRatio(self, step_ratio):
        self.step_ratio = step_ratio
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,
            'den_dim': self.den_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC,
            'featureD': self.featureD,
            'n_frames': self.n_frames,
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def getTemporalDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[..., 0].unsqueeze(dim=-1).expand(-1, -1, -1, self.n_frames))
        for i in range(gridSize[0]):
            alpha[i] = self.compute_temporal_alpha(dense_xyz[i].view(-1, 3), self.stepSize)[0].view((gridSize[1], gridSize[2], self.n_frames))
        return alpha, dense_xyz

    @torch.no_grad()
    def calc_init_alpha(self, gridSize):
        alpha, _ = self.getDenseAlpha(gridSize)
        self.init_alpha = alpha

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):
        delete_unoptimized = False

        alpha, dense_xyz = self.getDenseAlpha(gridSize)

        # START INIT EQUAL VALUES
        if delete_unoptimized:
            optimized = (alpha - self.init_alpha) < 0.0001
            alpha = alpha * optimized.float()
        # END INIT EQUAL VALUES

        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, all_stds, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-2])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered], all_stds[mask_filtered]

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_alpha(self, xyz_locs, length=1):
        return self.compute_mean_alpha(xyz_locs, length)

    def compute_temporal_alpha(self, xyz_locs, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        static_alpha, static_sigma = self.compute_mean_alpha(xyz_locs, length, return_density=True)

        xyz_sampled = self.normalize_coord(xyz_locs)

        dynamic_prediction = self.compute_dynamics(xyz_sampled) # -1
        dynamic_mask = torch.sigmoid(dynamic_prediction) > self.dynamic_threshold

        sigma = torch.zeros((xyz_sampled.shape[0], self.n_frames), device=xyz_sampled.device, dtype=(torch.float16 if self.amp else torch.float32))
        ray_valid = alpha_mask & dynamic_mask
        if ray_valid.any():
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            sigma_feature = self.renderDenModule(features=sigma_feature, temporal_mask=None, temporal_indices=None)
            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma
            if self.zero_dynamic_sigma:
                null_space = static_sigma < self.zero_dynamic_sigma_thresh
                alpha_for_prune = 1. - torch.exp(-sigma * length)
                static_space = (alpha_for_prune.max(dim=-1)[0] - alpha_for_prune.min(dim=-1)[
                    0]) < self.sigma_static_thresh
                sigma[null_space & static_space] = 0

        sigma = torch.where(dynamic_mask.unsqueeze(dim=-1).expand(-1, self.n_frames), sigma, static_sigma.unsqueeze(-1).expand(-1, self.n_frames))
        alpha = 1 - torch.exp(-sigma*length)

        return alpha, sigma

    def compute_mean_alpha(self, xyz_locs, length=1, return_density=False):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_static_density(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])
        if return_density:
            return alpha, sigma

        return alpha

    def forward_dynamics(self, rays_chunk, variance_train, is_train=False, ndc_ray=False, N_samples=-1, rgb_train=None):
        # sample points
        xyz_sampled, z_vals, ray_valid, dists, viewdirs = self.sampling_points(rays_chunk, ndc_ray, is_train, N_samples)
        xyz_sampled = self.normalize_coord(xyz_sampled)

        # calculate pixel variance for weighted sampling
        if variance_train is None:
            variance = rgb_train.std(dim=1, unbiased=False).mean(dim=1).to(xyz_sampled)
        else:
            variance = variance_train.to(xyz_sampled)

        dynamics_supervision = variance > self.temporal_variance_threshold

        Nr, ns, nc = xyz_sampled.shape
        dynamic_prediction = self.compute_dynamics(xyz_sampled.reshape((Nr * ns, nc))).reshape(Nr, ns)
        max_dynamic_prediction = dynamic_prediction.max(dim=1)[0]
        if self.dynamic_use_volumetric_render:
            static_sigma = torch.zeros(*xyz_sampled.shape[:2], device=xyz_sampled.device, dtype=(torch.float32 if self.amp else torch.float32))
            if ray_valid.any():
                # static branch
                static_sigma_feature = self.compute_static_density(xyz_sampled[ray_valid])
                valid_static_sigma = self.feature2density(static_sigma_feature)
                static_sigma[ray_valid] = valid_static_sigma
            static_alpha, static_weight, static_bg_weight = static_raw2alpha(static_sigma, dists * self.distance_scale)
            # static_weight Nr x Ns

            dynamic_prediction = torch.sum((static_weight.detach()).softmax(dim=1) * dynamic_prediction, dim=1)
            # dynamic_prediction[static_weight <= self.rayMarch_weight_thres] = -1e7
        return dynamic_prediction, dynamics_supervision, max_dynamic_prediction

    def inference_dynamics(self, xyz_sampled, dists=None, dynamic_granularity='ray_wise'):
        Nr, ns, nc = xyz_sampled.shape
        dynamic_prediction = self.compute_dynamics(xyz_sampled.reshape((Nr * ns, nc))).reshape(Nr, ns)

        ray_wise_dynamic_prediction = dynamic_prediction.max(dim=1)[0]
        if self.dynamic_use_volumetric_render:
            # static branch
            static_sigma_feature = self.compute_static_density(xyz_sampled.reshape(Nr * ns, nc))
            static_sigma = self.feature2density(static_sigma_feature).reshape(Nr, ns)
            static_alpha, static_weight, static_bg_weight = static_raw2alpha(static_sigma, dists * self.distance_scale)
            # static_weight Nr x Ns

            dynamic_prediction2 = torch.sum((static_weight.detach()).softmax(dim=1) * dynamic_prediction, dim=1)
            temporal_mask = torch.sigmoid(dynamic_prediction2) > self.dynamic_threshold
            # dynamic_prediction[static_weight <= self.rayMarch_weight_thres] = -1e7
        else:
            temporal_mask = torch.sigmoid(ray_wise_dynamic_prediction) > self.dynamic_threshold
            if dynamic_granularity == 'point_wise':
                smoothed_dynamic_prediction = self.maxpool1d(dynamic_prediction.unsqueeze(dim=1)).squeeze()
                temporal_mask = (torch.sigmoid(smoothed_dynamic_prediction) > self.point_wise_dynamic_threshold) & \
                                (temporal_mask.unsqueeze(dim=1).expand(-1, ns))
            else:
                assert dynamic_granularity == 'ray_wise'
        return temporal_mask, dynamic_prediction

    def forward(self, rays_chunk, std_train, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, rgb_train=None,
                temporal_indices=None, static_branch_only=False, remove_foreground=False, **kwargs):
        rays_chunk = rays_chunk.float()
        if std_train is not None:
            std_train = std_train.float()
        if static_branch_only:
            xyz_sampled, z_vals, ray_valid, dists, viewdirs = self.sampling_points(rays_chunk, ndc_ray, is_train,
                                                                                   N_samples)
            xyz_sampled = self.normalize_coord(xyz_sampled)
            static_sigma, static_rgb, static_rgb_map, static_depth_map, static_fraction, valid_static_sigma, \
            static_alpha, static_weight, static_acc_map = \
                self.forward_static_branch(rays_chunk, xyz_sampled, z_vals, dists, ray_valid, viewdirs, white_bg,
                                           is_train, remove_foreground=remove_foreground)
            retva = {
                'static_sigma': static_sigma,
                'static_rgb': static_rgb,
                'static_rgb_map': static_rgb_map,
                'static_depth_map': static_depth_map,
                'static_fraction': static_fraction,
                'valid_static_sigma': valid_static_sigma,
            }
            return retva
        return self.forward_seperatly(rays_chunk, std_train, white_bg, is_train, ndc_ray, N_samples, rgb_train, temporal_indices=temporal_indices, **kwargs)

    def sampling_points(self, rays_chunk, ndc_ray, is_train, N_samples, alpha_filte=True):
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                 N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                             N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if alpha_filte and self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        return xyz_sampled, z_vals, ray_valid, dists, viewdirs

    def generate_temporal_mask(self, rgb_train, variance_train, xyz_sampled, dists, temporal_indices, dynamic_granularity='ray_wise'):
        Nr, ns, nc = xyz_sampled.shape
        num_frames = self.n_frames if temporal_indices is None else self.n_train_frames
        if rgb_train is not None:
            # calculate pixel variance for weighted sampling
            if variance_train is None:
                variance = rgb_train.std(dim=1, unbiased=False).mean(dim=1).to(self.device)
            else:
                variance = variance_train.to(self.device)
            temporal_mask = variance > self.temporal_variance_threshold
            dynamics_supervision = temporal_mask
            temporal_mask = temporal_mask.unsqueeze(dim=1).expand(-1, ns)
            dynamic_prediction = None
        else:
            temporal_mask, dynamic_prediction = self.inference_dynamics(xyz_sampled, dists=dists, dynamic_granularity=dynamic_granularity)
            if dynamic_granularity == 'ray_wise':
                temporal_mask = temporal_mask.unsqueeze(dim=1).expand(-1, xyz_sampled.shape[1])
            dynamics_supervision = None
        temporal_mask = temporal_mask.unsqueeze(dim=-1).expand(-1, -1, num_frames)
        ray_wise_temporal_mask = temporal_mask.any(dim=1)

        return temporal_mask, dynamic_prediction, dynamics_supervision, ray_wise_temporal_mask

    def forward_static_branch(self, rays_chunk, xyz_sampled, z_vals, dists, ray_valid, viewdirs, white_bg, is_train, remove_foreground=False):
        # =======================static branch=====================
        static_sigma = torch.zeros(*xyz_sampled.shape[:2], device=xyz_sampled.device, dtype=(torch.float32 if self.amp else torch.float32))
        static_rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device, dtype=(torch.float16 if self.amp else torch.float32))
        if remove_foreground:
            temporal_mask, dynamic_prediction, dynamics_supervision, ray_wise_temporal_mask = \
                self.generate_temporal_mask(None, None, xyz_sampled, dists, None, dynamic_granularity=self.dynamic_granularity)
            sample_mask = temporal_mask.any(dim=-1)
        if ray_valid.any():
            # static branch
            static_sigma_feature = self.compute_static_density(xyz_sampled[ray_valid])
            valid_static_sigma = self.feature2density(static_sigma_feature)
            static_sigma[ray_valid] = valid_static_sigma
        else:
            valid_static_sigma = None
        if remove_foreground:
            static_sigma[sample_mask] = 0
        static_alpha, static_weight, static_bg_weight = static_raw2alpha(static_sigma, dists * self.distance_scale)
        static_app_mask = static_weight > self.rayMarch_weight_thres_static
        static_fraction = 0
        if static_app_mask.any():
            static_fraction = reduce(lambda a, b: a * b, list(xyz_sampled[static_app_mask].shape)) / \
                              reduce(lambda a, b: a * b, list(xyz_sampled.shape))
            static_app_features = self.compute_static_app(xyz_sampled[static_app_mask])
            valid_static_rgbs = self.renderStaticModule(xyz_sampled[static_app_mask], viewdirs[static_app_mask],
                                                        static_app_features)
            static_rgb[static_app_mask] = valid_static_rgbs

        static_acc_map = torch.sum(static_weight, dim=1)
        static_rgb_map = torch.sum(static_weight[..., None] * static_rgb, dim=1)
        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            static_rgb_map = static_rgb_map + (1. - static_acc_map[..., None])
        static_rgb_map = static_rgb_map.clamp(0, 1)
        with torch.no_grad():
            static_depth_map = torch.sum(static_weight * z_vals, dim=1)
            static_depth_map = static_depth_map + (1. - static_acc_map) * rays_chunk[..., -1]
        return static_sigma, static_rgb, static_rgb_map, static_depth_map, static_fraction, valid_static_sigma, \
               static_alpha, static_weight, static_acc_map

    def forward_seperatly(self, rays_chunk, std_train, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1,
                          rgb_train=None, composite_by_points=False, temporal_indices=None, diff_calc=False,
                          render_path=False, nodepth=False):
        timing = dict()
        _t = time.time()
        # if self.dynamic_granularity == 'point_wise':
        #     rgb_train = None
            # assert rgb_train is None
        xyz_sampled, z_vals, ray_valid, dists, viewdirs = self.sampling_points(rays_chunk, ndc_ray, is_train, N_samples)
        xyz_sampled = self.normalize_coord(xyz_sampled)
        # temporal mask
        num_frames = self.n_frames if temporal_indices is None else self.n_train_frames
        temporal_mask, dynamic_prediction, dynamics_supervision, ray_wise_temporal_mask = \
            self.generate_temporal_mask(None if self.dynamic_granularity == 'point_wise' else rgb_train,
                                        std_train, xyz_sampled, dists, temporal_indices,
                                        dynamic_granularity=self.dynamic_granularity)
        t_ = time.time()
        timing['preprocessing'] = t_ - _t
        _t = t_

        # ======================static branch==================
        static_sigma, static_rgb, static_rgb_map, static_depth_map, static_fraction, \
        valid_static_sigma, static_alpha, static_weight, static_acc_map = \
            self.forward_static_branch(rays_chunk, xyz_sampled, z_vals, dists, ray_valid, viewdirs, white_bg, is_train)
        # =====================================================
        t_ = time.time()
        timing['static'] = t_ - _t
        _t = t_

        # ======================dynamic branch==================
        sigma = torch.zeros((*xyz_sampled.shape[:2], num_frames), device=xyz_sampled.device, dtype=(torch.float16 if self.amp else torch.float32))
        # frequency_weight = torch.zeros((*xyz_sampled.shape[:2], 2*self.n_time_embedding+1), device=xyz_sampled.device, dtype=(torch.float16 if self.amp else torch.float32))
        rgb = torch.zeros((*xyz_sampled.shape[:2], num_frames, 3), device=xyz_sampled.device, dtype=sigma.dtype)
        # frequency_weight_rgb = torch.zeros((*xyz_sampled.shape[:2], 2*self.n_time_embedding+1, 3), device=xyz_sampled.device, dtype=(torch.float16 if self.amp else torch.float32))
        ray_valid = ray_valid & (temporal_mask.any(dim=-1))
        if ray_valid.any():
            # dynamic branch
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            # sigma_feature = self.renderDenModule(features=sigma_feature, temporal_mask=None if temporal_mask is None else temporal_mask[ray_valid])
            if temporal_indices is None:
                masked_temporal_indices = temporal_indices
            elif len(temporal_indices.shape) == 2:
                masked_temporal_indices = temporal_indices.unsqueeze(dim=1).expand(-1, xyz_sampled.shape[1], -1)
                masked_temporal_indices = masked_temporal_indices[ray_valid]
            elif len(temporal_indices.shape) == 1:
                masked_temporal_indices = temporal_indices
            else:
                raise NotImplementedError
            if self.time_head == 'forrier':
                sigma_feature, point_wise_frequencies = self.renderDenModule(features=sigma_feature, temporal_mask=None, temporal_indices=masked_temporal_indices)
            else:
                sigma_feature = self.renderDenModule(features=sigma_feature, temporal_mask=None, temporal_indices=masked_temporal_indices)
            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma
            # if self.time_head == 'forrier':
            #     frequency_weight[ray_valid] = point_wise_frequencies
            if self.zero_dynamic_sigma:
                null_space = static_sigma < self.zero_dynamic_sigma_thresh
                alpha_for_prune = sigma2alpha(sigma, dists*self.distance_scale)
                static_space = (alpha_for_prune.max(dim=-1)[0] - alpha_for_prune.min(dim=-1)[0]) < self.sigma_static_thresh
                sigma[null_space & static_space] = 0
                # ray_valid = ray_valid & (static_sigma > self.zero_dynamic_sigma_thresh)
                ray_valid = ray_valid & (~null_space | ~static_space)
        else:
            validsigma = None
            if render_path:
                ret = {
                    'comp_rgb_map': static_rgb_map.unsqueeze(dim=1).expand(-1, self.n_frames, -1),
                }
                if not nodepth:
                    ret.update({'comp_depth_map': static_depth_map.unsqueeze(dim=1).expand(-1, self.n_frames)})
                return ret

        # if self.time_head == 'forrier' and is_train and ray_valid.any():
        #     ray_fft = torch.fft.rfft(rgb_train.mean(dim=-1), dim=1).unsqueeze(dim=1).expand(-1, xyz_sampled.shape[1], -1)
        #     ray_fft = ray_fft[ray_valid]
        #     fft_mask = ray_fft.abs() < self.filter_thresh
        #     fft_mask = torch.cat([fft_mask[:,0:1],
        #                           torch.stack([fft_mask[:, 1:], fft_mask[:, 1:]], dim=-1).reshape(fft_mask.shape[0], -1)],
        #                          dim=1)
        #     filtered_frequencies = (frequency_weight[ray_valid].abs() * fft_mask.float()).mean()
        # else:
        #     filtered_frequencies = 0

        # sigma_diff = (sigma.mean(dim=-1)[temporal_mask.any(dim=-1)] - static_sigma.detach()[temporal_mask.any(dim=-1)])
        # sigma Nr x ns x T
        # static_sigma Nr x ns
        if not diff_calc:
            sigma_diff = (sigma - static_sigma.detach().unsqueeze(dim=-1))[ray_valid]
        # sigma_ray_wise = sigma[temporal_mask.any(dim=-1).any(dim=-1)]

        _sub_time = time.time()

        if self.dynamic_granularity == 'point_wise':
            # sigma = torch.where(temporal_mask, sigma,
            #                     (static_sigma.detach().unsqueeze(-1).expand(-1, -1, sigma.shape[2])
            #                      if self.static_point_detach else
            #                      static_sigma.unsqueeze(-1).expand(-1, -1, sigma.shape[2])
            #                      )
            #                     )
            # substitute of the above commented code
            sigma[~(temporal_mask.any(dim=-1))] = static_sigma.detach()[~(temporal_mask.any(dim=-1))].unsqueeze(dim=-1).to(sigma)

        # alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)
        # substitute of the above commented codes.
        ray_mask = temporal_mask.any(dim=-1).any(dim=-1)
        alpha = torch.zeros((*xyz_sampled.shape[:2], num_frames), device=xyz_sampled.device, dtype=(torch.float32))
        weight = torch.zeros((*xyz_sampled.shape[:2], num_frames), device=xyz_sampled.device, dtype=(torch.float32))
        valid_alpha, valid_weight, valid_bg_weight = raw2alpha(sigma[ray_mask], dists[ray_mask] * self.distance_scale)
        alpha[ray_mask] = valid_alpha
        alpha[~ray_mask] = static_alpha[~ray_mask].unsqueeze(dim=-1).detach()
        weight[ray_mask] = valid_weight
        weight[~ray_mask] = static_weight[~ray_mask].unsqueeze(dim=-1).detach()

        # alpha, weight [N_ray, N_sample, n_frames]
        app_mask = weight > self.rayMarch_weight_thres
        # Note app_mask is implicitly added by ray_valid
        # app_mask [N_ray, N_sample, n_frames]
        app_spatio_mask = app_mask.any(dim=2)
        timing['sub_test'] = time.time() - _sub_time

        t_ = time.time()
        timing['dy_sigma'] = t_ - _t
        _t = t_

        if self.dynamic_granularity == 'point_wise':
            app_spatio_mask = app_spatio_mask & temporal_mask.any(dim=2)
        fraction, temporal_fraction = 0, 0
        if app_spatio_mask.any():
            fraction = reduce(lambda a, b: a * b, list(xyz_sampled[app_spatio_mask].shape)) / \
                       reduce(lambda a, b: a * b, list(xyz_sampled.shape))
            temporal_fraction = (torch.sum(app_spatio_mask) / reduce(lambda a, b: a * b, list(app_spatio_mask.shape))).item()

            if temporal_indices is None:
                masked_temporal_indices = temporal_indices
            elif len(temporal_indices.shape) == 2:
                masked_temporal_indices = temporal_indices.unsqueeze(dim=1).expand(-1, xyz_sampled.shape[1], -1)
                masked_temporal_indices = masked_temporal_indices[app_spatio_mask]
            elif len(temporal_indices.shape) == 1:
                masked_temporal_indices = temporal_indices
            else:
                raise NotImplementedError

            rgb_query_start = time.time()
            app_features = self.compute_appfeature(xyz_sampled[app_spatio_mask])
            rgb_query_time = time.time() - rgb_query_start
            valid_rgbs = self.renderModule(features=app_features, viewdirs=viewdirs[app_spatio_mask],
                                           spatio_temporal_sigma_mask=app_mask[app_spatio_mask],
                                           temporal_mask=None, temporal_indices=masked_temporal_indices)
            rgb_head_time = time.time() - rgb_query_start - rgb_query_time
            # print(rgb_query_time, rgb_head_time)
            if self.time_head == 'forrier':
                valid_rgbs, point_wise_rgb_frequencies = valid_rgbs
            #     frequency_weight_rgb[app_spatio_mask] = point_wise_rgb_frequencies
            rgb[app_spatio_mask] = valid_rgbs

        # if self.time_head == 'forrier' and is_train and app_spatio_mask.any():
            # ray_fft = torch.fft.rfft(rgb_train.mean(dim=-1), dim=1).unsqueeze(dim=1).expand(-1, xyz_sampled.shape[1], -1)
            # ray_fft = ray_fft[app_spatio_mask]
            # fft_mask = ray_fft.abs() < self.filter_thresh
            # fft_mask = torch.cat([fft_mask[:,0:1],
            #                       torch.stack([fft_mask[:, 1:], fft_mask[:, 1:]], dim=-1).reshape(fft_mask.shape[0], -1)],
            #                      dim=1)
            # filtered_frequencies_rgb = (frequency_weight_rgb[app_spatio_mask].abs() *
            #                             fft_mask.float().unsqueeze(dim=-1).expand(-1, -1, 3)).mean()
            # filtered_loss = filtered_frequencies + filtered_frequencies_rgb
            # filtered_loss = 0

        if not diff_calc:
            rgb_diff = (rgb - static_rgb.detach().unsqueeze(dim=2))[app_spatio_mask]
        t_ = time.time()
        timing['dy_rgb'] = t_ - _t
        _t = t_

        if self.dynamic_granularity == 'point_wise':
            # rgb Nr x ns x T x 3
            # rgb = torch.where(temporal_mask.unsqueeze(dim=-1).expand(-1, -1, -1, rgb.shape[-1]), rgb,
            #                   (static_rgb.detach().unsqueeze(dim=2).expand(-1, -1, rgb.shape[2], -1)
            #                    if self.static_point_detach else
            #                    static_rgb.unsqueeze(dim=2).expand(-1, -1, rgb.shape[2], -1)
            #                   )
            #                  )
            # substitute of the above commented codes
            rgb[~(temporal_mask.any(dim=-1))] = static_rgb.detach()[~(temporal_mask.any(dim=-1))].unsqueeze(dim=1)

        # acc_map = torch.sum(weight, dim=1)
        # substitute of the above commented code
        acc_map = torch.zeros((xyz_sampled.shape[0], num_frames), device=xyz_sampled.device, dtype=(torch.float32))
        acc_map[ray_mask] = torch.sum(weight[ray_mask], dim=1)
        acc_map[~ray_mask] = static_acc_map.detach()[~ray_mask].unsqueeze(dim=-1)


        # rgb_map = torch.sum(weight[..., None] * rgb, dim=1)
        # substitute for the above commented code
        # rgb_map Nr T 3
        rgb_map = torch.zeros((xyz_sampled.shape[0], num_frames, 3), device=xyz_sampled.device, dtype=(torch.float32))
        rgb_map[ray_mask] = torch.sum(weight[ray_mask][..., None] * rgb[ray_mask], dim=1)
        rgb_map[~ray_mask] = static_rgb_map.detach()[~ray_mask].unsqueeze(dim=1)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)
        if nodepth == False:
            with torch.no_grad():
                # depth_map = torch.sum(weight * z_vals.unsqueeze(dim=-1), dim=1)
                # substitute of the above commented code
                depth_map = torch.zeros((xyz_sampled.shape[0], num_frames), device=xyz_sampled.device, dtype=(torch.float32))
                depth_map[ray_mask] = torch.sum(weight[ray_mask] * z_vals.unsqueeze(dim=-1), dim=1)
                depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1].unsqueeze(dim=-1)
                depth_map[~ray_mask] = static_depth_map[~ray_mask].detach().unsqueeze(dim=-1)
        # ===================================================

        if (ray_wise_temporal_mask is not None) and is_train:
            rgb_map = rgb_map[ray_wise_temporal_mask]
            # depth_map = depth_map[ray_wise_temporal_mask]

        t_ = time.time()
        timing['render_function'] = t_ - _t
        _t = t_

        if not is_train:
            # composite_by_rays rgb_map: Nr x T x 3
            # ray_wise_temporal_mask: Nr x T
            comp_rgb_map = rgb_map * ray_wise_temporal_mask.unsqueeze(dim=-1).float() \
                           + static_rgb_map.unsqueeze(dim=1) * (1.-ray_wise_temporal_mask.float()).unsqueeze(dim=-1)
            if nodepth == False:
                comp_depth_map = depth_map * ray_wise_temporal_mask.float() + \
                                 static_depth_map.unsqueeze(dim=1) * (1.-ray_wise_temporal_mask.float())
            if render_path:
                ret = {'comp_rgb_map': comp_rgb_map}
                if nodepth == False:
                    ret.update({'comp_depth_map': comp_depth_map})
                return ret

            # comp_sigma = sigma * temporal_mask.float() + static_sigma.unsqueeze(dim=-1) * (1-temporal_mask.float())

        total_time = sum(timing.values())
        timing = {k: round(timing[k]/total_time*100, 3) for k in timing.keys()}
        # print(timing)
        ret_values = {'rgb_map': rgb_map,
                      'fraction': fraction,
                      'temporal_fraction': temporal_fraction,
                      'ray_wise_temporal_mask': ray_wise_temporal_mask,
                      }
        if nodepth == False:
            ret_values.update({
                'depth_map': depth_map,
            })
        if not diff_calc:
            ret_values.update({
                'sigma_diff': sigma_diff,
                'rgb_diff': rgb_diff,
            })
        ret_values.update({
            'static_rgb_map': static_rgb_map,
            'static_depth_map': static_depth_map,
            'static_fraction': static_fraction,
        })

        if not is_train:
            ret_values.update({
                'comp_rgb_map': comp_rgb_map,
                'comp_depth_map': comp_depth_map,
            })
        return ret_values # rgb, sigma, alpha, weight, bg_weight

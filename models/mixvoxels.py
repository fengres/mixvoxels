import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import time
from .utils_model import static_raw2alpha, raw2alpha, sigma2alpha
from .timeHead import DirectDyRender, ForrierDyRender, MLPRender_Fea

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


class MixVoxels(torch.nn.Module):
    def __init__(self, args, aabb, gridSize, device,
                    density_n_comp = 8,
                    appearance_n_comp = 24,
                    app_dim = 27,
                    shadingMode = 'MLP_PE',
                    alphaMask = None,
                    near_far=[2.0,6.0],
                    density_shift = -10,
                    alphaMask_thres=0.001,
                    distance_scale=25,
                    rayMarch_weight_thres=0.0001,
                    rayMarch_weight_thres_static=0.0001,
                    pos_pe = 6,
                    view_pe = 6,
                    fea_pe = 6,
                    featureC=128,
                    step_ratio=2.0,
                    fea2denseAct = 'softplus',
                    den_dim=None,
                    featureD=128,
                    n_frames=300,
                    amp=False,
                    temporal_variance_threshold=0.1,
                    dynamic_threshold=0.9,
                    zero_dynamic_sigma=0,
                    zero_dynamic_sigma_thresh=0.001,
                    sigma_static_thresh=1.,
                    n_train_frames=0,
                    density_n_comp_dynamic=0,
                    app_n_comp_dynamic=0,
                    interpolation='bilinear',
                    point_wise_dynamic_threshold=None,
                    dynamic_pool_kernel_size=1,
                    time_head='dyrender',
                    # frequency parameters
                    static_featureC=128,
                 ):
        super(MixVoxels, self).__init__()
        self.args = args
        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.density_n_comp_dynamic = density_n_comp_dynamic or density_n_comp
        self.app_n_comp_dynamic = app_n_comp_dynamic or appearance_n_comp
        self.app_dim = app_dim
        self.den_dim = den_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device=device
        self.amp = amp
        self.n_frames = n_frames
        self.temporal_variance_threshold = temporal_variance_threshold
        self.dynamic_threshold = dynamic_threshold
        self.point_wise_dynamic_threshold = point_wise_dynamic_threshold
        self.zero_dynamic_sigma = zero_dynamic_sigma
        self.zero_dynamic_sigma_thresh = zero_dynamic_sigma_thresh
        self.sigma_static_thresh = sigma_static_thresh
        self.n_train_frames = n_train_frames
        self.interpolation = interpolation
        self.time_head = time_head
        self.static_featureC = static_featureC
        self.TimeHead = {'directdyrender': DirectDyRender, 'forrier': ForrierDyRender}[time_head]

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
        self.featureD = featureD

        self.init_svd_volume(gridSize[0], device)

        self.init_render_func(view_pe, featureC, device)
        self.init_render_den_func(featureD, device)
        self.init_static_render_func(device)

        self.maxpool1d = torch.nn.MaxPool1d(kernel_size=dynamic_pool_kernel_size, stride=1,
                                            padding=(dynamic_pool_kernel_size-1)//2)

    def init_render_func(self, view_pe, featureC, device):
        self.renderModule = self.TimeHead(self.app_dim, viewpe=view_pe, using_view=True, featureD=featureC,
                                        total_time=self.n_frames, net_spec=self.args.netspec_dy_color).to(device)
        print(self.renderModule)

    def init_render_den_func(self, featureD, device):
        self.renderDenModule = self.TimeHead(self.den_dim, viewpe=0, using_view=False, featureD=featureD,
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
        alpha, dense_xyz = self.getDenseAlpha(gridSize)

        # START INIT EQUAL VALUES
        delete_unoptimized = False
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
            sigma_feature = self.renderDenModule(features=sigma_feature, temporal_indices=None)
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
        return dynamic_prediction, dynamics_supervision, max_dynamic_prediction

    def inference_dynamics(self, xyz_sampled):
        Nr, ns, nc = xyz_sampled.shape
        dynamic_prediction = self.compute_dynamics(xyz_sampled.reshape((Nr * ns, nc))).reshape(Nr, ns)
        smoothed_dynamic_prediction = self.maxpool1d(dynamic_prediction.unsqueeze(dim=1)).squeeze()

        ray_wise_dynamic_prediction = dynamic_prediction.max(dim=1)[0]

        mask = torch.sigmoid(ray_wise_dynamic_prediction) > self.dynamic_threshold
        mask = (torch.sigmoid(smoothed_dynamic_prediction) > self.point_wise_dynamic_threshold) & \
                        (mask.unsqueeze(dim=1).expand(-1, ns))
        return mask

    def forward(self, rays_chunk, std_train, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, rgb_train=None,
                temporal_indices=None, **kwargs):
        rays_chunk = rays_chunk.float()
        if std_train is not None:
            std_train = std_train.float()
        return self.forward_seperatly(rays_chunk, std_train, white_bg, is_train, ndc_ray,
                                      N_samples, rgb_train, temporal_indices=temporal_indices,
                                      **kwargs)

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

    def generate_dynamic_mask(self, xyz_sampled, dists, temporal_indices):
        num_frames = self.n_frames if temporal_indices is None else self.n_train_frames
        point_dynamic_mask = self.inference_dynamics(xyz_sampled)
        ray_dynamic_mask = point_dynamic_mask.any(dim=1)

        return point_dynamic_mask, ray_dynamic_mask

    def forward_static_branch(self, rays_chunk, xyz_sampled, z_vals, dists, ray_valid, viewdirs, white_bg, is_train):
        # =======================static branch=====================
        static_sigma = torch.zeros(*xyz_sampled.shape[:2], device=xyz_sampled.device, dtype=torch.float32)
        static_rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device, dtype=torch.float16)
        if ray_valid.any():
            # static branch
            static_sigma_feature = self.compute_static_density(xyz_sampled[ray_valid])
            valid_static_sigma = self.feature2density(static_sigma_feature)
            static_sigma[ray_valid] = valid_static_sigma
        else:
            valid_static_sigma = None
        static_alpha, static_weight, static_bg_weight = static_raw2alpha(static_sigma, dists * self.distance_scale)
        static_app_mask = static_weight > self.rayMarch_weight_thres_static
        if static_app_mask.any():
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
        return static_sigma, static_rgb, static_rgb_map, static_depth_map, valid_static_sigma, \
               static_alpha, static_weight, static_acc_map

    def forward_seperatly(self, rays_chunk, std_train, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1,
                          rgb_train=None, temporal_indices=None, render_path=False):

        # Ray Marching:
        #  sampling points along rays, and filter the empty space (with ray_valid) through the alphaGridMask
        xyz_sampled, z_vals, ray_valid, dists, viewdirs = self.sampling_points(rays_chunk, ndc_ray, is_train, N_samples)
        xyz_sampled = self.normalize_coord(xyz_sampled)

        # Inference the dynamic points
        num_frames = self.n_frames if temporal_indices is None else self.n_train_frames
        point_dynamic_mask, ray_dynamic_mask = \
            self.generate_dynamic_mask(xyz_sampled, dists, temporal_indices)

        # ======================static branch==================
        static_sigma, static_rgb, static_rgb_map, static_depth_map, \
        valid_static_sigma, static_alpha, static_weight, static_acc_map = \
            self.forward_static_branch(rays_chunk, xyz_sampled, z_vals, dists, ray_valid, viewdirs, white_bg, is_train)
        # =====================================================

        # ======================dynamic branch==================
        sigma = torch.zeros((*xyz_sampled.shape[:2], num_frames), device=xyz_sampled.device, dtype=torch.float16)
        rgb = torch.zeros((*xyz_sampled.shape[:2], num_frames, 3), device=xyz_sampled.device, dtype=sigma.dtype)
        ray_valid = ray_valid & (point_dynamic_mask)

        if ray_valid.any():
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            assert temporal_indices is None or len(temporal_indices.shape) == 1, 'wrong temporal indices'
            sigma_feature = self.renderDenModule(features=sigma_feature, temporal_indices=temporal_indices)
            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma
            if self.zero_dynamic_sigma:
                null_space = static_sigma < self.zero_dynamic_sigma_thresh
                alpha_for_prune = sigma2alpha(sigma, dists*self.distance_scale)
                static_space = (alpha_for_prune.max(dim=-1)[0] - alpha_for_prune.min(dim=-1)[0]) < self.sigma_static_thresh
                sigma[null_space & static_space] = 0
        elif render_path: # if rendering, skip the following procedures.
            ret = {'comp_rgb_map': static_rgb_map.unsqueeze(dim=1).expand(-1, self.n_frames, -1), }
            ret.update({'comp_depth_map': static_depth_map.unsqueeze(dim=1).expand(-1, self.n_frames)})
            return ret

        # ======== Mixing densities ======
        # sigma Nr x ns x T
        # static_sigma Nr x ns
        sigma[~point_dynamic_mask] = static_sigma.detach()[~point_dynamic_mask].unsqueeze(dim=-1).to(sigma)
        # alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)
        # substitute of the above commented codes.
        alpha = torch.zeros((*xyz_sampled.shape[:2], num_frames), device=xyz_sampled.device, dtype=torch.float32)
        weight = torch.zeros((*xyz_sampled.shape[:2], num_frames), device=xyz_sampled.device, dtype=torch.float32)
        valid_alpha, valid_weight, valid_bg_weight = raw2alpha(sigma[ray_dynamic_mask], dists[ray_dynamic_mask] * self.distance_scale)
        alpha[ray_dynamic_mask] = valid_alpha
        alpha[~ray_dynamic_mask] = static_alpha[~ray_dynamic_mask].unsqueeze(dim=-1).detach()
        weight[ray_dynamic_mask] = valid_weight
        weight[~ray_dynamic_mask] = static_weight[~ray_dynamic_mask].unsqueeze(dim=-1).detach()

        # alpha, weight [N_ray, N_sample, n_frames]
        app_mask = weight > self.rayMarch_weight_thres
        app_mask = app_mask.any(dim=2) & point_dynamic_mask
        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            rgb[app_mask] = self.renderModule(features=app_features, viewdirs=viewdirs[app_mask],
                                           temporal_indices=temporal_indices)

        # substitute of the above commented codes
        rgb[~point_dynamic_mask] = static_rgb.detach()[~point_dynamic_mask].unsqueeze(dim=1)

        acc_map = torch.zeros((xyz_sampled.shape[0], num_frames), device=xyz_sampled.device, dtype=(torch.float32))
        acc_map[ray_dynamic_mask] = torch.sum(weight[ray_dynamic_mask], dim=1)
        acc_map[~ray_dynamic_mask] = static_acc_map.detach()[~ray_dynamic_mask].unsqueeze(dim=-1)

        # rgb_map Nr T 3
        rgb_map = torch.zeros((xyz_sampled.shape[0], num_frames, 3), device=xyz_sampled.device, dtype=(torch.float32))
        rgb_map[ray_dynamic_mask] = torch.sum(weight[ray_dynamic_mask][..., None] * rgb[ray_dynamic_mask], dim=1)
        rgb_map[~ray_dynamic_mask] = static_rgb_map.detach()[~ray_dynamic_mask].unsqueeze(dim=1)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)
        with torch.no_grad():
            depth_map = torch.zeros((xyz_sampled.shape[0], num_frames), device=xyz_sampled.device, dtype=(torch.float32))
            depth_map[ray_dynamic_mask] = torch.sum(weight[ray_dynamic_mask] * z_vals.unsqueeze(dim=-1), dim=1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1].unsqueeze(dim=-1)
            depth_map[~ray_dynamic_mask] = static_depth_map[~ray_dynamic_mask].detach().unsqueeze(dim=-1)
        # ===================================================

        if (ray_dynamic_mask is not None) and is_train:
            rgb_map = rgb_map[ray_dynamic_mask]

        if not is_train:
            # composite_by_rays rgb_map: Nr x T x 3
            comp_rgb_map = rgb_map * ray_dynamic_mask.unsqueeze(dim=-1).unsqueeze(dim=-1).float() \
                           + static_rgb_map.unsqueeze(dim=1) * (1.-ray_dynamic_mask.float()).unsqueeze(dim=-1).unsqueeze(dim=-1)
            comp_depth_map = depth_map * ray_dynamic_mask.unsqueeze(dim=-1).float() + \
                             static_depth_map.unsqueeze(dim=1) * (1.-ray_dynamic_mask.float()).unsqueeze(dim=-1)
            if render_path:
                ret = {'comp_rgb_map': comp_rgb_map}
                ret.update({'comp_depth_map': comp_depth_map})
                return ret

        ret_values = {
            'rgb_map': rgb_map,
            'ray_dynamic_mask': ray_dynamic_mask,
            'depth_map': depth_map,
            'static_rgb_map': static_rgb_map,
            'static_depth_map': static_depth_map,
        }

        if not is_train:
            ret_values.update({
                'comp_rgb_map': comp_rgb_map,
                'comp_depth_map': comp_depth_map,
            })
        return ret_values # rgb, sigma, alpha, weight, bg_weight

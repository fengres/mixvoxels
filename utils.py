import cv2,torch
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import scipy.signal
import random
import time

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def visualize_4d_sigma(sigma, z_layers=10, sigma_max=2, cmap=cv2.COLORMAP_VIRIDIS, line_width=2):
    # sigma [H W ns]
    start_depth = 0.7
    z_nums = sigma.shape[2]
    interval = int(z_nums*(1-start_depth))//z_layers
    sigma_layers = []
    for i in range(int(start_depth*z_nums), int(start_depth * z_nums) + interval*z_layers, interval):
        sigma_layer = sigma[:, :, i]
        sigma_layer = sigma_layer/sigma_max
        sigma_layer = (255*sigma_layer).astype(np.uint8)
        sigma_layer = cv2.applyColorMap(sigma_layer, cmap)
        sigma_layers.append(sigma_layer)
        sigma_layers.append(np.zeros((sigma_layer.shape[0], line_width, 3)))
    sigma_layer = np.concatenate(sigma_layers[:-1], axis=1)
    return sigma_layer

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W, T)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = [cv2.applyColorMap(x[...,i], cmap) for i in range(depth.shape[2])]
    return x_, [mi,ma]

def visualize_depth_numpy_static(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)




__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()
    # return __LPIPS__[net_name](gt, im).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


import torch.nn as nn
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class TVLossVoxel(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLossVoxel,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # batch_size = x.size()[0]
        d_x = x.size()[1]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_d = self._tensor_size(x[:,1:,:,:])
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])

        d_tv = torch.pow((x[:,1:,:,:]-x[:,:d_x-1,:,:]),2).sum()
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(d_tv/count_d + h_tv/count_h + w_tv/count_w)

    def _tensor_size(self,t):
        return t.size()[0]*t.size()[1]*t.size()[2]*t.size()[3]

def entropy_loss(sigma_ray_wise):
    # sigma: Nr x ns x T
    ns = sigma_ray_wise.shape[1]
    if len(sigma_ray_wise.shape) == 3:
        sigma = sigma_ray_wise.transpose(1, 2).reshape(-1, ns)
    else:
        sigma = sigma_ray_wise
    sigma = torch.nn.functional.softmax(sigma, dim=-1)
    ent = - (sigma * (sigma+1e-6).log()).sum(dim=1).mean()
    return ent

def consistency_loss(input_diff, thresh=0.1, rgb=False):
    # sigma_diff Ns x T
    # rgb=True then Ns x T x 3
    if rgb:
        diff = input_diff.abs().mean(dim=-1).reshape(-1)
    else:
        diff = input_diff.abs().reshape(-1)
    diff = diff[diff<thresh]
    # beta = -np.log(thresh)
    # loss = (diff * ( ((diff+1e-6).log().abs())**beta )).mean()
    loss = (diff**2).mean()
    return loss

import plyfile
import skimage.measure
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


# utils for debug

# validate if the dynamic branch will affect the static branch
class DebugGradient:
    def __init__(self, opt):
        self.static_optimizer = opt
        self.last_static_params = None

    def check(self):
        params = []
        diffs = []
        idx = 0
        for group in self.static_optimizer.param_groups:
            for p in group['params']:
                params.append(p.data)
                if self.last_static_params is not None:
                    diff = p.data - self.last_static_params[idx]
                    diffs.append(diff)
                idx = idx + 1
        if self.last_static_params is not None:
            total_diff = sum([diff.abs().sum() for diff in diffs])
            print(total_diff)
        self.last_static_params = params


class TemporalSampler:
    def __init__(self, total_frames, sample_frames):
        self.total_frames = total_frames
        self.sample_frames = sample_frames
        assert self.total_frames % self.sample_frames == 0
        self.n_choices = self.total_frames//self.sample_frames
        self.choices = [list(range(i, i+self.n_choices)) for i in range(0, self.total_frames, self.n_choices)]
        self.samples = list(range(total_frames))

    def sample(self, rgb_train, iteration):
        if self.total_frames != self.sample_frames:
            samples = []
            for i in range(self.sample_frames):
                samples.append(random.choice(self.choices[i]))
        else:
            samples = self.samples
        if rgb_train is not None:
            return np.array(samples), rgb_train.transpose(0, 1)[samples].transpose(0, 1)
        else:
            return np.array(samples)

    def sample_continously_include(self, idx, interval=1, n_frames=None, total_frames=300):
        """
        Designed for evaluation, for estimating the static space using relative small frames
        to accelerate evaluation_path.
        """
        if idx - n_frames // 2 < 0:
            left = 0
        else:
            left = idx - n_frames // 2
        right = left + n_frames
        if right > total_frames:
            right = total_frames
            left = right - n_frames
        print(left, right)
        indices = torch.arange(left, right).cuda()
        return indices, idx - left

    def sample_evenly_include(self, idx, interval=1, n_frames=None, total_frames=300):
        """
        Designed for evaluation, for estimating the static space using relative small frames
        to accelerate evaluation_path.
        """
        assert total_frames % n_frames == 0
        group_id = idx % (total_frames // n_frames)
        indices = list(range(group_id, total_frames, total_frames // n_frames))
        return indices, idx // (total_frames // n_frames)

class ContinousEvenTemporalSampler(TemporalSampler):
    def __init__(self, total_frames, sample_frames):
        self.total_frames = total_frames
        self.sample_frames = sample_frames
        assert self.total_frames % self.sample_frames == 0
        self.n_choices = self.total_frames//self.sample_frames
        self.choices = [list(range(i, i+self.sample_frames)) for i in range(0, self.total_frames, self.sample_frames)]
        self.even_choices = [list(range(i, i+self.n_choices)) for i in range(0, self.total_frames, self.n_choices)]

    def sample(self, rgb_train, iteration):
        if np.random.rand() < 0.5:
            samples = random.choice(self.choices)
            return np.array(samples), rgb_train.transpose(0, 1)[samples].transpose(0, 1)
        else:
            samples = []
            for i in range(self.sample_frames):
                samples.append(random.choice(self.even_choices[i]))
            return np.array(samples), rgb_train.transpose(0, 1)[samples].transpose(0, 1)

class ContinousTemporalSampler(TemporalSampler):
    def __init__(self, total_frames, sample_frames):
        self.total_frames = total_frames
        self.sample_frames = sample_frames
        assert self.total_frames % self.sample_frames == 0
        self.n_choices = self.total_frames//self.sample_frames
        self.choices = [list(range(i, i+self.sample_frames)) for i in range(0, self.total_frames, self.sample_frames)]

    def sample(self, rgb_train, iteration):
        samples = random.choice(self.choices)
        return np.array(samples), rgb_train.transpose(0, 1)[samples].transpose(0, 1)

class ImportanceTemporalSampler(TemporalSampler):
    def __init__(self, total_frames, sample_frames):
        super(ImportanceTemporalSampler, self).__init__(total_frames, sample_frames)
        # self.total_frames = total_frames
        # self.sample_frames = sample_frames
        # assert self.total_frames % self.sample_frames == 0

    def sample(self, rgb_train, iteration):
        if np.random.rand() < 0.5:
            return super().sample(rgb_train, iteration)
        differences = (rgb_train[:,1:,:] - rgb_train[:,:-1,:]).abs().mean(dim=-1)
        _, indices = differences.sort(dim=1, descending=True)
        indices = indices + 1
        indices = torch.cat([torch.zeros_like(indices[:, 0:1]), indices], dim=1)
        indices = indices[:, :self.sample_frames]
        indices, _ = indices.sort(dim=1)
        return_rgb = torch.gather(rgb_train, dim=1, index=indices.unsqueeze(dim=-1).expand(-1, -1, rgb_train.shape[-1]))
        return indices, return_rgb

class CombImportanceTemporalSampler(TemporalSampler):
    def __init__(self, total_frames, sample_frames):
        # super(CombImportanceTemporalSampler, self).__init__(total_frames, sample_frames)
        self.even_sampler = TemporalSampler(total_frames, sample_frames//2)
        self.total_frames = total_frames
        self.sample_frames = sample_frames
        assert sample_frames % 2 == 0
        # assert self.total_frames % self.sample_frames == 0

    def sample(self, rgb_train, iteration):
        even_indices = self.even_sampler.sample(None, iteration)
        even_indices = torch.from_numpy(even_indices).unsqueeze(dim=0).expand(rgb_train.shape[0], -1).to(rgb_train).long()

        differences = (rgb_train[:,1:,:] - rgb_train[:,:-1,:]).abs().mean(dim=-1)
        _, indices = differences.sort(dim=1, descending=True)
        indices = indices + 1
        indices = torch.cat([torch.zeros_like(indices[:, 0:1]), indices], dim=1)
        indices = indices[:, :self.sample_frames//2]

        indices = torch.cat([even_indices, indices], dim=1)
        indices, _ = indices.sort(dim=1)
        return_rgb = torch.gather(rgb_train, dim=1, index=indices.unsqueeze(dim=-1).expand(-1, -1, rgb_train.shape[-1]))
        return indices, return_rgb

class TemporalWeightedSampler:
    def __init__(self, total_frames, sample_frames, temp_start, temp_end, total_iteration, replace, eval_sample_frames=None, method='mean'):
        self.total_frames = total_frames
        self.sample_frames = sample_frames
        self.eval_sample_frames = sample_frames if eval_sample_frames is None else eval_sample_frames
        assert self.total_frames % self.sample_frames == 0
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.total_iteration = total_iteration
        self.replace = replace
        self.method = method

    def get_temp(self, iteration):
        temp = iteration/self.total_iteration * (self.temp_end - self.temp_start) + self.temp_start
        return temp

    def sample(self, rgb_train, iteration):
        # rgb_train Nr x T x 3
        # t_start = time.time()
        temp = self.get_temp(iteration)
        if self.method == 'mean':
            mean = rgb_train.mean(dim=1)
            diff = (rgb_train - mean.unsqueeze(dim=1)).abs().mean(dim=2)
        elif self.method == 'median':
            median = rgb_train.median(dim=1)[0]
            diff = (rgb_train - median.unsqueeze(dim=1)).abs().mean(dim=2)
        elif self.method == 'diff':
            differences = (rgb_train[:, 1:, :] - rgb_train[:,:-1,:]).abs().mean(dim=2) # Nr x (T-1)
            diff = torch.cat([differences.mean(dim=1).unsqueeze(dim=1), differences], dim=1)
        else:
            raise NotImplementedError

        #     pass
        p = torch.nn.functional.softmax(diff/temp, dim=1)
        return_indices = torch.multinomial(p, self.sample_frames, replacement=bool(self.replace))
        # return_rgb = rgb_train[return_indices]
        return_rgb = torch.gather(rgb_train, dim=1, index=return_indices.unsqueeze(dim=-1).expand(-1, -1, rgb_train.shape[-1]))
        # t_end = time.time()
        # print(t_end-t_start)
        return return_indices, return_rgb

    def sample_continuously_include(self, idx, interval=1, n_frames=None, total_frames=300):
        """
        Designed for evaluation, for estimating the static space using relative small frames
        to accelerate evaluation_path.
        """
        if idx - n_frames//2 < 0:
            left = 0
        else:
            left = idx - n_frames//2
        right = left + n_frames
        if right > total_frames:
            right = total_frames
            left = right - n_frames
        print(left, right)
        indices = torch.arange(left, right).cuda()
        return indices, idx-left

    def sample_evenly_include(self, idx, interval=1, n_frames=None, total_frames=300):
        """
        Designed for evaluation, for estimating the static space using relative small frames
        to accelerate evaluation_path.
        """
        assert total_frames % n_frames == 0
        group_id = idx % (total_frames//n_frames)
        indices = list(range(group_id, total_frames, total_frames // n_frames))
        return indices, idx//(total_frames//n_frames)


def get_ray_weight(rgb_train):
    # rgb_train Nr T 3
    median = rgb_train.median(dim=1)[0]
    diff = ((rgb_train - median.unsqueeze(dim=1)).abs()).mean(dim=1).mean(dim=1)
    # diff = (diff**2)/(diff**2 + gamma**2)
    return diff


class WeightedRaySampler:
    def __init__(self, total, batch, weights):
        self.total = total
        self.large_batch = batch * 64
        self.batch = batch
        self.curr = total
        self.ids = None
        self.weights = weights

    def nextids(self, gamma=0.02):
        self.curr+=self.large_batch
        if self.curr + self.large_batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0

        if gamma == 0:
            weights = self.weights[self.ids]
        else:
            weights = (self.weights ** 2)/(self.weights**2 + gamma**2)
        weights = weights.cuda()
        ids = torch.multinomial(weights, self.batch, replacement=False)
        return self.ids[ids]

# FFT utils
@torch.no_grad()
def find_last_true(tensor, dim):
    '''
    assume tensor is H W F with F different frequencies
    '''
    new_tensor = torch.flip(tensor, dim=dim)
    indices = tensor.shape[dim] - new_tensor.argmax(dim=dim)
    return indices

def base_dir(dir_path):
    if dir_path.endswith('/'):
        return dir_path.split('/')[-2]
    return dir_path.split('/')[-1]


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self, gamma=None):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


class TicTok:
    def __init__(self):
        self.last = None

    def tik(self):
        self.current = time.time()
        if self.last is not None:
            self.interval = self.current - self.last
        else:
            self.interval = 0
        self.last = self.current

    def print(self, s):
        print(f'Time {s}: {self.interval}')

    def tik_print(self, s):
        self.tik()
        print(f'Time {s}: {self.interval}')

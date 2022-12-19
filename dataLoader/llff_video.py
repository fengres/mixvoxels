import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import math
import os
import cv2
from PIL import Image
from torchvision import transforms as T
from utils import get_ray_weight, SimpleSampler, mp
import time
from .ray_utils import *

# ========= Resize images with factors ==========
def _minify(basedir, factors=[], resolutions=[], prefix='frames_'):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, prefix+'{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, prefix+'{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output
    for r in factors + resolutions:
        if isinstance(r, int):
            name = prefix + '{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = prefix + '{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        frame_dir = os.path.join(basedir, name)
        if os.path.exists(frame_dir):
            continue
        os.makedirs(frame_dir)
        print('Minifying', r, basedir)

        for sub_dir in os.listdir(os.path.join(basedir, 'frames')):
            imgdir = os.path.join(basedir, 'frames', sub_dir)
            imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
            imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
            imgdir_orig = imgdir

            wd = os.getcwd()
            target_img_dir = os.path.join(frame_dir, sub_dir)
            if not os.path.exists(target_img_dir):
                os.makedirs(target_img_dir)

            check_output('cp {}/* {}'.format(imgdir_orig, target_img_dir), shell=True)

            ext = imgs[0].split('.')[-1]
            args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
            print(args)
            os.chdir(target_img_dir)
            check_output(args, shell=True)
            os.chdir(wd)

            if ext != 'png':
                check_output('rm {}/*.{}'.format(target_img_dir, ext), shell=True)
                print('Removed duplicates')
            print('Done')


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=1, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


#def get_spiral(c2ws_all, near_fars, rads_scale=0.5, N_views=120):
def get_spiral(c2ws_all, near_fars, rads_scale=0.5, N_views=120):

    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    print(close_depth, inf_depth)
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)

class SSDDataset(Dataset):
    def __init__(self, ssd_dir, *args, **kwargs):
        self.ssd_dir = ssd_dir
        self.split = kwargs['split']
        n_batch = 4096 if self.split == 'train' else 1
        if not os.path.exists(os.path.join(ssd_dir, self.split)):
            os.makedirs(ssd_dir, exist_ok=True)
            os.makedirs(os.path.join(ssd_dir, self.split))
            self.dataset = LLFFVideoDataset(*args, **kwargs)

            self.scene_bbox = self.dataset.scene_bbox
            self.near_far = self.dataset.near_far
            self.white_bg = self.dataset.white_bg
            self.img_wh = self.dataset.img_wh
            self.n_frames = self.dataset.n_frames
            self.directions = self.dataset.directions
            self.focal = self.dataset.focal
            self.n_rays = self.dataset.all_rays.shape[0]
            self.render_path = self.dataset.render_path
            torch.save({'scene_bbox': self.scene_bbox,
                        'near_far': self.near_far,
                        'white_bg': self.white_bg,
                        'img_wh': self.img_wh,
                        'n_frames': self.n_frames,
                        'directions': self.directions,
                        'focal': self.focal,
                        'n_rays': self.n_rays,
                        'render_path': self.render_path,
                        }, os.path.join(ssd_dir, 'meta.pt'))
            self.sampler = SimpleSampler(self.n_rays, n_batch)
            self.n_saving = int(math.ceil(self.n_rays / n_batch))
            self.make_ssd_storage()
            del self.dataset
        else:
            self.meta = torch.load(os.path.join(ssd_dir, 'meta.pt'))
            self.scene_bbox = self.meta['scene_bbox']
            self.near_far = self.meta['near_far']
            self.white_bg = self.meta['white_bg']
            self.img_wh = self.meta['img_wh']
            self.n_frames = self.meta['n_frames']
            self.directions = self.meta['directions']
            self.focal = self.meta['focal']
            self.n_rays = self.meta['n_rays']
            self.render_path = self.meta['render_path']
            self.n_saving = len(glob.glob(os.path.join(ssd_dir, '*.pth')))
        self.id = None
        self.curr = self.n_saving
        self.total = self.n_saving
        self.batch = 1

    def make_ssd_storage(self):
        for i in range(self.n_saving):
            ids = self.sampler.nextids()
            rays = self.dataset.all_rays[ids]
            rgbs = self.dataset.all_rgbs[ids]
            stds = self.dataset.all_stds[ids]
            torch.save({'rays': rays, 'rgbs': rgbs, 'stds': stds}, os.path.join(self.ssd_dir, self.split, '{}.pth'.format(i)))

    def next(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            if self.split == 'train':
                self.id = torch.LongTensor(np.random.permutation(self.total))
            else:
                self.id = torch.arange(self.total).long()
            self.curr = 0
        if self.batch == 1:
            data = torch.load(os.path.join(self.ssd_dir, self.split, '{}.pth'.format(self.id[self.curr])))
        else:
            data = [torch.load(os.path.join(self.ssd_dir, self.split, '{}.pth'.format(self.id[curr])))
                    for curr in range(self.curr, self.curr+self.batch)]
            keys = data[0].keys()
            tdata = {}
            for k in keys:
                tdata[k] = torch.cat([d[k] for d in data], dim=0)
            data = tdata
        return data

    def get(self, idx):
        assert self.split == 'test'
        data = torch.load(os.path.join(self.ssd_dir, self.split, '{}.pth'.format(idx)))
        return data

    def __len__(self):
        return self.n_rays

    def reset(self):
        self.curr = self.n_saving
        self.total = self.n_saving
        self.batch = 1

class LLFFVideoDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_id=[0,], n_frames=100,
                 render_views=120, tmp_path='memory', scene_box=[-3.0, -1.67, -1.2], temporal_variance_threshold=1000,
                 frame_start=0, near=0.0, far=1.0, diffuse_kernel=0):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = datadir
        self.split = split
        self.hold_id = hold_id
        self.is_stack = is_stack
        self.downsample = downsample
        self.diffuse_kernel = diffuse_kernel
        self.define_transforms()
        self.render_views = render_views
        self.tmp_path = tmp_path
        self.temporal_variance_threshold = temporal_variance_threshold
        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.n_frames = n_frames
        self.frame_start = frame_start
        self.read_meta()
        self.white_bg = False

        # self.near_far = [np.min(self.near_fars[:,0]),np.max(self.near_fars[:,1])]
        self.near_far = [near, far]
        # self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        # TODO
        # self.scene_bbox = torch.tensor([[-2.0, -2.0, -1.0], [2.0, 2.0, 1.0]])
        # self.scene_bbox = torch.tensor([[-1.5, -3.0, -1.0], [1.5, 3.0, 1.0]])
        if scene_box is None:
            scene_box = [-3.0, -1.67, -1.2]
        self.scene_bbox = torch.tensor([scene_box, list(map(lambda x: -x, scene_box))])
        # self.scene_bbox = torch.tensor([[-4.0, -3.0, -2.0], [4.0, 3.0, 2.0]])
        # self.scene_bbox = torch.tensor([-1.67, -1.5, -1.0], [1.67, 1.5, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_meta(self):

        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        if self.downsample == 1.0:
            self.video_paths = sorted(glob.glob(os.path.join(self.root_dir, 'frames/*')))
        else:
            _minify(self.root_dir, factors=[int(self.downsample), ])
            self.video_paths = sorted(glob.glob(os.path.join(self.root_dir, 'frames_{}/*'.format(int(self.downsample)))))
            self.video_paths = list(filter(lambda x: not x.endswith('.npy'), self.video_paths))
        _calc_std(os.path.join(self.root_dir, 'frames'+('' if self.downsample == 1.0 else '_{}'.format(int(self.downsample)))),
                  os.path.join(self.root_dir, 'stds'+('' if self.frame_start==0 else str(self.frame_start))+('' if self.downsample == 1.0 else '_{}'.format(int(self.downsample)))),
                  frame_start=self.frame_start, n_frame=self.n_frames)
        if 'coffee_martini' in self.root_dir:
            print('====================deletting unsynchronized video==============')
            poses_bounds = np.concatenate([poses_bounds[:12], poses_bounds[13:]], axis=0)
            self.video_paths.pop(12)
        # load full resolution image then resize
        if self.split in ['train', 'test']:
            assert len(poses_bounds) == len(self.video_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        hwf = poses[:, :, -1]

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor

        # build rendering path
        N_views, N_rots = self.render_views, 2
        tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)

        self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views)

        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)

        average_pose = average_poses(self.poses)
        dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)


        i_test = np.array(self.hold_id)
        video_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))

        # use first N_images-1 to train, the LAST is val
        self.all_rays = []
        self.all_rgbs = []
        self.all_stds_without_diffusion = []
        self.all_rays_weight = []
        self.all_stds = []
        for i in video_list:
            video_path = self.video_paths[i]
            c2w = torch.FloatTensor(self.poses[i])
            frames_paths = sorted(os.listdir(video_path))[self.frame_start:self.frame_start+self.n_frames][::(self.n_frames//self.n_frames)]
            std_path = video_path.replace('frames', 'stds'+('' if self.frame_start==0 else str(self.frame_start))) + '_std.npy'
            assert os.path.isdir(video_path)
            assert os.path.isfile(std_path)
            frames = [Image.open(os.path.join(video_path, image_id)).convert('RGB') for image_id in frames_paths]
            # t1 = time.time()
            # frames = mp(func=lambda x: Image.open(x).convert('RGB'),
            #             parameters=[os.path.join(video_path, image_id) for image_id in frames_paths],
            #             n_workers=8)
            # t2 = time.time()
            # print(t2-t1)
            if self.downsample != 1.0:
                if list(frames[0].size) != list(self.img_wh):
                    frames = [img.resize(self.img_wh, Image.LANCZOS) for img in frames]

            frames = [self.transform(img) for img in frames]  # (T, 3, h, w)

            frames = [img.view(3, -1).permute(1, 0) for img in frames]  # (T, h*w, 3) RGB
            frames = torch.stack(frames, dim=1) # hw T 3

            if self.diffuse_kernel > 0:
                std_frames_without_diffuse = np.load(std_path)
                std_frames = diffuse(std_frames_without_diffuse, self.diffuse_kernel)
            else:
                std_frames_without_diffuse = None
                std_frames = np.load(std_path)
            std_frames = torch.from_numpy(std_frames).reshape(-1)
            if std_frames_without_diffuse is not None:
                std_frames_without_diffuse = torch.from_numpy(std_frames_without_diffuse).reshape(-1)

            rays_weight = get_ray_weight(frames)

            self.all_rays_weight.append(rays_weight.half())
            self.all_rgbs += [frames.half()]
            self.all_stds += [std_frames.half()]
            if std_frames_without_diffuse is not None:
                self.all_stds_without_diffusion += [std_frames_without_diffuse.half()]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
            # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

            self.all_rays += [torch.cat([rays_o, rays_d], 1).half()]  # (h*w, 6)

        if not self.is_stack:
            self.all_rays_weight = torch.cat(self.all_rays_weight, dim=0) # (Nr)
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, T, 3)
            self.all_stds = torch.cat(self.all_stds, 0)
            if len(self.all_stds_without_diffusion) > 0:
                self.all_stds_without_diffusion = torch.cat(self.all_stds_without_diffusion, 0)
            # calc the dynamic data
            dynamic_mask = self.all_stds > self.temporal_variance_threshold
            self.dynamic_rays = self.all_rays[dynamic_mask]
            self.dynamic_rgbs = self.all_rgbs[dynamic_mask]
            self.dynamic_stds = self.all_stds[dynamic_mask]
        else:
            self.all_rays_weight = torch.stack(self.all_rays_weight, dim=0) # (Nr)
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            T = self.all_rgbs[0].shape[1]
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], T, 3)  # (len(self.meta['frames]),h,w, T, 3)
            self.all_stds = torch.stack(self.all_stds, 0).reshape(-1,*self.img_wh[::-1])
            if len(self.all_stds_without_diffusion) > 0:
                self.all_stds_without_diffusion = torch.stack(self.all_stds_without_diffusion, 0).reshape(-1,*self.img_wh[::-1])

    def shift_stds(self):
        self.all_stds = self.all_stds_without_diffusion
        return self.all_stds

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample


def _calc_std(frame_path_root, std_path_root, frame_start=0, n_frame=300):
    # if frame_start != 0:
    #     std_path_root = std_path_root+str(frame_start)
    if os.path.exists(std_path_root):
        return
    os.makedirs(std_path_root)
    print(frame_path_root)
    for child in os.listdir(frame_path_root):
        if not child.startswith('cam'):
            continue
        frame_path = os.path.join(frame_path_root, child)
        std_path = os.path.join(std_path_root, child)
        frame_paths = sorted([os.path.join(frame_path, fn) for fn in os.listdir(frame_path)])[frame_start:frame_start+n_frame]

        frames = []
        for fp in frame_paths:
            frame = Image.open(fp).convert('RGB')
            frame = np.array(frame, dtype=np.float) / 255.
            frames.append(frame)
        frame = np.stack(frames, axis=0)
        std_map = frame.std(axis=0).mean(axis=-1)
        std_map_blur = (cv2.GaussianBlur(std_map, (31, 31), 0)).astype(np.float)
        np.save(std_path + '_std.npy', std_map_blur)
        print(frame_paths)
        # print(frame_path)
        # print(std_path + '_std.npy', frames[0].shape, std_map_blur.shape, std_map.shape)

def diffuse(std, kernel):
    h, w = std.shape
    oh, ow = h, w
    add_h = kernel - (h % kernel)
    add_w = kernel - (w % kernel)
    if add_h > 0:
        std = np.concatenate((std, np.zeros((add_h , w))), axis=0)
    if add_w > 0:
        std = np.concatenate((std, np.zeros((h+add_h, add_w))), axis=1)
    h, w = std.shape
    std = std.reshape(h//kernel, kernel, w//kernel, kernel).transpose(0, 2, 1, 3).max(axis=-1).max(axis=-1)
    std = std.reshape(h//kernel, 1, w//kernel, 1).repeat(kernel, axis=1).repeat(kernel, axis=3)
    std = std.reshape(h, w)[:oh, :ow]
    return std

if __name__ == '__main__':
    _minify(basedir=os.path.expanduser('~/project/nerf/data/coffee_martini'), factor=[4,])
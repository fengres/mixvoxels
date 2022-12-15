import random
import subprocess
import shlex
from skimage.metrics import structural_similarity as sk_ssim
import numpy as np
import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
from argparse import Namespace
import multiprocessing

def cuda_empty():
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

def cat_dic_list(list_of_dics, cat_dim=0):
    # note list_of_dicts should have at least 1 elements
    keys = list_of_dics[0].keys()
    ret_values = {}
    for k in keys:
        values = [d[k] for d in list_of_dics]
        if None in values:
            values = None
        elif isinstance(values[0], (float, int)):
            values = np.array(values).mean()
        elif len(values[0].shape) == 0:
            values = sum(values) / len(values)
        else:
            values = torch.cat(values, dim=cat_dim)
        ret_values[k] = values
    return ret_values


def OctreeRender_trilinear_fast(rays, tensorf, std_train, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda', rgb_train=None,
                                use_time='all', time=None, temporal_indices=None, with_grad=True, simplify=False, **kwargs):
    # tiktok = TicTok()
    N_rays_all = rays.shape[0]
    return_values = []
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        # tiktok.tik()
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        current_values = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray,
                                 N_samples=N_samples, rgb_train=rgb_train, temporal_indices=temporal_indices,
                                 std_train=std_train, **kwargs)
        # tiktok.tik_print('RENDER/rendering')
        if not with_grad:
            for k in current_values.keys():
                if 'map' not in k:
                    current_values[k] = None
                else:
                    if 'render_path' not in kwargs:
                        current_values[k] = current_values[k].cpu()
                    else:
                        current_values[k] = current_values[k]
        return_values.append(current_values)
        # tiktok.tik_print('RENDER/post')
    return cat_dic_list(return_values)


@torch.no_grad()
def evaluation(test_dataset, tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda', simplify=False):
    PSNRs, PSNRs_pf, PSNRs_STA, rgb_maps, depth_maps = [], [], [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])

        retva = renderer(rays, tensorf,  std_train=None, chunk=args.batch_size//2, N_samples=N_samples, ndc_ray=ndc_ray, white_bg = white_bg, device=device, with_grad=False,
                         simplify=simplify)
        retva = Namespace(**retva)

        retva.rgb_map = retva.rgb_map.clamp(0.0, 1.0)
        retva.comp_rgb_map = retva.comp_rgb_map.clamp(0.0, 1.0)
        retva.static_rgb_map = retva.static_rgb_map.clamp(0.0, 1.0)

        retva.rgb_map = retva.rgb_map.reshape(H, W, test_dataset.n_frames, 3).cpu()
        retva.depth_map = retva.depth_map.reshape(H, W, test_dataset.n_frames).cpu()

        retva.comp_rgb_map = retva.comp_rgb_map.reshape(H, W, test_dataset.n_frames, 3).cpu()
        retva.comp_depth_map = retva.comp_depth_map.reshape(H, W, test_dataset.n_frames).cpu()

        retva.static_rgb_map = retva.static_rgb_map.reshape(H, W, 3).cpu()
        retva.static_depth_map = retva.static_depth_map.reshape(H, W).cpu()

        retva.depth_map, _ = visualize_depth_numpy(retva.depth_map.numpy(),near_far)
        retva.comp_depth_map, _ = visualize_depth_numpy(retva.comp_depth_map.numpy(),near_far)
        retva.static_depth_map, _ = visualize_depth_numpy_static(retva.static_depth_map.numpy(),near_far)

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, test_dataset.n_frames, 3)
            gt_static_rgb = gt_rgb.mean(dim=2)
            # gt_rgb = gt_rgb[:,:,0,:]
            per_frame_loss = ((retva.comp_rgb_map - gt_rgb) ** 2).mean(dim=0).mean(dim=0).mean(dim=1)
            loss = per_frame_loss.mean()
            loss_static = torch.mean((retva.static_rgb_map - gt_static_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))
            PSNRs_pf.append((-10.0 * np.log(per_frame_loss.detach().cpu().numpy()) / np.log(10.0)).mean())
            PSNRs_STA.append(-10.0 * np.log(loss_static.item()) / np.log(10.0))

            for i_time in range(0, retva.comp_rgb_map.shape[2], 10):
                # ssim = rgb_ssim(retva.comp_rgb_map[:,:,i_time,:], gt_rgb[:,:,i_time,:], 1)
                ssim = sk_ssim(retva.comp_rgb_map[:,:,i_time,:].cpu().detach().numpy(), gt_rgb[:,:,i_time,:].cpu().detach().numpy(), multichannel=True)
                l_a = rgb_lpips(gt_rgb[:,:,i_time,:].numpy(), retva.comp_rgb_map[:,:,i_time,:].numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb[:,:,i_time,:].numpy(), retva.comp_rgb_map[:,:,i_time,:].numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)
            print('=================LPIPS==================')
            print(l_alex)
            print(l_vgg)

        for rgb_map, depth_map, name, is_video in [(retva.static_rgb_map, retva.static_depth_map, 'static', False),
                                                   (retva.rgb_map, retva.depth_map, 'moving', True),
                                                   (retva.comp_rgb_map, retva.comp_depth_map, 'comp', True)]:
            rgb_map = (rgb_map.numpy() * 255).astype('uint8')
            if is_video:
                rgb_maps = [rgb_map[:,:,i,:] for i in range(rgb_map.shape[2])]
                depth_maps = depth_map
                # if savePath is not None:
                # imageio.mimwrite(f'{savePath}/{prtx}_{name}_video.mp4', np.stack(rgb_maps), fps=30/(300/len(rgb_maps)), quality=10)
                imageio.mimwrite(f'{savePath}/{prtx}_{name}_video.mp4', np.stack(rgb_maps), fps=30, quality=10)
                # imageio.mimwrite(f'{savePath}/{prtx}_{name}_depthvideo.mp4', np.stack(depth_maps), fps=30/(300/len(rgb_maps)), quality=10)
                imageio.mimwrite(f'{savePath}/{prtx}_{name}_depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)
                rgb_depth_maps = [np.concatenate((rgb_map[:, :, i, :], depth_map[i]), axis=1) for i in range(rgb_map.shape[2])]
                # imageio.mimwrite(f'{savePath}/{prtx}_{name}_rgbdepthvideo.mp4', np.stack(rgb_depth_maps), fps=30 / (300 / len(rgb_maps)), quality=10)
                imageio.mimwrite(f'{savePath}/{prtx}_{name}_rgbdepthvideo.mp4', np.stack(rgb_depth_maps), fps=30, quality=10)
            else:
                imageio.imwrite(f'{savePath}/{prtx}_{name}_rgb.png', rgb_map)
                imageio.imwrite(f'{savePath}/{prtx}_{name}_depth.png', depth_map)
                imageio.imwrite(f'{savePath}/{prtx}_{name}_rgbdepth.png', np.concatenate([rgb_map, depth_map], axis=1))
        # calculate flip value
        gt_video = os.path.join(args.datadir, 'frames_{}'.format(int(args.downsample_train)), 'cam00')
        output_path = os.path.join(savePath, f'{prtx}_comp_video.mp4')
        try:
            flip_output = subprocess.check_output(shlex.split(
                f'python eval/main.py --output {output_path} --gt {gt_video} --downsample {int(args.downsample_train)} --tmp_dir /tmp/{args.expname} --start_frame {args.frame_start} --end_frame {args.frame_start + args.n_frames}'
            )).decode()
            flip_output = eval('{'+flip_output.split('{')[-1])['Mean']
        except:
            flip_output = 0.0
        # calculate jod
        try:
            jodcmd = f'python eval/main_jod.py --output {output_path} --gt {gt_video} --downsample {int(args.downsample_train)} --tmp_dir /tmp/{args.expname} --start_frame {args.frame_start} --end_frame {args.frame_start + args.n_frames}'
            print(jodcmd)
            jod_output = subprocess.check_output(shlex.split(
                jodcmd
            )).decode()
            jod_output = float(jod_output)
        except:
            jod_output = 0.0

    # if PSNRs:
    psnr = np.mean(np.asarray(PSNRs))
    psnr_pf = np.mean(np.asarray(PSNRs_pf))
    psnr_sta = np.mean(np.asarray(PSNRs_STA))

    ssim = np.mean(np.asarray(ssims))
    dssim = np.mean((1.-np.asarray(ssims))/2.)
    l_a = np.mean(np.asarray(l_alex))
    l_v = np.mean(np.asarray(l_vgg))
    np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, psnr_pf, psnr_sta, ssim, l_a, l_v, flip_output, jod_output]))
    print(f'SSIM: {ssim}, DSSIM: {dssim}')
    print(f'LPISIS AlexNet: {l_a}')
    print(f'LPISIS VGGNet: {l_v}')
    print(f'FLIP: {flip_output}')
    print(f'JOD: {jod_output}')
    total_results = {
        'ssim': ssim,
        'dssim': dssim,
        'lpisis_alex': l_a,
        'lpisis_vgg': l_v,
        'flip': flip_output,
        'jod': jod_output,
    }

    print('PSNR:{:.6f}, PSNR_PERFRAME:{:.6f}, PSNR_STA:{:.6f}'.format(psnr, psnr_pf, psnr_sta))
    return PSNRs, PSNRs_STA, total_results

@torch.no_grad()
def evaluation_path(test_dataset, tensorf, args, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, device='cuda', temporal_sampler=None, start_idx=0):

    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    W, H = test_dataset.img_wh
    n_frames = test_dataset.n_frames
    n_train_frames = temporal_sampler.sample_frames
    camera_per_frame = [int(i/n_frames*len(c2ws)) for i in range(n_frames)]
    frames_per_camera = [[] for i in range(len(c2ws))]
    for i_frame, i_camera in enumerate(camera_per_frame):
        frames_per_camera[i_camera].append(i_frame)

    tictok = TicTok()
    processings = []
    for idx, c2w in tqdm(enumerate(c2ws)):
        if idx < start_idx:
            continue

        tictok.tik()
        temporal_indices = torch.arange(n_frames).long().cuda()
        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
        tictok.tik_print('pre-render')
        retva = renderer(rays, tensorf, std_train=None, chunk=args.batch_size*4, N_samples=N_samples,
                         ndc_ray=ndc_ray, white_bg = white_bg, device=device, with_grad=False,
                         simplify=True, temporal_indices=temporal_indices, render_path=True)
        tictok.tik_print('render')
        retva = Namespace(**retva)

        # retva.rgb_map = retva.rgb_map.clamp(0.0, 1.0)
        nodepth=True
        retva.comp_rgb_map = retva.comp_rgb_map.clamp(0.0, 1.0)
        proc = multiprocessing.Process(target=write_video, args=(retva.comp_rgb_map.cpu(), savePath, idx, (None if nodepth else retva.comp_depth_map.cpu()), 30, 10,
                                                                 H, W, n_train_frames, near_far))
        processings.append(proc)
        proc.start()
        tictok.tik_print('post-render4')

    for proc in processings:
        proc.join()

def write_video(comp_rgb_map, savePath, idx, comp_depth_map=None, fps=30, quality=10, H=None, W=None, n_train_frames=None,
                near_far=None):
    # retva.rgb_map, retva.depth_map = retva.rgb_map.reshape(H, W, n_train_frames, 3).cpu(), retva.depth_map.reshape(H, W, n_train_frames).cpu()
    comp_rgb_map = comp_rgb_map.reshape(H, W, n_train_frames, 3).cpu()
    if comp_depth_map is not None:
        comp_depth_map = comp_depth_map.reshape(H, W, n_train_frames).cpu()
    # retva.static_rgb_map, retva.static_depth_map = retva.static_rgb_map.reshape(H, W, 3).cpu(), retva.static_depth_map.reshape(H, W).cpu()

    if comp_depth_map is not None:
        # retva.depth_map = np.stack(visualize_depth_numpy(retva.depth_map[:,:,:].numpy(), near_far)[0], axis=2)
        comp_depth_map = np.stack(visualize_depth_numpy(comp_depth_map[:, :, :].numpy(), near_far)[0], axis=2)
        # retva.static_depth_map, _ = visualize_depth_numpy_static(retva.static_depth_map.numpy(), near_far)

    # H W T 3
    # retva.rgb_map = (retva.rgb_map.numpy() * 255).astype('uint8')
    comp_rgb_map = (comp_rgb_map.numpy() * 255).astype('uint8').transpose(2, 0, 1, 3)
    if comp_depth_map is not None:
        comp_depth_map = comp_depth_map.transpose(2, 0, 1, 3)
    # retva.static_rgb_map = (retva.static_rgb_map.numpy() * 255).astype('uint8')
    # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
    imageio.mimwrite(f'{savePath}/cam_{idx}_comp_video.mp4', comp_rgb_map, fps=30, quality=quality)
    if comp_depth_map is not None:
        imageio.mimwrite(f'{savePath}/cam_{idx}_comp_depthvideo.mp4', comp_depth_map, fps=30, quality=quality)

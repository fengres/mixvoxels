import math
import os
import time
from argparse import Namespace

import torch
from tqdm.auto import tqdm
import pdb

import utils
from opt import config_parser
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from functools import partial


import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from dynamics import Dynamics

from dataLoader import dataset_dict
import sys
from torch.profiler import profile, record_function, ProfilerActivity


# torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast

def cuda_empty():
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)

@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True,
                           n_frames=args.n_frames, render_views=args.render_views, scene_box=args.scene_box,
                           frame_start=args.frame_start, near=args.near, far=args.far, diffuse_kernel=args.diffuse_kernel)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if args.temporal_sampler == 'simple':
        temporal_sampler = TemporalSampler(args.n_frames, args.n_train_frames)
    elif args.temporal_sampler == 'weighted':
        temporal_sampler = TemporalWeightedSampler(args.n_frames, args.n_train_frames, args.temperature_start,
                                                   args.temperature_end, args.n_iters, args.temporal_sampler_replace)

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    # kwargs.update({'device': device})
    # tensorf = eval(args.model_name)(**kwargs)
    tensorf = eval(args.model_name)(args, kwargs['aabb'], kwargs['gridSize'], device,
                                    density_n_comp=kwargs['density_n_comp'], appearance_n_comp=kwargs['appearance_n_comp'],
                                    app_dim=args.data_dim_color, near_far=kwargs['near_far'],
                                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre,
                                    density_shift=args.density_shift, distance_scale=args.distance_scale,
                                    rayMarch_weight_thres=args.rm_weight_mask_thre,
                                    rayMarch_weight_thres_static=args.rm_weight_mask_thre_static,
                                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
                                    featureC=args.featureC, step_ratio=kwargs['step_ratio'], fea2denseAct=args.fea2denseAct,
                                    den_dim=args.data_dim_density, densityMode=args.densityMode, featureD=args.featureD,
                                    rel_pos_pe=args.rel_pos_pe, n_frames=args.n_frames,
                                    amp=args.amp, temporal_variance_threshold=args.temporal_variance_threshold,
                                    n_frame_for_static=args.n_frame_for_static,
                                    dynamic_threshold=args.dynamic_threshold, n_time_embedding=args.n_time_embedding,
                                    static_dynamic_seperate=args.static_dynamic_seperate,
                                    zero_dynamic_sigma=args.zero_dynamic_sigma,
                                    zero_dynamic_sigma_thresh=args.zero_dynamic_sigma_thresh,
                                    sigma_static_thresh=args.sigma_static_thresh,
                                    n_train_frames=args.n_train_frames,
                                    net_layer_add=args.net_layer_add,
                                    density_n_comp_dynamic=args.n_lamb_sigma_dynamic,
                                    app_n_comp_dynamic=args.n_lamb_sh_dynamic,
                                    interpolation=args.interpolation,
                                    dynamic_granularity=args.dynamic_granularity,
                                    point_wise_dynamic_threshold=args.point_wise_dynamic_threshold,
                                    static_point_detach=args.static_point_detach,
                                    dynamic_pool_kernel_size=args.dynamic_pool_kernel_size,
                                    time_head=args.time_head, filter_thresh=args.filter_threshold,
                                    static_featureC=args.static_featureC,
                                    )
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.dense_alpha:
        with autocast(enabled=bool(args.amp)):
            alpha, sigma = tensorf.getTemporalDenseAlpha(gridSize=(300,150,150))
        convert_sdf_samples_to_ply(alpha.cpu()[...,150], f'{args.ckpt[:-3]}.ply', bbox=tensorf.aabb.cpu(), level=0.005)
        alpha = alpha.cpu().numpy()
        np.save(os.path.join(logfolder, 'dense_alpha.npy'), alpha)

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True,
                                n_frames=args.n_frames, scene_box=args.scene_box, temporal_variance_threshold=args.temporal_variance_threshold,
                                frame_start=args.frame_start, near=args.near, far=args.far, diffuse_kernel=args.diffuse_kernel)
        with autocast(enabled=bool(args.amp)):
            PSNRs_test, PSNRs_STA_test, all_metrics = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                    N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')
        print(f'======> {args.expname} test all psnr sta: {np.mean(PSNRs_STA_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        with autocast(enabled=bool(args.amp)):
            PSNRs_test, PSNRs_STA_test, all_metrics = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                    N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, simplify=(args.n_frames>0))
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')
        print(f'======> {args.expname} test all psnr sta: {np.mean(PSNRs_STA_test)} <========================')

    if args.render_path:
        cuda_empty()
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        with torch.no_grad():
            with autocast(enabled=bool(args.amp)):
                evaluation_path(test_dataset, tensorf, args, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
                                temporal_sampler=temporal_sampler,
                                start_idx=args.render_path_start)

def train_dynamics(args, tensorf, allrays, allrgbs, allstds, ndc_ray, nSamples, scaler, device, iter_ratio=1):
    DynamicCriterion = Dynamics(args, device, use_volumetric_render=args.dynamic_use_volumetric_render)
    dy_optimizer = torch.optim.Adam(tensorf.get_dynamic_optparam_groups(args.lr_init), betas=(0.9, 0.99))
    dy_lr_factor = args.lr_decay_target_ratio ** (1 / (args.n_dynamic_iters*iter_ratio))
    pbar_dynamic = tqdm(range(args.n_dynamic_iters*iter_ratio), miniters=args.progress_refresh_rate, file=sys.stdout)
    dy_Sampler = SimpleSampler(allrays.shape[0], args.batch_size * 10)
    tvreg = TVLoss() if args.model_name == 'TensorVMSplit' else TVLossVoxel()
    for iteration in pbar_dynamic:
        ray_idx = dy_Sampler.nextids()
        rays_train, rgb_train, variance_train = allrays[ray_idx].to(device).float(), allrgbs[ray_idx].to(device).float(), allstds[ray_idx].to(device).float()
        # rgb_map, alphas_map, depth_map, weights, uncertainty
        dy_optimizer.zero_grad()
        with autocast(enabled=bool(args.amp)):
            retva = tensorf.forward_dynamics(rays_train.to(device), is_train=True, variance_train=variance_train,
                                                                      ndc_ray=ndc_ray, N_samples=nSamples,
                                                                      rgb_train=rgb_train)
            dynamic_prediction_loss = DynamicCriterion.calculate_loss(*retva)
            # loss_tv = tensorf.TV_loss_dynamic(tvreg) * 2
            loss_tv = 0
            total_loss = dynamic_prediction_loss + loss_tv
            DynamicCriterion.compute_metrics()
        if args.amp:
            scaler.scale(total_loss).backward()
            scaler.step(dy_optimizer)
            scaler.update()
        else:
            total_loss.backward()
            dy_optimizer.step()
        for param_group in dy_optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * dy_lr_factor
        current_lr = dy_optimizer.param_groups[0]['lr']
        if iteration % args.progress_refresh_rate == 0:
            pbar_dynamic.set_description(f'Iteration {iteration:05d}: '
                                         + f' loss = {total_loss.item():.6f}'
                                         + f' lr = {current_lr:.6f}')
    DynamicCriterion.print_metrics()
    cuda_empty()


# evaluation
@torch.no_grad()
def eval_dynamics(args, tensorf, test_dataset, ndc_ray, nSamples, device):
    DynamicCriterion = Dynamics(args, device, use_volumetric_render=args.dynamic_use_volumetric_render)
    for idx, samples in tqdm(enumerate(test_dataset.all_rays), file=sys.stdout):
        rays_test = samples.reshape(-1, samples.shape[-1]).to(device).contiguous()
        rgb_test = test_dataset.all_rgbs[idx].reshape(-1, args.n_frames, 3).to(device).contiguous()
        std_test = test_dataset.all_stds[idx].reshape(-1).to(device).contiguous()

        # rgb_map, alphas_map, depth_map, weights, uncertainty
        N_rays_all = rays_test.shape[0]
        all_dynamics, all_dynamics_supervision, all_max_dynamics = [], [], []
        chunk = 256
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            with autocast(enabled=bool(args.amp)):
                dynamics, dynamics_supervision, max_dynamics = tensorf.forward_dynamics(rays_test[chunk_idx * chunk:(chunk_idx + 1) * chunk], is_train=False,
                                                                          ndc_ray=ndc_ray, N_samples=nSamples,
                                                                          rgb_train=rgb_test[chunk_idx * chunk:(chunk_idx + 1) * chunk], variance_train=std_test[chunk_idx * chunk:(chunk_idx + 1) * chunk])
                all_dynamics.append(dynamics)
                all_dynamics_supervision.append(dynamics_supervision)
                all_max_dynamics.append(max_dynamics)

        all_dynamics = torch.cat(all_dynamics, dim=0)
        all_dynamics_supervision = torch.cat(all_dynamics_supervision, dim=0)
        all_max_dynamics = torch.cat(all_max_dynamics, dim=0)
        dynamic_prediction_loss = DynamicCriterion.calculate_loss(all_dynamics, all_dynamics_supervision, all_max_dynamics)
        print('test loss: {:.4f}'.format(dynamic_prediction_loss.item()))
        DynamicCriterion.compute_metrics()
        DynamicCriterion.print_metrics()
    cuda_empty()

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    time_dataset_start = time.time()
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False,
                            n_frames=args.n_frames, scene_box=args.scene_box, temporal_variance_threshold=args.temporal_variance_threshold,
                            frame_start=args.frame_start, near=args.near, far=args.far, diffuse_kernel=args.diffuse_kernel)
    time_dataset_end = time.time()
    print(f'Loading Train Dataset: {time_dataset_end-time_dataset_start}s')
    time_dataset_start = time_dataset_end
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True,
                           n_frames=args.n_frames, render_views=args.render_views, scene_box=args.scene_box,
                           temporal_variance_threshold=args.temporal_variance_threshold,
                           frame_start=args.frame_start, near=args.near, far=args.far, diffuse_kernel=args.diffuse_kernel)
    time_dataset_end = time.time()
    print(f'Loading Test Dataset: {time_dataset_end-time_dataset_start}s')

    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    args.expname = os.path.basename(args.config.split('.')[0])
    # if args.meta_config is not None:
    #     args.expname = args.expname + '_' + os.path.basename(args.meta_config.split('.')[0])
    # args.expname = '_'.join([args.expname, utils.base_dir(args.datadir), str(args.downsample_train)])
    logfolder = '{}/{}'.format(args.basedir, args.expname)
    print(args.expname, logfolder)

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
    print(f'Sampling points: {nSamples}')

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device,
                       'amp': args.amp,
                       'temporal_variance_threshold': args.temporal_variance_threshold,
                       'dynamic_threshold': args.dynamic_threshold,
                       'n_time_embedding': args.n_time_embedding,
                       'static_dynamic_seperate': args.static_dynamic_seperate,
                       'n_frames': args.n_frames,
                       'dynamic_use_volumetric_render': args.dynamic_use_volumetric_render,
                       'sigma_static_thresh': args.sigma_static_thresh,
                       'zero_dynamic_sigma': args.zero_dynamic_sigma,
                       'zero_dynamic_sigma_thresh': args.zero_dynamic_sigma_thresh,
                       'n_train_frames': args.n_train_frames,
                       'net_layer_add': args.net_layer_add,
                       })
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(args, aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    rayMarch_weight_thres=args.rm_weight_mask_thre,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct,
                    den_dim=args.data_dim_density, densityMode=args.densityMode, featureD=args.featureD, rel_pos_pe=args.rel_pos_pe, n_frames=args.n_frames,
                    amp=args.amp, temporal_variance_threshold=args.temporal_variance_threshold, n_frame_for_static=args.n_frame_for_static,
                    dynamic_threshold=args.dynamic_threshold, n_time_embedding=args.n_time_embedding, static_dynamic_seperate=args.static_dynamic_seperate,
                    dynamic_use_volumetric_render=args.dynamic_use_volumetric_render, zero_dynamic_sigma=args.zero_dynamic_sigma,
                    zero_dynamic_sigma_thresh=args.zero_dynamic_sigma_thresh, sigma_static_thresh=args.sigma_static_thresh, n_train_frames=args.n_train_frames,
                    net_layer_add=args.net_layer_add,
                    density_n_comp_dynamic=args.n_lamb_sigma_dynamic,
                    app_n_comp_dynamic=args.n_lamb_sh_dynamic,
                    interpolation=args.interpolation,
                    dynamic_granularity=args.dynamic_granularity,
                    point_wise_dynamic_threshold=args.point_wise_dynamic_threshold,
                    static_point_detach=args.static_point_detach,
                    dynamic_pool_kernel_size=args.dynamic_pool_kernel_size,
                    time_head=args.time_head,
                    filter_thresh=args.filter_threshold,
                    static_featureC=args.static_featureC,
                    )

    grad_vars = tensorf.get_optparam_groups(args.lr_dynamic_init, args.lr_dynamic_basis)
    static_grad_vars = tensorf.get_static_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    opt_proto = {
        'sgd': torch.optim.SGD,
        'adam': partial(torch.optim.Adam, betas=(0.9, 0.99)),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.99)),
        'rmsp': partial(torch.optim.RMSprop, momentum=0.0),
    }[args.optimizer]
    optimizer = opt_proto(grad_vars, weight_decay=args.dynamic_weight_decay)
    static_optimizer = opt_proto(static_grad_vars)
    scaler = GradScaler()
    static_scaler = GradScaler()

    #linear in logrithmic space
    N_voxel_list = ( torch.round(
                        torch.exp(
                            torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1)
                        )
                     ).long()
                   ).tolist()[1:]


    torch.cuda.empty_cache()
    Metrics = {
        'PSNRs': [],
        'PSNRs_t': [0],
        'PSNRs_STA': [],
        'PSNRs_st': [0],
    }
    TESTKEYS = ['PSNRs_t', 'PSNRs_st']

    batch_factor = [1, 1, 1, 1] if args.batch_factor == [] else args.batch_factor
    allrays, allrgbs, allstds = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_stds
    if not args.ndc_ray:
        allrays, allrgbs, allstds = tensorf.filtering_rays(allrays, allrgbs, allstds, bbox_only=True)
    current_batch_size = int(args.batch_size * batch_factor[0])
    print("creating sammpler with batch size: {}".format(current_batch_size))
    assert args.ray_sampler == 'simple'
    print("=================SimpleRay========================")
    print('All Rays: {}'.format(allrays.shape[0]))
    trainingSampler = SimpleSampler(allrays.shape[0], current_batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    sparse_reg = lambda x: torch.abs(1-torch.exp(-args.sparsity_lambda*x))
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    # Training Dynamic Volumetric representations
    train_dynamics(args, tensorf, allrays, allrgbs, allstds, ndc_ray, nSamples, scaler, device)
    eval_dynamics(args, tensorf, test_dataset, ndc_ray, nSamples, device)
    assert args.temporal_sampler == 'simple'
    print("=================SimpleTemporal========================")
    temporal_sampler = TemporalSampler(args.n_frames, args.n_train_frames)

    # debugger = DebugGradient(static_optimizer)
    # debugger.check()
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    timing = {}

    # tensorf.calc_init_alpha(tuple(reso_cur))
    for iteration in pbar:
        _time = time.time()
        if args.use_cosine_lr_scheduler:
            lr_factor = math.cos((iteration + 1.0) / args.n_iters * math.pi / 2) / math.cos((iteration + 0.0) / args.n_iters * math.pi / 2)
        gamma_current = iteration/args.n_iters * (args.gamma_end - args.gamma_start) + args.gamma_start
        ray_idx = trainingSampler.nextids(gamma=gamma_current)
        rays_train, rgb_train, std_train = allrays[ray_idx].to(device).float(), allrgbs[ray_idx].to(device).float(), allstds[ray_idx].to(device).float()
        temporal_indices, supervision_rgb_train = temporal_sampler.sample(rgb_train, iteration)
        #rgb_map, alphas_map, depth_map, weights, uncertainty
        time_ = time.time()
        timing['pre'] = time_ - _time

        optimizer.zero_grad()
        static_optimizer.zero_grad()
        with autocast(enabled=bool(args.amp)):
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            retva = renderer(rays_train, tensorf, chunk=current_batch_size,
                                    N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray,
                                    device=device, is_train=True, rgb_train=rgb_train,
                                    temporal_indices=temporal_indices,
                                    std_train=std_train)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            retva = Namespace(**retva)

            # =============== dynamics prediction for points ===============
            # dynamics dynamics_supervision shape: Ns
            # dynamic_prediction_loss = DynamicCriterion.calculate_loss(dynamics, dynamics_supervision)
            # DynamicCriterion.compute_metrics()

            # ray_wise_temporal_mask [Nr x T]
            total_loss = 0
            total_static_loss = 0

            # supervision_rgb_train = rgb_train.transpose(0,1)[temporal_indices].transpose(0,1)
            if args.dy_loss == 'l2':
                loss_ray_wise = ((retva.rgb_map - supervision_rgb_train[retva.ray_dynamic_mask])**2)
            elif args.dy_loss == 'l1':
                loss_ray_wise = ((retva.rgb_map - supervision_rgb_train[retva.ray_dynamic_mask]).abs())
            else:
                raise NotImplementedError
            loss = loss_ray_wise.mean()

            total_loss += loss
            Metrics['PSNRs'].append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if args.static_type == 'mean':
                static_supervision = rgb_train.mean(dim=1)
            elif args.static_type == 'median':
                static_supervision = rgb_train.median(dim=1)[0]
            elif args.static_type == 'single_frame':
                static_supervision = torch.zeros(rgb_train.shape[0], rgb_train.shape[2]).to(rgb_train)
                ray_dynamic_mask = retva.ray_wise_temporal_mask.any(dim=1)
                static_supervision[ray_dynamic_mask] = rgb_train[ray_dynamic_mask].mean(dim=1)
                static_supervision[~ray_dynamic_mask] = rgb_train[~ray_dynamic_mask][:,0,:]
            else:
                raise NotImplementedError
            if args.static_loss == 'l2':
                loss_static = ((retva.static_rgb_map - static_supervision)**2).mean()
            elif args.static_loss == 'l1':
                loss_static = ((retva.static_rgb_map - static_supervision).abs()).mean()
            else:
                raise NotImplementedError
            total_static_loss += loss_static
            Metrics['PSNRs_STA'].append(-10.0 * np.log(loss_static.item()) / np.log(10.0))

            _time = time.time()
            timing['calc'] = _time - time_

            if args.sigma_entropy_weight > 0:
                total_loss += args.sigma_entropy_weight * entropy_loss(retva.sigma_ray_wise)
            if args.sigma_entropy_weight_static > 0:
                total_static_loss += args.sigma_entropy_weight_static * entropy_loss(retva.static_sigma)

            if args.sigma_diff_weight > 0:
                if args.sigma_diff_method == 'l2':
                    total_loss += args.sigma_diff_weight * (retva.sigma_diff.mean(dim=-1)**2).mean()
                elif args.sigma_diff_method == 'log':
                    total_loss += args.sigma_diff_weight * consistency_loss(retva.sigma_diff, thresh=args.sigma_diff_log_thresh)

            if args.rgb_diff_weight > 0:
                total_loss += args.rgb_diff_weight * consistency_loss(retva.rgb_diff, thresh=args.rgb_diff_log_thresh, rgb=True)

            if Ortho_reg_weight > 0:
                loss_reg = tensorf.vector_comp_diffs()
                total_loss += Ortho_reg_weight*loss_reg
                summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)

            if L1_reg_weight > 0:
                loss_reg_L1 = tensorf.density_L1(sparse_reg)
                total_loss = total_loss + L1_reg_weight * loss_reg_L1
                loss_reg_L1_static = tensorf.density_L1_static(sparse_reg)
                total_static_loss = total_static_loss + L1_reg_weight * loss_reg_L1_static
                summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

            if TV_weight_density>0 and iteration < args.TV_loss_end_iteration:
                TV_weight_density *= lr_factor
                if args.TV_dynamic_factor > 0:
                    loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density * args.TV_dynamic_factor
                    total_loss = total_loss + loss_tv
                if args.static_dynamic_seperate:
                    loss_tv = tensorf.TV_loss_static_density(tvreg) * TV_weight_density
                    total_static_loss = total_static_loss + loss_tv
                summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)

            if TV_weight_app>0 and iteration < args.TV_loss_end_iteration:
                TV_weight_app *= lr_factor
                if args.TV_dynamic_factor > 0:
                    loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app * args.TV_dynamic_factor
                    total_loss = total_loss + loss_tv
                if args.static_dynamic_seperate:
                    loss_tv = tensorf.TV_loss_static_app(tvreg) * TV_weight_app
                    total_static_loss = total_static_loss + loss_tv
                summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        time_ = time.time()
        timing['reg'] = time_ - _time

        if args.amp:
            static_scaler.scale(total_static_loss).backward()
            static_scaler.step(static_optimizer)
            static_scaler.update()

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # debugger.check()
        else:
            total_static_loss.backward()
            static_optimizer.step()

            total_loss.backward()
            optimizer.step()

        _time = time.time()
        timing['backward'] = _time - time_
        # print(timing)

        loss = loss.detach().item()
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)

        for key in Metrics.keys():
            if key in TESTKEYS:
                continue
            summary_writer.add_scalar('train/{}'.format(key), Metrics[key][-1], global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        for param_group in static_optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        first_group_lr = optimizer.param_groups[0]['lr']


        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            description = f'Iteration {iteration:05d}:' \
                      + f' mse:{loss:.2f}' \
                      + f' loss:{total_loss.item():.2f}' \
                      + f' LR_G1:{first_group_lr: .3f} '
            for key in Metrics.keys():
                description += '{}:{:.2f} '.format(key.lower(), float(np.mean(Metrics[key])))
            pbar.set_description(description)
            for key in Metrics.keys():
                if key not in TESTKEYS:
                    Metrics[key] = []

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            tensorf.save(f'{logfolder}/{args.expname}.th')
            cuda_empty()
            with autocast(enabled=bool(args.amp)):
                Metrics['PSNRs_t'], Metrics['PSNRs_st'], all_metrics = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False,
                                    simplify=True)
            summary_writer.add_scalar('test/psnr', np.mean(Metrics['PSNRs_t']), global_step=iteration)
            summary_writer.add_scalar('test/psnr_sta', np.mean(Metrics['PSNRs_st']), global_step=iteration)
            cuda_empty()

        if iteration in update_AlphaMask_list:
            # if reso_cur[0] * reso_cur[1] * reso_cur[2]<330**3:# update volume resolution
            reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            print(new_aabb)
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)

            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                # trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)
            cuda_empty()
            current_batch_size = int(batch_factor[update_AlphaMask_list.index(iteration)] * args.batch_size)
            print("re-creating sammpler with batch size: {}".format(current_batch_size))
            # trainingSampler = SimpleSampler(allrgbs.shape[0], current_batch_size)
            if args.ray_sampler == 'simple':
                trainingSampler = SimpleSampler(allrays.shape[0], current_batch_size)
            elif args.ray_sampler == 'weighted':
                trainingSampler = WeightedRaySampler(allrays.shape[0], current_batch_size, train_dataset.all_rays_weight)

        if args.ray_sampler == 'comp' and iteration == args.ray_sampler_shift:
            print('Shifting Training Sampler')
            trainingSampler = WeightedRaySampler(allrays.shape[0], current_batch_size, train_dataset.all_rays_weight)

        if iteration == args.shift_std:
            print('Shifting STDs')
            allstds = train_dataset.shift_stds()
            test_dataset.shift_stds()
            if iteration not in upsamp_list:
                train_dynamics(args, tensorf, allrays, allrgbs, allstds, ndc_ray, nSamples, scaler, device, iter_ratio=2)
                eval_dynamics(args, tensorf, test_dataset, ndc_ray, nSamples, device)
                cuda_empty()

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_dynamic_init*lr_scale, args.lr_dynamic_basis*lr_scale)
            optimizer = opt_proto(grad_vars, weight_decay=args.dynamic_weight_decay)

            static_grad_vars = tensorf.get_static_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            static_optimizer = opt_proto(static_grad_vars)

            train_dynamics(args, tensorf, allrays, allrgbs, allstds, ndc_ray, nSamples, scaler, device, iter_ratio=(2 if iteration==upsamp_list[-1] else 1))
            eval_dynamics(args, tensorf, test_dataset, ndc_ray, nSamples, device)
            cuda_empty()

        if args.update_stepratio_iters is not None and iteration in args.update_stepratio_iters:
            _idx = args.update_stepratio_iters.index(iteration)
            tensorf.update_stepRatio(args.update_stepratio[_idx])
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.update_stepratio[_idx]))
            print(f'Sampling points: {nSamples}')

        if args.n_iters > 1000 and (iteration % 1000 == 0 or iteration == args.n_iters-1):
            tensorf.save(f'{logfolder}/{args.expname}.th')

    cuda_empty()
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True,
                                scene_box=args.scene_box)
        with autocast(enabled=bool(args.amp)):
            PSNRs_test, PSNRs_STA_test, all_metrics = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                        N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')
        print(f'======> {args.expname} test all psnr sta: {np.mean(PSNRs_STA_test)} <========================')

    # evaluate images existing in dataset, can not generate a continuous video for llff data.
    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        with autocast(enabled=bool(args.amp)):
            PSNRs_test, PSNRs_STA_test, all_metrics = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                    N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
                                    simplify=(args.n_frames>0))
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        summary_writer.add_scalar('test/psnr_sta_all', np.mean(PSNRs_STA_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')
        print(f'======> {args.expname} test all psnr sta: {np.mean(PSNRs_STA_test)} <========================')
    # for llff data. without many images as ground truth, novel views are rendered without measuring metrics
    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        with autocast(enabled=bool(args.amp)):
            print("evaluating path")
            evaluation_path(test_dataset, tensorf, args, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                            N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device,
                            temporal_sampler=temporal_sampler)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path or args.dense_alpha):
        render_test(args)
    else:
        reconstruction(args)


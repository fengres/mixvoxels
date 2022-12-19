import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--meta_config', is_config_file=True, default=None,
                        help='config file path')
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--ssd_dir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'adamw', 'lars', 'rmsp'])
    parser.add_argument('--use_cosine_lr_scheduler', type=int, default=0)

    parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                        choices=['TensorVMSplit', 'NeuralVoxel'])
    parser.add_argument('--netspec_dy_density', type=str, default='i-d-d-o')
    parser.add_argument('--netspec_dy_color', type=str, default='i-d-d-o')
    parser.add_argument('--voxel_init_dynamic', type=float, default=0.1)
    parser.add_argument('--voxel_init_static', type=float, default=0.1)
    parser.add_argument("--sparsity_lambda", type=float, default=1.0)

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)
    parser.add_argument("--n_dynamic_iters", type=int, default=2000)
    parser.add_argument("--n_frames", type=int, default=100)
    parser.add_argument("--n_train_frames", type=int, default=10)
    parser.add_argument("--n_time_embedding", type=int, default=24)
    parser.add_argument("--render_views", type=int, default=120)
    parser.add_argument("--zero_dynamic_sigma", type=int, default=0)
    parser.add_argument("--zero_dynamic_sigma_thresh", type=float, default=0.001)

    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'llffvideo', 'nsvf', 'dtu','tankstemple', 'own_data', 'ssd'])
    parser.add_argument("--near", type=float, default=0.0)
    parser.add_argument("--far", type=float, default=1.0)
    parser.add_argument("--frame_start", type=int, default=0, help='frame start')
    parser.add_argument("--diffuse_kernel", type=int, default=0, help='diffuse kernel size')

    parser.add_argument("--render_path_start", type=int, default=0, help='diffuse kernel size')

    # dynamic prunning
    parser.add_argument('--static_branch_only_initial', type=int, default=0)
    parser.add_argument('--dynamic_only_ray_start_iteration', type=int, default=-1)
    parser.add_argument("--remove_foreground", type=int, default=0, help='remove foreground')
    parser.add_argument('--static_type', type=str, default='mean')
    parser.add_argument('--static_dynamic_seperate', type=int, default=1)
    parser.add_argument("--dynamic_reg_weight", type=float, default=0)
    parser.add_argument("--sigma_static_thresh", type=float, default=1.0)
    parser.add_argument("--dynamic_granularity", type=str, default='ray_wise', choices=['ray_wise', 'point_wise'])
    parser.add_argument("--point_wise_dynamic_threshold", type=float, default=0.03)
    parser.add_argument("--static_point_detach", type=int, default=1)
    parser.add_argument("--dynamic_pool_kernel_size", type=int, default=1)

    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02, help='learning rate')
    parser.add_argument("--lr_basis", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--lr_dynamic_init", type=float, default=0.02, help='learning rate')
    parser.add_argument("--lr_dynamic_basis", type=float, default=1e-3, help='learning rate')

    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument('--distortion_loss', type=float, default=0.)
    parser.add_argument('--gaussian', type=int, default=0)
    parser.add_argument('--dy_loss', type=str, default='l2', choices=['l2', 'l1'])
    parser.add_argument('--static_loss', type=str, default='l2', choices=['l2', 'l1'])
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--temporal_variance_threshold", type=float, default=0.03)
    parser.add_argument("--temporal_maxdiff_threshold", type=float, default=1000000)
    parser.add_argument("--dynamic_threshold", type=float, default=0.9)
    parser.add_argument("--loss_weight_static", type=float, default=1)
    parser.add_argument("--dynamic_use_volumetric_render", type=int, default=0)
    parser.add_argument("--loss_weight_thresh_start", type=float, default=0.0)
    parser.add_argument("--loss_weight_thresh_end", type=float, default=0.00)
    parser.add_argument("--simple_sample_weight", type=float, default=0)
    parser.add_argument("--simple_sample_weight_end", type=float, default=0)

    parser.add_argument("--ray_sampler", type=str, default='simple')
    parser.add_argument("--ray_sampler_shift", type=int, default=3000)
    parser.add_argument("--gamma_start", type=float, default=0.02)
    parser.add_argument("--gamma_end", type=float, default=0.02)

    parser.add_argument("--ray_weight_gamma", type=float, default=1)

    parser.add_argument("--filter_loss_weight", type=float, default=0)
    parser.add_argument("--filter_threshold", type=float, default=1.0)

    parser.add_argument("--temporal_sampler_method", type=str, default='mean')
    parser.add_argument("--temporal_sampler", type=str, default='simple')
    parser.add_argument("--temporal_sampler_replace", type=int, default=1)
    parser.add_argument("--temperature_start", type=float, default=10.0)
    parser.add_argument("--temperature_end", type=float, default=0.2)


    parser.add_argument("--dynamic_weight_decay", type=float, default=0.0)

    parser.add_argument("--sigma_diff_method", type=str, default='l2')
    parser.add_argument("--sigma_diff_weight", type=float, default=0.0)
    parser.add_argument("--sigma_diff_log_thresh", type=float, default=0.1)
    parser.add_argument("--rgb_diff_weight", type=float, default=0.0)
    parser.add_argument("--rgb_diff_log_thresh", type=float, default=0.02)

    parser.add_argument("--alpha_dynamic", type=int, default=0)
    parser.add_argument("--n_frame_for_static", type=int, default=2)
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_dynamic_factor", type=float, default=1.0,
                        help='TV loss factor')
    parser.add_argument("--TV_loss_end_iteration", type=int, default=100000)
    parser.add_argument("--sigma_decay", type=float, default=0.0, help='sigma decay')
    parser.add_argument("--sigma_decay_static", type=float, default=0.0, help='sigma decay')
    parser.add_argument("--sigma_decay_method", type=str, default="l2", choices=['l2', 'l1'], help='sigma decay method')

    parser.add_argument("--sigma_entropy_weight", type=float, default=0, help='sigma decay')
    parser.add_argument("--sigma_entropy_weight_static", type=float, default=0, help='sigma decay')

    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--n_lamb_sigma_dynamic", type=int, action="append", default=None)
    parser.add_argument("--n_lamb_sh_dynamic", type=int, action="append", default=None)
    parser.add_argument("--data_dim_color", type=int, default=27)
    parser.add_argument("--interpolation", type=str, default="bilinear")
    # FengADD
    parser.add_argument("--data_dim_density", type=int, default=27)
    parser.add_argument("--densityMode", type=str, default="None")
    parser.add_argument("--featureD", type=int, default=128, help='hidden feature channel in MLP')
    parser.add_argument("--net_layer_add", type=int, default=0, help='hidden feature channel in MLP')
    parser.add_argument("--rel_pos_pe", type=int, default=6)

    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--rm_weight_mask_thre_static", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')

    # network decoder
    parser.add_argument("--time_head", type=str, default='dyrender')
    parser.add_argument("--shadingMode", type=str, default="MLP_PE",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')
    parser.add_argument("--static_featureC", type=int, default=128,
                        help='hidden feature channel in MLP')



    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--dense_alpha", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)
    parser.add_argument('--ray_weighted',type=int,default=0)


    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')



    parser.add_argument('--shift_std', type=int, default=-1)
    parser.add_argument('--N_voxel_init', type=int, default=100**3)
    parser.add_argument('--N_voxel_final', type=int, default=300**3)
    parser.add_argument("--scene_box", type=float, action="append")
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--batch_factor", type=float, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")
    parser.add_argument("--update_stepratio", type=int, action="append")
    parser.add_argument("--update_stepratio_iters", type=int, action="append")

    parser.add_argument('--idx_view',
                        type=int,
                        default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000,
                        help='frequency of visualize the image')

    # initialization
    parser.add_argument("--init_static_voxel", type=str, default='none', help='initialization of static voxels')
    parser.add_argument("--init_static_mean", type=float, default=0.0, help='initialization of static voxels')
    parser.add_argument("--init_static_std", type=float, default=0.1, help='initialization of static voxels')
    parser.add_argument("--init_static_a", type=float, default=-0.1, help='initialization of static voxels')
    parser.add_argument("--init_static_b", type=float, default=0.1, help='initialization of static voxels')

    parser.add_argument("--init_dynamic_voxel", type=str, default='none', help='initialization of static voxels')
    parser.add_argument("--init_dynamic_mean", type=float, default=0.0, help='initialization of static voxels')
    parser.add_argument("--init_dynamic_std", type=float, default=0.1, help='initialization of static voxels')
    parser.add_argument("--init_dynamic_a", type=float, default=-0.1, help='initialization of static voxels')
    parser.add_argument("--init_dynamic_b", type=float, default=0.1, help='initialization of static voxels')

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
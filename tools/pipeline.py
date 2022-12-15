import os
import subprocess
import argparse

def show_results(expname):
    mean_file = f'log/{expname}/imgs_test_all/mean.txt'
    cmd = f'python eval/parse_txt.py --files {mean_file}'
    os.system(cmd)

def main(args, unknown):
    # train
    expname = os.path.basename(args.config.split('.')[0])
    stdout_file = os.path.join('log', expname, 'stdout.txt')
    os.makedirs(os.path.dirname(stdout_file), exist_ok=True)
    unknown_args = ' '.join(unknown)
    cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} ' \
          f'python train.py --config {args.config} --vis_every {args.vis_every} '\
          f'{unknown_args} '
    if args.stdout == 'file':
        cmd += f'> {stdout_file} 2>&1'
    if args.train:
        print(cmd)
        os.system(cmd)
    # show_results(expname)

    # make spirals
    if args.spirals:
        videos_path = f'log/{expname}/imgs_path_all/'
        target = f'log/{expname}/spirals'
        target_video = f'log/{expname}/spirals.mp4'
        cmd = f'python tools/make_spiral.py --videos_path {videos_path} ' \
              f'--target {target} ' \
              f'--target_video {target_video} ' \
              f'--spiral_mode circles --type v+d'
        print(cmd)
        os.system(cmd)

    # show statistics
    # show_results(expname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--vis_every', type=int, default=100000000)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--spirals', type=int, default=1)
    parser.add_argument('--stdout', type=str, default='file', choices=['file', 'cmd'])
    args, unknown = parser.parse_known_args()
    main(args, unknown)
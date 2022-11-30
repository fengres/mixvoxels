import os
import glob
import argparse

def capture_img(video_path, img_path, i_frame, fps):
    # Note i_frame starts from 0
    sec = i_frame // fps
    msec = round(i_frame/fps - sec, 5)
    sec = str(sec).zfill(2)
    msec = str(msec)[2:]
    cmd = f'ffmpeg -i {video_path} -ss 00:00:{sec}.{msec} -f image2 -r 1 -t 1 {img_path}'
    os.system(cmd)

def imgs_to_video(imgs_path, video_path, fps):
    cmd = f'ffmpeg -f image2 -r {fps} -i {imgs_path}/%04d.png {video_path}'
    os.system(cmd)

def one_circle(args):
    if 'v' in args.type:
        n_frames = args.fps * args.seconds
        for i in range(n_frames):
            n_view = round(i / n_frames * args.n_views)
            if n_view == args.n_views:
                n_view = n_view - 1
            video_path = os.path.join(args.videos_path, f'cam_{n_view}_comp_video.mp4')
            img_path = os.path.join(args.target, f'{str(i).zfill(4)}.png')
            capture_img(video_path, img_path, i, args.fps)
        imgs_to_video(args.target, args.target_video, args.fps)
    if 'd' in args.type:
        n_frames = args.fps * args.seconds
        for i in range(n_frames):
            n_view = round(i / n_frames * args.n_views)
            if n_view == args.n_views:
                n_view = n_view - 1
            video_path = os.path.join(args.videos_path, f'cam_{n_view}_comp_depthvideo.mp4')
            img_path = os.path.join(args.target_depth, f'{str(i).zfill(4)}.png')
            capture_img(video_path, img_path, i, args.fps)
        imgs_to_video(args.target_depth, args.target_video_depth, args.fps)

def circles(args):
    n_frames = args.fps * args.seconds
    if 'v' in args.type and 'd' in args.type:
        markers = ['', 'depth']
        targets = [args.target, args.target_depth]
        target_videos = [args.target_video, args.target_video_depth]
    elif 'v' in args.type:
        markers = ['', ]
        targets = [args.target, ]
        target_videos = [args.target_video, ]
    elif 'd' in args.type:
        markers = ['depth',]
        targets = [args.target_depth,]
        target_videos = [args.target_video_depth,]

    for marker, target, target_video in zip(markers, targets, target_videos):
        for i in range(n_frames):
            n_view = i % args.n_views
            video_path = os.path.join(args.videos_path, f'cam_{n_view}_comp_{marker}video.mp4')
            img_path = os.path.join(target, f'{str(i).zfill(4)}.png')
            capture_img(video_path, img_path, i, args.fps)
        imgs_to_video(target, target_video, args.fps)

def one_circle_img(args, base_frame, base_view):
    for i in range(args.n_views):
        current_view = (i + base_view) % args.n_views
        video_path = os.path.join(args.videos_path, f'cam_{current_view}_comp_video.mp4')
        img_path = os.path.join(args.target, f'{str(i+base_frame).zfill(4)}.png')
        capture_img(video_path, img_path, base_frame, args.fps)


def bullet_time(args):
    n_frames = args.fps * args.seconds
    frame_shift = 0
    for i in range(n_frames):
        n_view = i % args.n_views
        if i == args.bullet_time_frame:
            one_circle_img(args, args.bullet_time_frame, n_view)
            frame_shift += args.n_views
        video_path = os.path.join(args.videos_path, f'cam_{n_view}_comp_video.mp4')
        img_path = os.path.join(args.target, f'{str(i+frame_shift).zfill(4)}.png')
        capture_img(video_path, img_path, i, args.fps)
    imgs_to_video(args.target, args.target_video, args.fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--target_video', type=str, default=None)
    parser.add_argument('--n_views', type=int, default=120)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--seconds', type=int, default=10)
    parser.add_argument('--spiral_mode', type=str, default='circles')
    parser.add_argument('--bullet_time_frame', type=int, default=75)
    parser.add_argument('--type', type=str, default='v')
    args = parser.parse_args()
    args.target_depth = (args.target[:-1] if args.target.endswith('/') else args.target)+'_depth'
    args.target_video_depth = args.target_video[:-4] + '_depth.mp4'
    os.makedirs(args.target, exist_ok=True)
    os.makedirs(args.target_depth, exist_ok=True)
    if args.spiral_mode == 'one_circle':
        one_circle(args)
    if args.spiral_mode == 'circles':
        circles(args)
    if args.spiral_mode == 'bullet_time':
        bullet_time(args)

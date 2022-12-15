import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    --data_dir/
    -----videos/
    ---------video_1.mp4
    ---------video_2.mp4
    -----frames/
    ---------video_1/
    ---------video_2/
    """
    parser.add_argument('data_dir', type=str, help='folder for processing')
    parser.add_argument('--data_type', type=str, default='dynerf', choices=['dynerf', 'gopro'], help='folder for processing')
    args = parser.parse_args()

    video_dir = os.path.join(args.data_dir, 'videos')
    frames_dir = os.path.join(args.data_dir, 'frames')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_name)
        frame_path = os.path.join(frames_dir, video_name.split('.')[0])
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        cmd = 'ffmpeg -i {} {}'.format(video_path, os.path.join(frame_path, "%04d.jpg"))
        os.system(cmd)

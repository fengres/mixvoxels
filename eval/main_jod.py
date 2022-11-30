import re
import os
import subprocess
import argparse


def extract_from_result(text: str, prompt: str):
    m = re.search(prompt, text)
    return float(m.group(1))

def to_file(file_name, temp_name, WIDTH, HIEGHT, start_frame=None, end_frame=None, cache_dir=None):
    if os.path.isfile(file_name):
        return file_name
    else:
        for img_file in os.listdir(file_name):
            if img_file.endswith('.png'):
                img_index = int(img_file.split('.')[0])
                if img_index >= (start_frame + 1) and img_index < (end_frame + 1):
                    spth = os.path.join(file_name, img_file)
                    tpth = os.path.join(cache_dir, '%04d.png' % (img_index - start_frame))
                    cmd = f'cp {spth} {tpth}'
                    os.system(cmd)

        os.system('ffmpeg -i {}/%4d.png -frames:v 300 -s {}x{} -c:v libx264 -qp 0 {}'
                  .format(cache_dir, WIDTH, HEIGHT, temp_name))
        return temp_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--downsample', type=int, default=None)
    parser.add_argument('--tmp_dir', type=str, default='/tmp/nerf_metric_temp')
    parser.add_argument('--tmp_gt_dir', type=str, default='/tmp/nerf_metric_temp_gt')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=300)
    args = parser.parse_args()

    os.system(f'rm -rf {args.tmp_dir}')
    os.system(f'mkdir {args.tmp_dir}')
    if args.downsample == 2:
        WIDTH = 1360
        HEIGHT = 1024
    elif args.downsample == 4:
        WIDTH = 688
        HEIGHT = 512

    file1 = to_file(args.output, os.path.join(args.tmp_dir, 'nerf_metric_temp1.mp4'), WIDTH, HEIGHT)
    file2 = to_file(args.gt, os.path.join(args.tmp_dir, 'nerf_metric_temp2.mp4'), WIDTH, HEIGHT, start_frame=args.start_frame, end_frame=args.end_frame, cache_dir=args.tmp_dir)

    result = subprocess.check_output(['fvvdp', '--test', file1, '--ref', file2, '--gpu', '0', '--display', 'standard_fhd'])
    result = result.decode().strip()
    result = float(result.split('=')[1])
    print(result)
    os.system('stty echo')

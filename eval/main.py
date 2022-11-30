import re
import os
import subprocess
import glob
import argparse
import matplotlib.font_manager as fm
from PIL import Image, ImageFont, ImageDraw

def add_text(img, flip):
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), 72)
    im = Image.open(img)
    draw = ImageDraw.Draw(im)
    draw.text((30, 30), f"FLIP: {round(flip, 4)}", (255, 251, 0), font=font)
    im.save(img)

def extract_from_result(text: str, prompt: str):
    m = re.search(prompt, text)
    return float(m.group(1))

def to_directory(file_name, WIDTH, HEIGHT, tmp_dir, start_frame=None, end_frame=None):
    if os.path.isdir(file_name):
        # cp frames from
        for img_file in os.listdir(file_name):
            if img_file.endswith('.png'):
                img_index = int(img_file.split('.')[0])
                if img_index >= (start_frame+1) and img_index < (end_frame+1):
                    spth = os.path.join(file_name, img_file)
                    tpth = os.path.join(tmp_dir, '%04d.png'%(img_index-start_frame))
                    cmd = f'cp {spth} {tpth}'
                    print(cmd)
                    os.system(cmd)
        return tmp_dir
    else:
        os.system('ffmpeg -i {} -s {}x{} {}'.format(file_name, WIDTH, HEIGHT, os.path.join(tmp_dir, r'%4d.png')))
        return tmp_dir


def calc(directory_1, directory_2, flip_path, interval):
    ALL_PAIRS = []
    for file_name in os.listdir(directory_1):
        if not file_name.endswith('.png'):
            continue
        assert os.path.isfile(os.path.join(directory_1, file_name))
        assert os.path.isfile(os.path.join(directory_2, file_name))
        ALL_PAIRS.append((os.path.join(directory_1, file_name), os.path.join(directory_2, file_name)))

    all_results = []
    for file1, file2 in ALL_PAIRS:
        if int(file1.split('/')[-1].split('.')[0]) % interval != 0:
            continue
        print(file1, file2)
        if flip_path is not None:
            frame_flip_path = os.path.join(flip_path, file1.split('/')[-1].split('.')[0].zfill(5)+'.png')
            result = subprocess.check_output(
                ['python', 'eval/flip.py', '--reference', file1, '--test', file2, '--flip_path', frame_flip_path])
        else:
            frame_flip_path = None
            result = subprocess.check_output(['python', 'eval/flip.py', '--reference', file1, '--test', file2])
        result = result.decode()
        # print(result)
        all_results.append({
            'Mean': extract_from_result(result, r'Mean: (\d+\.\d+)'),
            'Weighted median': extract_from_result(result, r'Weighted median: (\d+\.\d+)'),
            '1st weighted quartile': extract_from_result(result, r'1st weighted quartile: (\d+\.\d+)'),
            '3rd weighted quartile': extract_from_result(result, r'3rd weighted quartile: (\d+\.\d+)'),
            'Min': extract_from_result(result, r'Min: (\d+\.\d+)'),
            'Max': extract_from_result(result, r'Max: (\d+\.\d+)'),
        })
        if flip_path is not None:
            add_text(frame_flip_path, all_results[-1]['Mean'])
        print(all_results[-1])

    all_results_processed = {k: [_[k] for _ in all_results] for k in all_results[0]}
    print(all_results_processed)
    all_results_processed = {k: sum(all_results_processed[k]) / len(all_results_processed[k]) for k in all_results_processed}
    print(all_results_processed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--downsample', type=int, default=None)
    parser.add_argument('--tmp_dir', type=str, default='/tmp/nerf_metric_temp')
    parser.add_argument('--tmp_gt_dir', type=str, default='/tmp/nerf_metric_temp_gt')
    parser.add_argument('--flip_path', type=str, default=None)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=300)
    args = parser.parse_args()

    WIDTH = 676 * 4 // args.downsample
    HEIGHT = 507 * 4 // args.downsample

    os.system(f'rm -rf {args.tmp_dir}')
    os.system(f'mkdir {args.tmp_dir}')
    os.system(f'rm -rf {args.tmp_gt_dir}')
    os.system(f'mkdir {args.tmp_gt_dir}')
    directory_1 = to_directory(args.output, WIDTH, HEIGHT, args.tmp_dir)
    directory_2 = to_directory(args.gt, WIDTH, HEIGHT, args.tmp_gt_dir, start_frame=args.start_frame, end_frame=args.end_frame)
    calc(directory_1, directory_2, args.flip_path, args.interval)
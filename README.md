# MixVoxels: Mixed Neural Voxels for Fast Multi-view Video Synthesis
### [Project Page]()  |  [Paper]()

Pytorch implementation for the paper: [Mixed Neural Voxels for Fast Multi-view Video Synthesis](). 

https://user-images.githubusercontent.com/43294876/204843741-0a2d10a8-f0b4-4f69-b262-7fc7983a20c5.mp4

We present MixVoxels to better represent the dynamic scenes with fast training speed and competitive rendering qualities. 
The proposed MixVoxels represents the 4D dynamic scenes as a mixture of static and dynamic voxels and processes them with different networks. 
In this way, the computation of the required modalities for static voxels can be processed by a lightweight model, 
which essentially reduces the amount of computation, 
especially for many daily dynamic scenes dominated by the static background. 
As a result, with 15 minutes of training for dynamic scenes with inputs of 300-frame videos, MixVoxels achieves better PSNR than previous methods.

<div align="center">
<img src="tools/mixvoxels.png" width="450" align="center"/>
</div>

## Installation

Install environment:
```
conda create -n mixvoxels python=3.8
conda activate mixvoxels
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard pyfvvdp
```

## Dataset
1. Download the [Plenoptic Video Dataset](https://github.com/facebookresearch/Neural_3D_Video)
2. Unzip to your directory DATADIR and run the following command:
  ```
  python tools/prepare_video.py ${DATADIR}
  ```

## Training

To train a dynamic scene, run the following commands, you can train different dynamic scenes by assign DATA to different scene name:
```
DATA=coffee_martini # [coffee_martini|cut_roasted_beef|cook_spinach|flame_salmon|flame_steak|sear_steak]
# MixVoxels-T
python train.py --config configs/schedule5000/${DATA}_5000.txt --render_path 0
# MixVoxels-S
python train.py --config configs/schedule7500/${DATA}_7500.txt --render_path 0
# MixVoxels-M
python train.py --config configs/schedule12500/${DATA}_12500.txt --render_path 0
# MixVoxels-L
python train.py --config configs/schedule25000/${DATA}_25000.txt --render_path 0
```

We provide the trained model:

<div align="center">
<table width="600">
<tr>
    <th rowspan="2">scene</th>
    <th colspan="2">PSNR</th>
    <th colspan="2">download</th>
</tr>
<tr>
<td style="text-align: center">MixVoxels-T (15min)</td>
<td style="text-align: center">MixVoxels-M (40min)</td>
<td style="text-align: center">MixVoxels-T (15min)</td>
<td style="text-align: center">MixVoxels-M (40min)</td>
</tr>
<tr>
<td>coffee-martini</td>
<td style="text-align: center">28.1339</td>
<td style="text-align: center">29.0186</td>
<td style="text-align: center">link</td>
<td style="text-align: center">link</td>
</tr>
<tr>
<td>flame-salmon</td>
<td style="text-align: center">28.7982</td>
<td style="text-align: center">29.2620</td>
<td style="text-align: center">link</td>
<td style="text-align: center">link</td>
</tr>
<tr>
<td>cook-spinach</td>
<td style="text-align: center">31.4499</td>
<td style="text-align: center">31.6433</td>
<td style="text-align: center">link</td>
<td style="text-align: center">link</td>
</tr>
<tr>
<td>cut-roasted-beef</td>
<td style="text-align: center">32.4078</td>
<td style="text-align: center">32.2800</td>
<td style="text-align: center">link</td>
<td style="text-align: center">link</td>
</tr>
<tr>
<td>flame-steak</td>
<td style="text-align: center">31.6508</td>
<td style="text-align: center">31.3052</td>
<td style="text-align: center">link</td>
<td style="text-align: center">link</td>
</tr>
<tr>
<td>sear-steak</td>
<td style="text-align: center">31.8203</td>
<td style="text-align: center">31.2136</td>
<td style="text-align: center">link</td>
<td style="text-align: center">link</td>
</tr>

</table>
</div>

## Rendering and Generating Spirals
The following command will generate 120 novel view videos, or you can set the render_path as 1 in the above training command.
```
python train.py --config your_config --render_only 1 --render_path 1 --ckpt log/your_config/your_config.ckpt
```
Generating spirals:
```
python tools/make_spiral.py --video_path log/your_config/img_path_all/ --target log/your_config/spirals --target_video log/your_config/spirals.mp4
```


[//]: # (## Citation)
[//]: # (If you find our code or paper helps, please consider citing:)
[//]: # (```)
[//]: # ()
[//]: # (```)

## Acknowledge
The codes are based on [TensoRF](https://github.com/apchenstu/TensoRF), many thanks to the authors. 

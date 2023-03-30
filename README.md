# MixVoxels: Mixed Neural Voxels for Fast Multi-view Video Synthesis
### [Project Page](https://fengres.github.io/mixvoxels/)  |  [Paper](https://arxiv.org/pdf/2212.00190.pdf)

Pytorch implementation for the paper: [Mixed Neural Voxels for Fast Multi-view Video Synthesis](https://arxiv.org/pdf/2212.00190.pdf). 

https://user-images.githubusercontent.com/43294876/204843741-0a2d10a8-f0b4-4f69-b262-7fc7983a20c5.mp4


More complicated scenes with fast movements and large areas of motions.


https://user-images.githubusercontent.com/43294876/228771137-ed67629f-e109-47cc-b0f1-50904435546d.mp4



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
> Please note that in your first running, the above command will first pre-process the dataset, including resizing the frames by a factor of 2 (to 1K resolution which is a standard), as well as calculating the std of each video and save them into your disk. The pre-processing will cost about 2 hours, but is only required at the first running. After the pre-processing, the command will automatically train your scenes.  


We provide the trained model:
<table>
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
<td style="text-align: center"><a href="https://drive.google.com/file/d/1rB2Qfcp3GUQg2KiygZZ8lE-hEIOmGvm3/view?usp=share_link">link</a></td>
<td style="text-align: center"><a href="https://drive.google.com/file/d/1REt2A9yFk_4QJvxNqqcxYkMlka2IeVN0/view?usp=sharing">link</a></td>
</tr>
<tr>
<td>flame-salmon</td>
<td style="text-align: center">28.7982</td>
<td style="text-align: center">29.2620</td>
<td style="text-align: center"><a href="https://drive.google.com/file/d/1SxeyqYS9mN7ySxL5vdSXKLvfbdAPzrz5/view?usp=share_link">link</a></td>
<td style="text-align: center"><a href="https://drive.google.com/file/d/1VsCJvHMrDqWiN2nVDGb6NldINF3111G2/view?usp=sharing">link</a></td>
</tr>
<tr>
<td>cook-spinach</td>
<td style="text-align: center">31.4499</td>
<td style="text-align: center">31.6433</td>
<td style="text-align: center"><a href="https://drive.google.com/file/d/1hyZeD8UZDe1XLfz4GTXC_LHKOnPqw86W/view?usp=sharing">link</a></td>
<td style="text-align: center"><a href="https://drive.google.com/file/d/11pwDxlqoKTEl1Qb2Lz8WTFH5ISB1z6Mb/view?usp=sharing">link</a></td>
</tr>
<tr>
<td>cut-roasted-beef</td>
<td style="text-align: center">32.4078</td>
<td style="text-align: center">32.2800</td>
<td style="text-align: center"><a href="https://drive.google.com/file/d/1dWA4OYDbwL0OFJ8WJ0rn1VuseP2yMjk7/view?usp=share_link">link</a></td>
<td style="text-align: center"><a href="https://drive.google.com/file/d/1hgVgJhkWqusqoLXYovr4CF35tKyd8y7N/view?usp=share_link">link</a></td>
</tr>
<tr>
<td>flame-steak</td>
<td style="text-align: center">31.6508</td>
<td style="text-align: center">31.3052</td>
<td style="text-align: center"><a href="https://drive.google.com/file/d/1k4x3Q0BYDFc-r6tWqKgNeXR9A-idZjyZ/view?usp=share_link">link</a></td>
<td style="text-align: center"><a href="https://drive.google.com/file/d/1oDAF1hObpB_8FwSn4TZhGsN2f_UG-bmu/view?usp=share_link">link</a></td>
</tr>
<tr>
<td>sear-steak</td>
<td style="text-align: center">31.8203</td>
<td style="text-align: center">31.2136</td>
<td style="text-align: center"><a href="https://drive.google.com/file/d/1qXTbOTd91-ZFKJXxUUa7N_T2VsaToz3j/view?usp=share_link">link</a></td>
<td style="text-align: center"><a href="https://drive.google.com/file/d/1noVyfJ00G8yIAGLRLBjWt1beTs8L6baE/view?usp=share_link">link</a></td>
</tr>

</table>

## Rendering and Generating Spirals
The following command will generate 120 novel view videos, or you can set the render_path as 1 in the above training command.
```
python train.py --config your_config --render_only 1 --render_path 1 --ckpt log/your_config/your_config.ckpt
```
Generating spirals:
```
python tools/make_spiral.py --videos_path log/your_config/imgs_path_all/ --target log/your_config/spirals --target_video log/your_config/spirals.mp4
```


## Citation
If you find our code or paper helps, please consider citing:
```
@article{wang2022mixed,
  title={Mixed Neural Voxels for Fast Multi-view Video Synthesis},
  author={Wang, Feng and Tan, Sinan and Li, Xinghang and Tian, Zeyue and Liu, Huaping},
  journal={arXiv preprint arXiv:2212.00190},
  year={2022}
}
```

## Acknowledge
The codes are based on [TensoRF](https://github.com/apchenstu/TensoRF), many thanks to the authors. 

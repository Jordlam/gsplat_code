# Jordlam's Changes to DGE

## Commands
Obtain pre-trained Gaussians via https://github.com/Jordlam/dgd_edits

—> Copy dataset to gsplat_code:
```cp -r ~/dgd_edits/data/hypernerf/cookie_DINO_40000 ./gsplat_data/hypernerf && cp -r ~/dgd_edits/data/hypernerf/split-cookie ./gsplat_data/hypernerf```

For editing:
```python launch.py --config configs/dge.yaml --train data.source=gsplat_data/hypernerf/split-cookie system.gs_source=gsplat_data/hypernerf/cookie_DINO_40000/point_cloud/iteration_40000/point_cloud.ply system.deform_source=gsplat_data/hypernerf/cookie_DINO_40000 system.prompt_processor.prompt="change all to pizza" system.ratio=4 system.points="(135,170)" system.thetas="0.55"```

For rendering2 (```cd``` into dgd_edits):
```cp ~/gsplat_code/gsplat_data/hypernerf/cookie_DINO_40000/edit/point_cloud.ply ./data/hypernerf/cookie_DINO_40000/point_cloud/iteration_40000/point_cloud.ply && python render2.py -s ./data/hypernerf/split-cookie/ -m ./data/hypernerf/cookie_DINO_40000 --fundation_model "DINOv2" --semantic_dimension 384 --iterations 40_000 --frame 39 --novel_views -1 --total 233 --ratio 4```

Additional data for DGD inputs and pretrained Gaussians can be found at https://drive.google.com/drive/folders/1swQeiLSrIcP3jqn-ycHVylfPYsSaqpw-?usp=sharing

## Installation
Environment and dependencies should follow DGE installation, but matches scalar versions.

```
# Install conda env
conda create -n DGE python=3.11
conda activate DGE

# Install torch
# CUDA 12.2
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu122
pip install -r requirements.txt
```

## Making changes to the code
Due to how DGE runs, if you make any changes to ```submodules```, you must rebuild the corresponding submodule in order to see your changes reflected. For example, if I made changes to ```submodules/diff-gaussian-rasterization```

```
pip uninstall diff-gaussian-rasterization
cd gaussiansplatting/submodules/diff-gaussian-rasterization
python setup.py install
```

You should now see ```diff-gaussian-rasterization``` as a pip module if you run ```conda list```

# DGE Reference: Direct Gaussian 3D Editing by Consistent Multi-view Editing

[Minghao Chen](https://silent-chen.github.io), [Iro Laina](), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/)

[Paper](https://arxiv.org/abs/2404.18929) | [Webpage](https://silent-chen.github.io/DGE/) 

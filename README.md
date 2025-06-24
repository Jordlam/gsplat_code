# Jordlam's Changes to DGE
Changes made to DGE to perform editing of dynamic 3D Gaussians.

![cookie_ground_truth](./demo/cookie_gt.gif)
![cookie_baseline](./demo/cookie_baseline.gif)
![cookie_pizza](./demo/cookie_pizza.gif)

![cookie_chocolate](./demo/cookie_chocolate.gif)
![cookie_velvet](./demo/cookie_velvet.gif)

![hand_ground_truth](./demo/hand_gt.gif)
![hand_baseline](./demo/hand_baseline.gif)
![hand_ironman](./demo/hand_ironman.gif)

![torch_ground_truth](./demo/torch_gt.gif)
![torch_baseline](./demo/torch_baseline.gif)
![torch_stone](./demo/torch_stone.gif)

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


## Commands
Obtain pre-trained Gaussians via https://github.com/Jordlam/dgd_edits

Removing edit cache:
```rm -r ./.threestudio_cache && rm -r ./edit_cache && rm -r ./outputs/dge/*```

We copy segmented 3D Gaussians to avoid mutating them:
```cp ./data/hypernerf/cookie_DINO_40000/point_cloud/iteration_40000/point_cloud.ply ./data/hypernerf/cookie_DINO_40000/point_cloud/iteration_40000/orig_point_cloud.ply```

For editing: (please remove cache before)
```python launch.py --config configs/dge.yaml --train data.source=data/hypernerf/split-cookie system.gs_source=data/hypernerf/cookie_DINO_40000/point_cloud/iteration_40000/point_cloud.ply system.deform_source=data/hypernerf/cookie_DINO_40000 system.prompt_processor.prompt="change cookie to pizza" system.ratio=4 system.points="(135,170)" system.thetas="0.55"```

For rendering2 (```cd``` into dgd_edits):
```cp ./data/hypernerf/cookie_DINO_40000/edits/point_cloud_pizza.ply ./data/hypernerf/cookie_DINO_40000/point_cloud/iteration_40000/point_cloud.ply && python render2.py -s ./data/hypernerf/split-cookie/ -m ./data/hypernerf/cookie_DINO_40000 --fundation_model "DINOv2" --semantic_dimension 384 --iterations 40_000 --frame 39 --novel_views -1 --total 223 --ratio 4```

## Reference the original repository

[Minghao Chen](https://silent-chen.github.io), [Iro Laina](), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/)

[Paper](https://arxiv.org/abs/2404.18929) | [Webpage](https://silent-chen.github.io/DGE/) 

from dataclasses import dataclass, field

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import sys
import shutil
import torch
import threestudio
import os
from threestudio.systems.base import BaseLift3DSystem

from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel, DeformModel # DGD edit

from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from omegaconf import OmegaConf

from argparse import ArgumentParser
from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.sam import LangSAMTextSegmentor
from sklearn.decomposition import PCA
from gaussiansplatting.utils.sh_utils import RGB2SH

@threestudio.register("dge-system")
class DGE(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        gs_source: str = None
        # TODO hard coded deform model path
        deform_source: str = "gsplat_data/hypernerf/cookie_DINO_10000"

        per_editing_step: int = -1
        edit_begin_step: int = 0
        edit_until_step: int = 4000

        densify_until_iter: int = 4000
        densify_from_iter: int = 0
        densification_interval: int = 100
        max_densify_percent: float = 0.01

        gs_lr_scaler: float = 1
        gs_final_lr_scaler: float = 1
        color_lr_scaler: float = 1
        opacity_lr_scaler: float = 1
        scaling_lr_scaler: float = 1
        rotation_lr_scaler: float = 1

        # lr
        mask_thres: float = 0.5
        max_grad: float = 1e-7
        min_opacity: float = 0.005

        seg_prompt: str = ""

        # cache
        cache_overwrite: bool = True
        cache_dir: str = ""


        # anchor
        anchor_weight_init: float = 0.1
        anchor_weight_init_g0: float = 1.0
        anchor_weight_multiplier: float = 2
        
        training_args: dict = field(default_factory=dict)

        use_masked_image: bool = False
        local_edit: bool = False

        # guidance 
        camera_update_per_step: int = 500
        added_noise_schedule: List[int] = field(default_factory=[999, 200, 200, 21])    
        

    cfg: Config

    def configure(self) -> None:
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=self.cfg.anchor_weight_init_g0,
            anchor_weight_init=self.cfg.anchor_weight_init,
            anchor_weight_multiplier=self.cfg.anchor_weight_multiplier,
        )
        self.deform = DeformModel(False, False)
        print("We initialized deform model")
        # print("We are not loading pretrained deform model")
        self.deform.load_weights(self.cfg.deform_source)
        print("We loaded pretrained deform model")

        bg_color = [1, 1, 1] if False else [0, 0, 0]
        self.background_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.text_segmentor = LangSAMTextSegmentor().to(get_device())

        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join("edit_cache", self.cfg.cache_dir)
        else:
            self.cache_dir = os.path.join("edit_cache", self.cfg.gs_source.replace("/", "-"))
    
    def get_feature(self, x, y, view, gaussians, pipeline, background, scaling_modifier, override_color, d_xyz, d_rotation, d_scaling, patch=None):
        # TODO patch: currently experiment without using d_xyz, d_rotation, and d_scaling
        with torch.no_grad():
            render_feature_dino_pkg = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, override_color = override_color)
            # render_feature_dino_pkg = render(view, gaussians, pipeline, background, scaling_modifier = scaling_modifier, override_color = override_color)
            image_feature_dino = render_feature_dino_pkg["feature_map"]
        if patch is None:
            return image_feature_dino[:, y, x]
        else:
            a = image_feature_dino[:, y:y+patch[1], x:x+patch[0]]
            return a.mean(dim=(1,2))

    def calculate_selection_score_DINOv2(self, features, query_feature, score_threshold=0.8):
        # features /= features.norm(dim=-1, keepdim=True)
        # query_feature /= query_feature.norm(dim=-1, keepdim=True)
        # scores = features.half() @ query_feature.half()

        # clamping added so not divide by 0
        features /= features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        query_feature /= query_feature.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scores = features @ query_feature
        scores = scores[:, 0]
        mask = (scores >= score_threshold).float()
        return mask
    
    @torch.no_grad()
    def update_masks(self, save_name="mask") -> None:
        print(f"Segment with prompt: {self.cfg.seg_prompt}")
        mask_cache_dir = os.path.join(
            self.cache_dir, self.cfg.seg_prompt + f"_{save_name}_{self.view_num}_view"
        )
        gs_mask_path = os.path.join(mask_cache_dir, "gs_mask.pt")
        # TODO for now just grab all the masks:
        all_the_masks = []

        if not os.path.exists(gs_mask_path) or self.cfg.cache_overwrite:
            if os.path.exists(mask_cache_dir):
                shutil.rmtree(mask_cache_dir)
            os.makedirs(mask_cache_dir)
            threestudio.info(f"Segmentation with prompt: {self.cfg.seg_prompt}")
        
        for id in tqdm(self.view_list):
            cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
            pca = PCA(n_components=3)
            semantic_features = self.gaussian.get_semantic_feature

            pca.fit(semantic_features[:,0,:].detach().cpu())
            pca_features = pca.transform(semantic_features[:,0,:].detach().cpu())
            for i in range(3):
                pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())
            pca_features = torch.tensor(pca_features, dtype=torch.float, device = 'cuda', requires_grad = True)

            view = cur_cam
            fid = view.fid
            xyz = self.gaussian.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            with torch.no_grad():
                d_xyz, d_rotation, d_scaling = self.deform.step(xyz.detach(), time_input)                

            # point "(270,370)" hardcoded to be the cookie
            # --thetas "0.55" hardcoded to be best threshold
            i = 0 # lazy
            points = [(270, 370)]
            thetas = [0.55] # this might be a universal threshold

            query_feature = self.get_feature(points[i][0], points[i][1], view, self.gaussian, self.pipe, self.background_tensor, 1.0,
                                        semantic_features[:,0,:], d_xyz, d_rotation, d_scaling, patch = (5,5))
            mask = self.calculate_selection_score_DINOv2(semantic_features, query_feature, score_threshold = thetas[i])
            # indices_to_mask = np.where(mask.cpu().numpy() >= thetas[i])[0]
            # The indices_to_mask are just the indices and not the mask itself
            mask = (mask.cpu() >= thetas[i]).to("cuda")
            all_the_masks.append(mask)
        
        selected_mask = all_the_masks[0] # TODO just one mask for now
        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)

    @torch.no_grad()
    def update_mask(self, save_name="mask") -> None:
        print(f"Segment with prompt: {self.cfg.seg_prompt}")
        mask_cache_dir = os.path.join(
            self.cache_dir, self.cfg.seg_prompt + f"_{save_name}_{self.view_num}_view"
        )
        gs_mask_path = os.path.join(mask_cache_dir, "gs_mask.pt")
        if not os.path.exists(gs_mask_path) or self.cfg.cache_overwrite:
            if os.path.exists(mask_cache_dir):
                shutil.rmtree(mask_cache_dir)
            os.makedirs(mask_cache_dir)
            weights = torch.zeros_like(self.gaussian._opacity)
            weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
            threestudio.info(f"Segmentation with prompt: {self.cfg.seg_prompt}")
            for id in tqdm(self.view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_path_viz = os.path.join(
                    mask_cache_dir, "viz_{:0>4d}.png".format(id)
                )

                mask = self.text_segmentor(self.origin_frames[id], self.cfg.seg_prompt)[
                    0
                ].to(get_device())

                mask_to_save = (
                        mask[0]
                        .cpu()
                        .detach()[..., None]
                        .repeat(1, 1, 3)
                        .numpy()
                        .clip(0.0, 1.0)
                        * 255.0
                ).astype(np.uint8)
                cv2.imwrite(cur_path, mask_to_save)

                masked_image = self.origin_frames[id].detach().clone()[0]
                masked_image[mask[0].bool()] *= 0.3
                masked_image_to_save = (
                        masked_image.cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                ).astype(np.uint8)
                masked_image_to_save = cv2.cvtColor(
                    masked_image_to_save, cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(cur_path_viz, masked_image_to_save)
                self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)

            weights /= weights_cnt + 1e-7

            selected_mask = weights > self.cfg.mask_thres
            selected_mask = selected_mask[:, 0]
            torch.save(selected_mask, gs_mask_path)
        else:
            print("load cache")
            for id in tqdm(self.view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_mask = cv2.imread(cur_path)
                cur_mask = torch.tensor(
                    cur_mask / 255, device="cuda", dtype=torch.float32
                )[..., 0][None]
            selected_mask = torch.load(gs_mask_path)

        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)

    def on_validation_epoch_end(self):
        pass

    def forward(self, batch: Dict[str, Any], renderbackground=None, local=False) -> Dict[str, Any]:
        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        semantics = []
        masks = []
        self.viewspace_point_list = []
        self.gaussian.localize = local
        for id, cam in enumerate(batch["camera"]):
            view = cam
            fid = view.fid
            xyz = self.gaussian.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            with torch.no_grad():
                d_xyz, d_rotation, d_scaling = self.deform.step(xyz.detach(), time_input)

            # render_pkg = render(cam, self.gaussian, self.pipe, renderbackground)
            render_pkg = render(cam, self.gaussian, self.pipe, renderbackground, d_xyz, d_rotation, d_scaling)
            image, viewspace_point_tensor, _, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)

            semantic_map = render(
                cam,
                self.gaussian,
                self.pipe,
                renderbackground,
                d_xyz,
                d_rotation,
                d_scaling,
                override_color=self.gaussian.mask[..., None].float().repeat(1, 3),
            )["render"]
            semantic_map = torch.norm(semantic_map, dim=0)
            semantic_map = semantic_map > 0.8
            semantic_map_viz = image.detach().clone()
            semantic_map_viz = semantic_map_viz.permute(
                1, 2, 0
            )  # 3 512 512 to 512 512 3
            semantic_map_viz[semantic_map] = 0.40 * semantic_map_viz[
                semantic_map
            ] + 0.60 * torch.tensor([1.0, 0.0, 0.0], device="cuda")
            semantic_map_viz = semantic_map_viz.permute(
                2, 0, 1
            )  # 512 512 3 to 3 512 512

            semantics.append(semantic_map_viz)
            masks.append(semantic_map)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

        self.gaussian.localize = False  # reverse

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        semantics = torch.stack(semantics, dim=0)
        masks = torch.stack(masks, dim=0)

        render_pkg["semantic"] = semantics
        render_pkg["masks"] = masks
        self.visibility_filter = self.radii > 0.0
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)

        return {
            **render_pkg,
        }

    def render_all_view(self, cache_name):
        cache_dir = os.path.join(self.cache_dir, cache_name)
        os.makedirs(cache_dir, exist_ok=True)
        with torch.no_grad():
            for id in tqdm(range(self.trainer.datamodule.train_dataset.total_view_num)):
                cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                if not os.path.exists(cur_path) or self.cfg.cache_overwrite:
                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                    cur_batch = {
                        "index": id,
                        "camera": [cur_cam],
                        "height": self.trainer.datamodule.train_dataset.height,
                        "width": self.trainer.datamodule.train_dataset.width,
                    }
                    out = self(cur_batch)["comp_rgb"]
                    out_to_save = (
                            out[0].cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    out_to_save = cv2.cvtColor(out_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_path, out_to_save)
                cached_image = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
                self.origin_frames[id] = torch.tensor(
                    cached_image / 255, device="cuda", dtype=torch.float32
                )[None]

    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            if self.true_global_step < self.cfg.densify_until_iter:
                viewspace_point_tensor_grad = torch.zeros_like(
                    self.viewspace_point_list[0]
                )
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = (
                            viewspace_point_tensor_grad
                            + self.viewspace_point_list[idx].grad
                    )
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter],
                    self.radii[self.visibility_filter],
                )
                self.gaussian.add_densification_stats(
                    viewspace_point_tensor_grad, self.visibility_filter
                )
                # Densification
                if (
                        self.true_global_step >= self.cfg.densify_from_iter
                        and self.true_global_step % self.cfg.densification_interval == 0
                ):  # 500 100
                    self.gaussian.densify_and_prune(
                        self.cfg.max_grad,
                        self.cfg.max_densify_percent,
                        self.cfg.min_opacity,
                        self.cameras_extent,
                        5,
                    )

    def validation_step(self, batch, batch_idx):
        batch["camera"] = [
            self.trainer.datamodule.train_dataset.scene.cameras[idx]
            for idx in batch["index"]
        ]
        out = self(batch)
        for idx in range(len(batch["index"])):
            cam_index = batch["index"][idx].item()
            self.save_image_grid(
                f"it{self.true_global_step}-val/{batch['index'][idx]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": self.origin_frames[cam_index][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                        {
                            "type": "rgb",
                            "img": self.edit_frames[cam_index][0]
                            if cam_index in self.edit_frames
                            else torch.zeros_like(self.origin_frames[cam_index][0]),
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name=f"validation_step_{idx}",
                step=self.true_global_step,
            )
            self.save_image_grid(
                f"render_it{self.true_global_step}-val/{batch['index'][idx]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][idx],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][idx],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["semantic"][idx].moveaxis(0, -1),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "semantic" in out
                    else []
                ),
                name=f"validation_step_render_{idx}",
                step=self.true_global_step,
            )

    def test_step(self, batch, batch_idx):
        only_rgb = True  # TODO add depth test step
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        batch["camera"] = [
            self.trainer.datamodule.val_dataset.scene.cameras[batch["index"]]
        ]
        testbackground_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )

        out = self(batch, testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=5,
            name="test",
            step=self.true_global_step,
        )
        save_list = []
        for index, image in sorted(self.edit_frames.items(), key=lambda item: item[0]):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
                    "kwargs": {"data_format": "HWC"},
                },
            )
        if len(save_list) > 0:
            self.save_image_grid(
                f"edited_images.png",
                save_list,
                name="edited_images",
                step=self.true_global_step,
            )
        save_list = []
        for index, image in sorted(
                self.origin_frames.items(), key=lambda item: item[0]
        ):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
                    "kwargs": {"data_format": "HWC"},
                },
            )
        self.save_image_grid(
            f"origin_images.png",
            save_list,
            name="origin",
            step=self.true_global_step,
        )

        save_path = self.get_save_path(f"last.ply")
        print("save_path", save_path)
        self.gaussian.save_ply(save_path)

    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
        self.view_list = self.trainer.datamodule.train_dataset.n2n_view_index
        self.view_num = len(self.view_list)
        opt = OptimizationParams(self.parser, self.trainer.max_steps, self.cfg.gs_lr_scaler, self.cfg.gs_final_lr_scaler, self.cfg.color_lr_scaler,
                                 self.cfg.opacity_lr_scaler, self.cfg.scaling_lr_scaler, self.cfg.rotation_lr_scaler, )
        self.gaussian.load_ply(self.cfg.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        self.cameras_extent = self.trainer.datamodule.train_dataset.scene.cameras_extent
        self.gaussian.spatial_lr_scale = self.cameras_extent

        self.pipe = PipelineParams(self.parser)
        opt = OmegaConf.create(vars(opt))
        opt.update(self.cfg.training_args)
        self.gaussian.training_setup(opt)

        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret
    
    def edit_all_view(self, original_render_name, cache_name, update_camera=False, global_step=0):
        # if self.true_global_step >= self.cfg.camera_update_per_step * 2:
        #     self.guidance.use_normal_unet()
        
        self.edited_cams = []
        if update_camera:
            # this is in the case that you resume training with new dataset cameras
            self.trainer.datamodule.train_dataset.update_cameras(random_seed = global_step + 1)
            self.view_list = self.trainer.datamodule.train_dataset.n2n_view_index
            sorted_train_view_list = sorted(self.view_list)
            selected_views = torch.linspace(
                0, len(sorted_train_view_list) - 1, self.trainer.datamodule.val_dataset.n_views, dtype=torch.int
            )
            self.trainer.datamodule.val_dataset.selected_views = [sorted_train_view_list[idx] for idx in selected_views]

        self.edit_frames = {}
        cache_dir = os.path.join(self.cache_dir, cache_name)
        original_render_cache_dir = os.path.join(self.cache_dir, original_render_name)
        os.makedirs(cache_dir, exist_ok=True)

        cameras = []
        images = []
        original_frames = []
        t_max_step = self.cfg.added_noise_schedule
        self.guidance.max_step = t_max_step[min(len(t_max_step)-1, self.true_global_step//self.cfg.camera_update_per_step)]
        with torch.no_grad():
            for id in self.view_list:
                cameras.append(self.trainer.datamodule.train_dataset.scene.cameras[id])
            sorted_cam_idx = self.sort_the_cameras_idx(cameras)
            view_sorted = [self.view_list[idx] for idx in sorted_cam_idx]
            cams_sorted = [cameras[idx] for idx in sorted_cam_idx]     
                   
            for id in view_sorted:
                cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                original_image_path = os.path.join(original_render_cache_dir, "{:0>4d}.png".format(id))
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                cur_batch = {
                    "index": id,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                out_pkg = self(cur_batch)
                out = out_pkg["comp_rgb"]
                if self.cfg.use_masked_image:
                    out = out * out_pkg["masks"].unsqueeze(-1)
                images.append(out)
                assert os.path.exists(original_image_path)
                cached_image = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)
                frame_t = torch.tensor(
                    cached_image / 255, device="cuda", dtype=torch.float32
                )[None]
                # self.origin_frames[id] = torch.tensor(
                #     cached_image / 255, device="cuda", dtype=torch.float32
                # )[None]
                self.origin_frames[id] = frame_t
                original_frames.append(self.origin_frames[id])
            images = torch.cat(images, dim=0)
            original_frames = torch.cat(original_frames, dim=0)

            edited_images = self.guidance(
                images,
                original_frames,
                self.prompt_processor(),
                cams = cams_sorted
            )
            images_t = edited_images["edit_images"]

            for view_index_tmp in range(len(self.view_list)):
                self.edit_frames[view_sorted[view_index_tmp]] = edited_images['edit_images'][view_index_tmp].unsqueeze(0).detach().clone() # 1 H W C
    
    def sort_the_cameras_idx(self, cams):
        foward_vectos = [cam.R[:, 2] for cam in cams]
        foward_vectos = np.array(foward_vectos)
        cams_center_x = np.array([cam.camera_center[0].item() for cam in cams])
        most_left_vecotr = foward_vectos[np.argmin(cams_center_x)]
        distances = [np.arccos(np.clip(np.dot(most_left_vecotr, cam.R[:, 2]), 0, 1)) for cam in cams]
        sorted_cams = [cam for _, cam in sorted(zip(distances, cams), key=lambda pair: pair[0])]
        reference_axis = np.cross(most_left_vecotr, sorted_cams[1].R[:, 2])
        distances_with_sign = [np.arccos(np.dot(most_left_vecotr, cam.R[:, 2])) if np.dot(reference_axis,  np.cross(most_left_vecotr, cam.R[:, 2])) >= 0 else 2 * np.pi - np.arccos(0 * np.dot(most_left_vecotr, cam.R[:, 2])) for cam in cams]

        sorted_cam_idx = [idx for _, idx in sorted(zip(distances_with_sign, range(len(cams))), key=lambda pair: pair[0])]

        return sorted_cam_idx

    def on_fit_start(self) -> None:
        # START HERE
        super().on_fit_start()
        print("Rendering all views...")
        self.render_all_view(cache_name="origin_render")
        # render_all_view just renders all the dataset images (without editing) and saves them to the cache
        # note: it is called "view" here because each image in the dataset is a view of the scene

        # if len(self.cfg.seg_prompt) > 0:
        #     self.update_mask()
        # update_mask here is a function that uses the text_segmentor to segment the images in the dataset
        # it then uses the cuda render functions via .apply_weights() to render an image of the black and white mask
        # the masks then get saved via gaussian.set_mask and gaussian.apply_grad_mask
        # where the mask is an average of all the calculated masks
        # where apply_grad_mask is a function that appends hooks to the gradients so they are calculated differently
        # so basically, the mask makes each parameter of the gaussian model calculate with the mask via the hook

        # TODO let's change it so we get all the masks for each time frame
        print("Updating masks...")
        if len(self.cfg.seg_prompt) > 0:
            self.update_masks()

        print("Initializing prompt processor...")
        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        # this processes the editing prompt (not the segmentation)
        # for example cfg.prompt_processor is a dictionary like so:
        # {
        #   'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
        #   'prompt': 'turn the cookie to red'
        # }
        # and cfg.prompt_processor_type is 'stable-diffusion-prompt-processor'
        # where the class was registered in the dictionary __modules__ using the python decorator
        # so the default registered classes are found in prompt_processors/__init__.py
        # Thus, self.prompt_processor is the class __init__ with self.cfg.prompt_processor

        print("Initializing guidance...")
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0 or self.cfg.loss.use_sds:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        # this is the editing heuristic
        # after this on_fit_start, the regular training cycle is called
        # where the below training_step is called for each batch

    def training_step(self, batch, batch_idx):
        # print("Editing all view...")
        if self.true_global_step % self.cfg.camera_update_per_step == 0 and self.cfg.guidance_type == 'dge-guidance' and not self.cfg.loss.use_sds:
            self.edit_all_view(original_render_name='origin_render', cache_name="edited_views", update_camera=self.true_global_step >= self.cfg.camera_update_per_step, global_step=self.true_global_step) 
        # now that on_fit_start has finished, we can edit the rendered views
        # basically, in self.edit_all_view, the dataset is updated first just in case
        # then, the forward step is called where the output is a dict pkg with the rendered image and mask as separate
        # edit_all_view takes these two outputs and multiplies them together
        # note!!! since the rendered image and rendered mask is needed in edit_all_view, this is why render_all_view and update_mask is called first
        # the new images are then saved to a new self.edit_frames (so we can view the edit only output)
        # where each edited image is just an image of the rendered segmentation portion (everything else is masked out)

        self.gaussian.update_learning_rate(self.true_global_step)
        batch_index = batch["index"]

        if isinstance(batch_index, int):
            batch_index = [batch_index]
        if self.cfg.guidance_type == 'dge-guidance': 
            for img_index, cur_index in enumerate(batch_index):
                if cur_index not in self.edit_frames:
                    batch_index[img_index] = self.view_list[img_index]

        # print("Forward pass...")
        out = self(batch, local=self.cfg.local_edit)
        # now, we set the local gaussian_model.localize attribute which is used for the masked rendering
        # iterating over the views again, the forward method is called once again
        # such that the render function is called twice as always, but the localize attribute
        # makes the render only affect the gaussians with the masked indices
        # the second call the render here is just redundant as the masked masked is just the mask itself
        # then everything below is just hueristics for loss calculation

        images = out["comp_rgb"]
        mask = out["masks"].unsqueeze(-1)
        loss = 0.0
        # nerf2nerf loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            prompt_utils = self.prompt_processor()
            gt_images = []
            for img_index, cur_index in enumerate(batch_index):
                # if cur_index not in self.edit_frames:
                #     # cur_index = self.view_list[0]
                if (cur_index not in self.edit_frames or (
                        self.cfg.per_editing_step > 0
                        and self.cfg.edit_begin_step
                        < self.global_step
                        < self.cfg.edit_until_step
                        and self.global_step % self.cfg.per_editing_step == 0
                )) and 'dge' not in str(self.cfg.guidance_type) and not self.cfg.loss.use_sds:
                    result = self.guidance(
                        images[img_index][None],
                        self.origin_frames[cur_index],
                        prompt_utils,
                    )
                 
                    self.edit_frames[cur_index] = result["edit_images"].detach().clone()

                gt_images.append(self.edit_frames[cur_index])
            gt_images = torch.concatenate(gt_images, dim=0)
            if self.cfg.use_masked_image:
                guidance_out = {
                "loss_l1": torch.nn.functional.l1_loss(images * mask, gt_images * mask),
                # "loss_p": self.perceptual_loss(
                #     (images * mask).permute(0, 3, 1, 2).contiguous(),
                #     (gt_images * mask ).permute(0, 3, 1, 2).contiguous(),
                # ).sum(),
                } 
            else:
                guidance_out = {
                    "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                    # "loss_p": self.perceptual_loss(
                    #     images.permute(0, 3, 1, 2).contiguous(),
                    #     gt_images.permute(0, 3, 1, 2).contiguous(),
                    # ).sum(),
                }
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )
        # sds loss
        if self.cfg.loss.use_sds:
            prompt_utils = self.prompt_processor()
            self.guidance.cfg.use_sds = True
            guidance_out = self.guidance(
                out["comp_rgb"],
                torch.concatenate(
                    [self.origin_frames[idx] for idx in batch_index], dim=0
                ),
                prompt_utils) 
            loss += guidance_out["loss_sds"] * self.cfg.loss.lambda_sds 

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}
    
    # Debugging
    def backward(self, loss):
        # print("We are entering the backward step")
        if loss.isnan() or loss.isinf():
            print("Bad batch found.")
            return
        loss.backward()
        # print("Finished backward step")

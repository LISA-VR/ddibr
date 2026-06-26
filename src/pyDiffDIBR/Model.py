"""
This file is part of "∂DIBR", a differentiable renderer based on Depth Image Based Rendering (DIBR) techniques for fast Novel View Synthesis.
Copyright (C) 2026 "Université Libre de Bruxelles (ULB)". All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact:
    - Armand Losfeld - armand.lfd.pro@proton.me
    - Daniele Bonatto - daniele.bonatto@ulb.be
"""

from typing import Union, TypedDict, Any, Optional

import slangpy as spy

from ._utils import ArrayUtils, Logger, IOUtils, TypeUtils
from .Camera import Camera, ImagePair
from .Dataset import Dataset, DatasetIO, DatasetCameraSelectorUtils, DatasetTrainingUtils

from .RenderingDevice import RenderingDevice
from .RenderingApp import RenderingApp, RenderingObjectType, RenderingStageType
from .FwdRendering import FwdRenderingApp
from .BwdRendering import BwdRenderingApp

from .Loss import AnisotropicTotalVariation, LossDataParameterDict, Loss


ModelParameterDict = TypedDict("ModelParameterDict", {
    "name" : str,
    "size" : tuple[int, int, int],
    "data" : Any,
    "grad" : spy.Buffer,
    "uniform" : TypeUtils.UniformGPUData_t,
}, total = True)


class DiffDIBR:
    def __init__(
        self,
        model_cams : list[Camera],
        nb_active_cameras : Optional[int] = None,
        is_training : bool = False,
        toml_path : Optional[str] = None,
        mesh_reduction : float | None = None,
        nb_samples_ms : int = RenderingApp.DEFAULT_NB_SAMPLES_MS
    ):
        if len(model_cams) <= 0: 
            raise ValueError("Cannot run without any input camera")
        self.model_cameras = model_cams
        self.total_cameras = len(self.model_cameras)
        self.nb_active_cameras : int = nb_active_cameras if nb_active_cameras is not None else len(model_cams)
        self.is_training = is_training

        self.fwd_app = FwdRenderingApp(nb_samples=nb_samples_ms) if toml_path is None else FwdRenderingApp.from_toml(toml_path, nb_samples_ms=nb_samples_ms)
        self._bwd_app : Optional[BwdRenderingApp] = None
        if is_training:
            if toml_path is not None:
                self._bwd_app = BwdRenderingApp.from_toml(self.fwd_app, toml_path) 
            else:
                self._bwd_app = BwdRenderingApp(self.fwd_app) 

        self.fwd_app.load_program()
        if is_training: 
            self.bwd_app.load_program()
        
        self._mesh_reduction : float = 0.0 if mesh_reduction is None else mesh_reduction
        self.load_input()
    
    def simplify_mesh(self):
        if self._mesh_reduction <= 0.0:
            return
        for input_idx in range(len(self.fwd_app.input_views)):
            self.fwd_app.simplify_input_mesh(input_idx, self._mesh_reduction)
    
    def load_input(self):
        self.app.release_input_views()
        for cam in self.model_cameras:
            self.app.load_input_view(cam)
        self.simplify_mesh()
    
    @property
    def app(self) -> RenderingApp:
        if self.is_training:
            return self.bwd_app
        else:
            return self.fwd_app

    @property
    def bwd_app(self) -> BwdRenderingApp:
        if self._bwd_app is None:
            raise ValueError("BwdApp is none.")
        return self._bwd_app

    @classmethod
    def from_dataset(
        cls,
        dataset : Dataset, 
        nb_cams : int,
        nb_active_cams : Optional[int] = None,
        is_training : bool = False,
        ratio_rnd_color : float = 0.1, ratio_rnd_depth : float = 0.05,
        project_cam_center : bool = True,
        toml_path : Optional[str] = None,
        mesh_reduction : float | None = None,
        nb_samples_ms : int = RenderingApp.DEFAULT_NB_SAMPLES_MS
    ):
        model_cams : list[Camera] = []
        
        if nb_cams == len(dataset):
            model_cams = [DatasetTrainingUtils.copy_model_camera_for_training(cam, ratio_rnd_color, ratio_rnd_depth) for cam in dataset.cameras]
        else:
            model_cams = DatasetTrainingUtils.find_model_cameras_with_PAM(dataset, nb_cams, ratio_rnd_color, ratio_rnd_depth, project_center=project_cam_center)
            #model_cams = DatasetTrainingUtils.find_model_cameras_with_MVCP(dataset, nb_cams, ratio_rnd_color, ratio_rnd_depth) # todo: remove completely?
        
        app = cls(model_cams, nb_active_cams, is_training, toml_path, mesh_reduction, nb_samples_ms)
        return app

    def fwd(
        self,
        target_cam_idx : int,
        active_cam_idx : Union[list[int], None] = None,
    ):
        if active_cam_idx is None:
            active_cam_idx = DatasetCameraSelectorUtils.select_input_views(self.model_cameras, self.app.target_views[target_cam_idx]["camera"], self.nb_active_cameras, True, False)
        self.fwd_app.clear_blend(target_cam_idx)

        for i in active_cam_idx:
            self.fwd_app.render(i, target_cam_idx)
            self.fwd_app.blend_view(i,target_cam_idx)
        self.fwd_app.normalize_blending(target_cam_idx)

    def bwd(
        self,
        target_cam_idx : int,
        active_cam_idx : list[int],
    ):    
        for i in active_cam_idx:
            self.bwd_app.backward_blending(i, target_cam_idx)
            self.bwd_app.bwd_render(i, target_cam_idx)
        self.bwd_app.clear_intermediate()

    def get_result(
        self, target_cam_idx : int
    ):
        return {
            "color" : self.get_result_color(target_cam_idx),
            "depth" : self.get_result_depth(target_cam_idx),
            "quality" : self.get_result_quality(target_cam_idx),
        }

    def get_result_color(
        self, target_cam_idx : int
    ):
        return self.fwd_app.get_result_color(target_cam_idx)

    def get_result_depth(
        self, target_cam_idx : int
    ):
        return self.fwd_app.get_result_depth(target_cam_idx)

    def get_result_quality(
        self, target_cam_idx : int
    ):
        return self.fwd_app.get_result_quality(target_cam_idx)
    
    def get_result_mask(
        self, target_cam_idx : int
    ):
        return self.fwd_app.get_result_mask(target_cam_idx)
    
    def get_result_normal(
        self, target_cam_idx : int
    ):
        return self.fwd_app.get_result_normal(target_cam_idx)
    
    def save(
        self, out_dir : str | None, img_format : str = "{}_{}.yuv", camera_json_name : str = "camera.json"
    ):
        self._save_ckpt(out_dir, img_format, camera_json_name)
    
    def _save_ckpt(
        self, out_dir : str | None, img_format : str = "{}_{}.yuv", camera_json_name : str = "camera.json"
    ):
        if out_dir is None: return
        IOUtils.create_dirs(out_dir)

        Logger.info("-" * 10 + f" Save Model Parameters (Color & Depth) " + "-" * 10 )
        for i in range(len(self.app.input_views)):
            input_view = self.app.input_views[i]
            learned_color = input_view["camera"].image_pair.color.texture.to_numpy()
            learned_color = ArrayUtils.sigmoid(learned_color)
            learned_depth = input_view["camera"].image_pair.depth.texture.to_numpy()
            learned_depth = ArrayUtils.sigmoid(learned_depth, shift = input_view["camera"].cpu_params.depth_range[0], scale = input_view["camera"].cpu_params.depth_range[1] - input_view["camera"].cpu_params.depth_range[0])

            image_pair = ImagePair(learned_color, learned_depth, 'float32', False, "", False)
            image_pair.save_color(out_dir + img_format.format("trained_color", input_view["camera"].name))
            image_pair.save_depth(out_dir + img_format.format("trained_depth", input_view["camera"].name), input_view["camera"].cpu_params.depth_range)

        Logger.info("-" * 10 + f" Save Camera JSON " + "-" * 10 )
        DatasetIO.save_to_json(Dataset(self.model_cameras, 0, False), out_dir + camera_json_name)

    def update_model_cpu_storage(self):
        self.app.device_handler.wait()
        for camera in self.model_cameras:
            camera.update_cpu()

    def _release_batch(self, batch : list[Camera]):
        Logger.debug(f"Release current batch", str(self.__class__))
        self.app.release_target_views()
        
        for cam in batch:
            cam.image_pair.color.release_texture()
            cam.image_pair.depth.release_texture()

    def _reload_batch(self, batch : list[Camera], load_gpu : bool = False):
        Logger.debug(f"Reload batch", str(self.__class__))

        for cam in batch:
            if load_gpu:
                cam.image_pair.color.reload_gpu_texture()
                cam.image_pair.depth.reload_gpu_texture()

            self.app.load_target_view(cam)
        
        if self.is_training:    
            self.bwd_app.load_intermediate_results()

    def _retrieve_loss_data(self, input_idx : int, output_idx : int, loss : Loss) -> list[LossDataParameterDict]:
        loss_data = []
        loss_keys = []

        for obj in loss.objects:
            uniform_data = None
            metadata = None
            camera : Camera | None = None

            if obj.is_input():
                camera = self.bwd_app._input_views[input_idx]["camera"]
                uniform_data = self.bwd_app._input_views[input_idx]["uniform"]

                if obj.type == RenderingObjectType.Color:
                    uniform_data = uniform_data["color_texture"]
                elif obj.type == RenderingObjectType.Depth:
                    uniform_data = uniform_data["depth_texture"]
                    if isinstance(loss, AnisotropicTotalVariation):
                        metadata = {}
                        metadata["scale_sigmoid"] = camera.cpu_params.depth_range[1] - camera.cpu_params.depth_range[0]
                        metadata["shift_sigmoid"] = camera.cpu_params.depth_range[0]
                else:
                    raise NotImplementedError("") #todo : 
            elif obj.is_target():
                camera = self.bwd_app._target_views[output_idx]["camera"]
                uniform_data = self.bwd_app._target_views[output_idx]["uniform"]
                if metadata is None: metadata = {}
                metadata["mask"] = self.bwd_app._target_views[output_idx]["uniform"]["blendingResult"]["mask"]

                if obj.type == RenderingObjectType.Color:
                    uniform_data = uniform_data["color_texture"]
                elif obj.type == RenderingObjectType.Depth:
                    uniform_data = uniform_data["depth_texture"]
                else:
                    raise NotImplementedError("") #todo : 
            elif obj.stage == RenderingStageType.BlendedResult:
                camera = self.bwd_app._target_views[output_idx]["camera"]
                uniform_data = self.bwd_app._target_views[output_idx]["uniform"]["blendingResult"]

                if obj.type == RenderingObjectType.Color:
                    uniform_data = uniform_data["color"]
                elif obj.type == RenderingObjectType.Depth:
                    uniform_data = uniform_data["depth"]
                elif obj.type == RenderingObjectType.Quality:
                    uniform_data = uniform_data["quality"]
                else:
                    raise NotImplementedError("") #todo : 
            elif obj.stage == RenderingStageType.IntermediateResult:
                camera = self.bwd_app._input_views[input_idx]["camera"]
                uniform_data = self.bwd_app.get_intermediate_result_uniform(input_idx, output_idx)["drawResult"]
                if metadata is None: metadata = {}
                metadata["mask"] = uniform_data["mask"]

                if obj.type == RenderingObjectType.Color:
                    uniform_data = uniform_data["color"]
                elif obj.type == RenderingObjectType.Depth:
                    uniform_data = uniform_data["depth"]
                elif obj.type == RenderingObjectType.Quality:
                    uniform_data = uniform_data["quality"]
                else:
                    raise NotImplementedError("") #todo : 
                            
            if uniform_data is None: raise ValueError("Uniform could not be retrieved.")
            if camera is None: raise ValueError("Camera could not be retrieved.")

            loss_data.append(LossDataParameterDict({"name" : camera.name, "size" : camera.cpu_params.resolution, "uniform_data" : uniform_data, "metadata" : metadata}))
            loss_keys.append(obj.stage.value)

        loss_data : list[LossDataParameterDict] = [data for data, _ in sorted(zip(loss_data, loss_keys), key = lambda d : d[1])]

        return loss_data
    
    def get_texture_parameters(self, idx_input_view : int, texture_type : RenderingObjectType = RenderingObjectType.Color) -> ModelParameterDict:
        if (texture_type !=  RenderingObjectType.Color) and (texture_type !=  RenderingObjectType.Depth):
            raise ValueError("This function can only be called to get texture (Color, Depht) parameters.")
        
        label = "color" if texture_type ==  RenderingObjectType.Color else "depth"
        in_view = self.bwd_app._input_views[idx_input_view]
        in_view["propagate_grad"][label] = True
        in_view["uniform"] = self.bwd_app.get_input_view_uniform(in_view)

        data : spy.Texture
        grad : spy.Buffer
        size : tuple[int, int, int]
        uniform : TypeUtils.UniformGPUData_t
        if texture_type ==  RenderingObjectType.Color:
            data = in_view["camera"].image_pair.color.texture
            grad = in_view["camera"].image_pair.color.grad
            size = (in_view["camera"].cpu_params.resolution[0], in_view["camera"].cpu_params.resolution[1], 4)
            uniform = {"diff_tex" : in_view["uniform"][f"{label}_texture"]}
        elif texture_type ==  RenderingObjectType.Depth:
            data = in_view["camera"].image_pair.depth.texture
            grad = in_view["camera"].image_pair.depth.grad
            size = (in_view["camera"].cpu_params.resolution[0], in_view["camera"].cpu_params.resolution[1], 1)
            uniform = {"diff_tex" : in_view["uniform"][f"{label}_texture"]}
        else:
            raise Exception(f"Error - {BwdRenderingApp.__name__}: color|depth texture are the only learnable features.")
        

        return ModelParameterDict({
            "name" : f"{label}_{in_view['camera'].name}",
            "size" : size,
            "data": data,
            "grad": grad,
            "uniform" : uniform
        })
    
    def get_camera_parameters(self, idx_input_view : int, *selected_var_names) -> ModelParameterDict:
        in_view = self.bwd_app._input_views[idx_input_view]
        max_size = in_view["camera"].gpu_params.cam_param_gpu.size // 4
        selected = in_view["camera"].select_parameters_for_optimization(selected_var_names)
        in_view["propagate_grad"]["camera_parameter"] = True
        in_view["uniform"] = self.bwd_app.get_input_view_uniform(in_view)

        selected_params = RenderingDevice().create_buffer(selected, label=f"param_select_cam_{in_view['camera'].name}", usage=spy.BufferUsage.shader_resource)

        return ModelParameterDict({
            "name" : f"cam_params_{in_view['camera'].name}",
            "size" : (max_size, 1, 1),
            "data": in_view["camera"].gpu_params.cam_param_gpu,
            "grad": in_view["camera"].gpu_params.grad_buffer_for_cam_params,
            "uniform" : {
                "selector" : selected_params,
                "diff_buffer": in_view["uniform"]["params"]["param"]
            }
        })

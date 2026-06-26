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

from enum import Enum
from typing import TypedDict, Any
import numpy as np 
import slangpy as spy

from ._utils import ImageUtils, UniformUtils, IOUtils, Logger, ArrayUtils
from .Image import Image, RenderedImage, MultiSampledRenderedImage
from .Camera import Camera, RenderedViewpoint, BlendedViewpoint
from .MeshBuffer import SimpleFlatMeshBuffer, MeshBuffer, SimplifiableMeshBuffer
from .RenderingDevice import RenderingDevice, SubmitId
from .RenderingApp import RenderingApp
from .Blending import BlendingMethod, Blending

class ViewSynthesisMethod(Enum):
    Triangle = 1

class QualityParameter(TypedDict):
    Lambda  : float
    Power   : float

class QualityList(TypedDict):
    DiscardingTreshold  : float

    ShapeBased          : QualityParameter
    SideBased           : QualityParameter
    DepthBased          : QualityParameter
    DepthTestBased      : QualityParameter   
    NormalBased         : QualityParameter   
    CameraAlignBased    : QualityParameter   

UniformType = dict[str, Any]

TextureDict = TypedDict("TextureDict",
{
    "color"         : spy.Texture,
    "color_view"    : spy.TextureView,
    "depth"         : spy.Texture,
    "depth_view"    : spy.TextureView,
}, total = True)

IntermediateResultTextureDict = TypedDict("IntermediateResultTextureDict",
{
    "color"             : RenderedImage | Image,
    "depth"             : RenderedImage | Image,
    "quality"           : RenderedImage | Image,
    "mask"              : RenderedImage | Image,
    "ms_color"          : MultiSampledRenderedImage | Image,
    "ms_depth"          : MultiSampledRenderedImage | Image,
    "ms_quality"        : MultiSampledRenderedImage | Image,
    "ms_mask"           : MultiSampledRenderedImage | Image,
}, total = True)

InputViewDict = TypedDict("InputViewDict", 
{
    "camera"                : Camera,
    "mesh_buffer"           : MeshBuffer,
    "uniform"               : UniformType
}, total = True)


DepthBufferDict = TypedDict("DepthBufferDict", {
    "texture"       : spy.Texture,
    "texture_view"  : spy.TextureView
})

TargetOutputViewDict = TypedDict("TargetOutputViewDict", 
{
    "camera"                : Camera,
    "depth_buffer"          : DepthBufferDict,
    "rendered_viewpoint"    : RenderedViewpoint,
    "blended_viewpoint"     : BlendedViewpoint,
    "uniform"               : UniformType
}, total = True)

# ------------------------------------ UNIFORMS ------------------------------------ 

TensorizedBufferUniformDict = TypedDict("TensorizedBufferUniformDict", {
    "buffer" : spy.Buffer,
    "size" : int
})

CsteBufferUniformDict = TypedDict("CsteBufferUniformDict", {
    "buff_" : TensorizedBufferUniformDict
})

CameraParametersUniformDict = TypedDict("CameraParametersUniformDict", 
{
    "proj_type": int,
    "imgSize" : tuple[int, int],
    "focal" : tuple[float, float],
    "princ_p" : tuple[float, float],
    "min_max_depth": tuple[float, float],
    "pos": tuple[float, float, float],
    "rot": tuple[float, float, float],
    "param" : CsteBufferUniformDict,
}, total = False)

CstTextureUniformDict = TypedDict("CstTextureUniformDict", {
    "texture" : spy.Texture
}, total = False)

CameraUniformDict = TypedDict("CameraUniformDict", 
{
    "params" : CameraParametersUniformDict,
    "depth_texture": CstTextureUniformDict,
    "color_texture": CstTextureUniformDict,
    "sampler_tex": spy.Sampler,
}, total = False)

SpaceTransformUniformDict = TypedDict("SpaceTransformUniformDict", 
{
    "R_t" : CsteBufferUniformDict,
}, total = False)

QualityParametersUniformDict = TypedDict("QualityParametersUniformDict", 
{
    "lambda_quality_shape"              : float,
    "lambda_quality_normal"             : float,
    "lambda_quality_depth"              : float,
    "lambda_quality_side"               : float,
    "lambda_quality_foreground"         : float,
    "lambda_quality_face_camera"        : float,
    "scaling_power_quality_shape"              : float,
    "scaling_power_quality_normal"             : float,
    "scaling_power_quality_depth"              : float,
    "scaling_power_quality_side"               : float,
    "scaling_power_quality_foreground"         : float,
    "scaling_power_quality_face_camera"        : float,
    "quality_treshold"                  : float,
}, total = False)

UniformsDict = TypedDict("UniformsDict", 
{
    "transform"         : SpaceTransformUniformDict,
    "out_cam_params"    : CameraParametersUniformDict,
    "in_diff_cam"       : CameraUniformDict,
    "qParams"           : QualityParametersUniformDict
}, total = False)

class FwdRenderingApp(RenderingApp):
    VERTEX_SEMANTIC_NAME    = "POSITION"

    _RENDER_PROGRAM_FNAME = "synthesis.slang"

    _KERNEL_RT_COMPUTATION_NAME = "SpaceTransformer.slang"
    _KERNEL_RT_COMPUTATION_EP   = "single_thread_kernel_compute_R_t"
                        
    _QUALITY_TOML_KEY = "quality"
    _QUALITY_0_TOML_KEY = "threshold"
    _QUALITY_SCALE_TOML_KEY = "scale"
    _QUALITY_POWER_TOML_KEY = "power"
    
    _DEPTH_QUALITY_TOML_KEY = "depth"
    _FOREGROUND_QUALITY_TOML_KEY = "foreground"
    _SIDE_QUALITY_TOML_KEY = "side"
    _SHAPE_QUALITY_TOML_KEY = "shape"
    _FACE_CAMERA_QUALITY_TOML_KEY = "face_camera"
    _NORMAL_QUALITY_TOML_KEY = "normal"
                                                                                          
    def __init__(
        self, 
        blend_m : BlendingMethod = BlendingMethod.Masked,
        quality_shape        : QualityParameter = QualityParameter(Lambda=1.0, Power=1.0),
        quality_normal       : QualityParameter = QualityParameter(Lambda=1.0, Power=1.0),
        quality_depth        : QualityParameter = QualityParameter(Lambda=1.0, Power=1.0),
        quality_side         : QualityParameter = QualityParameter(Lambda=1.0, Power=1.0),
        quality_foreground   : QualityParameter = QualityParameter(Lambda=1.0, Power=1.0),
        quality_face_cam     : QualityParameter = QualityParameter(Lambda=1.0, Power=1.0),
        quality_treshold     : float = 0.15,
        default_t4_color     : tuple[float, float, float, float] = (0, 0, 0, 0),
        default_t1_color     : tuple[float, ] = (0, ),
        nb_samples           : int = RenderingApp.DEFAULT_NB_SAMPLES_MS,
        ):
        self.Precision = 1.0
        self.ViewSynthesisMethod = ViewSynthesisMethod.Triangle
        self.blending = Blending.from_enum(blend_m, default_value_t4=default_t4_color, default_value_t1=default_t1_color) #Blending(blend_m)

        super().__init__()

        self.entry_point_names = []
        self.device_handler = RenderingDevice()

        self._input_views : list[InputViewDict] = [] 
        self._target_views : list[TargetOutputViewDict] = [] 
        self._intermediate_results  : dict[tuple[int, int], RenderedViewpoint] | None = None
        
        self.sampler = self.device_handler.create_sampler() #todo: move it somewhere else

        self.nb_samples = nb_samples

        self.quality = QualityList(
            DiscardingTreshold=quality_treshold,
            ShapeBased=quality_shape,
            SideBased = quality_side,
            DepthBased = quality_depth,
            DepthTestBased = quality_foreground,
            NormalBased = quality_normal,
            CameraAlignBased = quality_face_cam,
        )
        
        for info in self.info: Logger.debug(info, str(self.__class__))

    @property
    def target_views(self) -> list[TargetOutputViewDict]:
        return self._target_views
    
    @property
    def input_views(self) -> list[InputViewDict]:
        return self._input_views

    @property
    def info(self) -> list[str]:    
        info = []
        info.append("-"*5 + "FwdApp" + "-"*5)

        return info

    @classmethod
    def from_toml(cls, path_toml : str, nb_samples_ms : int = RenderingApp.DEFAULT_NB_SAMPLES_MS):
        data = IOUtils.read_toml(path_toml)

        fwd_app_data = data.get(RenderingApp._APP_TOML_KEY)
        if fwd_app_data is None:
            return FwdRenderingApp()
        
        th = data[RenderingApp._APP_TOML_KEY][FwdRenderingApp._QUALITY_TOML_KEY][FwdRenderingApp._QUALITY_0_TOML_KEY]
        blending = data[RenderingApp._APP_TOML_KEY][Blending._BLENDING_TOML_KEY]
        
        depth = data[RenderingApp._APP_TOML_KEY][FwdRenderingApp._QUALITY_TOML_KEY][FwdRenderingApp._DEPTH_QUALITY_TOML_KEY]
        foreground = data[RenderingApp._APP_TOML_KEY][FwdRenderingApp._QUALITY_TOML_KEY][FwdRenderingApp._FOREGROUND_QUALITY_TOML_KEY]
        side = data[RenderingApp._APP_TOML_KEY][FwdRenderingApp._QUALITY_TOML_KEY][FwdRenderingApp._SIDE_QUALITY_TOML_KEY]
        shape = data[RenderingApp._APP_TOML_KEY][FwdRenderingApp._QUALITY_TOML_KEY][FwdRenderingApp._SHAPE_QUALITY_TOML_KEY]
        face_camera = data[RenderingApp._APP_TOML_KEY][FwdRenderingApp._QUALITY_TOML_KEY][FwdRenderingApp._FACE_CAMERA_QUALITY_TOML_KEY]
        normal = data[RenderingApp._APP_TOML_KEY][FwdRenderingApp._QUALITY_TOML_KEY][FwdRenderingApp._NORMAL_QUALITY_TOML_KEY]

        return FwdRenderingApp(
            quality_depth           = QualityParameter(Lambda=depth[FwdRenderingApp._QUALITY_SCALE_TOML_KEY]       , Power=depth[FwdRenderingApp._QUALITY_POWER_TOML_KEY]),
            quality_foreground      = QualityParameter(Lambda=foreground[FwdRenderingApp._QUALITY_SCALE_TOML_KEY]  , Power=foreground[FwdRenderingApp._QUALITY_POWER_TOML_KEY]),
            quality_side            = QualityParameter(Lambda=side[FwdRenderingApp._QUALITY_SCALE_TOML_KEY]        , Power=side[FwdRenderingApp._QUALITY_POWER_TOML_KEY]),
            quality_shape           = QualityParameter(Lambda=shape[FwdRenderingApp._QUALITY_SCALE_TOML_KEY]       , Power=shape[FwdRenderingApp._QUALITY_POWER_TOML_KEY]),
            quality_face_cam        = QualityParameter(Lambda=face_camera[FwdRenderingApp._QUALITY_SCALE_TOML_KEY] , Power=face_camera[FwdRenderingApp._QUALITY_POWER_TOML_KEY]),
            quality_normal          = QualityParameter(Lambda=normal[FwdRenderingApp._QUALITY_SCALE_TOML_KEY]      , Power=normal[FwdRenderingApp._QUALITY_POWER_TOML_KEY]),
            quality_treshold        = th,
            blend_m                 = BlendingMethod[blending[Blending._METHOD_BLENDING_TOML_KEY]],
            default_t4_color        = blending[Blending._BCK_T4_BLENDING_TOML_KEY],
            default_t1_color        = blending[Blending._BCK_T1_BLENDING_TOML_KEY],
            nb_samples              = nb_samples_ms,
        )

    def load_program(
            self, 
            program_file : str = _RENDER_PROGRAM_FNAME, 
            entry_point_names : list[str] = [RenderingApp._RENDER_VERTEX_SHADER_EP, RenderingApp._RENDER_GEOMETRY_SHADER_EP, RenderingApp._RENDER_FRAGMENT_SHADER_EP]
        ):
        # Store entry_point_name
        for ep_name in entry_point_names : self.entry_point_names.append(ep_name)

        self.rt_compute_kernel = self.device_handler.load_kernel(self._KERNEL_RT_COMPUTATION_NAME, self._KERNEL_RT_COMPUTATION_EP)

        self.render_program = RenderingApp.create_render_prog(self.device_handler.device, program_file, entry_point_names)

        self.input_layout = self.device_handler.create_input_layout(
            FwdRenderingApp.VERTEX_SEMANTIC_NAME,
            3,
            "float32",
        )

        self.target_layout = [
            spy.ColorTargetDesc({
                "format" : spy.Format.rgba32_float,
                "write_mask" : spy.RenderTargetWriteMask.all
            }), # Color
            spy.ColorTargetDesc({
                "format" : spy.Format.r32_float,
                "write_mask" : spy.RenderTargetWriteMask.red
            }),# Depth
            spy.ColorTargetDesc({
                "format" : spy.Format.r32_float,
                "write_mask" : spy.RenderTargetWriteMask.red
            }),# Quality
            spy.ColorTargetDesc({
                "format" : spy.Format.r32_float,
                "write_mask" : spy.RenderTargetWriteMask.red
            }),# Mask
            spy.ColorTargetDesc({
                "format" : spy.Format.r32_float,
                "write_mask" : spy.RenderTargetWriteMask.red
            })# Normal
        ]

        self.render_pipeline = self.device_handler.device.create_render_pipeline(
            RenderingApp.gen_rendering_pipeline_desc(self.render_program, self.input_layout, self.target_layout, self.nb_samples)
        )

    def simplify_input_mesh(self, input_idx : int, mesh_reduction : float):
        input_view = self.input_views[input_idx]
        if isinstance(input_view["mesh_buffer"], SimplifiableMeshBuffer):
            input_view["mesh_buffer"].simplify(
                ArrayUtils.sigmoid(
                    input_view["camera"].image_pair.depth.arr, 
                    input_view["camera"].cpu_params.depth_range[0],
                    input_view["camera"].cpu_params.depth_range[1] - input_view["camera"].cpu_params.depth_range[0]
                ),
                mesh_reduction = mesh_reduction
            )

    def load_input_view(self, view : Camera):
        Logger.debug(f"Load Input View: {view.name}", str(self.__class__))

        #mesh = SimpleFlatMeshBuffer(view.cpu_params.resolution[0], view.cpu_params.resolution[1], label=view.name)
        mesh = SimplifiableMeshBuffer(view.cpu_params.resolution[0], view.cpu_params.resolution[1], label=view.name)
        view_dict = InputViewDict({
            "camera":view,
            "mesh_buffer" : mesh,
            "uniform" : {} 
        })
        view_dict["uniform"] = self.get_input_view_uniform(view_dict)
        self._input_views.append(view_dict)
    
    def load_target_view(self, view : Camera):
        Logger.debug(f"Load Target View: {view.name}", str(self.__class__))

        # ------------------ Target Cam ------------------ 
        rendered_view = RenderedViewpoint(view, self.nb_samples, always_reload=True)

        # ------------------ Blended Cam ------------------ 
        blended_view = BlendedViewpoint(view, always_reload=True)

        # ------------------ Depth Testing ------------------ 
        depth_buffer = RenderingApp.use_depth_testing(self.device_handler, view.cpu_params.resolution[0], view.cpu_params.resolution[1], view.name, nb_samples=self.nb_samples, label = f"{FwdRenderingApp.__name__}")

        # ------------------ Fwd Objects ------------------ 

        view_dict = TargetOutputViewDict({
            "camera" : view,
            "depth_buffer": {"texture" : depth_buffer, "texture_view" : depth_buffer.create_view()},
            "rendered_viewpoint"    : rendered_view,
            "blended_viewpoint"     : blended_view,
            "uniform" : {} 
        })
        view_dict["uniform"] = self.get_target_view_uniform(view_dict)

        self._target_views.append(view_dict)

    def release_input_views(self):
        self._input_views = []

    def release_target_views(self):
        Logger.debug(f"Release all target Views.", str(self.__class__))
        self._target_views = []

    def compute_rt_matrix(self, command_encoder : spy.CommandEncoder, input_idx : int, target_idx : int):
        Logger.debug(f"Compute RT matrix", str(self.__class__))

        target_out_pair_view = self._target_views[target_idx]
        input_view = self._input_views[input_idx]
        
        self.rt_compute_kernel["kernel"].dispatch(
            [1, 1, 1], 
            command_encoder = command_encoder,
            param_cam_1 = input_view["uniform"]["params"]["param"],
            param_cam_2 = target_out_pair_view["uniform"]["params"]["param"],
            R_t = input_view["uniform"]["R_t"]
        )

    def blend_view(self, input_idx : int, target_idx : int, wait_submit_id : SubmitId = None) -> SubmitId:
        target_out_pair_view = self._target_views[target_idx]
        #input_view = self.input_views[input_idx]

        return self.blending.blend_color_depth_pair(
            thread_count=(target_out_pair_view["camera"].cpu_params.resolution[0], target_out_pair_view["camera"].cpu_params.resolution[1], 1),
            view_color=target_out_pair_view["uniform"]["renderingResult"]["color"],
            view_depth=target_out_pair_view["uniform"]["renderingResult"]["depth"],
            view_quality=target_out_pair_view["uniform"]["renderingResult"]["quality"],
            view_mask=target_out_pair_view["uniform"]["renderingResult"]["mask"],
            view_normal=target_out_pair_view["uniform"]["renderingResult"]["normal"],
            blended_view_color=target_out_pair_view["uniform"]["blendingResult"]["color"],
            blended_view_depth=target_out_pair_view["uniform"]["blendingResult"]["depth"],
            blended_view_quality=target_out_pair_view["uniform"]["blendingResult"]["quality"],
            blended_view_mask=target_out_pair_view["uniform"]["blendingResult"]["mask"],
            blended_view_normal=target_out_pair_view["uniform"]["blendingResult"]["normal"],
            wait_submit_id=wait_submit_id
        )

    def normalize_blending(self, target_idx : int, wait_submit_id : SubmitId = None) -> SubmitId:
        target_out_pair_view = self._target_views[target_idx]

        return self.blending.normalize_blended_pair(
            (target_out_pair_view["camera"].cpu_params.resolution[0], target_out_pair_view["camera"].cpu_params.resolution[1], 1),
            target_out_pair_view["uniform"]["blendingResult"]["color"],
            target_out_pair_view["uniform"]["blendingResult"]["depth"],
            target_out_pair_view["uniform"]["blendingResult"]["quality"],
            target_out_pair_view["uniform"]["blendingResult"]["mask"],
            target_out_pair_view["uniform"]["blendingResult"]["normal"],
            self.quality["DiscardingTreshold"],
            wait_submit_id=wait_submit_id
        )

    def clear_blend(self, target_idx : int, wait_submit_id : SubmitId = None) -> SubmitId:
        Logger.debug(f"Clear blending. IDX: {target_idx}", str(self.__class__))

        target_out_pair_view = self._target_views[target_idx]

        command_encoder = self.device_handler.create_command_encoder()

        # ------------------------------------------- CLEAR BLENDED VIEW -------------------------------------------
        if(self.blending is not None): 
            command_encoder.clear_texture_float(target_out_pair_view["blended_viewpoint"].blended_image_pair.color.texture, clear_value= spy.float4(0.0))
            command_encoder.clear_texture_float(target_out_pair_view["blended_viewpoint"].blended_image_pair.depth.texture, clear_value= spy.float4(0.0))
            command_encoder.clear_texture_float(target_out_pair_view["blended_viewpoint"].blended_quality.texture, clear_value= spy.float4(0.0))
            command_encoder.clear_texture_float(target_out_pair_view["blended_viewpoint"].blended_mask.texture, clear_value= spy.float4(0.0))
            command_encoder.clear_texture_float(target_out_pair_view["blended_viewpoint"].blended_normal.texture, clear_value= spy.float4(0.0))
        
        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

    def render(self, input_idx : int, target_idx : int, wait_submit_id : SubmitId = None) -> SubmitId:
        if(len(self._target_views) == 0): raise Exception("Error - FwdApp: cannot start a render pass if no target view(s) was loaded.")
        if(self.render_pipeline is None): raise Exception("Error - FwdApp: cannot start a render pass if no render pipeline was created.")
        if(target_idx < 0 or target_idx >= len(self._target_views)) : raise Exception("Error - FwdApp: cannot start a render pass if the target view does not exist.")
        if(input_idx < 0 or input_idx >= len(self._input_views)) : raise Exception("Error - FwdApp: cannot start a render pass if the input view does not exist.")

        target_out_pair_view = self._target_views[target_idx]
        input_view = self._input_views[input_idx]

        command_encoder = self.device_handler.create_command_encoder()

        # ------------------------------------------- COMP [Rt] matrix -------------------------------------------
        self.compute_rt_matrix(command_encoder, input_idx, target_idx)

        # ------------------------------------------- DRAW RENDERING PASS -------------------------------------------
        color_attach = [
            #* Color Output
            spy.RenderPassColorAttachment({
                "view"              : target_out_pair_view["rendered_viewpoint"].rendered_image_pair.ms_color.texture_v,
                "resolve_target"    : target_out_pair_view["rendered_viewpoint"].rendered_image_pair.color.texture_v,
                "store_op" : spy.StoreOp.store,
                "clear_value" : spy.float4(0.0)
                }
            ),
            #* Depth Map Output
            spy.RenderPassColorAttachment({
                "view"              : target_out_pair_view["rendered_viewpoint"].rendered_image_pair.ms_depth.texture_v,
                "resolve_target"    : target_out_pair_view["rendered_viewpoint"].rendered_image_pair.depth.texture_v,
                "clear_value" : spy.float4(0.0),
                "store_op" : spy.StoreOp.store
                }
            ),
            #* Quality Map Output
            spy.RenderPassColorAttachment({
                "view"              : target_out_pair_view["rendered_viewpoint"].rendered_quality.ms_img.texture_v,
                "resolve_target"    : target_out_pair_view["rendered_viewpoint"].rendered_quality.dst_img.texture_v,
                "clear_value" : spy.float4(0.0),
                "store_op"          : spy.StoreOp.store
                }
            ),
            #* Vertices Mask Output
            spy.RenderPassColorAttachment({
                "view"              : target_out_pair_view["rendered_viewpoint"].rendered_mask.ms_img.texture_v,
                "resolve_target"    : target_out_pair_view["rendered_viewpoint"].rendered_mask.dst_img.texture_v,
                "clear_value"       : spy.float4(0.0),
                "store_op"          : spy.StoreOp.store
                }
            ),
            #* Normal Mask Output
            spy.RenderPassColorAttachment({
                "view"              : target_out_pair_view["rendered_viewpoint"].rendered_normal.ms_img.texture_v,
                "resolve_target"    : target_out_pair_view["rendered_viewpoint"].rendered_normal.dst_img.texture_v,
                "clear_value"       : spy.float4(0.0),
                "store_op"          : spy.StoreOp.store
                }
            )
        ]

        r_p_e = command_encoder.begin_render_pass({
            "color_attachments": color_attach,
            "depth_stencil_attachment":
            {
                "depth_load_op":spy.LoadOp.clear,
                "depth_store_op" : spy.StoreOp.store,
                "stencil_load_op" : spy.LoadOp.dont_care,
                "stencil_store_op" : spy.StoreOp.dont_care,
                "view" : target_out_pair_view["depth_buffer"]["texture_view"],
            }
        })

        sh_o = r_p_e.bind_pipeline(self.render_pipeline)

        r_p_e.set_render_state(
            FwdRenderingApp.create_render_state(
                target_out_pair_view["camera"].cpu_params.resolution[0],
                target_out_pair_view["camera"].cpu_params.resolution[1], 
                input_view["mesh_buffer"].vtx_buffer,
                input_view["mesh_buffer"].idx_buffer,
                )
            )

        uniforms = self.get_uniforms(input_view, target_out_pair_view)
        UniformUtils.apply_uniforms_to_shader(self.render_program.layout, sh_o, dict(uniforms))

        r_p_e.draw_indexed({"vertex_count": (input_view["mesh_buffer"].idx_buffer.size // (4))}) # 4 bytes 
        r_p_e.end()

        self._copy_rendering_result_to_intermediate_storage(command_encoder, input_idx, target_idx)

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

    def _copy_rendering_result_to_intermediate_storage(self, command_encoder : spy.CommandEncoder, input_idx : int, target_idx : int):
        if self._intermediate_results is not None and len(self._intermediate_results) > 0:
            rendered_viewpoint = self._target_views[target_idx]["rendered_viewpoint"]
            intermediate_result=  self._intermediate_results[(input_idx, target_idx)]
            if rendered_viewpoint.camera.cpu_params.resolution != intermediate_result.camera.cpu_params.resolution:
                raise ValueError("Intermediate result and RenderedViewpoint does not have the same resolution!")
            
            # ---- Copy intermediate result ---- 
            self.device_handler.copy_tex(
                command_encoder, 
                rendered_viewpoint.rendered_image_pair.ms_color.texture, 
                intermediate_result.rendered_image_pair.ms_color.texture
            )
            self.device_handler.copy_tex(
                command_encoder, 
                rendered_viewpoint.rendered_image_pair.ms_depth.texture, 
                intermediate_result.rendered_image_pair.ms_depth.texture
            )
            self.device_handler.copy_tex(
                command_encoder, 
                rendered_viewpoint.rendered_quality.ms_img.texture,      
                intermediate_result.rendered_quality.ms_img.texture
            )
            self.device_handler.copy_tex(
                command_encoder, 
                rendered_viewpoint.rendered_mask.ms_img.texture,
                intermediate_result.rendered_mask.ms_img.texture
            )

            self.device_handler.copy_tex(
                command_encoder, 
                rendered_viewpoint.rendered_image_pair.color.texture, 
                intermediate_result.rendered_image_pair.color.texture
            )
            self.device_handler.copy_tex(
                command_encoder, 
                rendered_viewpoint.rendered_image_pair.depth.texture, 
                intermediate_result.rendered_image_pair.depth.texture
            )
            self.device_handler.copy_tex(
                command_encoder, 
                rendered_viewpoint.rendered_quality.dst_img.texture,  
                intermediate_result.rendered_quality.dst_img.texture
            )
            self.device_handler.copy_tex(
                command_encoder, 
                rendered_viewpoint.rendered_mask.dst_img.texture,     
                intermediate_result.rendered_mask.dst_img.texture
            )

    def bwd_render(self, input_idx: int, target_idx: int, wait_submit_id: SubmitId = None) -> SubmitId:
        raise RuntimeError("FwdApp should only be used for fwd. If bwd calls are needed, use BwdApp.")

    def get_uniforms(self, input_view : InputViewDict, target_output_view : TargetOutputViewDict) -> UniformsDict:
        uniforms = UniformsDict(
            {
                "out_cam_params": target_output_view["uniform"]["params"],
                "transform":{
                    "R_t" : input_view["uniform"]["R_t"]
                },
                "in_diff_cam" : {
                    "params"        : input_view["uniform"]["params"],
                    "color_texture" : input_view["uniform"]["color_texture"],
                    "depth_texture" : input_view["uniform"]["depth_texture"],
                    "sampler_tex"   : input_view["uniform"]["sampler_tex"],
                },
                "qParams" : self.get_quality_uniform()
            }
        )

        return uniforms
    
    def get_result_color(self, target_idx : int) -> spy.Texture:
        return self._target_views[target_idx]["blended_viewpoint"].blended_image_pair.color.texture
    
    def get_result_depth(self, target_idx : int) -> spy.Texture:
        return self._target_views[target_idx]["blended_viewpoint"].blended_image_pair.depth.texture
    
    def get_result_quality(self, target_idx : int) -> spy.Texture:
        return self._target_views[target_idx]["blended_viewpoint"].blended_quality.texture
    
    def get_result_mask(self, target_idx : int) -> spy.Texture:
        return self._target_views[target_idx]["blended_viewpoint"].blended_mask.texture
    
    def get_result_normal(self, target_idx : int) -> spy.Texture:
        return self._target_views[target_idx]["blended_viewpoint"].blended_normal.texture
        #return self._target_views[target_idx]["rendered_viewpoint"].rendered_normal.dst_img.texture
    
    def get_quality_uniform(self) -> QualityParametersUniformDict:
        return QualityParametersUniformDict({
                    "lambda_quality_depth"                      : self.quality["DepthBased"]["Lambda"],
                    "lambda_quality_shape"                      : self.quality["ShapeBased"]["Lambda"],
                    "lambda_quality_normal"                     : self.quality["NormalBased"]["Lambda"],
                    "lambda_quality_side"                       : self.quality["SideBased"]["Lambda"],
                    "lambda_quality_foreground"                 : self.quality["DepthTestBased"]["Lambda"],
                    "lambda_quality_face_camera"                : self.quality["CameraAlignBased"]["Lambda"],
                    "scaling_power_quality_shape"               : self.quality["ShapeBased"]["Power"],
                    "scaling_power_quality_normal"              : self.quality["NormalBased"]["Power"],
                    "scaling_power_quality_depth"               : self.quality["DepthBased"]["Power"],
                    "scaling_power_quality_side"                : self.quality["SideBased"]["Power"],
                    "scaling_power_quality_foreground"          : self.quality["DepthTestBased"]["Power"],
                    "scaling_power_quality_face_camera"         : self.quality["CameraAlignBased"]["Power"],
                    "quality_treshold"                          : self.quality["DiscardingTreshold"],
                })
        
    def get_input_view_uniform(self, input_view : InputViewDict) -> UniformType:
        view = input_view["camera"]
        
        uniform = {
            "params" : CameraParametersUniformDict({
                "imgSize"           : view.cpu_params.resolution,
                "proj_type"         : view.cpu_params.projection.value,
                "min_max_depth"     : view.cpu_params.depth_range,
                "param":{
                    "buff_" : {
                        "buffer" : view.gpu_params.cam_param_gpu,
                        "size"   : view.gpu_params.cam_param_gpu.size // 4,
                    }
                }
            }),
            "color_texture" : CstTextureUniformDict({
                "texture"           : view.image_pair.color.texture,
            }),
            "depth_texture" : CstTextureUniformDict({
                "texture"           : view.image_pair.depth.texture,
            }),
            "sampler_tex" : self.sampler,
            "R_t" : CsteBufferUniformDict({
                "buff_" : {
                    "buffer" : view.gpu_params.rt_buffer,
                    "size"   : view.gpu_params.rt_buffer.size //4,
                },
            }),
        }

        return uniform
    
    def get_target_view_uniform(self, target_view : TargetOutputViewDict) -> UniformType:
        view = target_view["camera"]
        blended_view = target_view["blended_viewpoint"]
        rendered_view = target_view["rendered_viewpoint"]
        
        uniform = {
            "params" : CameraParametersUniformDict({
                "imgSize"       : view.cpu_params.resolution,
                "proj_type"     : view.cpu_params.projection.value,
                "min_max_depth" : view.cpu_params.depth_range,
                "param" : {
                    "buff_":{
                        "buffer"    : view.gpu_params.cam_param_gpu,
                        "size"      : view.gpu_params.cam_param_gpu.size // 4, 
                    }
                },
            }),
            "blendingResult" : { 
                "color"     : CstTextureUniformDict({
                    "texture"           : blended_view.blended_image_pair.color.texture,
                }),
                "depth"     : CstTextureUniformDict({
                    "texture"           : blended_view.blended_image_pair.depth.texture,
                }),
                "quality"     : CstTextureUniformDict({
                    "texture"           : blended_view.blended_quality.texture,
                }),
                "mask"     : CstTextureUniformDict({
                    "texture"           : blended_view.blended_mask.texture,
                }),
                "normal"     : CstTextureUniformDict({
                    "texture"           : blended_view.blended_normal.texture,
                })
            },
            "renderingResult" : { 
                "color"     : CstTextureUniformDict({
                    "texture"           : rendered_view.rendered_image_pair.color.texture,
                }),
                "depth"     : CstTextureUniformDict({
                    "texture"           : rendered_view.rendered_image_pair.depth.texture,
                }),
                "quality"     : CstTextureUniformDict({
                    "texture"           : rendered_view.rendered_quality.dst_img.texture,
                }),
                "mask"     : CstTextureUniformDict({
                    "texture"           : rendered_view.rendered_mask.dst_img.texture,
                }),
                "normal"     : CstTextureUniformDict({
                    "texture"           : rendered_view.rendered_normal.dst_img.texture,
                })
            }
        }

        return uniform
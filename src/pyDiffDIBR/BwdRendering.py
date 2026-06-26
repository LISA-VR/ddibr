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

from typing import TypedDict, Any
import slangpy as spy

from ._utils import ArrayUtils, ImageUtils, UniformUtils, TypeUtils
from .Camera import Camera, RenderedViewpoint, BlendedViewpoint
from .MeshBuffer import SimpleFlatMeshBuffer, MeshBuffer, SimplifiableMeshBuffer
from .RenderingDevice import SubmitId, RenderingDevice
from .RenderingApp import RenderingApp

from .FwdRendering import FwdRenderingApp
from .FwdRendering import (
    InputViewDict as FwdInputViewDict,
    DepthBufferDict as FwdDepthBufferDict,
    QualityParametersUniformDict as FwdQualityParametersUniformDict
)


# ------------------------------------ FAST TYPE DEFs ------------------------------------ 
TextureDict = TypedDict("TextureDict",
{
    "color": spy.Texture,
    "color_view" : spy.TextureView,
    "depth": spy.Texture,
    "depth_view" : spy.TextureView,
}, total = True)

PropagateGradientDict = TypedDict("PropagateGradientDict", {
    "camera_parameter"  : bool,
    "color"             : bool,
    "depth"             : bool
}, total = True)

InputViewDict = TypedDict("InputViewDict", 
{
    "camera"                : Camera,
    "mesh_buffer"           : SimplifiableMeshBuffer,
    "propagate_grad"        : PropagateGradientDict,
    "uniform"               : TypeUtils.UniformGPUData_t
}, total = True)


DepthBufferDict = TypedDict("DepthBufferDict", {
    "texture" : spy.Texture,
    "texture_view" : spy.TextureView
}, total = True)

TargetOutputViewDict = TypedDict("TargetOutputViewDict", 
{
    "camera"                : Camera,
    "depth_buffer"          : FwdDepthBufferDict,
    "blended_viewpoint"     : BlendedViewpoint,
    "uniform"               : TypeUtils.UniformGPUData_t
}, total = True)

# ------------------------------------ UNIFORMS ------------------------------------ 

TensorizedBufferUniformDict = TypedDict("TensorizedBufferUniformDict", {
    "buffer" : spy.Buffer,
    "size" : int
})

GradTensorizedBufferUniformDict = TypedDict("GradTensorizedBufferUniformDict", {
    "grad_scaling" : spy.Buffer,
    "grad" : spy.Buffer,
    "size" : int,
    "no_grad" : bool,
})

ChunkedGradTensorizedBufferUniformDict = TypedDict("ChunkedGradTensorizedBufferUniformDict", {
    "grad_scaling" : spy.Buffer,
    "grad" : spy.Buffer,
    "size" : int,
    "no_grad" : bool,
    "nb_blocks" : tuple[int, int]
})

CsteBufferUniformDict = TypedDict("CsteBufferUniformDict", {
    "buff_" : TensorizedBufferUniformDict
})

CameraParametersUniformDict = TypedDict("CameraParametersUniformDict", 
{
    "proj_type": int,
    "imgSize" : tuple[int, int],
    "min_max_depth": tuple[float, float],
    "param" : CsteBufferUniformDict,
}, total = False)

DiffBufferUniformDict = TypedDict("DiffBufferUniformDict", {
    "buff_" : TensorizedBufferUniformDict,
    "grad_" : GradTensorizedBufferUniformDict
})

ChunkedDiffBufferUniformDict = TypedDict("ChunkedDiffBufferUniformDict", {
    "buff_" : TensorizedBufferUniformDict,
    "grad_" : ChunkedGradTensorizedBufferUniformDict
})

DiffCameraParametersUniformDict = TypedDict("DiffCameraParametersUniformDict", 
{
    "proj_type": int,
    "imgSize" : tuple[int, int],
    "min_max_depth": tuple[float, float],
    #"param" : DiffBufferUniformDict,
    "param" : ChunkedDiffBufferUniformDict,
}, total = False)

DiffTextureUniformDict = TypedDict("DiffTextureUniformDict", {
    "accumulateBuffer" : spy.Buffer,
    "range" : float,
    "texture" : spy.Texture,
    "no_grad" : bool,
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

DiffCameraUniformDict = TypedDict("DiffCameraUniformDict", 
{
    "params" : DiffCameraParametersUniformDict,
    "depth_texture": DiffTextureUniformDict,
    "color_texture": DiffTextureUniformDict,
    "sampler_tex": spy.Sampler,
}, total = False)

SpaceTransformUniformDict = TypedDict("SpaceTransformUniformDict", 
{
    "R_t" : ChunkedDiffBufferUniformDict,
}, total = False)

MeshBufferUniformsDict = TypedDict("MeshBufferUniformsDict",
{
    "indices" : spy.Buffer,
    "vertices" : spy.Buffer
}, total = False)

DrawResultCopyUniformsDict = TypedDict("DrawResultCopyUniformsDict",
{
    "color"     : DiffTextureUniformDict,
    "depth"     : DiffTextureUniformDict,
    "quality"   : DiffTextureUniformDict,
    "mask"      : DiffTextureUniformDict,
    "ms_color"     : spy.Texture,
    "ms_depth"     : spy.Texture,
    "ms_quality"   : spy.Texture,
    "ms_mask"   : spy.Texture,
}, total = False)

BlendingResultUniformsDict = TypedDict("BlendingResultUniformsDict",
{
    "color"     : DiffTextureUniformDict,#spy.Texture,
    "depth"     : DiffTextureUniformDict,#spy.Texture,
    "quality"   : DiffTextureUniformDict,#spy.Texture,
    "mask"      : DiffTextureUniformDict,#spy.Texture,
}, total = False)

OptimizationParameterUniformsDict = TypedDict("OptimizationParameterUniformsDict", {
    "depth_check_tol" : float,
}, total = True)

UniformsDict = TypedDict("UniformsDict", 
{
    "transform"         : SpaceTransformUniformDict,
    "out_cam_params"    : CameraParametersUniformDict,
    "in_diff_cam"       : DiffCameraUniformDict,
    "target_cam"        : CameraUniformDict,
    "meshBuffer"         : MeshBufferUniformsDict,
    "drawResult"        : DrawResultCopyUniformsDict,
    "blendingResult"    : BlendingResultUniformsDict,
    "opti_param"        : OptimizationParameterUniformsDict,
    "qParams"           : FwdQualityParametersUniformDict
}, total = False)

class BwdRenderingApp(RenderingApp):
    VERTEX_SEMANTIC_NAME    = "POSITION"

    _RENDER_PROGRAM_FNAME = "synthesis.slang"
    _RENDER_FRAGMENT_SHADER_EP = "diff_fragment_main"

    _KERNEL_RT_COMPUTATION_NAME = "SpaceTransformer.slang"
    _KERNEL_BWD_RT_COMPUTATION_EP   = "bwd_single_thread_kernel_compute_R_t"

    _KERNEL_BWD_DIFF_BUFFER_NAME = "DiffBuffer.slang"
    _KERNEL_BWD_DIFF_BUFFER_EP   = "main_add_chunked_buffer"

    def __init__(self, fwd_app : FwdRenderingApp, check_depth_tol : float = 1e-7):

        self.fwd_app = fwd_app

        super().__init__()

        self.render_program : spy.ShaderProgram
        self.entry_point_names = []
        self.device_handler : RenderingDevice = fwd_app.device_handler
        
        # ---- CST----
        self._check_depth_tol = check_depth_tol
        # ---- CST----

        self._input_views           : list[InputViewDict] = [] 
        self._target_views          : list[TargetOutputViewDict] = []
        self._intermediate_results  : dict[tuple[int, int], RenderedViewpoint] = {}
        
        self.sampler = self.device_handler.create_sampler()

    @property
    def target_views(self) -> list[TargetOutputViewDict]:
        return self._target_views
    
    @property
    def input_views(self) -> list[InputViewDict]:
        return self._input_views

    @classmethod
    def from_toml(cls, fwd_app : FwdRenderingApp, path_toml : str):
        #data = IOUtils.read_toml(path_toml)
        return BwdRenderingApp(
            fwd_app
        )

    def load_program(
            self, 
            program_file : str = _RENDER_PROGRAM_FNAME, 
            entry_point_names : list[str] = [RenderingApp._RENDER_VERTEX_SHADER_EP, RenderingApp._RENDER_GEOMETRY_SHADER_EP, _RENDER_FRAGMENT_SHADER_EP]
        ):
        # Store entry_point_name
        for ep_name in entry_point_names : self.entry_point_names.append(ep_name)

        self.grad_rt_compute_kernel = self.device_handler.load_kernel(self._KERNEL_RT_COMPUTATION_NAME, self._KERNEL_BWD_RT_COMPUTATION_EP)
        self.combine_chunked_kernel = self.device_handler.load_kernel(self._KERNEL_BWD_DIFF_BUFFER_NAME, self._KERNEL_BWD_DIFF_BUFFER_EP)

        self.render_program = RenderingApp.create_render_prog(self.device_handler.device, program_file, entry_point_names)

        self.input_layout = self.device_handler.create_input_layout(
            FwdRenderingApp.VERTEX_SEMANTIC_NAME,
            3,
            "float32",
        )

        self.target_layout = []

        self.render_pipeline = self.device_handler.device.create_render_pipeline(
            RenderingApp.gen_rendering_pipeline_desc(self.render_program, self.input_layout, self.target_layout, self.fwd_app.nb_samples)
        )

    def load_input_view(self, view : Camera):
        mesh_buffer = SimplifiableMeshBuffer(view.cpu_params.resolution[0], view.cpu_params.resolution[1], label=view.name)
        
        propagate_grad = PropagateGradientDict({
            "camera_parameter" : False,
            "color" : False,
            "depth" : False
        })

        view_dict = InputViewDict({
            "camera"                : view,
            "mesh_buffer"           : mesh_buffer,
            "propagate_grad"        : propagate_grad,
            "uniform"               : dict()
        })
        view_dict["uniform"] = self.get_input_view_uniform(view_dict)

        self._input_views.append(view_dict)

        fwd_view_dict = FwdInputViewDict({"camera":view, "mesh_buffer" : mesh_buffer, "uniform" : {}})
        fwd_view_dict["uniform"] = self.fwd_app.get_input_view_uniform(fwd_view_dict)
        self.fwd_app.input_views.append(fwd_view_dict)
    
    def load_target_view(self, view : Camera):
        # ----- FwdApp ----- 
        self.fwd_app.load_target_view(view)

        ## ------------------ Blended Cam ------------------ 
        blended_view = self.fwd_app.target_views[-1]["blended_viewpoint"]
        blended_view.blended_image_pair.color.requires_grad = True
        blended_view.blended_image_pair.color.reload_gpu_texture()
        blended_view.blended_image_pair.depth.requires_grad = True
        blended_view.blended_image_pair.depth.reload_gpu_texture()
        blended_view.blended_quality.requires_grad = True
        blended_view.blended_quality.reload_gpu_texture()
        blended_view.blended_mask.requires_grad = True
        blended_view.blended_mask.reload_gpu_texture()
        self.fwd_app.target_views[-1]["uniform"] = self.fwd_app.get_target_view_uniform(self.fwd_app.target_views[-1]) # Reload Uniform GPU pointer

        # ------------------ Depth Testing ------------------ 
        depth_buffer = RenderingApp.use_depth_testing(self.device_handler, view.cpu_params.resolution[0], view.cpu_params.resolution[1], view.name, nb_samples=self.fwd_app.nb_samples, label = f"{BwdRenderingApp.__name__}")

        view_dict = TargetOutputViewDict({
            "camera":view,
            "depth_buffer": {"texture" : depth_buffer, "texture_view" : depth_buffer.create_view()},
            "blended_viewpoint":blended_view,
            "uniform" : {}
        })
        view_dict["uniform"] = self.get_target_view_uniform(view_dict)

        self._target_views.append(view_dict)
        
    def load_intermediate_results(self):
        # -------------------------------- Intermediate Result -------------------------------- 
        for idx_i, input_view in enumerate(self._input_views):
            for idx_t, target_view in enumerate(self._target_views):
                if (idx_i, idx_t) not in self._intermediate_results or self._intermediate_results[(idx_i, idx_t)].camera.cpu_params.resolution != target_view["camera"].cpu_params.resolution:
                    # Load only if data differs (---> We only care about the storage)
                    # Or if it does not yet exist
                    intermediate_result = RenderedViewpoint(target_view["camera"], self.fwd_app.nb_samples, requires_grad = True, always_reload=True)
                    self._intermediate_results[(idx_i, idx_t)] = intermediate_result

        self.fwd_app._intermediate_results = self._intermediate_results

    def release_intermediate_results(self):
        self._intermediate_results = {}
        self.fwd_app._intermediate_results = {}

    def release_input_views(self):
        self._input_views = []
        self.fwd_app.release_input_views()

    def release_target_views(self):
        self._target_views = []
        self.fwd_app.release_target_views()

    def combine_chunked_buffer(self, command_encoder : spy.CommandEncoder, input_idx : int, target_idx : int):
        input_view = self._input_views[input_idx]
        if(not input_view["propagate_grad"]["camera_parameter"]): return
        
        nb_chunks = input_view["camera"].nb_chunks

        self.combine_chunked_kernel["kernel"].dispatch(
            [nb_chunks[0], 1, 1], 
            command_encoder = command_encoder,
            buffer = input_view["uniform"]["R_t"]["grad_"]
        )

        self.combine_chunked_kernel["kernel"].dispatch(
            [nb_chunks[0], 1, 1], 
            command_encoder = command_encoder,
            buffer = input_view["uniform"]["params"]["param"]["grad_"]
        )

    def compute_grad_rt_matrix(self, command_encoder : spy.CommandEncoder, input_idx : int, target_idx : int):
        target_view = self._target_views[target_idx]
        input_view = self._input_views[input_idx]
        if(not input_view["propagate_grad"]["camera_parameter"]): return
        
        self.grad_rt_compute_kernel["kernel"].dispatch(
            [1, 1, 1], 
            command_encoder = command_encoder,
            param_cam_1 = input_view["uniform"]["params"]["param"],
            param_cam_2 = target_view["uniform"]["params"]["param"],
            R_t = input_view["uniform"]["R_t"]
        )

        command_encoder.clear_buffer(input_view["camera"].gpu_params.grad_buffer_for_Rt_matrix)

    def render(self, input_idx: int, target_idx: int, wait_submit_id: SubmitId = None) -> SubmitId:
        raise RuntimeError("BwdApp should only be used for bwd calls. If fwd calls are needed, please use Fwdapp.")

    def bwd_render(self, input_idx : int, target_idx : int, wait_submit_id : SubmitId = None) -> SubmitId:
        if(len(self._target_views) == 0): raise ValueError("BwdApp: cannot start a render pass if no target view(s) was loaded.")
        if(self.render_pipeline is None): raise ValueError("BwdApp: cannot start a render pass if no render pipeline was created.")
        if(target_idx < 0 or target_idx >= len(self._target_views)) : raise ValueError("BwdApp: cannot start a render pass if the target view does not exist.")
        if(input_idx < 0 or input_idx >= len(self._input_views)) : raise ValueError("BwdApp: cannot start a render pass if the input view does not exist.")

        command_encoder : spy.CommandEncoder = self.device_handler.create_command_encoder()

        target_out_pair_view = self._target_views[target_idx]
        input_view = self._input_views[input_idx]

        # -------------------------------------- DIFFERENTIABLE RENDERING PASS --------------------------------------
        color_attach = []

        r_p_e = command_encoder.begin_render_pass({
            "color_attachments": color_attach,
            "depth_stencil_attachment":
            {
                "depth_load_op":spy.LoadOp.clear,
                "depth_store_op" : spy.StoreOp.dont_care,
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

        uniforms = self.get_uniforms(input_idx, target_idx)
        UniformUtils.apply_uniforms_to_shader(self.render_program.layout,sh_o, dict(uniforms))

        r_p_e.draw_indexed({"vertex_count": (input_view["mesh_buffer"].idx_buffer.size // (4))}) # 4 bytes 
        r_p_e.end()

        # -------------------------------------- RECOMBINE CHUNKED GRAD  --------------------------------------
        self.combine_chunked_buffer(command_encoder, input_idx, target_idx)

        # -------------------------------------- PROPAGATE [Rt] GRAD to POS|ROT  --------------------------------------
        self.compute_grad_rt_matrix(command_encoder, input_idx, target_idx)

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

    def backward_blending(self,  input_idx : int, output_idx : int, wait_submit_id : SubmitId = None) -> SubmitId:
        target_view = self._target_views[output_idx]

        uniform_intermediate_result = self.get_intermediate_result_uniform(input_idx, output_idx)

        return self.fwd_app.blending.bwd_blend_color_depth_pair(
            thread_count = (target_view["camera"].cpu_params.resolution[0], target_view["camera"].cpu_params.resolution[1], 1),
            view_color              = uniform_intermediate_result["drawResult"]["color"],
            view_depth              = uniform_intermediate_result["drawResult"]["depth"],
            view_quality            = uniform_intermediate_result["drawResult"]["quality"],
            view_mask               = uniform_intermediate_result["drawResult"]["mask"],
            blended_view_color      = target_view["uniform"]["blendingResult"]["color"],
            blended_view_depth      = target_view["uniform"]["blendingResult"]["depth"],
            blended_view_quality    = target_view["uniform"]["blendingResult"]["quality"],
            blended_view_mask       = target_view["uniform"]["blendingResult"]["mask"],
            quality_treshold        = self.fwd_app.quality["DiscardingTreshold"],
            wait_submit_id=wait_submit_id
        )

    def clear_grads(self) -> SubmitId:
        c_e = self.device_handler.create_command_encoder()

        for o in range(len(self._target_views)):
            target_view = self._target_views[o]
            
            c_e.clear_buffer(target_view["blended_viewpoint"].blended_image_pair.color.grad)
            c_e.clear_buffer(target_view["blended_viewpoint"].blended_image_pair.depth.grad)
            c_e.clear_buffer(target_view["blended_viewpoint"].blended_quality.grad)

        for i in range(len(self._input_views)):
            input_view = self._input_views[i]

            c_e.clear_buffer(input_view["camera"].image_pair.color.grad)
            c_e.clear_buffer(input_view["camera"].image_pair.depth.grad)

            c_e.clear_buffer(input_view["camera"].gpu_params.grad_buffer_for_cam_params)
            c_e.clear_buffer(input_view["camera"].gpu_params.grad_buffer_for_Rt_matrix)
            for t in range(len(self._target_views)):
                c_e.clear_buffer(self._intermediate_results[(i, t)].rendered_image_pair.color.grad)
                c_e.clear_buffer(self._intermediate_results[(i, t)].rendered_image_pair.depth.grad)
                c_e.clear_buffer(self._intermediate_results[(i, t)].rendered_quality.dst_img.grad)
                c_e.clear_buffer(self._intermediate_results[(i, t)].rendered_mask.dst_img.grad)

        return self.device_handler.submit_command(c_e)

    def clear_intermediate(self, wait_submit_id : SubmitId = None) -> SubmitId:
        command_encoder = self.device_handler.create_command_encoder()
        for idx_i in range(len(self._input_views)):
            for idx_t in range(len(self._target_views)):
                inter_r = self._intermediate_results[(idx_i, idx_t)]
            
                command_encoder.clear_buffer(inter_r.rendered_image_pair.color.grad)
                command_encoder.clear_buffer(inter_r.rendered_image_pair.depth.grad)
                command_encoder.clear_buffer(inter_r.rendered_quality.dst_img.grad)
                command_encoder.clear_buffer(inter_r.rendered_mask.dst_img.grad)
        
        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

    def get_uniforms(self, input_idx : int, target_idx : int) -> UniformsDict:
        target_output_view = self._target_views[target_idx]
        input_view = self._input_views[input_idx]
        uniforms = UniformsDict(
            {
                "out_cam_params": target_output_view["uniform"]["params"],
                "transform":{
                    "R_t" : input_view["uniform"]["R_t"],
                },
                "in_diff_cam" : {
                    "params"        : input_view["uniform"]["params"],
                    "color_texture" : input_view["uniform"]["color_texture"],
                    "depth_texture" : input_view["uniform"]["depth_texture"],
                    "sampler_tex"   : input_view["uniform"]["sampler_tex"],
                },
                "target_cam": {
                    "color_texture" : target_output_view["uniform"]["color_texture"],
                    "depth_texture" : target_output_view["uniform"]["depth_texture"],
                    "params"        : target_output_view["uniform"]["params"],
                    "sampler_tex"   : target_output_view["uniform"]["sampler_tex"]
                },
                "meshBuffer"        : {"indices" : input_view["mesh_buffer"].idx_buffer, "vertices" : input_view["mesh_buffer"].vtx_buffer},
                "drawResult"        : self.get_intermediate_result_uniform(input_idx, target_idx)["drawResult"],
                "blendingResult"    : target_output_view["uniform"]["blendingResult"],
                "opti_param" : {
                    "depth_check_tol"               : self._check_depth_tol
                },
                "qParams" : self.fwd_app.get_quality_uniform(),
            }
        )

        return uniforms
    
    def get_input_view_uniform(self, input_view : InputViewDict) -> TypeUtils.UniformGPUData_t:
        view = input_view["camera"]
        propagate_grad = input_view["propagate_grad"]
        
        uniform = {
            "params" : DiffCameraParametersUniformDict({
                "imgSize"           : view.cpu_params.resolution,
                "proj_type"         : view.cpu_params.projection.value,
                "min_max_depth"     : view.cpu_params.depth_range,
                "param":{
                    "buff_" : {
                        "buffer" : view.gpu_params.cam_param_gpu,
                        "size"   : view.gpu_params.cam_param_gpu.size // 4,
                    },
                    "grad_" : {
                        "grad"          : view.gpu_params.grad_buffer_for_cam_params,
                        "grad_scaling"  : view.gpu_params.grad_scaling_buffer_for_cam_params,
                        "size"          : view.gpu_params.cam_param_gpu.size // 4,
                        "no_grad"       : not propagate_grad["camera_parameter"],
                        "nb_blocks"     : view.nb_chunks
                    }
                }
            }),
            "color_texture" : DiffTextureUniformDict({
                "texture"           : view.image_pair.color.texture,
                "accumulateBuffer"  : view.image_pair.color.grad,
                "no_grad"           : not propagate_grad["color"],
                "range"             : view.image_pair.color.type_range
            }),
            "depth_texture" : DiffTextureUniformDict({
                "texture"           : view.image_pair.depth.texture,
                "accumulateBuffer"  : view.image_pair.depth.grad,
                "no_grad"           : not propagate_grad["depth"],
                "range"             : view.image_pair.depth.type_range
            }),
            "sampler_tex" : self.sampler,
            "R_t" : ChunkedDiffBufferUniformDict({
                "buff_" : {
                    "buffer" : view.gpu_params.rt_buffer,
                    "size"   : view.gpu_params.rt_buffer.size //4,
                },
                "grad_" : {
                    "grad"          : view.gpu_params.grad_buffer_for_Rt_matrix,
                    "grad_scaling"  : view.gpu_params.grad_scaling_buffer_for_Rt_matrix,
                    "size"          : view.gpu_params.rt_buffer.size // 4,
                    "no_grad"       : not propagate_grad["camera_parameter"],
                    "nb_blocks"     : view.nb_chunks
                }
            }),
        }

        return uniform
    
    def get_intermediate_result_uniform(self, idx_i : int, idx_t : int) -> TypeUtils.UniformGPUData_t:
        intermediate_result = self._intermediate_results[(idx_i, idx_t)]
        
        uniform = {
            "drawResult" : DrawResultCopyUniformsDict({
                    "ms_color"     : intermediate_result.rendered_image_pair.ms_color.texture,
                    "ms_depth"     : intermediate_result.rendered_image_pair.ms_depth.texture,
                    "ms_quality"   : intermediate_result.rendered_quality.ms_img.texture,
                    "ms_mask"      : intermediate_result.rendered_mask.ms_img.texture,
                    "color"        : {
                        "texture"           : intermediate_result.rendered_image_pair.color.texture,
                        "accumulateBuffer"  : intermediate_result.rendered_image_pair.color.grad,
                        "no_grad"           : False,
                        "range"             : intermediate_result.rendered_image_pair.color.type_range
                    },
                    "depth"        : {
                        "texture"           : intermediate_result.rendered_image_pair.depth.texture,
                        "accumulateBuffer"  : intermediate_result.rendered_image_pair.depth.grad,
                        "no_grad"           : False,
                        "range"             : intermediate_result.rendered_image_pair.depth.type_range
                    },
                    "quality"        : {
                        "texture"           : intermediate_result.rendered_quality.dst_img.texture,
                        "accumulateBuffer"  : intermediate_result.rendered_quality.dst_img.grad,
                        "no_grad"           : False,
                        "range"             : intermediate_result.rendered_quality.dst_img.type_range
                    },
                    "mask"        : {
                        "texture"           : intermediate_result.rendered_mask.dst_img.texture,
                        "accumulateBuffer"  : intermediate_result.rendered_mask.dst_img.grad,
                        "no_grad"           : False,
                        "range"             : intermediate_result.rendered_mask.dst_img.type_range
                    },
            }),
        }
        
        return uniform
    
    def get_target_view_uniform(self, target_view : TargetOutputViewDict) -> TypeUtils.UniformGPUData_t:
        view = target_view["camera"]
        blended_view = target_view["blended_viewpoint"]
        
        uniform = {
            "blendingResult" : BlendingResultUniformsDict({
                "color"     : {
                    "texture"           : blended_view.blended_image_pair.color.texture,
                    "accumulateBuffer"  : blended_view.blended_image_pair.color.grad,
                    "no_grad"           : False,
                    "range"             : blended_view.blended_image_pair.color.type_range
                },
                "depth"     : {
                    "texture"           : blended_view.blended_image_pair.depth.texture,
                    "accumulateBuffer"  : blended_view.blended_image_pair.depth.grad,
                    "no_grad"           : False,
                    "range"             : blended_view.blended_image_pair.depth.type_range
                },
                "quality"     : {
                    "texture"           : blended_view.blended_quality.texture,
                    "accumulateBuffer"  : blended_view.blended_quality.grad,
                    "no_grad"           : False,
                    "range"             : blended_view.blended_quality.type_range
                },
                "mask"     : {
                    "texture"           : blended_view.blended_mask.texture,
                    "accumulateBuffer"  : blended_view.blended_mask.grad,
                    "no_grad"           : True,
                    "range"             : blended_view.blended_mask.type_range
                },
            }),
            "color_texture" : CstTextureUniformDict({"texture":view.image_pair.color.texture}),
            "depth_texture" : CstTextureUniformDict({"texture":view.image_pair.depth.texture}),
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
            "sampler_tex": self.sampler
        }

        return uniform
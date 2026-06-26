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

from abc import ABC, abstractmethod
from typing import Sequence, TypedDict
from enum import Enum

import numpy as np
import slangpy as spy

from ._utils import Logger
from .RenderingDevice import RenderingDevice, SubmitId
from .Camera import Camera

class RenderingObjectType(Enum):
    NONE                = 0
    Color               = 1
    Depth               = 2
    Focal               = 3
    Principal_Point     = 4
    Position            = 5
    Rotation            = 6
    
    Quality             = 7

class RenderingStageType(Enum):
    NONE = 0
    Target = 1
    Input = 2
    IntermediateResult = 3
    BlendedResult = 4

class RenderingObject:
    def __init__(self, type : RenderingObjectType, stage : RenderingStageType):
        self.type = type
        self.stage = stage

    def is_input(self) -> bool:
        return self.stage == RenderingStageType.Input

    def is_target(self) -> bool:
        return self.stage == RenderingStageType.Target
    
    def is_result(self) -> bool:
        return (self.stage == RenderingStageType.BlendedResult) or (self.stage == RenderingStageType.IntermediateResult)

class RenderingApp(ABC):
    DEFAULT_NB_SAMPLES_MS = 4

    _RENDER_PROGRAM_FNAME = "unknown.slang"
    _RENDER_VERTEX_SHADER_EP    = "vertex_main"
    _RENDER_FRAGMENT_SHADER_EP  = "fragment_main"
    _RENDER_GEOMETRY_SHADER_EP  = "geometry_main"
    
    _APP_TOML_KEY = "DiffDIBR"

    def __init__(self):
        self.device_handler : RenderingDevice
        self.entry_point_names : list[str]
        self.render_program : spy.ShaderProgram
        self.render_pipeline : spy.RenderPipeline
        self.input_layout : spy.InputLayout

    def close(self):
        self.device_handler.close()

    @property
    def input_views(self):
        raise NotImplementedError()

    @property
    def target_views(self):
        raise NotImplementedError()

    @abstractmethod
    def load_input_view(self, view : Camera):
        raise NotImplementedError("The base RenderingApp class should not be used for rendering.")
    
    @abstractmethod
    def load_target_view(self, view : Camera):
        raise NotImplementedError("The base RenderingApp class should not be used for rendering.")

    @abstractmethod
    def release_target_views(self, ):
        raise NotImplementedError("The base RenderingApp class should not be used for rendering.")
        
    @abstractmethod
    def release_input_views(self, ):
        raise NotImplementedError("The base RenderingApp class should not be used for rendering.")
        
    @abstractmethod
    def get_uniforms(self, input_view, target_output_view):
        raise NotImplementedError("The base RenderingApp class should not be used for rendering.")

    @abstractmethod
    def render(self, input_idx : int, target_idx : int, wait_submit_id : SubmitId = None) -> SubmitId:
        raise NotImplementedError("The base RenderingApp class should not be used for rendering.")
    
    @abstractmethod
    def bwd_render(self, input_idx : int, target_idx : int, wait_submit_id : SubmitId = None) -> SubmitId:
        raise NotImplementedError("The base RenderingApp class should not be used for rendering or differentiation.")

    @staticmethod
    def create_render_prog(device : spy.Device, program_file : str = _RENDER_PROGRAM_FNAME, entry_point_names : list[str] = [_RENDER_VERTEX_SHADER_EP, _RENDER_GEOMETRY_SHADER_EP, _RENDER_FRAGMENT_SHADER_EP]) -> spy.ShaderProgram:
        # Load program
        render_program = device.load_program(module_name=program_file, entry_point_names=entry_point_names)

        Logger.debug(f"Create Rendering Program. fname:{program_file}, entrypoints: {entry_point_names}", str(RenderingApp))

        return render_program

    @staticmethod
    def gen_rendering_pipeline_desc(program : spy.ShaderProgram, input_layout : spy.InputLayout, target_layout : Sequence[spy.ColorTargetDesc], nb_samples : int = DEFAULT_NB_SAMPLES_MS) -> spy.RenderPipelineDesc:
        rend_pip_desc = spy.RenderPipelineDesc(
            {
                "program" : program,
                "input_layout": input_layout,
                "targets" : target_layout,
                "rasterizer": {"multisample_enable" : True, "forced_sample_count" : nb_samples}, #see: https://github.com/shader-slang/slang-rhi/blob/86d193d1df5a44b6716023e50c54f3affa54d6c3/src/vulkan/vk-pipeline.cpp#L460
                "multisample" : {
                    "sample_count" : nb_samples,
                },
                "depth_stencil":{
                    "format" : spy.Format.d32_float,
                    "depth_test_enable": True,
                    "depth_write_enable": True,
                    "depth_func": spy.ComparisonFunc.less,
                    "stencil_enable": False,
                }
            }
        )

        return rend_pip_desc

    @staticmethod
    def use_depth_testing(device_h : RenderingDevice, width : int, height :int, cam_name : str, nb_samples : int = DEFAULT_NB_SAMPLES_MS, label = "RenderApp"):
        depth_test_buffer = device_h.create_depth_buffer_tex(width, height, label= f"{label}_depth_buffer_{cam_name}", sample_count=nb_samples)

        Logger.debug(f"Enable Depth Testing.", str(RenderingApp))

        return depth_test_buffer

    @staticmethod
    def create_render_state(width : int, height : int, vertex_buffer : spy.Buffer, index_buffer: spy.Buffer):
        if(width < 0 or height < 0): raise Exception("Error - App: cannot create viewports and scissor_rect for negative resolution.")

        return spy.RenderState(
            {
                "viewports":[spy.Viewport.from_size(width, height)],
                "scissor_rects" : [spy.ScissorRect.from_size(width, height)],
                "vertex_buffers": [vertex_buffer],
                "index_buffer": index_buffer,
                "index_format": spy.IndexFormat.uint32,
            }
        )

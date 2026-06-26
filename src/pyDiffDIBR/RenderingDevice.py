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

from typing import Union, TypedDict
import os

import numpy as np
import slangpy as spy

from ._utils import ImageUtils, ArrayUtils, TypeUtils, Singleton, GPUUtils, Logger, IOUtils

SubmitId = Union[list[int], int, None]

KernelProgramDict = TypedDict("KernelProgramDict", {
    "program" : spy.ShaderProgram,
    "kernel" : spy.ComputeKernel
})

class RenderingDevice(metaclass = Singleton):
    """
    RenderingDevice class abstracts GPU device operations using SlangPy.
    Provides methods for creating textures, buffers, and submitting commands.
    Implements Singleton pattern to avoid multiple device instances.
    """ 
    _DEFINE_GPU_NB_SAMPLES  = "NB_SAMPLES"
    _DEFINE_GPU_TRAINING    = "__TRAINING__"


    def __init__(self, device_type = spy.DeviceType.automatic, debug = False, defines:dict[str, str] = {}):
        if( device_type in [spy.DeviceType.cuda, spy.DeviceType.cpu, spy.DeviceType.metal, spy.DeviceType.wgpu]): raise Exception("Error - RenderingDevice: cannot a device with unknown behavior. Device: " + device_type.name)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.compiler_options = spy.SlangCompilerOptions({
                "include_paths": [
                    spy.SHADER_PATH,
                    "shaders",
                    "src/shaders",
                    "src/pyDiffDIBR/shaders",
                    current_dir + "/src/pyDiffDIBR/shaders",
                    current_dir + "/src/shaders",
                    current_dir + "/shaders",
                ],
                "defines":defines
            }
        )
        self.debug = debug
        self.device = spy.Device(    
            enable_debug_layers=debug,
            compiler_options=self.compiler_options,
            type=device_type,
        )

        GPUUtils.print_info(self.device)
        for info in self.info: Logger.debug(info,str(RenderingDevice))

    @staticmethod
    def get_sampling_defines(nb_samples : int) -> dict[str, str]:
        return {RenderingDevice._DEFINE_GPU_NB_SAMPLES : str(nb_samples)}
    
    @staticmethod
    def get_training_defines() -> dict[str, str]:
        return {RenderingDevice._DEFINE_GPU_TRAINING : "1"}

    @property
    def info(self) -> list[str]:
        info = []
        info.append("-"*5 + "RenderingDevice" + "-"*5 )
        info.append(f"defines: {self.compiler_options.defines}")
        info.append(f"include: {self.compiler_options.include_paths}")
        return info

    def close(self):
        self.wait()
        self.device.close()

    def wait(self):
        Logger.debug('wait for all GPU jobs to finish.', str(self.__class__))
        self.device.wait()
    
    def wait_id(self, id : int):
        Logger.debug(f'wait for GPU job ID:{id} to finish.', str(self.__class__))
        self.device.wait_for_submit(id)

    def wait_submit_id(self, id : SubmitId):
        if id is None: 
            return
        elif isinstance(id, list):
            for id_ in id: 
                self.wait_id(id_)
        elif isinstance(id, int):
            self.wait_id(id)

    def create_command_encoder(self) -> spy.CommandEncoder:
        return self.device.create_command_encoder()

    def create_sampler(self, label="basic_sampler") -> spy.Sampler:
        sampler = self.device.create_sampler(
            label=label
        )

        return sampler
    
    def load_kernel(self, path : str, entry_point : str) -> KernelProgramDict:
        Logger.debug(f"Load GPU program '{entry_point}' from file '{path}'.", str(self.__class__))

        program = self.device.load_program(module_name=path, entry_point_names=[entry_point])
        kernel  = self.device.create_compute_kernel(program)

        return KernelProgramDict({"kernel" : kernel, "program" : program})


    def create_shader_texture(
            self,
            np_texture : np.ndarray,
            label = "tex", 
            usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access  # Allow shader access and unordered writes
            ) -> spy.Texture:
        Logger.debug(f"Create texture with label: {label}", str(self.__class__))
        tex = self.device.create_texture(
            data=np_texture,
            width=np_texture.shape[1],
            height=np_texture.shape[0],
            format=ImageUtils.get_image_slangpy_format(np_texture), # Could not create RGB32float tex ?
            usage=usage,
            sample_count=1,
            type=spy.TextureType.texture_2d,
            label = label
        )
        return tex
    
    def create_render_texture(
            self,
            width : int,
            height : int,
            format : spy.Format,
            label = "render_tex", 
            usage = spy.TextureUsage.render_target  # Usage flag for render target
            ) -> spy.Texture:
        Logger.debug(f"Create texture with label: {label}", str(self.__class__))
        tex = self.device.create_texture(
            width=width,
            height=height,
            format=format, # Could not create RGB32float tex ?
            usage=usage,
            type=spy.TextureType.texture_2d,
            label = label
        )
        return tex
    
    def create_multisample_render_texture(
            self,
            width : int,
            height : int,
            format : spy.Format,
            label = "ms_render_tex", 
            usage = spy.TextureUsage.render_target,  # Usage flag for render target | spy.TextureUsage.resolve_source,
            sample_count : int = 1,
            ) -> spy.Texture:
        Logger.debug(f"Create texture with label: {label}", str(self.__class__))
        tex = self.device.create_texture(
            width=width,
            height=height,
            format=format, # Could not create RGB32float tex ?
            usage=usage,
            sample_quality = 1,  # Fixed sample quality for MSAA,
            sample_count=sample_count,
            type=spy.TextureType.texture_2d_ms,
            label = label
        )
        return tex
    
    def create_depth_buffer_tex(
        self,
        width : int,
        height : int,
        label = "depth_buffer", 
        usage = spy.TextureUsage.depth_stencil | spy.TextureUsage.copy_source, #if usage.render_target, then driver_error "cannot create image d32_sfloat"
        sample_count : int = 1,
    ) -> spy.Texture:
        tex = self.device.create_texture(
            format=spy.Format.d32_float,
            usage=usage,
            width=width,
            height=height,
            label=label,
            sample_count= sample_count,
            sample_quality=1,
            type= spy.TextureType.texture_2d if(sample_count == 1) else spy.TextureType.texture_2d_ms
        )    
        return tex
    
    def create_buffer(
            self,
            np_buffer : np.ndarray,
            label = "buffer", 
            usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
            ) -> spy.Buffer:
        Logger.debug(f"Create buffer with label: {label}", str(self.__class__))
        buff = self.device.create_buffer(
            data=np_buffer,
            format=ArrayUtils.get_array_slangpy_format(np_buffer.shape[-1], np_buffer.dtype),
            usage=usage,
            label = label
        )
        return buff
    
    def create_chunked_buffer(
            self,
            np_buffer : np.ndarray,
            nb_chunks : list[int] | tuple[int, int] = [1, 1],
            label = "buffer", 
            usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
            ) -> spy.Buffer:
        Logger.debug(f"Create buffer with label: {label}", str(self.__class__))
        buff = self.create_buffer(
            np_buffer.repeat(nb_chunks[0] * nb_chunks[1], axis=0).repeat(1, axis=1),
            usage=usage,
            label = label
        )
        return buff
    
    def create_gradient_buffer_from_image(
            self,
            np_tex : np.ndarray,
            label = "buffer", 
            ) -> spy.Buffer:
        buff = self.device.create_buffer(
            data=np.zeros((np_tex.shape[0] *np_tex.shape[1] * np_tex.shape[2]), dtype="int32"),
            format=spy.Format.r32_sint,
            usage= spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            label = label
        )
        return buff
        

    def create_vertex(self, data : np.ndarray, label="vertex_buffer") -> spy.Buffer:
        vertex_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
            label=label,
            data=data,
        )
        return vertex_buffer
    
    def create_index(self, data : np.ndarray, label = 'index_buffer') -> spy.Buffer:
        index_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource,
            label=label,
            data=data,
        )
        return index_buffer
    
    @staticmethod
    def create_input_element(semantic_name : str, format : spy.Format) -> spy.InputElementDesc:
        return spy.InputElementDesc(
            {
                "semantic_name": semantic_name,
                #"semantic_index": 0, #either by name or by index
                "format": format,
            }
        )
    
    @staticmethod
    def create_vertex_stream(nb_byte : int, nb_coords : int) -> spy.VertexStreamDesc:
        return spy.VertexStreamDesc({"stride": nb_byte * nb_coords})

    def create_input_layout(self, sematic_name : str, nb_coords : int, dtype : str) -> spy.InputLayout:
        format = ArrayUtils.get_array_slangpy_format(nb_coords, dtype)
        nb_byte = TypeUtils.get_type_byte(dtype)

        in_el = RenderingDevice.create_input_element(sematic_name, format)
        vx_st = RenderingDevice.create_vertex_stream(nb_byte, nb_coords)
        input_layout = self.device.create_input_layout(
            input_elements=[in_el,],
            vertex_streams=[vx_st,],
        )
        return input_layout
    
    def submit_command(self,cmd_enc : spy.CommandEncoder, wait_submit_id : SubmitId = None, wait = False) -> int:
        """
        Submit a command encoder to the GraphicQueue.
        Note that, both `wait_submit_id` and `wait` create a fence.
        """
        Logger.debug(f"Submit new command to GPU.", str(self.__class__))

        self.wait_submit_id(wait_submit_id)
        
        id = self.device.submit_command_buffer(cmd_enc.finish())
        
        if(wait) : self.wait()
        return id
    
    def submit_commands(self, cmd_encs : list[spy.CommandEncoder], wait = False):
        self.device.submit_command_buffers(command_buffers=[c_e.finish() for c_e in cmd_encs])
        if(wait) : self.wait()

    def create_surface(self, window : spy.Window):
        return self.device.create_surface(window)
    
    def copy_tex(self, command_encoder : spy.CommandEncoder, src_tex : spy.Texture, dst_tex : spy.Texture):
        command_encoder.copy_texture(
            dst_tex, 
            spy.SubresourceRange({"layer": 0,"layer_count": dst_tex.layer_count,"mip": 0,"mip_count": dst_tex.mip_count}),
            spy.uint3(0),
            src_tex,
            spy.SubresourceRange({"layer": 0,"layer_count": src_tex.layer_count,"mip": 0,"mip_count": src_tex.mip_count}),
            spy.uint3(0)
        )
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
from typing import TypedDict, Any

import numpy as np
import slangpy as spy

from ._utils import TypeUtils, Logger
from .RenderingDevice import RenderingDevice, SubmitId, KernelProgramDict
from .RenderingApp import RenderingObject, RenderingObjectType

LossDataParameterDict = TypedDict("LossDataParameterDict",{
    "uniform_data"      : TypeUtils.UniformGPUData_t,
    "name"              : str,
    "size"              : tuple[int, int],
    "metadata"          : dict[str, Any] | None
})

LossTextureUniform = TypedDict("LossTextureUniform", {
    "texture" : spy.Texture
})

class Loss(ABC):
    _KERNEL_FNAME = "Loss.slang"
    _KERNEL_FWD_EP = "None"
    _KERNEL_BWD_EP = "None"
    _KERNEL_FWD_BWD_EP = "None" 

    _KERNEL_MEAN_FNAME = "Mean.slang"
    _KERNEL_MEAN_FWD_EP= "main_basic_mean_texture_t4"

    def __init__(
            self, 
            scale : float, 
            objects : list[RenderingObject], 
            slang_fname : str = _KERNEL_FNAME,
            fwd_kernel_entry_point_names        : str | None = None,
            bwd_kernel_entry_point_names        : str | None = None,
            fwd_bwd_kernel_entry_point_names    : str | None = None,
            label : str = "abstract-loss"):
        self.device_handler : RenderingDevice = RenderingDevice()
        
        self.scale = scale

        self._textures      : list[spy.Texture] = []
        self._textures_v    : list[spy.TextureView] = []

        self._fwd_kernel     : KernelProgramDict | None = None if fwd_kernel_entry_point_names is None else self.device_handler.load_kernel(slang_fname, fwd_kernel_entry_point_names)
        self._bwd_kernel     : KernelProgramDict | None = None if bwd_kernel_entry_point_names is None else self.device_handler.load_kernel(slang_fname, bwd_kernel_entry_point_names)
        self._fwd_bwd_kernel : KernelProgramDict | None = None if fwd_bwd_kernel_entry_point_names is None else self.device_handler.load_kernel(slang_fname, fwd_bwd_kernel_entry_point_names)

        self.objects = objects
        self.label = label

        self._mean_buffer       : spy.Buffer = self.device_handler.create_buffer(np.zeros((64 * 64, 4), dtype="float32"), label=f"{label}_mean_buffer")
        self._mean_fwd_kernel   : KernelProgramDict = self.device_handler.load_kernel(Loss._KERNEL_MEAN_FNAME, Loss._KERNEL_MEAN_FWD_EP)

    @abstractmethod
    def __call__(self, *args: Any, **kwargs : Any) -> SubmitId:
        """
        Forward Call to loss.
        For the required paremeters, please check the implementation of your non-generic loss.
        """
        raise NotImplementedError("You should not call a function of an abstract class.")
    
    @abstractmethod
    def fwd(self, *args: Any, **kwargs : Any) -> SubmitId:
        """
        Forward Call to loss.
        For the required paremeters, please check the implementation of your non-generic loss.
        """
        raise NotImplementedError("You should not call a function of an abstract class.")

    @abstractmethod
    def bwd(self, *args: Any, **kwargs : Any) -> SubmitId:
        """
        Backware Call to loss.
        For the required paremeters, please check the implementation of your non-generic loss.
        """
        raise NotImplementedError("You should not call a function of an abstract class.")

    @abstractmethod
    def fwd_bwd(self, *args: Any, **kwargs : Any) -> SubmitId:
        """
        Forward & Backward Call to loss.
        For the required paremeters, please check the implementation of your non-generic loss.
        """
        raise NotImplementedError("You should not call a function of an abstract class.")
    
    @property
    def fwd_kernel(self) -> KernelProgramDict:
        if(self._fwd_kernel is None): raise ValueError("Loss Forward kernel was not loaded.")
        return self._fwd_kernel
    
    @property
    def bwd_kernel(self) -> KernelProgramDict:
        if(self._bwd_kernel is None): raise ValueError("Loss Backward kernel was not loaded.")
        return self._bwd_kernel
    
    @property
    def fwd_bwd_kernel(self) -> KernelProgramDict:
        if(self._fwd_bwd_kernel is None): raise ValueError("Loss Forward/Backward kernel was not loaded.")
        return self._fwd_bwd_kernel

    def clear(self) -> SubmitId:
        command_encoder = self.device_handler.create_command_encoder()
        for tex in self._textures:
            command_encoder.clear_texture_float(tex, clear_value=spy.float4(0.0))
        return self.device_handler.submit_command(command_encoder)
    
    def _initialize_loss_texture(self, w : int, h : int, label : str, dtype='float32'):
        self._textures.append(self.device_handler.create_shader_texture(
            np.zeros((h,w,4), dtype=dtype),
            #label = f"{type(self).__name__}_loss_tex"
            label = label
            )
        )
        self._textures_v.append(self._textures[-1].create_view())
    
    def _release_loss(self):
        self._textures = []
        self._textures_v = []

    def texture(self, idx : int) -> spy.Texture:
        if(idx >= len(self._textures)):
            raise IndexError(f"{Loss.__name__}: loss texture does not exist.")
        return self._textures[idx]
    
    def texture_view(self, idx : int) -> spy.TextureView:
        if(idx >= len(self._textures_v)):
            raise IndexError(f"{Loss.__name__}: loss texture view does not exist.")
        return self._textures_v[idx]
    
    def build_loss_texture_label(self, *vars : str) -> str:
        label = f"{self.label}_loss_tex"
        for var in vars:
            label += f"_{var}"
        return label
    
    def find_texture_by_label(self, label : str) -> spy.Texture:
        for texture in self._textures:
            if label == texture.desc.label:
                return texture
        raise KeyError("No loss texture exists with this label.") 
    
    def find_or_init_loss_texture(self, label : str, w : int, h : int) -> spy.Texture:
        try:
            return self.find_texture_by_label(label)
        except KeyError:
            Logger.debug("Loss texture could not be find so the texture was created.", str(self.__class__))
            self._initialize_loss_texture(w, h, label)
            return self.texture(-1)
    
    def mean(self, label : str) -> float:
        loss_tex = self.find_texture_by_label(label)

        command_encoder = self.device_handler.create_command_encoder()

        kernel = self._mean_fwd_kernel["kernel"]

        thread_count = [
            64,
            64,
            1
        ]

        kernel.dispatch(
            thread_count = thread_count,
            inputTexture = loss_tex,
            outputBuffer = self._mean_buffer
        )

        id_ = self.device_handler.submit_command(command_encoder)
        self.device_handler.wait_submit_id(id_)

        data_cpu = self._mean_buffer.to_numpy()

        command_encoder = self.device_handler.create_command_encoder()
        command_encoder.clear_buffer(self._mean_buffer)
        self.device_handler.submit_command(command_encoder)

        return (data_cpu.sum().item())

class LossUtils(ABC):
    
    @staticmethod
    def find_loss_by_label(loss_list : list[Loss], label_loss : str) -> Loss:
        loss_ : Loss | None = None
        for loss in loss_list:
            if(label_loss == loss.label):
               loss_= loss
        if loss_ is None : 
            raise KeyError("Impossible to find the loss in the list of losses.")
        return loss_

class L1Norm(Loss):
    _KERNEL_FNAME = "Loss.slang"
    _KERNEL_FWD_COLOR_EP = None
    _KERNEL_BWD_COLOR_EP = "main_bwd_l1_norm_color"
    _KERNEL_FWD_BWD_COLOR_EP = "main_fwd_bwd_l1_norm_color"
    _KERNEL_FWD_DEPTH_EP = None
    _KERNEL_BWD_DEPTH_EP = None
    _KERNEL_FWD_BWD_CST_DIFF_DEPTH_EP = "main_fwd_bwd_l1_norm_cst_diff_depth"
    _KERNEL_FWD_BWD_DIFF_DIFF_DEPTH_EP = "main_fwd_bwd_l1_norm_diff_diff_depth"

    def __init__(
            self, 
            scale: float, 
            objects : list[RenderingObject], 
            slang_fname : str = _KERNEL_FNAME,
            fwd_kernel_entry_point_name        : str | None = None,
            bwd_kernel_entry_point_name        : str | None = None,
            fwd_bwd_kernel_entry_point_name    : str | None = None,
            label : str = "l1-norm", 
        ):
        super().__init__(scale, objects, label=label, slang_fname=slang_fname, fwd_kernel_entry_point_names=fwd_kernel_entry_point_name,bwd_kernel_entry_point_names=bwd_kernel_entry_point_name, fwd_bwd_kernel_entry_point_names=fwd_bwd_kernel_entry_point_name)

    def __call__(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture) -> SubmitId:
        raise NotImplementedError("")
    
    def fwd(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture) -> SubmitId:
        raise NotImplementedError("")

    def bwd(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture, grad_dst : spy.Buffer) -> SubmitId:
        raise NotImplementedError("")

    def fwd_bwd(
        self,
        target          : LossDataParameterDict,
        input           : LossDataParameterDict,
        dst             : TypeUtils.UniformGPUData_t | None = None, 
        thread_count    : TypeUtils.ThreadCount_t    | None = None,
        wait_submit_id : SubmitId = None
        ) -> SubmitId:
        command_encoder = self.device_handler.create_command_encoder()

        if thread_count is None : thread_count = (target["size"][0], target["size"][1], 1)

        if dst is None:
            label_loss = self.build_loss_texture_label(target['name'], input['name'])
            texture = self.find_or_init_loss_texture(label_loss, target["size"][0], target["size"][1])
            loss = LossTextureUniform({"texture" : texture})
        else: loss = dst
        
        kernel : spy.ComputeKernel = self.fwd_bwd_kernel["kernel"]

        kernel.dispatch(
            thread_count, 
            command_encoder = command_encoder,
            target          = target["uniform_data"],
            input           = input["uniform_data"],
            loss            = loss,
            scale_loss      = self.scale
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

class MaskedL1Norm(Loss):
    _KERNEL_FNAME = "Loss.slang"
    _KERNEL_FWD_COLOR_EP = None
    _KERNEL_BWD_COLOR_EP = None
    _KERNEL_FWD_BWD_COLOR_EP = "main_fwd_bwd_masked_l1_norm_color"
    _KERNEL_FWD_DEPTH_EP = None
    _KERNEL_BWD_DEPTH_EP = None
    _KERNEL_FWD_BWD_CST_DIFF_DEPTH_EP = None
    _KERNEL_FWD_BWD_DIFF_DIFF_DEPTH_EP = "main_fwd_bwd_masked_l1_norm_diff_diff_depth"

    def __init__(
            self, 
            scale: float, 
            objects : list[RenderingObject], 
            slang_fname : str = _KERNEL_FNAME,
            fwd_kernel_entry_point_name        : str | None = None,
            bwd_kernel_entry_point_name        : str | None = None,
            fwd_bwd_kernel_entry_point_name    : str | None = None,
            label : str = "l1-norm", 
        ):
        super().__init__(scale, objects, label=label, slang_fname=slang_fname, fwd_kernel_entry_point_names=fwd_kernel_entry_point_name,bwd_kernel_entry_point_names=bwd_kernel_entry_point_name, fwd_bwd_kernel_entry_point_names=fwd_bwd_kernel_entry_point_name)

    def __call__(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture) -> SubmitId:
        raise NotImplementedError("")
    
    def fwd(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture) -> SubmitId:
        raise NotImplementedError("")

    def bwd(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture, grad_dst : spy.Buffer) -> SubmitId:
        raise NotImplementedError("")

    def fwd_bwd(
        self,
        target          : LossDataParameterDict,
        input           : LossDataParameterDict,
        mask            : TypeUtils.UniformGPUData_t,
        dst             : TypeUtils.UniformGPUData_t | None = None, 
        thread_count    : TypeUtils.ThreadCount_t    | None = None,
        wait_submit_id : SubmitId = None
        ) -> SubmitId:
        command_encoder = self.device_handler.create_command_encoder()

        if thread_count is None : thread_count = (target["size"][0], target["size"][1], 1)

        if dst is None:
            label_loss = self.build_loss_texture_label(target['name'], input['name'])
            texture = self.find_or_init_loss_texture(label_loss, target["size"][0], target["size"][1])
            loss = LossTextureUniform({"texture" : texture})
        else: loss = dst
        
        kernel : spy.ComputeKernel = self.fwd_bwd_kernel["kernel"]

        kernel.dispatch(
            thread_count, 
            command_encoder = command_encoder,
            target          = target["uniform_data"],
            input           = input["uniform_data"],
            mask            = mask,
            loss            = loss,
            scale_loss      = self.scale
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)


class L2Norm(Loss):
    _KERNEL_FNAME = "Loss.slang"
    _KERNEL_FWD_COLOR_EP = None
    _KERNEL_BWD_COLOR_EP = None
    _KERNEL_FWD_BWD_COLOR_EP = "main_call_main_bwd_l2_norm_color"

    def __init__(
            self, 
            scale: float, 
            objects : list[RenderingObject], 
            slang_fname : str = _KERNEL_FNAME,
            fwd_kernel_entry_point_name        : str | None = None,
            bwd_kernel_entry_point_name        : str | None = None,
            fwd_bwd_kernel_entry_point_name    : str | None = None,
            label : str = "l2-norm", 
        ):
        super().__init__(scale, objects, label=label, slang_fname=slang_fname, fwd_kernel_entry_point_names=fwd_kernel_entry_point_name,bwd_kernel_entry_point_names=bwd_kernel_entry_point_name, fwd_bwd_kernel_entry_point_names=fwd_bwd_kernel_entry_point_name)

    def __call__(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture) -> SubmitId:
        raise NotImplementedError("")
    
    def fwd(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture) -> SubmitId:
        raise NotImplementedError("")

    def bwd(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture, grad_dst : spy.Buffer) -> SubmitId:
        raise NotImplementedError("")

    def fwd_bwd(
        self,
        target          : LossDataParameterDict,
        input           : LossDataParameterDict,
        dst             : TypeUtils.UniformGPUData_t | None = None, 
        thread_count    : TypeUtils.ThreadCount_t | None = None,
        wait_submit_id : SubmitId = None
        ) -> SubmitId:
        command_encoder = self.device_handler.create_command_encoder()

        if thread_count is None: thread_count = (target["size"][0], target["size"][1], 1) 

        if dst is None:
            label_loss = self.build_loss_texture_label(target['name'], input['name'])
            texture = self.find_or_init_loss_texture(label_loss, target["size"][0], target["size"][1])
            loss = LossTextureUniform({"texture" : texture})
        else: loss = dst
        
        kernel : spy.ComputeKernel = self.fwd_bwd_kernel["kernel"]

        kernel.dispatch(
            thread_count, 
            command_encoder = command_encoder,
            target          = target["uniform_data"],
            input           = input["uniform_data"],
            loss            = loss,
            scale_loss      = self.scale
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

class MaskedL2Norm(Loss):
    _KERNEL_FNAME = "Loss.slang"
    _KERNEL_FWD_COLOR_EP = None
    _KERNEL_BWD_COLOR_EP = None
    _KERNEL_FWD_BWD_COLOR_EP = "main_call_main_bwd_masked_l2_norm_color"

    def __init__(
            self, 
            scale: float, 
            objects : list[RenderingObject], 
            slang_fname : str = _KERNEL_FNAME,
            fwd_kernel_entry_point_name        : str | None = None,
            bwd_kernel_entry_point_name        : str | None = None,
            fwd_bwd_kernel_entry_point_name    : str | None = None,
            label : str = "masked-l2-norm", 
        ):
        super().__init__(scale, objects, label=label, slang_fname=slang_fname, fwd_kernel_entry_point_names=fwd_kernel_entry_point_name,bwd_kernel_entry_point_names=bwd_kernel_entry_point_name, fwd_bwd_kernel_entry_point_names=fwd_bwd_kernel_entry_point_name)

    def __call__(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture) -> SubmitId:
        raise NotImplementedError("")
    
    def fwd(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture) -> SubmitId:
        raise NotImplementedError("")

    def bwd(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture, grad_dst : spy.Buffer) -> SubmitId:
        raise NotImplementedError("")

    def fwd_bwd(
        self,
        target          : LossDataParameterDict,
        input           : LossDataParameterDict,
        mask            : TypeUtils.UniformGPUData_t,
        dst             : TypeUtils.UniformGPUData_t | None = None, 
        thread_count    : TypeUtils.ThreadCount_t | None = None,
        wait_submit_id : SubmitId = None
        ) -> SubmitId:
        command_encoder = self.device_handler.create_command_encoder()

        if thread_count is None: thread_count = (target["size"][0], target["size"][1], 1) 

        if dst is None:
            label_loss = self.build_loss_texture_label(target['name'], input['name'])
            texture = self.find_or_init_loss_texture(label_loss, target["size"][0], target["size"][1])
            loss = LossTextureUniform({"texture" : texture})
        else: loss = dst
        
        kernel : spy.ComputeKernel = self.fwd_bwd_kernel["kernel"]

        kernel.dispatch(
            thread_count, 
            command_encoder = command_encoder,
            target          = target["uniform_data"],
            input           = input["uniform_data"],
            mask            = mask,
            loss            = loss,
            scale_loss      = self.scale
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

class AnisotropicTotalVariation(Loss):
    _KERNEL_FNAME = "Loss.slang"
    _KERNEL_FWD_DEPTH_EP = None
    _KERNEL_BWD_DEPTH_EP = None
    _KERNEL_FWD_BWD_DEPTH_EP = "main_call_main_bwd_atv_depth"

    def __init__(
            self, 
            scale: float, 
            object : RenderingObject, 
            slang_fname : str = _KERNEL_FNAME,
            fwd_kernel_entry_point_name        : str | None = None,
            bwd_kernel_entry_point_name        : str | None = None,
            fwd_bwd_kernel_entry_point_name    : str | None = None,
            label : str = "atv", 
        ):
        super().__init__(scale, [object], label=label, slang_fname=slang_fname, fwd_kernel_entry_point_names=fwd_kernel_entry_point_name,bwd_kernel_entry_point_names=bwd_kernel_entry_point_name, fwd_bwd_kernel_entry_point_names=fwd_bwd_kernel_entry_point_name)
    
    def __call__(self, input : spy.Texture, dst : spy.Texture):
        raise NotImplementedError("")
    
    def fwd(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture) -> SubmitId:
        raise NotImplementedError("")

    def bwd(self, input : spy.Texture, dst : spy.Texture, grad_dst : spy.Buffer):
        raise NotImplementedError("")

    def fwd_bwd(
        self,
        input           : LossDataParameterDict,
        dst             : TypeUtils.UniformGPUData_t    | None = None, 
        thread_count    : TypeUtils.ThreadCount_t       | None = None,
        wait_submit_id : SubmitId = None, *args: Any, **kwargs : Any
        ) -> SubmitId:
        command_encoder = self.device_handler.create_command_encoder()

        if thread_count is None: thread_count = (input["size"][0], input["size"][1], 1)

        if dst is None:
            label_loss  = self.build_loss_texture_label(input["name"])
            texture     = self.find_or_init_loss_texture(label_loss, input["size"][0], input["size"][1])
            loss        = LossTextureUniform({"texture" : texture})
        else: loss = dst

        shift_sigmoid   = kwargs["shift_sigmoid"] if ("sigmoid" in kwargs) else False
        shift_sigmoid   = kwargs["shift_sigmoid"] if ("shift_sigmoid" in kwargs) else 0.0
        scale_sigmoid   = kwargs["scale_sigmoid"] if ("scale_sigmoid" in kwargs) else 1.0

        kernel : spy.ComputeKernel = self.fwd_bwd_kernel["kernel"]

        kernel.dispatch(
            thread_count, 
            command_encoder = command_encoder,
            input           = input["uniform_data"],
            loss            = loss,
            scale_loss      = self.scale,
            shift_sigmoid   = shift_sigmoid,
            scale_sigmoid   = scale_sigmoid
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

class DepthConsistency(Loss):
    _KERNEL_FNAME = "Loss.slang"
    _KERNEL_FWD_EP = None
    _KERNEL_BWD_EP = None
    _KERNEL_FWD_BWD_EP = "main_fwd_bwd_depth_consistency_loss"

    def __init__(
            self, 
            scale: float, 
            objects : list[RenderingObject], 
            slang_fname : str = _KERNEL_FNAME,
            fwd_kernel_entry_point_name        : str | None = None,
            bwd_kernel_entry_point_name        : str | None = None,
            fwd_bwd_kernel_entry_point_name    : str | None = None,
            label : str = "l1-norm", 
        ):
        super().__init__(scale, objects, label=label, slang_fname=slang_fname, fwd_kernel_entry_point_names=fwd_kernel_entry_point_name,bwd_kernel_entry_point_names=bwd_kernel_entry_point_name, fwd_bwd_kernel_entry_point_names=fwd_bwd_kernel_entry_point_name)

    def __call__(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture) -> SubmitId:
        raise NotImplementedError("")
    
    def fwd(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture) -> SubmitId:
        raise NotImplementedError("")

    def bwd(self, target : spy.Texture, output : spy.Texture, dst : spy.Texture, grad_dst : spy.Buffer) -> SubmitId:
        raise NotImplementedError("")

    def fwd_bwd(
        self,
        rendered_depth_per_input              : LossDataParameterDict,
        blended_depth                         : LossDataParameterDict,
        rendered_quality_per_input            : LossDataParameterDict,
        blended_quality                       : LossDataParameterDict,
        dst             : TypeUtils.UniformGPUData_t | None = None, 
        thread_count    : TypeUtils.ThreadCount_t    | None = None,
        wait_submit_id : SubmitId = None
        ) -> SubmitId:
        command_encoder = self.device_handler.create_command_encoder()

        if thread_count is None : thread_count = (blended_depth["size"][0], blended_depth["size"][1], 1)

        if dst is None:
            label_loss = self.build_loss_texture_label(blended_depth['name'], rendered_depth_per_input['name'])
            texture = self.find_or_init_loss_texture(label_loss, blended_depth["size"][0], blended_depth["size"][1])
            loss = LossTextureUniform({"texture" : texture})
        else: loss = dst
        
        kernel : spy.ComputeKernel = self.fwd_bwd_kernel["kernel"]

        kernel.dispatch(
            thread_count, 
            command_encoder = command_encoder,
            rendered_depth_per_input = rendered_depth_per_input["uniform_data"],
            blended_depth = blended_depth["uniform_data"],
            rendered_quality_per_input = rendered_quality_per_input["uniform_data"],
            blended_quality = blended_quality["uniform_data"],
            loss            = loss,
            scale_loss      = self.scale
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)


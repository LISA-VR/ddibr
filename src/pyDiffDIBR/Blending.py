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
from abc import ABC, abstractmethod
from typing import Any

import slangpy as spy

from ._utils import TypeUtils, Logger
from .RenderingDevice import KernelProgramDict, RenderingDevice, SubmitId

class BlendingMethod(Enum):
    NONE = 0
    Simple = 1
    Masked = 2

class Blending(ABC):
    _PROGRAM_FNAME                  = "Blending.slang"
    _NONE_STR                       = "none"
    _KERNEL_NAME_FWD_BLEND_T4_T1    = _NONE_STR
    _KERNEL_NAME_FWD_NORM_T4_T1     = _NONE_STR
    _KERNEL_NAME_BWD_BLEND_T4       = _NONE_STR
    _KERNEL_NAME_BWD_BLEND_T1       = _NONE_STR
    _KERNEL_NAME_BWD_BLEND_T4_T1    = _NONE_STR
    
    _BLENDING_TOML_KEY = "blending"
    _METHOD_BLENDING_TOML_KEY = "method"
    _BCK_T4_BLENDING_TOML_KEY = "default_t4_color"
    _BCK_T1_BLENDING_TOML_KEY = "default_t1_color"

    def __init__(self,
        fname : str = _PROGRAM_FNAME, 
        ep_name_fwd_blend_t4_t1 : str   = _KERNEL_NAME_FWD_BLEND_T4_T1,
        ep_name_fwd_norm_t4_t1 : str    = _KERNEL_NAME_FWD_NORM_T4_T1,
        ep_name_bwd_blend_t4 : str      = _KERNEL_NAME_BWD_BLEND_T4,
        ep_name_bwd_blend_t1 : str      = _KERNEL_NAME_BWD_BLEND_T1,
        ep_name_bwd_blend_t4_t1 : str   = _KERNEL_NAME_BWD_BLEND_T4_T1,
        default_value_t4 : tuple[float,float,float,float] = (0,0,0,0),
        default_value_t1 : tuple[float, ] = (0, ),
        ):
        self.device_handler = RenderingDevice()

        self.blend_tex4_tex1_compute_kernel = self.device_handler.load_kernel(fname, ep_name_fwd_blend_t4_t1)
        self.norm_tex4_tex1_compute_kernel  = self.device_handler.load_kernel(fname, ep_name_fwd_norm_t4_t1)
        self.bwd_blend_t4_compute_kernel : KernelProgramDict | None    = None if ep_name_bwd_blend_t4 == Blending._NONE_STR else self.device_handler.load_kernel(fname, ep_name_bwd_blend_t4)
        self.bwd_blend_t1_compute_kernel : KernelProgramDict | None    = None if ep_name_bwd_blend_t1 == Blending._NONE_STR else self.device_handler.load_kernel(fname, ep_name_bwd_blend_t1)
        self.bwd_blend_t4_t1_compute_kernel = self.device_handler.load_kernel(fname, ep_name_bwd_blend_t4_t1)
        
        self.default_value_t4 : tuple[float,float,float,float] = default_value_t4
        self.default_value_t1 : tuple[float, ]                 = default_value_t1

    @staticmethod
    def from_enum(
        blending_method : BlendingMethod,
        default_value_t4 : tuple[float,float,float,float] = (0,0,0,0),
        default_value_t1 : tuple[float, ] = (0, ),
    ):
        fname : str
        ep_name_fwd_blend_t4_t1 : str
        ep_name_fwd_norm_t4_t1 : str
        ep_name_bwd_blend_t4 : str
        ep_name_bwd_blend_t1 : str
        ep_name_bwd_blend_t4_t1 : str
        if blending_method == BlendingMethod.Simple:
            fname                     = SimpleBlending._PROGRAM_FNAME
            ep_name_fwd_blend_t4_t1   = SimpleBlending._KERNEL_NAME_FWD_BLEND_T4_T1
            ep_name_fwd_norm_t4_t1    = SimpleBlending._KERNEL_NAME_FWD_NORM_T4_T1
            ep_name_bwd_blend_t4      = SimpleBlending._KERNEL_NAME_BWD_BLEND_T4
            ep_name_bwd_blend_t1      = SimpleBlending._KERNEL_NAME_BWD_BLEND_T1
            ep_name_bwd_blend_t4_t1   = SimpleBlending._KERNEL_NAME_BWD_BLEND_T4_T1
    
            Logger.debug("SimpleBlending class is created.",str(Blending))
            return SimpleBlending(
                fname=fname,
                ep_name_fwd_blend_t4_t1 = ep_name_fwd_blend_t4_t1,
                ep_name_fwd_norm_t4_t1 = ep_name_fwd_norm_t4_t1,
                ep_name_bwd_blend_t4 = ep_name_bwd_blend_t4,
                ep_name_bwd_blend_t1 = ep_name_bwd_blend_t1,
                ep_name_bwd_blend_t4_t1 = ep_name_bwd_blend_t4_t1,
                default_value_t4 = default_value_t4,
                default_value_t1 = default_value_t1,
            )
        elif blending_method == BlendingMethod.Masked:
            fname                     = MaskedBlending._PROGRAM_FNAME
            ep_name_fwd_blend_t4_t1   = MaskedBlending._KERNEL_NAME_FWD_BLEND_T4_T1
            ep_name_fwd_norm_t4_t1    = MaskedBlending._KERNEL_NAME_FWD_NORM_T4_T1
            ep_name_bwd_blend_t4      = MaskedBlending._KERNEL_NAME_BWD_BLEND_T4
            ep_name_bwd_blend_t1      = MaskedBlending._KERNEL_NAME_BWD_BLEND_T1
            ep_name_bwd_blend_t4_t1   = MaskedBlending._KERNEL_NAME_BWD_BLEND_T4_T1

            Logger.debug("MaskedBlending class is created.",str(Blending))
            return MaskedBlending(
                fname=fname,
                ep_name_fwd_blend_t4_t1 = ep_name_fwd_blend_t4_t1,
                ep_name_fwd_norm_t4_t1 = ep_name_fwd_norm_t4_t1,
                ep_name_bwd_blend_t4 = ep_name_bwd_blend_t4,
                ep_name_bwd_blend_t1 = ep_name_bwd_blend_t1,
                ep_name_bwd_blend_t4_t1 = ep_name_bwd_blend_t4_t1,
                default_value_t4 = default_value_t4,
                default_value_t1 = default_value_t1,
            )
        else:
            raise ValueError("Blending method does not exist.")

    @abstractmethod
    def blend_color_depth_pair(
            self,
            wait_submit_id : SubmitId = None,
            *args: Any,
            **kwargs : Any
        ) -> SubmitId:
        raise NotImplementedError("You should not call a function of an abstract class.")

    @abstractmethod
    def normalize_blended_pair(
            self,
            wait_submit_id : SubmitId = None,
            *args: Any,
            **kwargs : Any
        ) -> SubmitId:
        raise NotImplementedError("You should not call a function of an abstract class.")

    @abstractmethod
    def bwd_blend_color(
            self,
            wait_submit_id : SubmitId = None,
            *args: Any,
            **kwargs : Any
        ):
       raise NotImplementedError("You should not call a function of an abstract class.")
    
    @abstractmethod
    def bwd_blend_depth(
            self,
            wait_submit_id : SubmitId = None,
            *args: Any,
            **kwargs : Any
        ):
        raise NotImplementedError("You should not call a function of an abstract class.")

    @abstractmethod
    def bwd_blend_color_depth_pair(
            self,
            wait_submit_id : SubmitId = None,
            *args: Any,
            **kwargs : Any
        ):
        raise NotImplementedError("You should not call a function of an abstract class.")

    @abstractmethod
    def blend_view(self, view_to_add : spy.Texture, quality_map : spy.Texture, dst_view : spy.Texture, dst_quality : spy.Texture, channels : int = 4):
        raise NotImplementedError("You should not call a function of an abstract class.")

    @abstractmethod
    def bwd_blend_view(self, view_to_add : spy.Texture, quality_map : spy.Texture, dst_view : spy.Texture, dst_quality : spy.Texture, channels : int = 4):
        raise NotImplementedError("You should not call a function of an abstract class.")

class SimpleBlending(Blending):
    _PROGRAM_FNAME                  = "Blending.slang"
    _KERNEL_NAME_FWD_BLEND_T4_T1    = "main_blend_t4_t1_view"
    _KERNEL_NAME_FWD_NORM_T4_T1     = "main_normalize_t4_t1_blending"
    _KERNEL_NAME_BWD_BLEND_T4       = Blending._NONE_STR
    _KERNEL_NAME_BWD_BLEND_T1       = Blending._NONE_STR
    _KERNEL_NAME_BWD_BLEND_T4_T1    = "main_bwd_blending_t4_t1"

    def __init__(self,
        fname : str = _PROGRAM_FNAME, 
        ep_name_fwd_blend_t4_t1 : str   = _KERNEL_NAME_FWD_BLEND_T4_T1,
        ep_name_fwd_norm_t4_t1 : str    = _KERNEL_NAME_FWD_NORM_T4_T1,
        ep_name_bwd_blend_t4 : str      = _KERNEL_NAME_BWD_BLEND_T4,
        ep_name_bwd_blend_t1 : str      = _KERNEL_NAME_BWD_BLEND_T1,
        ep_name_bwd_blend_t4_t1 : str   = _KERNEL_NAME_BWD_BLEND_T4_T1,
        default_value_t4 : tuple[float,float,float,float] = (0,0,0,0),
        default_value_t1 : tuple[float, ] = (0, ),
        ):
        super().__init__(
            fname = fname,
            ep_name_fwd_blend_t4_t1 = ep_name_fwd_blend_t4_t1,
            ep_name_fwd_norm_t4_t1 = ep_name_fwd_norm_t4_t1,
            ep_name_bwd_blend_t4 = ep_name_bwd_blend_t4,
            ep_name_bwd_blend_t1 = ep_name_bwd_blend_t1,
            ep_name_bwd_blend_t4_t1 = ep_name_bwd_blend_t4_t1,
            default_value_t4 = default_value_t4,
            default_value_t1 = default_value_t1,
        )
        
    def blend_color_depth_pair(
            self, 
            thread_count            : TypeUtils.ThreadCount_t,
            view_color              : TypeUtils.UniformGPUData_t,
            view_depth              : TypeUtils.UniformGPUData_t,   
            view_quality            : TypeUtils.UniformGPUData_t,
            view_normal             : TypeUtils.UniformGPUData_t,      
            blended_view_color      : TypeUtils.UniformGPUData_t,
            blended_view_depth      : TypeUtils.UniformGPUData_t, 
            blended_view_quality    : TypeUtils.UniformGPUData_t, 
            blended_view_normal     : TypeUtils.UniformGPUData_t,
            wait_submit_id : SubmitId = None,
            *args: Any,
            **kwargs : Any
        ) -> SubmitId:
        
        command_encoder = self.device_handler.create_command_encoder()

        self.blend_tex4_tex1_compute_kernel["kernel"].dispatch(
                thread_count, 
                command_encoder = command_encoder,
                view_to_add_t4  = view_color,
                view_to_add_t1  = view_depth,
                quality_map     = view_quality,
                normal_map      = view_normal,
                blended_view_t4 = blended_view_color,
                blended_view_t1 = blended_view_depth,
                blended_quality = blended_view_quality,
                blended_normal  = blended_view_normal,
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

    def normalize_blended_pair(
            self,
            thread_count            : TypeUtils.ThreadCount_t,
            blended_view_color      : TypeUtils.UniformGPUData_t,
            blended_view_depth      : TypeUtils.UniformGPUData_t, 
            blended_view_quality    : TypeUtils.UniformGPUData_t, 
            blended_view_mask       : TypeUtils.UniformGPUData_t, 
            blended_view_normal     : TypeUtils.UniformGPUData_t,  
            quality_treshold        : float,
            wait_submit_id  : SubmitId = None,
            **vars,
        ) -> SubmitId:
        command_encoder = self.device_handler.create_command_encoder()

        self.norm_tex4_tex1_compute_kernel["kernel"].dispatch(
                thread_count, 
                command_encoder = command_encoder,
                blended_view_t4 = blended_view_color,
                blended_view_t1 = blended_view_depth,
                blended_quality = blended_view_quality,
                blended_mask    = blended_view_mask,
                blended_normal  = blended_view_normal,
                default_value_t4 = self.default_value_t4,
                default_value_t1 = self.default_value_t1,
                quality_treshold = (quality_treshold, ) 
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

    def bwd_blend_color(
            self,
            thread_count    : TypeUtils.ThreadCount_t,
            view_to_add     : TypeUtils.UniformGPUData_t, quality_map       : TypeUtils.UniformGPUData_t,
            blended_view    : TypeUtils.UniformGPUData_t, blended_quality   : TypeUtils.UniformGPUData_t,
            wait_submit_id : SubmitId = None
        ):
        raise NotImplementedError(f"{self.bwd_blend_color.__name__} is not yet implemented")
    
    def bwd_blend_depth(
            self,
            thread_count    : TypeUtils.ThreadCount_t,
            view_to_add     : TypeUtils.UniformGPUData_t, quality_map       : TypeUtils.UniformGPUData_t,
            blended_view    : TypeUtils.UniformGPUData_t, blended_quality   : TypeUtils.UniformGPUData_t,
            wait_submit_id : SubmitId = None
        ):
        raise NotImplementedError(f"{self.bwd_blend_depth.__name__} is not yet implemented")

    def bwd_blend_color_depth_pair(
            self,
            thread_count            : TypeUtils.ThreadCount_t,
            view_color              : TypeUtils.UniformGPUData_t,
            view_depth              : TypeUtils.UniformGPUData_t,   
            view_quality            : TypeUtils.UniformGPUData_t,   
            blended_view_color      : TypeUtils.UniformGPUData_t,
            blended_view_depth      : TypeUtils.UniformGPUData_t, 
            blended_view_quality    : TypeUtils.UniformGPUData_t,
            wait_submit_id : SubmitId = None,
            *args: Any,
            **kwargs : Any
        ):
        command_encoder = self.device_handler.create_command_encoder()

        self.bwd_blend_t4_t1_compute_kernel["kernel"].dispatch(
                thread_count, 
                command_encoder = command_encoder,
                color = view_color,
                depth = view_depth,
                quality_map = view_quality,
                blended_quality = blended_view_quality,
                blended_color = blended_view_color,
                blended_depth = blended_view_depth
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id)

    def blend_view(self, view_to_add : spy.Texture, quality_map : spy.Texture, dst_view : spy.Texture, dst_quality : spy.Texture, channels : int = 4):
        raise NotImplementedError(f"{self.blend_view.__name__} is not yet implemented")

    def bwd_blend_view(self, view_to_add : spy.Texture, quality_map : spy.Texture, dst_view : spy.Texture, dst_quality : spy.Texture, channels : int = 4):
        raise NotImplementedError(f"{self.bwd_blend_view.__name__} is not yet implemented")

class MaskedBlending(Blending):
    _PROGRAM_FNAME                  = "Blending.slang"
    _KERNEL_NAME_FWD_BLEND_T4_T1    = "main_masked_blend_t4_t1_view"
    _KERNEL_NAME_FWD_NORM_T4_T1     = "main_normalize_t4_t1_blending"
    _KERNEL_NAME_BWD_BLEND_T4       = Blending._NONE_STR
    _KERNEL_NAME_BWD_BLEND_T1       = Blending._NONE_STR
    _KERNEL_NAME_BWD_BLEND_T4_T1    = "main_bwd_masked_blending_t4_t1"

    def __init__(self,
        fname : str = _PROGRAM_FNAME, 
        ep_name_fwd_blend_t4_t1 : str   = _KERNEL_NAME_FWD_BLEND_T4_T1,
        ep_name_fwd_norm_t4_t1 : str    = _KERNEL_NAME_FWD_NORM_T4_T1,
        ep_name_bwd_blend_t4 : str      = _KERNEL_NAME_BWD_BLEND_T4,
        ep_name_bwd_blend_t1 : str      = _KERNEL_NAME_BWD_BLEND_T1,
        ep_name_bwd_blend_t4_t1 : str   = _KERNEL_NAME_BWD_BLEND_T4_T1,
        default_value_t4 : tuple[float,float,float,float] = (0,0,0,0),
        default_value_t1 : tuple[float, ] = (0, ),
        ):
        super().__init__(
            fname = fname,
            ep_name_fwd_blend_t4_t1 = ep_name_fwd_blend_t4_t1,
            ep_name_fwd_norm_t4_t1 = ep_name_fwd_norm_t4_t1,
            ep_name_bwd_blend_t4 = ep_name_bwd_blend_t4,
            ep_name_bwd_blend_t1 = ep_name_bwd_blend_t1,
            ep_name_bwd_blend_t4_t1 = ep_name_bwd_blend_t4_t1,
            default_value_t4 = default_value_t4,
            default_value_t1 = default_value_t1,
        )

    def blend_color_depth_pair(
            self, 
            thread_count            : TypeUtils.ThreadCount_t,
            view_color              : TypeUtils.UniformGPUData_t,
            view_depth              : TypeUtils.UniformGPUData_t,   
            view_quality            : TypeUtils.UniformGPUData_t,
            view_mask               : TypeUtils.UniformGPUData_t,
            view_normal             : TypeUtils.UniformGPUData_t,   
            blended_view_color      : TypeUtils.UniformGPUData_t,
            blended_view_depth      : TypeUtils.UniformGPUData_t, 
            blended_view_quality    : TypeUtils.UniformGPUData_t,
            blended_view_mask       : TypeUtils.UniformGPUData_t,
            blended_view_normal       : TypeUtils.UniformGPUData_t,
            wait_submit_id : SubmitId = None,
            *args: Any,
            **kwargs : Any
        ) -> SubmitId:
        
        command_encoder = self.device_handler.create_command_encoder()

        self.blend_tex4_tex1_compute_kernel["kernel"].dispatch(
                thread_count, 
                command_encoder = command_encoder,
                view_to_add_t4  = view_color,
                view_to_add_t1  = view_depth,
                quality_map     = view_quality,
                mask_map        = view_mask,
                normal_map      = view_normal,
                blended_view_t4 = blended_view_color,
                blended_view_t1 = blended_view_depth,
                blended_quality = blended_view_quality,
                blended_normal  = blended_view_normal,
                blended_mask    = blended_view_mask,
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

    def normalize_blended_pair(
            self,
            thread_count            : TypeUtils.ThreadCount_t,
            blended_view_color      : TypeUtils.UniformGPUData_t,
            blended_view_depth      : TypeUtils.UniformGPUData_t, 
            blended_view_quality    : TypeUtils.UniformGPUData_t, 
            blended_view_mask       : TypeUtils.UniformGPUData_t, 
            blended_view_normal     : TypeUtils.UniformGPUData_t,  
            quality_treshold        : float,
            wait_submit_id  : SubmitId = None,
            **vars,
        ) -> SubmitId:
        command_encoder = self.device_handler.create_command_encoder()

        self.norm_tex4_tex1_compute_kernel["kernel"].dispatch(
                thread_count, 
                command_encoder = command_encoder,
                blended_view_t4 = blended_view_color,
                blended_view_t1 = blended_view_depth,
                blended_quality = blended_view_quality,
                blended_mask    = blended_view_mask,
                blended_normal  = blended_view_normal,
                default_value_t4 = self.default_value_t4,
                default_value_t1 = self.default_value_t1,
                quality_treshold = (0.1 * quality_treshold, ) 
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

    def bwd_blend_color(
            self,
            thread_count    : TypeUtils.ThreadCount_t,
            view_to_add     : TypeUtils.UniformGPUData_t, quality_map       : TypeUtils.UniformGPUData_t,
            blended_view    : TypeUtils.UniformGPUData_t, blended_quality   : TypeUtils.UniformGPUData_t,
            wait_submit_id : SubmitId = None
        ):
        raise NotImplementedError(f"{self.bwd_blend_color.__name__} is not yet implemented")
    
    def bwd_blend_depth(
            self,
            thread_count    : TypeUtils.ThreadCount_t,
            view_to_add     : TypeUtils.UniformGPUData_t, quality_map       : TypeUtils.UniformGPUData_t,
            blended_view    : TypeUtils.UniformGPUData_t, blended_quality   : TypeUtils.UniformGPUData_t,
            wait_submit_id : SubmitId = None
        ):
        raise NotImplementedError(f"{self.bwd_blend_depth.__name__} is not yet implemented")

    def bwd_blend_color_depth_pair(
            self,
            thread_count            : TypeUtils.ThreadCount_t,
            view_color              : TypeUtils.UniformGPUData_t,
            view_depth              : TypeUtils.UniformGPUData_t,   
            view_quality            : TypeUtils.UniformGPUData_t,
            view_mask               : TypeUtils.UniformGPUData_t,   
            blended_view_color      : TypeUtils.UniformGPUData_t,
            blended_view_depth      : TypeUtils.UniformGPUData_t, 
            blended_view_quality    : TypeUtils.UniformGPUData_t,
            blended_view_mask       : TypeUtils.UniformGPUData_t,
            quality_treshold        : float,
            wait_submit_id : SubmitId = None,
            *args: Any,
            **kwargs : Any
        ):
        command_encoder = self.device_handler.create_command_encoder()

        self.bwd_blend_t4_t1_compute_kernel["kernel"].dispatch(
                thread_count, 
                command_encoder = command_encoder,
                color               = view_color,
                depth               = view_depth,
                quality_map         = view_quality,
                mask_map            = view_mask,
                blended_color       = blended_view_color,
                blended_depth       = blended_view_depth,
                blended_quality     = blended_view_quality,
                blended_mask        = blended_view_mask,
                default_value_t4    = self.default_value_t4, 
                default_value_t1    = self.default_value_t1,
                quality_treshold = (0.1 * quality_treshold, ) 
        )

        return self.device_handler.submit_command(command_encoder, wait_submit_id)

    def blend_view(self, view_to_add : spy.Texture, quality_map : spy.Texture, dst_view : spy.Texture, dst_quality : spy.Texture, channels : int = 4):
        raise NotImplementedError(f"{self.blend_view.__name__} is not yet implemented")

    def bwd_blend_view(self, view_to_add : spy.Texture, quality_map : spy.Texture, dst_view : spy.Texture, dst_quality : spy.Texture, channels : int = 4):
        raise NotImplementedError(f"{self.bwd_blend_view.__name__} is not yet implemented")
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

import numpy as np
import slangpy as spy

from .RenderingDevice import RenderingDevice, SubmitId

from .Camera import Camera, CameraParamDict
from ._utils import Logger

class TextureType(Enum):
    RenderColor = 0
    RenderDepth = 1
    RenderQuality = 2

class Window:
    _QUIT = False

    _KERNEL_COPY_TEXTURE_NAME   = "CopyTexture.slang"
    _KERNEL_COPY_TEXTURE_EP     = "main_copy_tex1_to_tex2"

    _TEXTURE_TYPE = TextureType.RenderColor

    def __init__(self, device_handler : RenderingDevice | None, w : int = 1024, h : int = 1024, title : str = "DiffDIBR"):
        self.device_handler = device_handler if(isinstance(device_handler, RenderingDevice)) else RenderingDevice()

        self.window : spy.Window = spy.Window(w, h, title)

        self.surface = self.device_handler.create_surface(self.window)

        s_format = spy.Format.rgba8_unorm 
        if (self.device_handler.device.desc.type == spy.DeviceType.vulkan):
            s_format = spy.Format.bgrx8_unorm

        self.surface.configure(
            w,
            h,
            s_format,
            usage=spy.TextureUsage.copy_destination | spy.TextureUsage.render_target
        )
        self.window.on_keyboard_event = Window.handle_keyboard

        self.tmp_texture = self.device_handler.create_shader_texture(np.zeros((h,w,4),dtype='float32'),label="tmp_window_tex", usage=spy.TextureUsage.copy_destination | spy.TextureUsage.copy_source | spy.TextureUsage.shader_resource | spy.TextureUsage.render_target | spy.TextureUsage.unordered_access)
        self.copy_t1_t4_kernel = self.device_handler.load_kernel(self._KERNEL_COPY_TEXTURE_NAME, self._KERNEL_COPY_TEXTURE_EP)
        
    def update_frame(self, new_texture : spy.Texture, wait_submit_id : SubmitId = None):
        if(Window._QUIT):
            return 
        
        texture = self.surface.acquire_next_image()
        if not texture:
            return

        command_encoder = self.device_handler.create_command_encoder()
        if (new_texture.format == spy.Format.r32_float):
            self.copy_t1_t4_kernel["kernel"].dispatch(
                [self.tmp_texture.width, self.tmp_texture.height, 1],
                tex_1 = new_texture,
                tex_4 = self.tmp_texture,
                size = (self.tmp_texture.width, self.tmp_texture.height),
                command_encoder=command_encoder
            )
            command_encoder.blit(texture, self.tmp_texture, spy.TextureFilteringMode.linear)
        else:
            command_encoder.blit(texture, new_texture, spy.TextureFilteringMode.linear)
        _ = self.device_handler.submit_command(command_encoder, wait_submit_id=wait_submit_id)

        self.surface.present()

    def __call__(self):
        if(self.window.should_close()):
            Window._QUIT = True
        
        if(Window._QUIT):
            return

        self.window.process_events()

    def __del__(self):
        self.close()

    @staticmethod
    def handle_keyboard(event: spy.KeyboardEvent) -> None:
        Logger.debug(f"Keyboard event detected... Key: {event.key}; pressed? {event.is_key_press()}; repeated? {event.is_key_repeat()}; released? {event.is_key_release()}", str(Window))

        if event.key == spy.KeyCode.escape and event.is_key_press():
            Window._QUIT = True
        elif ((event.key == spy.KeyCode.i)  and event.is_key_press()):
            Window._TEXTURE_TYPE = TextureType.RenderColor
        elif ((event.key == spy.KeyCode.o) and event.is_key_press()):
            Window._TEXTURE_TYPE = TextureType.RenderDepth
        elif ((event.key == spy.KeyCode.p) and event.is_key_press()):
            Window._TEXTURE_TYPE = TextureType.RenderQuality

    def close(self):
        self.surface.unconfigure()
        self.window.close()

class FreeViewpointNavigatorWindow(Window):
    _CAMERA : Camera
    
    _STATIC_CAMERA = True
    _RESET_CAMERA  = False
    
    _CST_SHIFT_POS = 0.01
    _CST_SHIFT_ROT = 0.1 * 3.14/2.0
    _CST_SHIFT_FOCAL = 10

    def __init__(self, device_handler : RenderingDevice | None, w : int = 1024, h : int = 1024, title : str = "DiffDIBR"):
        super(FreeViewpointNavigatorWindow, self).__init__(device_handler, w, h, title)
        self.window.on_keyboard_event = FreeViewpointNavigatorWindow.handle_keyboard
        FreeViewpointNavigatorWindow.reset_camera(w, h)

        Logger.info("-" * 10 + "Free-viewpoint Keybinds" + "-" * 10)
        Logger.info("LEFT/RIGHT: horizontal shift")
        Logger.info("DOWN/UP: vertical shift")
        Logger.info("S/Z: backward/forward")
        Logger.info("A/E: rotate around vertical-axis")
        Logger.info("Q/D: rotate around horizontal-axis")
        Logger.info("W/X: rotate around back/ford-axis")
        Logger.info("P_DOWN/P_UP: zoom-out/in (change focal)")
        
    def __call__(self):
        if(FreeViewpointNavigatorWindow._RESET_CAMERA): 
            FreeViewpointNavigatorWindow.reset_camera(self.window.width, self.window.height)

        Window.__call__(self)
    
    def __del__(self):
        Window.__del__(self)

    @staticmethod
    def reset_camera(w : int, h : int):
        FreeViewpointNavigatorWindow._CAMERA = Camera (
            "free-view", 
            CameraParamDict({
                "Resolution"    : (w, h),
                "Focal"         : (min(w,h), min(w,h)),
                "Principle_point": (int(w//2), int(h//2)),
                "Depth_range"   : (0.5, 2.0),
                "Position" : (0, 0, 0),
                "Rotation" : (0, 0, 0),
                "Projection": "Perspective",
                "BitDepthColor" : 8,
                "BitDepthDepth" : 8,
                "ColorSpace" : "YUV444",
                "DepthColorSpace" : "YUV400",
                "Name" : "Free-Viewpoint camera"
            }), False, 1, True)
        FreeViewpointNavigatorWindow._RESET_CAMERA = False

    @staticmethod
    def handle_keyboard(event: spy.KeyboardEvent) -> None:
        need_cam_update = False
        trigger_motion = (event.is_key_repeat() or event.is_key_press())

        Window.handle_keyboard(event)

        if event.key == spy.KeyCode.delete and event.is_key_press():
            FreeViewpointNavigatorWindow._RESET_CAMERA = True
        elif event.key == spy.KeyCode.right and trigger_motion:
            _pos = FreeViewpointNavigatorWindow._CAMERA.cpu_params.position
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.position = (_pos[0], _pos[1] - FreeViewpointNavigatorWindow._CST_SHIFT_POS, _pos[2])
            need_cam_update = True
        elif event.key == spy.KeyCode.left and trigger_motion:
            _pos = FreeViewpointNavigatorWindow._CAMERA.cpu_params.position
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.position = (_pos[0], _pos[1] + FreeViewpointNavigatorWindow._CST_SHIFT_POS, _pos[2])
            need_cam_update = True
        elif event.key == spy.KeyCode.up and trigger_motion:
            _pos = FreeViewpointNavigatorWindow._CAMERA.cpu_params.position
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.position = (_pos[0], _pos[1], _pos[2] + FreeViewpointNavigatorWindow._CST_SHIFT_POS)
            need_cam_update = True
        elif event.key == spy.KeyCode.down and trigger_motion:
            _pos = FreeViewpointNavigatorWindow._CAMERA.cpu_params.position
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.position = (_pos[0], _pos[1], _pos[2] - FreeViewpointNavigatorWindow._CST_SHIFT_POS)
            need_cam_update = True
        elif event.key == spy.KeyCode.z and trigger_motion:
            _pos = FreeViewpointNavigatorWindow._CAMERA.cpu_params.position
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.position = (_pos[0] + FreeViewpointNavigatorWindow._CST_SHIFT_POS, _pos[1], _pos[2])
            need_cam_update = True
        elif event.key == spy.KeyCode.s and trigger_motion:
            _pos = FreeViewpointNavigatorWindow._CAMERA.cpu_params.position
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.position = (_pos[0] - FreeViewpointNavigatorWindow._CST_SHIFT_POS, _pos[1], _pos[2])
            need_cam_update = True
        elif event.key == spy.KeyCode.a and trigger_motion:
            _rot = FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation = (_rot[0] + FreeViewpointNavigatorWindow._CST_SHIFT_ROT, _rot[1], _rot[2])
            need_cam_update = True
        elif event.key == spy.KeyCode.e and trigger_motion:
            _rot = FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation = (_rot[0] - FreeViewpointNavigatorWindow._CST_SHIFT_ROT, _rot[1], _rot[2])
            need_cam_update = True
        elif event.key == spy.KeyCode.q and trigger_motion:
            _rot = FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation = (_rot[0], _rot[1] + FreeViewpointNavigatorWindow._CST_SHIFT_ROT, _rot[2])
            need_cam_update = True
        elif event.key == spy.KeyCode.d and trigger_motion:
            _rot = FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation = (_rot[0], _rot[1] - FreeViewpointNavigatorWindow._CST_SHIFT_ROT, _rot[2])
            need_cam_update = True
        elif event.key == spy.KeyCode.w and trigger_motion:
            _rot = FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation = (_rot[0], _rot[1], _rot[2] + FreeViewpointNavigatorWindow._CST_SHIFT_ROT)
            need_cam_update = True
        elif event.key == spy.KeyCode.x and trigger_motion:
            _rot = FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.rotation = (_rot[0], _rot[1], _rot[2] - FreeViewpointNavigatorWindow._CST_SHIFT_ROT)
            need_cam_update = True
        elif event.key == spy.KeyCode.page_up and trigger_motion:
            _focal = FreeViewpointNavigatorWindow._CAMERA.cpu_params.focal
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.focal = (_focal[0] + FreeViewpointNavigatorWindow._CST_SHIFT_FOCAL, _focal[1] + FreeViewpointNavigatorWindow._CST_SHIFT_FOCAL)
            need_cam_update = True
        elif event.key == spy.KeyCode.page_down and trigger_motion:
            _focal = FreeViewpointNavigatorWindow._CAMERA.cpu_params.focal
            FreeViewpointNavigatorWindow._CAMERA.cpu_params.focal = (_focal[0] - FreeViewpointNavigatorWindow._CST_SHIFT_FOCAL, _focal[1] - FreeViewpointNavigatorWindow._CST_SHIFT_FOCAL)
            need_cam_update = True
        elif event.key == spy.KeyCode.enter and event.is_key_press():
            FreeViewpointNavigatorWindow._STATIC_CAMERA = not FreeViewpointNavigatorWindow._STATIC_CAMERA

        if need_cam_update:
            FreeViewpointNavigatorWindow._CAMERA.update_gpu()

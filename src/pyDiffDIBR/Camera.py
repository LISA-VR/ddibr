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

from typing import TypedDict
from enum import Enum
from copy import deepcopy

import numpy as np
import slangpy as spy

from ._utils import ImageUtils, Logger, MathUtils
from .RenderingDevice import RenderingDevice
from .Image import (
    RWImage,
    RenderedMultiSampleImageComposite,
    
    ImagePair,
    RenderedMultiSampleImagePairComposite,
    RWImagePair
)



class CameraParamDict(TypedDict):
    """Camera Parameter dictionnary to transfer these parameters easily through file or other programs.
    It follows the Camera JSON object of MPEG camera parameter config file. 
    """

    Resolution        : tuple[int, int]
    Focal             : tuple[float, float]
    Principle_point   : tuple[float, float]
    Position          : tuple[float, float, float]
    Rotation          : tuple[float, float, float]
    Depth_range       : tuple[float, float]
    Projection        : str
    Name              : str
    BitDepthColor     : int
    BitDepthDepth     : int
    ColorSpace        : str
    DepthColorSpace   : str


class ProjectionType(Enum):
    """A list of all handled camera projections.
    """
    NONE = 0
    Perspective = 1

class CPUCameraParameters:
    """Stores camera parameters on CPU."""

    def __init__(self, name: str, resolution: tuple[int, int], focal: tuple[float, float],
                 principal_point: tuple[float, float], position: tuple[float, float, float],
                 rotation: tuple[float, float, float], depth_range: tuple[float, float],
                 projection: str = "Perspective"):
        self.name = name
        self.resolution = resolution
        self.focal = focal
        self.principal_point = principal_point
        self.position = position
        self.rotation = rotation
        self.depth_range = depth_range
        self.projection = ProjectionType[projection]

    def to_numpy(self) -> np.ndarray:
        """Convert parameters to NumPy array for GPU upload."""
        return np.array([*self.focal, *self.principal_point, *self.position, *self.rotation], dtype="float32").reshape((2 + 2 + 3 + 3, 1))
    
    def update_from_numpy(self, data : np.ndarray):
        if len(data.shape) > 2 or len(data.shape) <= 0 or data.shape[0] < 10:
            raise ValueError("Data should be a flat 2D array (10, 1)")
        
        self.focal              = (float(data[0].item()), float(data[1].item()))
        self.principal_point    = (float(data[2].item()),float(data[3].item()))
        self.position           = (float(data[4].item()),float(data[5].item()),float(data[6].item()))
        self.rotation           = (float(data[7].item()),float(data[8].item()),float(data[9].item()))
    
    def to_dict(
            self, 
            color_bit : int = ImageUtils.YUVFormat.yuv444p.bit, depth_bit : int = ImageUtils.YUVFormat.gray16le.bit, 
            color_space : str = ImageUtils.YUVFormat.yuv444p.code, depth_space : str = ImageUtils.YUVFormat.gray16le.code, 
        ) -> CameraParamDict:
        """Returns all the camera parameters into the MPEG JSON camera object format.
        """
        return CameraParamDict({
            "Name"              : self.name,
            "Position"          : self.position,
            "Rotation"          : self.rotation,
            "BitDepthColor"     : color_bit,
            "BitDepthDepth"     : depth_bit,
            "ColorSpace"        : color_space,
            "DepthColorSpace"   : depth_space,
            "Depth_range"       : self.depth_range,
            "Focal"             : self.focal,
            "Principle_point"   : self.principal_point,
            "Projection"        : self.projection.name,
            "Resolution"        : self.resolution,
        }
        )
    
    def __copy__(self,):
        params = CPUCameraParameters(self.name, self.resolution, self.focal, self.principal_point, self.position, self.rotation, self.depth_range, self.projection.name)
        return params
    
    def __deepcopy__(self):
        params = CPUCameraParameters(deepcopy(self.name), deepcopy(self.resolution), deepcopy(self.focal), deepcopy(self.principal_point), deepcopy(self.position), deepcopy(self.rotation), deepcopy(self.depth_range), deepcopy(self.projection.name))
        return params
    
    def downscale(self, scale : float):
        assert scale > 1.0
        old_res = self.resolution
        self.focal = (self.focal[0] / scale, self.focal[1] / scale)
        self.resolution = (int(self.resolution[0] / scale), int(self.resolution[1] / scale))
        self.principal_point = (self.principal_point[0] / old_res[0] * self.resolution[0], self.principal_point[1] / old_res[1] * self.resolution[1])
        
    def upscale(self, scale : float):
        assert scale > 1.0
        old_res = self.resolution
        self.focal = (self.focal[0] * scale, self.focal[1] * scale)
        self.resolution = (int(self.resolution[0] * scale), int(self.resolution[1] * scale))
        self.principal_point = (self.principal_point[0] / old_res[0] * self.resolution[0], self.principal_point[1] / old_res[1] * self.resolution[1])

class GPUCameraParameters:
    """Manages GPU buffers for camera parameters and gradients."""

    _MAX_GRADIENT_VALUE = 65536.0
    """Constant value for scaling gradient and avoid gradient vanishing/exploding problems.
    """
    
    def __init__(self, device: RenderingDevice, params: CPUCameraParameters, requires_grad: bool = True, nb_chunks: tuple[int, int] = (256, 256)):
        self.device = device
        self.requires_grad = requires_grad
        self.nb_chunks = nb_chunks

        self.cam_param_gpu          : spy.Buffer = device.create_buffer(params.to_numpy(), label=f"input_params_{params.name}")
        self._scaling_cam_buffer    : spy.Buffer | None = None
        self._grad_cam_buffer       : spy.Buffer | None = None

        self.rt_buffer              : spy.Buffer = device.create_buffer(np.zeros((12, 1), dtype="float32"), label=f"rt_matrix_{params.name}")
        self._grad_rt_buffer        : spy.Buffer | None = None
        self._scaling_rt_buffer     : spy.Buffer | None = None

        if(not requires_grad): return
       
        self._scaling_cam_buffer = device.create_buffer(GPUCameraParameters._MAX_GRADIENT_VALUE / GPUCameraParameters.cpu_range(params),label=f"input_scaling_params_{params.name}")
        self._grad_cam_buffer    = device.create_chunked_buffer(np.zeros_like(params.to_numpy(), dtype="int32"), nb_chunks = self.nb_chunks, label=f"input_grad_params_{params.name}")
       
        self._grad_rt_buffer     = device.create_chunked_buffer(np.zeros((9 + 3, 1), dtype="int32"), label = f"grad_rt_matrix_{params.name}", nb_chunks=self.nb_chunks)
        self._scaling_rt_buffer  = device.create_buffer((np.ones((9 + 3, 1), dtype="float32") * 65536.0 / (params.resolution[0] * params.resolution[1])).astype("float32"), label = f"scaling_rt_matrix_{params.name}")

    @staticmethod
    def cpu_range(params: CPUCameraParameters) -> np.ndarray:
        """Range of each parameter stored in a numpy array.
        This function can be used to scale gradient.

        Args:
            params (CPUCameraParameters): The CPU parameters.

        Returns:
            np.ndarray: Range of each parameter stored in a numpy array.
        """
        return np.array(
            [
               params.resolution[0], params.resolution[0], 
               *params.resolution, 
               *[GPUCameraParameters._MAX_GRADIENT_VALUE for _ in range(6)]
            ], dtype="float32"
        ).reshape((2 + 2 + 3 + 3, 1))
    
    def to_numpy(self) -> np.ndarray:
        gpu_data = self.cam_param_gpu.to_numpy()
        return gpu_data
    
    def update_buffer(self, params: CPUCameraParameters):
        """Copy the CPU parameters to the GPU buffer.

        Args:
            params (CPUCameraParameters): CPU parameters.
        """
        rd = RenderingDevice()
        c_e = rd.create_command_encoder()
        data = params.to_numpy()
        c_e.upload_buffer_data(self.cam_param_gpu, 0, data)
        rd.submit_command(c_e)

    @property
    def grad_scaling_buffer_for_cam_params(self) -> spy.Buffer:
        """Getter for `grad_scaling_buffer_for_cam_params`.
        `grad_scaling_buffer_for_cam_params` is a GPU buffer storing the scaling values for the gradient.

        Raises:
            RuntimeError: If the GPU buffer was not created.

        Returns:
            spy.Buffer: GPU buffer
        """
        if(self._scaling_cam_buffer is None): raise RuntimeError('Required GPU buffer does not exist.')
        return self._scaling_cam_buffer
    
    @grad_scaling_buffer_for_cam_params.setter
    def grad_scaling_buffer_for_cam_params(self, value : spy.Buffer | None):
        self._scaling_cam_buffer = value
    
    @property
    def grad_buffer_for_cam_params(self) -> spy.Buffer:
        """Getter for `grad_buffer_for_cam_params`.
        `grad_buffer_for_cam_params` is a GPU buffer storing the gradient of the camera parameters.

        Raises:
            RuntimeError: If the GPU buffer was not created.

        Returns:
            spy.Buffer: GPU buffer
        """
        if(self._grad_cam_buffer is None): raise RuntimeError('Required GPU buffer does not exist.')
        return self._grad_cam_buffer
    
    @grad_buffer_for_cam_params.setter
    def grad_buffer_for_cam_params(self, value : spy.Buffer | None):
        self._grad_cam_buffer = value

    @property
    def grad_buffer_for_Rt_matrix(self) -> spy.Buffer:
        """Getter for `grad_buffer_for_Rt_matrix`.
        `grad_buffer_for_Rt_matrix` is a GPU buffer storing the gradient of the [Rt] matrix (for world transformation).

        Raises:
            RuntimeError: If the GPU buffer was not created.

        Returns:
            spy.Buffer: GPU buffer
        """
        if(self._grad_rt_buffer is None): raise RuntimeError('Required GPU buffer does not exist.')
        return self._grad_rt_buffer
    
    @grad_buffer_for_Rt_matrix.setter
    def grad_buffer_for_Rt_matrix(self, value : spy.Buffer | None):
        self._grad_rt_buffer = value
    
    @property
    def grad_scaling_buffer_for_Rt_matrix(self) -> spy.Buffer:
        """Getter for `grad_scaling_buffer_for_Rt_matrix`.
        `grad_scaling_buffer_for_Rt_matrix` is a GPU buffer storing the scaling values for the gradient of the [Rt] matrix (for world transformation).

        Raises:
            RuntimeError: If the GPU buffer was not created.

        Returns:
            spy.Buffer: GPU buffer
        """
        if(self._scaling_rt_buffer is None): raise RuntimeError('Required GPU buffer does not exist.')
        return self._scaling_rt_buffer
    
    @grad_scaling_buffer_for_Rt_matrix.setter
    def grad_scaling_buffer_for_Rt_matrix(self, value : spy.Buffer | None):
        self._scaling_rt_buffer = value

class Camera:
    """A RGBD camera with CPU/GPU camera parameters and color&depth CPU/GPU data.
    """
    T_Pos = tuple[float, float, float]

    def __init__(self, name : str, cam_params : CameraParamDict, requires_grad : bool = True, nb_chunks : tuple[int, int] | int = 256, always_reload : bool = False):
        rd = RenderingDevice()
        
        self.name = name
        self.nb_chunks = (nb_chunks, nb_chunks) if (isinstance(nb_chunks, int)) else nb_chunks
        self.always_reload = always_reload
        self.requires_grad = requires_grad

        # --------------------------- Python Objects --------------------------- 
        self.cpu_params = CPUCameraParameters(
            cam_params.get('Name', name),
            cam_params.get('Resolution', (0, 0)),
            cam_params.get('Focal', (0, 0)),
            cam_params.get('Principle_point', (0, 0)),
            cam_params.get("Position", [0 for _ in range(3)]),
            cam_params.get("Rotation", [0 for _ in range(9)]),
            cam_params.get("Depth_range", [0 for _ in range(2)]),
            projection = cam_params.get("Projection", "None")
        )

        self.image_pair : ImagePair  # List of ImagePair objects
        self._cam_center : Camera.T_Pos | None = None

        # --------------------------- Device Objects ---------------------------
        self.gpu_params = GPUCameraParameters(rd, self.cpu_params, self.requires_grad, self.nb_chunks)

        # ----- Utils ------
        for info in self.info : Logger.debug(info, str(self.__class__))
    
    @property
    def info(self) -> list[str]:
        """Returns a list of information (if gradient, camera parameters, etc.) about the camera.
        Use for debugging purposes.

        Returns:
            list[str]: list of information
        """
        info = []
        info.append("-" * 5 + f"Camera: {self.name}" + "-" * 5)
        info.append(f"Gradient: {self.requires_grad}")
        info.append("Parameters:")
        info.append(str(self.cam_params))

        return info

    @property
    def cam_params(self) -> CameraParamDict:
        """Returns the camera parameter as a dict.

        Returns:
            CameraParamDict: camera parameter as a dict.
        """
        return self.cpu_params.to_dict()
    
    def cam_center(self, sigmoid : bool = False) -> T_Pos:
        # Quite ugly but will work for now
        if self._cam_center is None:
            z = self.cpu_params.depth_range[0] + (self.cpu_params.depth_range[1] - self.cpu_params.depth_range[0]) / 2.0
        
            if hasattr(self, 'image_pair'):
                null_value = 0
                if sigmoid:
                    null_value = MathUtils.sigmoid(null_value, self.cpu_params.depth_range[0], (self.cpu_params.depth_range[1] - self.cpu_params.depth_range[0]))
                    
                depth = self.image_pair.depth.arr
                if (depth != null_value).any():
                    z = np.median(depth).item()
                    if sigmoid:
                        z = MathUtils.sigmoid(z, self.cpu_params.depth_range[0], (self.cpu_params.depth_range[1] - self.cpu_params.depth_range[0]))
                        
            self._cam_center = (
                z, 
                self.cpu_params.resolution[0]/2 - self.cpu_params.principal_point[0], 
                self.cpu_params.resolution[1]/2 - self.cpu_params.principal_point[1],
            ) # z, x, y
        
        return self._cam_center
        
    def __copy__(self,  requires_grad : bool = True, nb_chunks : tuple[int, int] | int = 256, always_reload : bool = False):
        cam_params = self.cam_params.copy()
        cam_ = Camera(self.name, cam_params, requires_grad=requires_grad, nb_chunks=nb_chunks, always_reload=always_reload)
        cam_.image_pair = self.image_pair.__copy__(cam_.requires_grad, cam_.name, cam_.always_reload)
        return cam_

    def __deepcopy__(self, memo,  requires_grad : bool = True, nb_chunks : tuple[int, int] | int = 256, always_reload : bool = False):
        cam_params = deepcopy(self.cam_params)
        cam_ = Camera(deepcopy(self.name), cam_params, requires_grad=requires_grad, nb_chunks=nb_chunks, always_reload=always_reload)
        cam_.image_pair = self.image_pair.__copy__(cam_.requires_grad, cam_.name, cam_.always_reload)
        return cam_

    def load_image_pair(self, color_path : str, depth_path : str, yuv_format : tuple[ImageUtils.YUVFormat, ImageUtils.YUVFormat] = (ImageUtils.YUVFormat.yuv420p, ImageUtils.YUVFormat.gray16le)):
        """Add new image pair to camera"""
        new_pair = ImagePair.from_files(color_path, depth_path, self.cpu_params.resolution, self.cpu_params.depth_range, self.requires_grad, self.name, self.always_reload, yuv_format = yuv_format)
        self.image_pair = new_pair
    
    def load_numpy_pair(self, color_arr : np.ndarray, depth_arr : np.ndarray, dtype : str = "float32"):
        new_pair = ImagePair(color_arr, depth_arr, dtype, self.requires_grad, self.name, self.always_reload)
        self.image_pair = new_pair
    
    def load_texture_pair(self, color_texture : spy.Texture, depth_texture : spy.Texture):
        """Add new image pair to camera"""
        new_pair = ImagePair.from_textures(color_texture, depth_texture, self.requires_grad, self.name, self.always_reload)
        self.image_pair = new_pair
    
    def load_empty_pair(self, dtype = "float32"):
        """Add new image pair to camera"""
        new_pair = ImagePair.empty(self.cpu_params.resolution, self.requires_grad, self.name, self.always_reload, dtype)
        self.image_pair = new_pair

    def save_pair(self, out_color_path : str, out_depth_path : str):
        """Save the color and depth texture to the given paths.

        Args:
            out_color_path (str): Path to store the color data.
            out_depth_path (str): Path to store the depth data.
        """
        self.image_pair.save_color(out_color_path)
        self.image_pair.save_depth(out_depth_path, self.cpu_params.depth_range)

    def params_to_array(self) -> np.ndarray:
        """Returns the camera parameters into a numpy array.

        Returns:
            np.ndarray: camera parameters stored into a numpy array.
        """
        return self.cpu_params.to_numpy()
    
    def select_parameters_for_optimization(self, names : tuple) -> np.ndarray:
        """Select the parameters from the given names for the optimization.

        Args:
            names (tuple): The names of the parameters selected for optimization. ["focal", "principal_point", "pos", "rot"]

        Returns:
            np.ndarray: A numpy array where 1 means the parameter was selected.
        """
        selected = np.zeros((2 + 2 + 3 + 3, 1), dtype="int32")

        for name in names:
            if not isinstance(name, str): continue

            if (name == "focal"):
                selected[0:2,:] = 1
            elif (name == "principal_point"):
                selected[2:4,:] = 1
            elif (name == "pos"):
                selected[4:7,:] = 1
            elif (name == "rot"):
                selected[7:10,:] = 1
            else:
                Logger.warning(f"Camera: Unable to find parameter: {name}. Parameter will not be selected for optimization or learning...")

        return selected
    
    def update_gpu(self):
        """Update the GPU buffer(s) with the cpu data.
        """
        self.gpu_params.update_buffer(self.cpu_params)
    
    def update_cpu(self):
        """Update the CPU data with the GPU data.
        """
        self.cpu_params.update_from_numpy(self.gpu_params.to_numpy())
        self.image_pair.color.update_cpu()
        self.image_pair.depth.update_cpu()
        self._cam_center  = None
    
    def downscale(self, scale : float):
        self.cpu_params.downscale(scale)
        self.image_pair.resize(self.cpu_params.resolution)
        self.update_gpu()
        self._cam_center  = None
        
    def upscale(self, scale : float):
        self.cpu_params.upscale(scale)
        self.image_pair.resize(self.cpu_params.resolution)
        self.update_gpu()
        self._cam_center  = None
    
class RenderedViewpoint:
    """Virtual Camera handler for rendered viewpoints. 
    """

    def __init__(self, camera : Camera, nb_samples : int, always_reload : bool = False, requires_grad : bool = False):
        self.camera = camera

        self.nb_samples = nb_samples
        self.always_reload = always_reload
        self.requires_grad = requires_grad

        self.rendered_image_pair    = RenderedMultiSampleImagePairComposite(self.camera.cpu_params.resolution, "float32", nb_samples, name = self.camera.name, always_reload=always_reload, requires_grad  = requires_grad)
        
        arr_zeros = np.zeros((self.camera.cpu_params.resolution[1], self.camera.cpu_params.resolution[0], 1), dtype="float32")
        self.rendered_quality       = RenderedMultiSampleImageComposite(arr_zeros, nb_samples, f"quality_{self.camera.name}", always_reload=always_reload, requires_grad=requires_grad)
        self.rendered_mask          = RenderedMultiSampleImageComposite(arr_zeros, nb_samples, f"mask_{self.camera.name}", always_reload=always_reload, requires_grad=requires_grad)
        self.rendered_normal        = RenderedMultiSampleImageComposite(arr_zeros, nb_samples, f"normal_{self.camera.name}", always_reload=always_reload, requires_grad=requires_grad)

class BlendedViewpoint:
    """Virtual Camera handler for blended viewpoints. 
    """

    def __init__(self, camera : Camera, requires_grad : bool = False,always_reload : bool = False):
        self.camera = camera

        self.requires_grads= requires_grad
        self.always_reload = always_reload

        self.blended_image_pair     = RWImagePair(self.camera.cpu_params.resolution, 'float32', f"blended_{self.camera.name}", requires_grad, self.always_reload)
        zero_arr_ch_1               = np.zeros((self.camera.cpu_params.resolution[1], self.camera.cpu_params.resolution[0], 1), dtype="float32")
        self.blended_quality        = RWImage(zero_arr_ch_1, requires_grad, f"blended_quality_{self.camera.name}", self.always_reload)
        self.blended_mask           = RWImage(zero_arr_ch_1, requires_grad, f"blended_mask_{self.camera.name}", self.always_reload)
        self.blended_normal         = RWImage(zero_arr_ch_1, requires_grad, f"blended_normal_{self.camera.name}", self.always_reload)
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

from copy import deepcopy

import cv2
import numpy as np
import torch
import slangpy as spy

from ._utils import ImageUtils, StrUtils, ArrayUtils, Logger
from .RenderingDevice import RenderingDevice

class Image:
    """Image base class. It stored the CPU data in a numpy array, and if needed the GPU data in a slangpy texture.
    If `requires_grad` is True, then it also creates a slangpy buffer to store its gradient.

    The range is used to scale the gradient and avoids vanishing/exploding gradients during differentiation. 
    """

    _GPU_TEXTURE_USAGE  = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
    _GPU_TEXTURE_LABEL  = "{name}"
    _GPU_GRADIENT_LABEL = "grad_{name}"

    def __init__(self, arr : np.ndarray, requires_grad : bool, name : str, always_reload : bool, type_range : float = 1.0):
        self.arr     : np.ndarray = arr
        
        self.requires_grad = requires_grad
        self.name = name
        self.always_reload = always_reload
        self.type_range = type_range

        self._texture       : spy.Texture | None        = None
        self._texture_v     : spy.TextureView | None    = None
        self._grad          : spy.Buffer | None         = None
        if(always_reload): self.reload_gpu_texture()

        Logger.debug(f"Create Image: {self.name}", str(self.__class__))
    
    @property
    def grad(self) -> spy.Buffer:
        """Returns the gradient buffer if it exists.

        Raises:
            RuntimeError: Returns an error if not.

        Returns:
            spy.Buffer: Gradient Buffer.
        """
        if(self._grad is None): raise RuntimeError('Gradient buffer does not exist.')
        return self._grad
    
    @property
    def texture(self) -> spy.Texture:
        """Returns the data texture if it exists.

        Raises:
            RuntimeError: Returns an error if not.

        Returns:
            spy.Buffer: Data texture.
        """
        if(self._texture is None): raise RuntimeError('Texture does not exist.')
        return self._texture
    
    @property
    def texture_v(self) -> spy.TextureView:
        """Returns a view to the texture data if it exists.

        Raises:
            RuntimeError: Returns an error if not.

        Returns:
            spy.Buffer: Texture view.
        """
        if(self._texture_v is None): raise RuntimeError('Texture view does not exist.')
        return self._texture_v
    
    @texture.setter
    def texture(self, val : spy.Texture | None):
        self._texture = val
    
    @texture_v.setter
    def texture_v(self, val : spy.TextureView | None):
        self._texture_v = val
    
    @grad.setter
    def grad(self, value : spy.Buffer | None):
        self._grad = value
    
    def release_texture(self):
        """Release all GPU data.
        """
        self.texture = None
        self.texture_v = None
        self.grad = None

    def release(self):
        """Release both CPU and GPU data.
        """
        self.arr = np.zeros(())
        if(self.always_reload): self.reload_gpu_texture()
        else: self.release_texture()

    def clear(self):
        """Clear the CPU data. If `always_reload` is True, it also copies the new CPU data to the GPU storage. 
        """
        self.arr = np.zeros_like(self.arr, dtype = self.arr.dtype)
        if(self.always_reload): self.reload_gpu_texture()

    def rand(self, rand_scale : float = 1.0, range_val : tuple[float, float] = (0.0, 1.0)):
        """Adds uniform random noise to the CPU data. If `always_reload` is True, it also copies the new CPU data to the GPU storage.

        Args:
            rand_scale (float, optional): cale parameter controlling how much noise it adds (i.e., $\alpha$). Defaults to 1.0.
            range_val (tuple[float, float], optional): Scale the noise in this range. Defaults to (0.0, 1.0).
        """
        rand = np.random.rand(*(self.arr.shape)).astype(self.arr.dtype) * (range_val[1] - range_val[0]) + range_val[0]
        self.arr = (1.0 - rand_scale) * self.arr + rand_scale * rand
        if(self.always_reload): self.reload_gpu_texture()

    def update_cpu(self):
        old_shape = self.arr.shape
        self.arr = self.texture.to_numpy()
        if self.arr.shape != old_shape:
            self.arr = self.arr.reshape(old_shape)

    def reload_gpu_texture(self, type_range : float | None = None):
        """Reloads all CPU data into the GPU storage.

        .. warning:: It actually (re)creatres all the GPU data storage (i.e., texture, buffer, etc.) and copies the CPU data to it.

        Args:
            type_range (float | None, optional): Set the range of value. Defaults to None.
        """
        self.texture = RenderingDevice().create_shader_texture(self.arr, label = self._GPU_TEXTURE_LABEL.format(name=self.name), usage=self._GPU_TEXTURE_USAGE)
        self.grad = None if not self.requires_grad else RenderingDevice().create_gradient_buffer_from_image(self.arr, label = self._GPU_GRADIENT_LABEL.format(name=self.name))
        self.texture_v = self.texture.create_view()
        self.type_range = type_range if type_range is not None else self.type_range

    def to_torch(self, device : str = "cpu") -> torch.Tensor:
        """Returns the image data in a Torch tensor. It already swaps the dimensions to fit image pytorch requirements. 

        Args:
            device (str, optional): Device on which the data will be uploaded to. Defaults to "cpu".

        Returns:
            torch.Tensor: Torch data tensor (CxHxW).
        """
        return torch.from_numpy(np.transpose(self.arr, (2, 0, 1))).to(device=device)

    def resize(self, size : tuple[int, int], dtype="float32"):
        self.arr = ImageUtils.resize(self.arr, size, dtype)
        self.reload_gpu_texture()
    
class RWImage(Image):
    """Read-Write Image class, inherits the Image base class.
    Main differences with the Image base class are the GPU usage flags.
    """
    _GPU_TEXTURE_USAGE  = spy.TextureUsage.unordered_access | spy.TextureUsage.copy_source | spy.TextureUsage.shader_resource
    _GPU_TEXTURE_LABEL  = "rw_{name}"
    _GPU_GRADIENT_LABEL = "grad_rw_{name}"

    def __init__(self, arr : np.ndarray, requires_grad : bool, name : str, always_reload : bool, type_range : float = 1.0):
        super().__init__(arr, requires_grad, name, always_reload, type_range=type_range)

class RenderedImage(Image):
    """Rendered Image class, inherits the Image base class.
    Main differences with the Image base class are the GPU usage flags.
    """
    _GPU_TEXTURE_USAGE  = spy.TextureUsage.render_target | spy.TextureUsage.resolve_destination | spy.TextureUsage.copy_source | spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
    _GPU_TEXTURE_LABEL  = "render_{name}"
    _GPU_GRADIENT_LABEL = "grad_render_{name}"
    
    def __init__(self, arr : np.ndarray, name : str, always_reload : bool, requires_grad : bool = False, type_range : float = 1.0):
        super().__init__(arr, requires_grad, name, always_reload, type_range=type_range)

    def __copy__(self):
        copy_ = RenderedImage(self.arr, f"copy_{self.name}", self.always_reload)
        copy_.reload_gpu_texture()
        return copy_

    def __deepcopy__(self, memo):
        copy_ = RenderedImage(deepcopy(self.arr, memo), f"copy_{self.name}", deepcopy(self.always_reload, memo))
        copy_.reload_gpu_texture()
        return copy_

class MultiSampledRenderedImage(Image):
    """MultiSampleRendered Image class, inherits the Image base class.
    Main differences with the Image base class are the GPU usage flags.
    """
    _GPU_TEXTURE_USAGE  = spy.TextureUsage.render_target | spy.TextureUsage.shader_resource | spy.TextureUsage.resolve_source | spy.TextureUsage.copy_source | spy.TextureUsage.copy_destination
    _GPU_TEXTURE_LABEL  = "ms_render_{name}"

    def __init__(self, arr : np.ndarray, nb_samples : int, name : str, always_reload : bool):
        self.nb_samples = nb_samples
        super().__init__(arr, False, name, always_reload)
    
    def reload_gpu_texture(self, type_range : float | None = None):
        self.texture = RenderingDevice().create_multisample_render_texture(
            width   = self.arr.shape[1],
            height  = self.arr.shape[0],
            format  = ImageUtils.get_image_slangpy_format(self.arr),
            label   = self._GPU_TEXTURE_LABEL.format(name=self.name),
            sample_count = self.nb_samples,
            usage   = self._GPU_TEXTURE_USAGE
        )
        self.texture_v = self.texture.create_view()
        self.type_range = type_range if type_range is not None else self.type_range
    
    def __copy__(self):
        copy_ = MultiSampledRenderedImage(self.arr, self.nb_samples, f"copy_{self.name}", self.always_reload)
        copy_.reload_gpu_texture()
        return copy_

    def __deepcopy__(self, memo):
        copy_ = MultiSampledRenderedImage(deepcopy(self.arr, memo), deepcopy(self.nb_samples, memo), f"copy_{self.name}", deepcopy(self.always_reload, memo))
        copy_.reload_gpu_texture()
        return copy_


class RenderedMultiSampleImageComposite:
    """Creates a pair of images: the rendered multi-sampled image (MultiSampledRenderedImage) and the destination rendered image pair (RenderedImagePair).
    """
    
    def __init__(self, arr : np.ndarray, nb_samples : int, name : str, always_reload : bool, requires_grad : bool = False, type_range : float = 1.0):
        self.always_reload = always_reload
        self.nb_samples = nb_samples
        self.requires_grad = requires_grad

        self.dst_img : RenderedImage = RenderedImage(arr, name, always_reload=always_reload, requires_grad=requires_grad, type_range=type_range)
        self.ms_img  : MultiSampledRenderedImage = MultiSampledRenderedImage(arr, nb_samples, name, always_reload=always_reload)

class ImagePair:
    """ImagePair base class usef to store a pair of color and depth images (i.e., RGBD cameras). 
    """
        
    COLOR_CHANNELS = 4
    DEPTH_CHANNELS = 1

    def __init__(self, color_arr : np.ndarray, depth_arr : np.ndarray, dtype:str, requires_grad : bool, name : str, always_reload : bool):
        self.always_reload = always_reload
        self.requires_grad = requires_grad
        self.dtype = dtype
        
        color = ArrayUtils.convert_to_dtype(color_arr,dtype) 
        depth = ArrayUtils.convert_to_dtype(depth_arr,dtype)
        
        color = ImageUtils.set_array_to_image_format(color, ImagePair.COLOR_CHANNELS)
        depth = ImageUtils.set_array_to_image_format(depth, ImagePair.DEPTH_CHANNELS)

        self.color : Image = Image(color, requires_grad=requires_grad, name=f"{name}_color", always_reload=always_reload)
        self.depth : Image = Image(depth, requires_grad=requires_grad, name=f"{name}_depth", always_reload=always_reload)

        Logger.debug(f"Create ImagePair.", str(self.__class__))

    def __copy__(self, requires_grad : bool, name : str, always_reload : bool):
        copy_ = ImagePair(self.color.arr.copy(), self.depth.arr.copy(), self.dtype, requires_grad, name, always_reload)
        return copy_

    def __deepcopy__(self, memo, requires_grad : bool, name : str, always_reload : bool):
        copy_ = ImagePair(deepcopy(self.color.arr), deepcopy(self.depth.arr), deepcopy(self.dtype), requires_grad, name, always_reload)
        return copy_

    @classmethod
    def empty(
        cls, 
        resolution : tuple[int, int], 
        requires_grad : bool, 
        name : str, 
        always_reload : bool, 
        dtype : str, 
        ):
        """Creates an empty image pair, i.e., CPU/GPU data are filled with zeros.

        Args:
            resolution (tuple[int, int]): Resolution of the images.
            requires_grad (bool): If true, gradient buffer will be created.
            name (str): Name of the image pair.
            always_reload (bool): If true, every time the CPU data is modified (through the Image class functions), the CPU data is reloaded into the GPU.
            dtype (str): numpy data type.

        Returns:
            ImagePair: Returns the created ImagePair instance.
        """
        c = np.zeros((resolution[1], resolution[0], ImagePair.COLOR_CHANNELS), dtype=dtype)
        d = np.zeros((resolution[1], resolution[0], ImagePair.DEPTH_CHANNELS), dtype=dtype)
        return cls(c,d, dtype, requires_grad=requires_grad, name=name, always_reload = always_reload)

    @classmethod
    def from_files(
        cls,
        color_path : str, 
        depth_path : str, 
        resolution : tuple[int, int], 
        depth_range : tuple[float, float],
        requires_grad : bool, 
        name : str, 
        always_reload : bool, 
        yuv_format : tuple[ImageUtils.YUVFormat, ImageUtils.YUVFormat] = (ImageUtils.YUVFormat.yuv420p, ImageUtils.YUVFormat.gray16le), 
        dtype = "float32", 
        ):
        """Loads an image pair from two files. The image files can be any image file format (i.e, .yuv, .png, .exr, .jpeg, etc.) supported by the ImageUtils class.
        The correct resolution and YUV formats must be given if YUV files must be loaded.
        It also converts the depth values to the required RVS depth values. Check documentation of `ImageUtils.convert_depth_2_rvs_depth` for more details.

        Args:
            color_path (str): Path to the color file.
            depth_path (str): Path to the depth file.
            resolution (tuple[int, int]): Image resolution if YUV.
            depth_range (tuple[float, float]): Depth range needed to rescale depth values.
            requires_grad (bool): If true, gradient buffer will be created.
            name (str): Name of the image pair.
            always_reload (bool): If true, every time the CPU data is modified (through the Image class functions), the CPU data is reloaded into the GPU.
            yuv_format (tuple[ImageUtils.YUVFormat, ImageUtils.YUVFormat], optional): YUV formats if YUV. Defaults to (ImageUtils.YUVFormat.yuv420p, ImageUtils.YUVFormat.gray16le).
            dtype (str, optional): numpy data type. Defaults to "float32".

        Returns:
            ImagePair: Returns the created ImagePair instance.
        """

        c, d = ImagePair.load_image_pair_from_files(color_path, depth_path, resolution, dtype=dtype, yuv_format=yuv_format)
        d = ImageUtils.convert_normalized_mpeg_disparity_to_depth(d, depth_range[0], depth_range[1])
        
        return cls(c,d, dtype, requires_grad=requires_grad, name=name, always_reload = always_reload) 

    @classmethod
    def from_textures(
        cls, 
        color_texture : spy.Texture, 
        depth_texture : spy.Texture, 
        requires_grad : bool, 
        name : str, 
        always_reload : bool, 
        dtype = "float32",
        ):
        """Load the ImagePair from a pair of slangpy/GPU textures.

        Args:
            color_texture (spy.Texture): Salngpy color texture storage.
            depth_texture (spy.Texture): Salngpy depth texture storage.
            requires_grad (bool): If true, gradient buffer will be created.
            name (str): Name of the image pair.
            always_reload (bool): If true, every time the CPU data is modified (through the Image class functions), the CPU data is reloaded into the GPU.
            dtype (str): numpy data type. Defaults to "float32".

        Returns:
            ImagePair: Returns the created ImagePair instance.
        """
        c = color_texture.to_numpy().astype(dtype=dtype)
        d = depth_texture.to_numpy().astype(dtype=dtype)
        return cls(c,d, dtype, requires_grad=requires_grad, name=name, always_reload = always_reload)         
        
    @staticmethod
    def load_image_pair_from_files(color_path : str, depth_path : str, resolution : tuple[int, int], yuv_format : tuple[ImageUtils.YUVFormat, ImageUtils.YUVFormat] = (ImageUtils.YUVFormat.NONE, ImageUtils.YUVFormat.NONE), dtype = "float32"):
        """Load color image and depth map from disk.
        The image files can be any image file format (i.e, .yuv, .png, .exr, .jpeg, etc.) supported by the ImageUtils class.

        Args:
            color_path (str): Path to the color file.
            depth_path (str): Path to the depth file.
            resolution (tuple[int, int]): Image resolution if YUV.
            yuv_format (tuple[ImageUtils.YUVFormat, ImageUtils.YUVFormat], optional): YUV formats if YUV. Defaults to (ImageUtils.YUVFormat.NONE, ImageUtils.YUVFormat.NONE).
            dtype (str, optional): numpy data type. Defaults to "float32".

        Raises:
            FileNotFoundError: If the color numpy array is none.
            ValueError: If the given resolution does not equal the resolution of the color numpy array.
            FileNotFoundError: If the depth numpy array is none.
            ValueError: If the given resolution does not equal the resolution of the depth numpy array.

        Returns:
            tuple[np.ndarray, np.ndarray]: Color and Depth CPU numpy array data. 
        """
        color = ImageUtils.read_img(color_path, resolution, yuv_format=yuv_format[0], dtype=dtype, convert_2_rgb=True)
        depth = ImageUtils.read_img(depth_path, resolution, yuv_format=yuv_format[1], dtype=dtype, convert_2_rgb=False)

        return color, depth
    
    def resize(self, size : tuple[int, int]):
        self.color.resize(size, self.dtype)
        self.depth.resize(size, self.dtype)
    
    def release(self):
        """Release color and depth memory"""
        self.color.release()
        self.depth.release()

    def save_color(self, path : str, out_dtype : str = "uint8" , out_yuv_format : ImageUtils.YUVFormat = ImageUtils.YUVFormat.yuv444p):
        """Save color image only.
        For more details about image saving, check documentation of `ImageUtils.save_img`.

        Args:
            path (str): Output path where the image will be saved.
            out_dtype (str, optional): Output data type, if not YUV. Defaults to "uint8".
            out_yuv_format (ImageUtils.YUVFormat, optional): Output YUV format if YUV. Defaults to ImageUtils.YUVFormat.yuv444p.
        """
        ImageUtils.save_img(self.color.arr, path, self.COLOR_CHANNELS, out_dtype=out_dtype, out_yuv_format=out_yuv_format)
    
    def save_depth(self, path : str, depth_range : tuple[float, float], out_dtype : str = "uint16" , out_yuv_format : ImageUtils.YUVFormat = ImageUtils.YUVFormat.gray16le):
        """Save Depth image only.
        For more details about image saving, check documentation of `ImageUtils.save_img`.

        .. warning:: It converts the RVS depth values into normalized depth values. For more details, check documentation of `ImageUtils.convert_depth_to_normalized_mpeg_disparity`.

        Args:
            path (str): Output path where the image will be saved.
            out_dtype (str, optional): Output data type, if not YUV. Defaults to "uint16".
            out_yuv_format (ImageUtils.YUVFormat, optional): Output YUV format if YUV. Defaults to ImageUtils.YUVFormat.gray16le.
        """

        depth = ImageUtils.convert_depth_to_normalized_mpeg_disparity(self.depth.arr, depth_range[0], depth_range[1])
        ImageUtils.save_img(depth, path, self.DEPTH_CHANNELS, out_dtype=out_dtype, out_yuv_format=out_yuv_format)

class RenderedImagePair(ImagePair):
    """Creates a RGBD pair of RenderedImage.
    """
    def __init__(self, resolution : tuple[int, int], dtype:str, name : str, always_reload: bool, requires_grad : bool = False):
        self.always_reload = always_reload
        
        color = np.zeros((resolution[1], resolution[0], ImagePair.COLOR_CHANNELS), dtype=dtype)
        depth = np.zeros((resolution[1], resolution[0], ImagePair.DEPTH_CHANNELS), dtype=dtype)

        self.color : Image = RenderedImage(color, name=f"{name}_color", always_reload=always_reload, requires_grad = requires_grad)
        self.depth : Image = RenderedImage(depth, name=f"{name}_depth", always_reload=always_reload, requires_grad = requires_grad)

        Logger.debug(f"Create ImagePair.", str(self.__class__))

    @classmethod
    def from_array(cls, color : np.ndarray, depth : np.ndarray, always_reload: bool, name : str, requires_grad : bool = False): 
        color = ImageUtils.set_array_to_image_format(color, ImagePair.COLOR_CHANNELS)
        depth = ImageUtils.set_array_to_image_format(depth, ImagePair.DEPTH_CHANNELS)
        
        pair = cls((color.shape[0],color.shape[1]), str(color.dtype), name, always_reload, requires_grad)
        pair.color.arr = color
        pair.depth.arr = depth
        return pair

    def save_depth(self, path : str, depth_range : tuple[float, float], out_dtype : str = "uint16" , out_yuv_format : ImageUtils.YUVFormat = ImageUtils.YUVFormat.gray16le):
        """Save Depth image only.
        For more details about image saving, check documentation of `ImageUtils.save_img`.

        .. note:: It rescaled the depth values between $[D_{min}, D_{max}]$ before converting them to disparity values.

        .. warning:: It converts the depth values into normalized disparity values. For more details, check documentation of `ImageUtils.convert_depth_to_normalized_mpeg_disparity`.

        Args:
            path (str): Output path where the image will be saved.
            out_dtype (str, optional): Output data type, if not YUV. Defaults to "uint16".
            out_yuv_format (ImageUtils.YUVFormat, optional): Output YUV format if YUV. Defaults to ImageUtils.YUVFormat.gray16le.
        """
        depth = self.depth.arr
        depth = (depth * (depth_range[1] - depth_range[0])) + depth_range[0]
        depth = ImageUtils.convert_depth_to_normalized_mpeg_disparity(depth, depth_range[0], depth_range[1])
        ImageUtils.save_img(depth, path, self.DEPTH_CHANNELS, out_dtype=out_dtype, out_yuv_format=out_yuv_format)

class MultiSampledRenderedImagePair(ImagePair):
    """Creates a RGBD pair of MultiSampledRenderedImage.
    """
    def __init__(self, resolution : tuple[int, int], dtype:str, nb_samples : int, name : str, always_reload : bool, requires_grad : bool = False):
        self.always_reload = always_reload
        self.nb_samples = nb_samples
        self.requires_grad = requires_grad
        
        color = np.zeros((resolution[1], resolution[0], ImagePair.COLOR_CHANNELS), dtype=dtype)
        depth = np.zeros((resolution[1], resolution[0], ImagePair.DEPTH_CHANNELS), dtype=dtype)

        self.color : Image = MultiSampledRenderedImage(color, name=f"{name}_color", always_reload=always_reload, nb_samples = nb_samples)
        self.depth : Image = MultiSampledRenderedImage(depth, name=f"{name}_depth", always_reload=always_reload, nb_samples = nb_samples)

        Logger.debug(f"Create ImagePair.", str(self.__class__))

class RWImagePair(ImagePair):
    """Creates a RGBD pair of RWImage.
    """
    def __init__(self, resolution : tuple[int, int], dtype:str, name : str, requires_grad : bool, always_reload: bool):
        self.always_reload = always_reload
        
        color = np.zeros((resolution[1], resolution[0], ImagePair.COLOR_CHANNELS), dtype=dtype)
        depth = np.zeros((resolution[1], resolution[0], ImagePair.DEPTH_CHANNELS), dtype=dtype)

        self.color : Image = RWImage(color, requires_grad, name=f"{name}_color", always_reload=always_reload)
        self.depth : Image = RWImage(depth, requires_grad, name=f"{name}_depth", always_reload=always_reload)

        Logger.debug(f"Create ImagePair.", str(self.__class__))

class RenderedMultiSampleImagePairComposite:
    """Creates two pair of images: the rendered multi-sampled image pair (MultiSampledRenderedImagePair) and the destination rendered image pair (RenderedImagePair).
    """
    
    def __init__(self, resolution : tuple[int, int], dtype:str, nb_samples : int, name : str, always_reload : bool, requires_grad : bool = False):
        self.always_reload = always_reload
        self.nb_samples = nb_samples
        self.requires_grad = requires_grad

        self.dst_img_pair : RenderedImagePair = RenderedImagePair(resolution, dtype, name, always_reload=always_reload, requires_grad=requires_grad)
        self.ms_img_pair  : MultiSampledRenderedImagePair = MultiSampledRenderedImagePair(resolution, dtype, nb_samples, name, always_reload=always_reload, requires_grad=requires_grad)

    @property
    def color(self) -> Image:
        return self.dst_img_pair.color
    
    @property
    def depth(self) -> Image:
        return self.dst_img_pair.depth
    
    @property
    def ms_color(self) -> Image:
        return self.ms_img_pair.color
    
    @property
    def ms_depth(self) -> Image:
        return self.ms_img_pair.depth
    
    @color.setter
    def color(self, value : RenderedImage):
        self.dst_img_pair.color = value
    
    @depth.setter
    def depth(self, value : RenderedImage):
        self.dst_img_pair.depth = value
    
    @ms_color.setter
    def ms_color(self, value : MultiSampledRenderedImage):
        self.ms_img_pair.color = value
    
    @ms_depth.setter
    def ms_depth(self, value : MultiSampledRenderedImage):
        self.ms_img_pair.depth = value
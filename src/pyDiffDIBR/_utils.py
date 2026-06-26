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

import re
from enum import Enum
from math import sin, cos, tan, log, atan2, exp
from typing import TypedDict, _TypedDictMeta, Any, Sequence, Union, Mapping # type: ignore
from types import UnionType
import pathlib
import tomllib
import json
import random
import datetime
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import yuvio
import cv2
from skimage.metrics import structural_similarity as ssim
import slangpy as spy
import torch


class StrUtils:
    """Utility functions for string operations related to file paths and data types."""

    @staticmethod
    def get_file_extension(path : str) -> str:
        """Return the file extension.

        Args:
            path (str): the path to the file.

        Raises:
            ValueError: If there is no extension '.ext'.

        Returns:
            str: the extension
        """
        a = path.split(".")
        if(len(a) <= 1): raise ValueError('No extension found in path.')
        return a[-1]
    
    @staticmethod
    def remove_bit_depth_suffix(dtype :str) -> str:
        """Remove the bit suffix from a dtype string.

        Args:
            dtype (str): the dtype string (e.g., "float32")

        Raises:
            ValueError: if there is no bit information.
            ValueError: if there is no type information.

        Returns:
            str: the type (e.g., float)
        """
        a = re.split(r'[0-9]+', dtype)
        if(len(a) <= 1): raise ValueError('No depth found in dtype.')
        if(a[0] == ""): raise ValueError('No type found in dtype.')
        return a[0]
    
    @staticmethod
    def get_bit_depth_suffix(dtype :str) -> int:
        """Get the bit suffix from a dtype string.

        Args:
            dtype (str): the dtype string (e.g., "float32")

        Raises:
            ValueError: if there is no bit information.
            ValueError: if there is no type information.

        Returns:
            int: the bit depth (e.g., 32 for 'float32')
        """
        a = re.split(r'[a-z]+', dtype)
        if(len(a) <= 1): raise ValueError('No depth found in dtype.')
        if(a[1] == ""): raise ValueError('No type found in dtype.')
        return int(a[1])

    @staticmethod
    def build_yuvview_path(path : str) -> str:
        return path.split(".yuv")[0] + "_{}x{}_{}.yuv"


class TypeUtils:
    """Utility functions for handling data types and their properties."""

    ThreadCount_t       = tuple[int, int ,int] | spy.uint3
    UniformGPUData_t    = dict[str, Any]

    @staticmethod
    def get_type_byte(dtype : str) -> int:
        """Returns the bytes of the dtype string.

        Args:
            dtype (str): the dtype string.

        Raises:
            ValueError: if the type is unknown.

        Returns:
            int: byte number.
        """
        match dtype:
            case "float32":
                return 4
            case "int32":
                return 4
            case "uint32":
                return 4
            case "float16":
                return 2
            case "int16":
                return 2
            case "uint16":
                return 2
            case "uint8":
                return 1
            case "float64":
                return 8
            case "int64":
                return 8
        raise ValueError('Unknown dtype.')
        return -1

    @staticmethod
    def get_max_value(dtype : str) -> float:
        """
        Return the maximum representable value for a given dtype.
        
        Args:
            dtype (str): Data type string.
        Returns:
            float: Maximum value.
        Raises:
            ValueError: If type is unknown.
        """
        byte = TypeUtils.get_type_byte(dtype)
        type = StrUtils.remove_bit_depth_suffix(dtype)

        match type:
            case "int":
                return (2**(4 * byte) - 1)
            case "uint":
                return (2**(8 * byte) - 1)
            case "float":
                return 1.0
            
        raise ValueError('Unknown type.')
        return 0.0

class MathUtils:
    
    @staticmethod
    def sigmoid(value : float, shift : float = 0.0, scale : float = 1.0, lambda_ : float = 1.0) -> float:
        r"""Compute the sigmoid of a single float element.
        Shift and scale parameters can be given to apply a shift-scale operation after the sigmoid.
        I.e., $f(x) = g(x) * a + b$ with $g(x) = sigmoid(x, \lambda)$  # pyright: ignore[reportInvalidStringEscapeSequence]

        Args:
            arr (np.ndarray): numpy array
            shift (float, optional): shift value (b). Defaults to 0.0.
            scale (float, optional): scale value (a). Defaults to 1.0.
            l (float, optional): lambda value in sigmoid function. Defaults to 1.0.

        Returns:
            float: Array after applying shift-scale sigmoid element-wise.
        """
        return scale / (1.0 + exp(-lambda_ * value)) + shift

    @staticmethod
    def inv_sigmoid(value : float, shift : float = 0.0, scale : float = 1.0, lambda_ : float = 1.0) -> float: 
        tmp = (scale / (value - shift + 1e-8) - 1.0)
        tmp = 0.0 if tmp <= 0.0 else tmp
        return - log(tmp + 1e-8) / lambda_

class ArrayUtils:
    """Utility functions for array operations such as scaling, format conversion, and metrics."""
            
    @staticmethod
    def upsample(arr : np.ndarray,scale : int = 2) -> np.ndarray:
        """Upsample an array by repeating elements.

        Args:
            arr (np.ndarray): Input array.
            scale (int): Upscaling factor.

        Returns:
            np.ndarray: Upscaled array.

        Raises:
            ValueError: If scale < 1 or array is empty.
        """
        if scale < 1: raise ValueError('Scale must be >= 1 for upsampling.')
        if len(arr.shape) < 1: raise ValueError('Scale must be >= 1 for downsampling.')
        elif len(arr.shape) >= 2:
            res = arr.repeat(scale, axis=0).repeat(scale, axis=1)
            return res
        elif len(arr.shape) == 1:
            res = arr.repeat(scale, axis=0)
            return res
        raise RuntimeError('Undefined array operation.')
        return np.zeros(())

    @staticmethod
    def downscale(arr : np.ndarray,scale : int = 2, avg : bool = False) -> np.ndarray:
        """Downscale an array by skipping elements or averaging.

        Args:
            arr (np.ndarray): Input array.
            scale (int): Downscaling factor.
            avg (bool): Whether to average blocks.

        Returns:
            np.ndarray: Downscaled array.

        Raises:
            ValueError: If scale < 1 or array is empty."""
                
        if scale < 1: raise ValueError('Scale must be >= 1 for downsampling.')
        if len(arr.shape) < 1: raise ValueError('Scale must be >= 1 for upsampling.')
        elif len(arr.shape) >= 2:
            scaling_shape = [1.0 for _ in range(len(arr.shape))]
            scaling_shape[0] = 1 / scale
            scaling_shape[1] = 1 / scale
            in_sh = [arr.shape[i] for i in range(len(arr.shape))]
            res = np.zeros([int(in_sh[i] * scaling_shape[i]) for i in range(len(arr.shape))], dtype = arr.dtype)
            res = arr[::scale, ::scale, ...]
            if avg:
                for i in range(1,scale):
                    for j in range(1,scale):
                        res += arr[i::scale, j::scale, ...]
                res = (res / int(scale)).astype(arr.dtype)
                            
            return res
        elif len(arr.shape) == 1:
            res = np.zeros((int(arr.shape[0] / scale)))
            res = arr[::scale]
            return res
        raise RuntimeError('Undefined array operation.')
        return np.zeros(())


    @staticmethod
    def get_array_slangpy_format(nb_el : int, dtype : str) -> spy.Format:
        """Returns the slangpy format for a given numpy.array dtype and a given nb of channels.

        Args:
            nb_el (int): number of channels.
            dtype (str): numpy.array dtype string.

        Raises:
            ValueError: if the array format is unknown.

        Returns:
            spy.Format: Array format in slangpy.format
        """
        if(dtype == 'float32' and nb_el == 3):
            return spy.Format.rgb32_float
        elif(dtype == 'float32' and nb_el == 4):
            return spy.Format.rgba32_float
        elif(dtype == 'float32' and nb_el == 2):
            return spy.Format.rg32_float
        elif(dtype == 'float32' and nb_el == 1):
            return spy.Format.r32_float
        elif(dtype == 'int32' and nb_el == 3):
            return spy.Format.rgb32_sint
        elif(dtype == 'int32' and nb_el == 4):
            return spy.Format.rgba32_sint
        elif(dtype == 'int32' and nb_el == 2):
            return spy.Format.rg32_sint
        elif(dtype == 'int32' and nb_el == 1):
            return spy.Format.r32_sint
        elif(dtype == 'uint8' and nb_el == 4):
            return spy.Format.rgba8_uint
        elif(dtype == 'uint8' and nb_el == 2):
            return spy.Format.rg8_uint
        elif(dtype == 'uint8' and nb_el == 1):
            return spy.Format.r8_uint
        
        raise ValueError('Array format not found.')
        return spy.Format.undefined

    @staticmethod
    def convert_to_dtype(arr : np.ndarray, dtype : str) -> np.ndarray:
        """Convert a given array to a given dtype, involving a scaling operation based on the maximum values.

        Args:
            arr (np.ndarray): input np array.
            dtype (str): target np array dtype string.

        Returns:
            np.ndarray: Array in target dtype
        """
        max_val     = TypeUtils.get_max_value(arr.dtype.name)
        max_val_    = TypeUtils.get_max_value(dtype)
        
        if StrUtils.remove_bit_depth_suffix(dtype) in ["int", "uint"]:
            return np.round((arr.astype("float64") / max_val) * max_val_).astype(dtype)
        else:
            return ((arr.astype("float64") / max_val) * max_val_).astype(dtype)
    
    @staticmethod
    def mse(arr_1 : np.ndarray, arr_2 : np.ndarray) -> float:
        """Compute MSE between two arrays. These arrays are converted to 'float32' to avoid clipping.

        Args:
            arr_1 (np.ndarray): numpy array 1
            arr_2 (np.ndarray): numpy array 2

        Raises:
            ValueError: If arrays do not have same shape.

        Returns:
            float: MSE
        """
        if(arr_1.shape != arr_2.shape): raise ValueError('Arrays must have the same shape for MSE.')
        return np.mean((arr_1.astype("float32") - arr_2.astype("float32"))**2).item() # convert to float32 to avoid clipping
    
    @staticmethod
    def psnr(arr_1 : np.ndarray, arr_2 : np.ndarray) -> float:
        """Compute PSNR between two arrays. These arrays are first scaled to uint8, then converted to 'float32' to avoid clipping.

        Args:
            arr_1 (np.ndarray): numpy array 1
            arr_2 (np.ndarray): numpy array 2

        Raises:
            ValueError: If arrays do not have same shape.

        Returns:
            float: PSNR value
        """
        if(arr_1.shape != arr_2.shape): raise ValueError('Arrays must have the same shape for MSE.')
        arr_1 = ArrayUtils.convert_to_dtype(arr_1, 'uint8')
        arr_2 = ArrayUtils.convert_to_dtype(arr_2, 'uint8')
        mse = ArrayUtils.mse(arr_1, arr_2)
        
        return 10.0 * log((255.0**2) / (mse+1e-7), 10.0)
    
    @staticmethod
    def sigmoid(arr : np.ndarray, shift : float = 0.0, scale : float = 1.0, l : float = 1.0) -> np.ndarray:
        r"""Compute the sigmoid element-wise.
        Shift and scale parameters can be given to apply a shift-scale operation after the sigmoid.
        I.e., $f(x) = g(x) * a + b$ with $g(x) = sigmoid(x, \lambda)$  # pyright: ignore[reportInvalidStringEscapeSequence]

        Args:
            arr (np.ndarray): numpy array
            shift (float, optional): shift value (b). Defaults to 0.0.
            scale (float, optional): scale value (a). Defaults to 1.0.
            l (float, optional): lambda value in sigmoid function. Defaults to 1.0.

        Returns:
            np.ndarray: Array after applying shift-scale sigmoid element-wise.
        """
        return scale / (1.0 + np.exp(-l * arr)) + shift

    @staticmethod
    def inv_sigmoid(arr : np.ndarray, shift : float = 0.0, scale : float = 1.0, l : float = 1.0) -> np.ndarray: 
        tmp = (scale / (arr - shift + 1e-8) - 1.0)
        tmp[tmp <= 0] = 0.0
        return - np.log(tmp + 1e-8) / l
        
    @staticmethod
    def softmax(value : np.ndarray, temperature : float) -> np.ndarray:
        EPS = 1e-7
        if temperature < EPS:
            temperature = EPS
        
        prob = np.exp(value / temperature)
        sum_prob = np.sum(prob).item()
        if sum_prob < EPS:
            prob = np.ones_like(value, dtype="float32") / len(value)
        else:
            prob = prob / sum_prob
        return prob 

class DepthEstimatorUtils:
    
    _DEPTH_SCALE_METERS_TO_MM   = 1e3
    _DEPTH_SCALE_MM_TO_METERS   = 1.0 / _DEPTH_SCALE_METERS_TO_MM
    _DEPTH_SCALE_METERS_TO_SAVE = 1e2
    _DEPTH_SCALE_SAVE_TO_MM     = _DEPTH_SCALE_METERS_TO_MM / _DEPTH_SCALE_METERS_TO_SAVE
    
    _PROGRAM_GENERAL_IMPORT = """
# -------------- GENERAL IMPORT -------------- 

import glob, os, torch
import numpy as np
from PIL import Image
import json
from math import sin, cos
from typing import TypedDict, Union    

    """
    
    _PROGRAM_GENERAL_DATA_LOADING = """
# -------------- DATA LOADING -------------- 

def get_cam_name_from_fname(img_fname : str) -> str:
    return ".".join(os.path.basename(img_fname).split(".")[:-1])

def rotation_matrix_from_euler_angles(yaw : float, pitch : float, roll : float) -> np.ndarray:
    return np.matmul(
        rotation_matrix_from_rotation_around_z(yaw),
        np.matmul(
            rotation_matrix_from_rotation_around_y(pitch), 
            rotation_matrix_from_rotation_around_x(roll)
            )
    )

def rotation_matrix_from_rotation_around_x(rx : float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(rx), -sin(rx)],
            [0.0, sin(rx), cos(rx)]
        ], dtype="float32"
    )


def rotation_matrix_from_rotation_around_y(ry : float) -> np.ndarray:
    return np.array(
        [
            [cos(ry), 0.0, sin(ry)],
            [0.0, 1.0, 0.0],
            [-sin(ry), 0.0, cos(ry)]
        ], dtype="float32"
    )


def rotation_matrix_from_rotation_around_z(rz : float) -> np.ndarray:
    return np.array(
        [
            [cos(rz), -sin(rz), 0.0],
            [sin(rz), cos(rz), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype="float32"
    )


class DataDict(TypedDict):
    fname : str
    intrinsic    : Union[np.ndarray, None]
    extrinsic    : Union[np.ndarray, None]
    size         : Union[np.ndarray, None]
    depth_range  : Union[np.ndarray, None]

def load_data(image_dir : str, camera_paremeter_path : Union[str, None]) -> dict[str, DataDict]:
    data = {}

    image_fnames = sorted(glob.glob(os.path.join(image_dir, "*.[jpJP][npNP][egEG]")))
    for image_fname in image_fnames:
        #data[get_cam_name_from_fname(image_fname)] = [image_fname, None, None, None, None]
        data[get_cam_name_from_fname(image_fname)] = DataDict(fname=image_fname, intrinsic=None, extrinsic=None, size=None, depth_range=None)

    if camera_paremeter_path is not None:
        if not os.path.exists(camera_paremeter_path):
            return data

        with open(camera_paremeter_path) as file:
            json_data = json.load(file)

        cameras = json_data["cameras"]
        for camera in cameras:
            if camera["Name"] not in data: continue
        
            position            = camera["Position"]
            rotation            = camera["Rotation"]
            focal               = camera["Focal"]
            princripal_point    = camera["Principle_point"]
            resolution          = camera["Resolution"]
            depth_range          = camera["Depth_range"]

            intrinsic = np.array(
                [
                    [focal[0], 0, princripal_point[0]],
                    [0, focal[1], princripal_point[1]],
                    [0, 0, 1],
                ]
            )

            #permute  = np.array([ [0, 0, 1], [-1, 0, 0], [0, -1, 0] ] ) # Permutation OpenCV/Colmap -> OMAF
            permute     = np.array([ [0, -1, 0], [0, 0, -1], [1, 0, 0] ] ) # OMAF-> OpenCV/Colmap
            rotation    = 0.01745329252 * np.array(rotation)
            rot_mat     = rotation_matrix_from_euler_angles(rotation[0], rotation[1], rotation[2])
            translation = permute.dot(np.array(position))
            rot_mat   = ( (permute).dot( (rot_mat[0:3,0:3]) ).dot( np.transpose(permute) ) ) 

            extrinsic = np.eye(4,4)
            extrinsic[:3, :3]   = rot_mat[:3,:3]
            extrinsic[:3, 3]    = translation

            size = np.array([resolution[0], resolution[1]])

            data[camera["Name"]]["intrinsic"] = intrinsic
            data[camera["Name"]]["extrinsic"] = extrinsic
            data[camera["Name"]]["size"] = size
            data[camera["Name"]]["depth_range"] = depth_range

    print(data)

    return data

    """
    
    _PROGRAM_GENERAL_IMG_UTILS = """
# ------------------ Image Utils ------------------ 

def save_raw_16bit(depth, fpath="raw.png"):
    
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    #depth = depth * 256  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    
    print(np.min(depth),np.max(depth))
    depth = Image.fromarray(depth)
    depth.save(fpath)
    print("Saved raw depth to", fpath)

def resize(depth : np.ndarray, size : Union[tuple[int, int], np.ndarray]) -> np.ndarray:
    image = Image.fromarray(depth)
    return np.array(image.resize((size[0], size[1])))

def build_depth_path(img_path : str) -> str:
    return "".join(os.path.basename(img_path).split(".png")[:-1]) + "_depth.png"

def rescale_depth(depth : np.ndarray, min_d, max_d, depth_range):
    depth_scale = (depth_range[0] / (min_d + 1e-7) + depth_range[1] / (max_d + 1e-7)) / 2.0
    print("depth scale ", depth_scale)
    
    depth = depth * depth_scale
    # keeps depth value reaching depth ranges --- todo: find a better way maybe...
    d_min = 1.01 * depth_range[0]
    d_max = 0.99 * depth_range[1]
    depth = np.clip(depth, d_min, d_max)
    return depth

    """
    
    class Method(Enum):
        NONE            = 0
        Zoe             = 1
        DepthPro        = 2
        DepthAnything3  = 3
        
    @staticmethod
    def from_array(img : np.ndarray, tmp_dir : str = "/tmp/", out_path : str | None = None, estimation_method : Method = Method.Zoe) -> np.ndarray:
        r"""Compute the depth from a color image. Different estimation methods can be used. (see submodules; path:external/).

        Args:
            img (np.ndarray): numpy array
            out_path (str, None): Path to the estimated depth map. (defaults = None)
            tmp_dir (str): Tmp directory to save tmp results. (defaults = /tmp)
            estimation_method (DepthEstimatorUtils.Method): the method with which the depth will be estimated. (defaults = DepthEstimatorUtils.Method.Zoe)

        Returns:
            np.ndarray: Depth values [(H,W,1), float32] in millimeters.
        """
        with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
            path_color = os.path.join(tmp, "img.png")
            ImageUtils.save_img(img, path_color, 4)

            path_depth : str | None = None
            match estimation_method:
                case DepthEstimatorUtils.Method.Zoe:
                    path_depth = DepthEstimatorUtils._zoe(os.path.dirname(path_color), os.path.dirname(out_path) if out_path is not None else tmp, tmp_dir=tmp)[1][0]
                case DepthEstimatorUtils.Method.DepthPro:
                    path_depth = DepthEstimatorUtils._depth_pro(path_color, out_path if out_path is not None else os.path.join(tmp,"depthpro_depth.png"), tmp_dir=tmp)
                case DepthEstimatorUtils.Method.DepthAnything3:
                    path_depth = DepthEstimatorUtils._depth_anything_3(os.path.dirname(path_color), os.path.dirname(out_path) if out_path is not None else tmp, tmp_dir=tmp)[1][0]
                case DepthEstimatorUtils.Method.NONE:
                    path_depth = DepthEstimatorUtils._none(os.path.dirname(path_color), os.path.dirname(out_path) if out_path is not None else tmp, tmp_dir=tmp)[1][0]

            if path_depth is None or not os.path.exists(path_depth):
                raise RuntimeError(f"The depth image path does not exist: {path_depth}. Something occured...")

            depth = cv2.imread(path_depth, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise FileNotFoundError("Cannot read file: {path_depth}")

            depth = depth.astype("float32") * DepthEstimatorUtils._DEPTH_SCALE_SAVE_TO_MM #in mm
            depth = ImageUtils.set_array_to_image_format(depth, 1)
        return depth
        
    @staticmethod
    def from_arrays(imgs : list[np.ndarray], tmp_dir : str = "/tmp/", out_dir : str | None = None, estimation_method : Method = Method.Zoe, parameters : list[Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None) -> list[np.ndarray]:
        r"""Compute the depth from a color image. Different estimation methods can be used. (see submodules; path:external/).

        Args:
            img (np.ndarray): numpy array
            out_path (str, None): Path to the estimated depth map. (defaults = None)
            tmp_dir (str): Tmp directory to save tmp results. (defaults = /tmp)
            estimation_method (DepthEstimatorUtils.Method): the method with which the depth will be estimated. (defaults = DepthEstimatorUtils.Method.Zoe)

        Returns:
            np.ndarray: Depth values [(H,W,1), float32] in millimeters.
        """
        depths : list[np.ndarray] = []
        with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
            color_paths_sorted = []
            for idx in range(len(imgs)):
                name = parameters[idx]["Name"] if parameters is not None else f"img_{idx}"
                path_color = os.path.join(tmp, f"{name}.png")
                color_paths_sorted.append(path_color)
                ImageUtils.save_img(imgs[idx], path_color, 4)
            
            parameter_path = None if parameters is None else os.path.join(tmp,"camera.json")
            if parameter_path is not None and parameters is not None: 
                IOUtils.save_json(parameter_path, {"cameras": parameters})
            
            color_paths : list[str] | None = None
            depth_paths : list[str] | None = None
            match estimation_method:
                case DepthEstimatorUtils.Method.Zoe:
                    (color_paths, depth_paths) = DepthEstimatorUtils._zoe(tmp, out_dir if out_dir is not None else tmp, tmp, parameter_path)
                case DepthEstimatorUtils.Method.DepthPro:
                    raise NotImplementedError("DepthPro Estimation from multi-view is not yet supported.")
                case DepthEstimatorUtils.Method.DepthAnything3:
                    (color_paths, depth_paths) = DepthEstimatorUtils._depth_anything_3(tmp, out_dir if out_dir is not None else tmp, tmp, parameter_path)
                case DepthEstimatorUtils.Method.NONE:
                    (color_paths, depth_paths) = DepthEstimatorUtils._none(tmp, out_dir if out_dir is not None else tmp, tmp, parameter_path)
            
            if color_paths is None or depth_paths is None:
                raise ValueError("No data was generated.")
            
            for idx in range(len(imgs)):
                color_path = color_paths_sorted[idx]
                idx_non_sorted = color_paths.index(color_path)
                depth_path = depth_paths[idx_non_sorted]
                
                if depth_path is None or not os.path.exists(depth_path):
                    raise RuntimeError(f"The depth image path does not exist: {depth_path}. Something occured...")
                        
                depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth is None:
                    raise FileNotFoundError("Cannot read file: {path_depth}")
                    
                depth = depth.astype("float32") * DepthEstimatorUtils._DEPTH_SCALE_SAVE_TO_MM #in mm
                depth = ImageUtils.set_array_to_image_format(depth, 1)
                depths.append(depth)
        return depths
    
    @staticmethod
    def _get_program_load_data(color_dir : str, camera_params_path : str | None) -> str:
        
        return f"""
image_dir = "{color_dir}"
camera_params_path = "{camera_params_path}"
images_data = load_data(image_dir,camera_params_path)

cam_names = []
images_fnames = []
intrinsics      : list[np.ndarray] = []
extrinsics      : list[np.ndarray] = []
sizes           : list[np.ndarray] = []
nears_fars      : list[np.ndarray] = []
#for cam_name, (fpath, intrinsic, extrinsic, sizes, nears_fars) in sorted(images_data.items(), key=lambda item : item[0]):
for cam_name, cam_dict in sorted(images_data.items(), key=lambda item : item[0]):
    fname = cam_dict["fname"]
    intrinsic = cam_dict["intrinsic"]
    extrinsic = cam_dict["extrinsic"]
    size = cam_dict["size"]
    near_far = cam_dict["depth_range"]
    
    cam_names.append(cam_names)
    images_fnames.append(fname)
    if intrinsic is not None: intrinsics.append(intrinsic)
    if extrinsic is not None: extrinsics.append(extrinsic)
    if size is not None: sizes.append(size)
    if near_far is not None: nears_fars.append(near_far)

if len(images_fnames) == 0:
    raise ValueError("No data found.")
intrinsics_arr  = np.stack(intrinsics, axis=0)   if len(intrinsics) == len(images_fnames) else None
extrinsics_arr  = np.stack(extrinsics, axis=0)   if len(extrinsics) == len(images_fnames) else None
sizes_arr       = np.stack(sizes, axis=0)             if len(sizes) == len(images_fnames) else None
nears_fars_arr  = np.stack(nears_fars, axis=0)        if len(nears_fars) == len(images_fnames) else None
process_res     = np.max(sizes_arr).item() if sizes_arr is not None else 504

        """
    
    @staticmethod
    def _none(color_dir : str, out_dir : str, tmp_dir : str, camera_params_path : str | None = None) -> tuple[list[str], list[str]]:
        r"""Returns empty depth maps.

        Args:
            path_color (str): path to the color image
            out_path (str): Path to the estimated depth map.
            tmp_dir (str): Tmp directory to save tmp results.

        Returns:
            str: Path to the estimated depth map.
        """
        
        program : str = DepthEstimatorUtils._PROGRAM_GENERAL_IMPORT + DepthEstimatorUtils._PROGRAM_GENERAL_DATA_LOADING + DepthEstimatorUtils._PROGRAM_GENERAL_IMG_UTILS + DepthEstimatorUtils._get_program_load_data(color_dir, camera_params_path) + f"""
depths = []
for idx, fname in enumerate(images_fnames):
    image = np.asarray(Image.open(fname).convert("RGB"))  # load
    depth = np.ones((image.shape[0], image.shape[1]), dtype="float32")
    depth *= np.mean(nears_fars_arr[idx]).item()
    depth_numpy : np.ndarray = depth
    depths.append(depth_numpy)

assert len(depths) == len(images_fnames)

depths = np.stack(depths, axis=0)
min_d = depths.min()
max_d = depths.max()
print(min_d, max_d)
print(nears_fars_arr)

out_dir = "{out_dir}"
if not os.path.exists(out_dir): os.makedirs(out_dir)

depth_paths = []
for idx, fname in enumerate(images_fnames):
    depth_ = depths[idx]

    # Save raw
    fpath = os.path.join(out_dir, build_depth_path(images_fnames[idx]))
    save_raw_16bit(depth_ * {DepthEstimatorUtils._DEPTH_SCALE_METERS_TO_SAVE}, fpath) #rescale to millimeters

    depth_paths.append(fpath)

print(images_fnames)
print(depth_paths)
        """
        
        py_fname = os.path.join(tmp_dir, "none_py_depth.py")
        with open(py_fname, "+w") as py_file:
            py_file.write(program)
        
        py_out = SubProcessUtils.conda_pyrun_from_file(DepthEstimatorUtils.Method.Zoe.name.lower(), py_fname)            
        backslash = "\\"
        Logger.debug(f"NONEDEPTH: {py_out.stdout.split(f'{backslash}n')}", str(DepthEstimatorUtils))
        if len(py_out.stderr) > 1 or py_out.stderr != '':
            Logger.error(f"NONEDEPTH ERR: {py_out.stderr}")
        
        color_paths = py_out.stdout.split('\n')[-3][2:-2].split("', '")
        depth_paths = py_out.stdout.split('\n')[-2][2:-2].split("', '")
        
        Logger.debug(f"Color paths: {color_paths}.", DepthEstimatorUtils.__name__)
        Logger.debug(f"Depth paths: {depth_paths}.", DepthEstimatorUtils.__name__)
        
        return (color_paths, depth_paths)
    
    @staticmethod
    def _zoe(color_dir : str, out_dir : str, tmp_dir : str, camera_params_path : str | None = None) -> tuple[list[str], list[str]]:
        r"""Compute the depth from ZoeDepth (see submodule; external/ZoeDepth).

        Args:
            path_color (str): path to the color image
            out_path (str): Path to the estimated depth map.
            tmp_dir (str): Tmp directory to save tmp results.

        Returns:
            str: Path to the estimated depth map.
        """
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        zoe_dir = os.path.join(current_dir, "..", "..", "external/ZoeDepth/")
        Logger.debug(zoe_dir)
        program : str = DepthEstimatorUtils._PROGRAM_GENERAL_IMPORT + DepthEstimatorUtils._PROGRAM_GENERAL_DATA_LOADING + DepthEstimatorUtils._PROGRAM_GENERAL_IMG_UTILS + DepthEstimatorUtils._get_program_load_data(color_dir, camera_params_path) + f"""
# Zoe_N
model_zoe = torch.hub.load("{zoe_dir}", "ZoeD_N", source="local", pretrained=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe.to(DEVICE)

depths = []
for idx, fname in enumerate(images_fnames):
    image = Image.open(fname).convert("RGB")  # load
    depth = zoe.infer_pil(image, pad_input=False)  # as numpy, pad_input=False for more accuracte results (https://github.com/isl-org/ZoeDepth/issues/10)

    depth_numpy : np.ndarray
    if isinstance(depth, torch.Tensor):
        depth_numpy = depth.squeeze().cpu().numpy()
    else:
        depth_numpy = depth
    depths.append(depth_numpy)

assert len(depths) == len(images_fnames)

depths = np.stack(depths, axis=0)
min_d = depths.min()
max_d = depths.max()
print(min_d, max_d)
print(nears_fars_arr)

out_dir = "{out_dir}"
if not os.path.exists(out_dir): os.makedirs(out_dir)

depth_paths = []
for idx, fname in enumerate(images_fnames):
    depth_ = depths[idx] 
    if nears_fars_arr is not None:
        depth_ = rescale_depth(depth_, min_d, max_d, nears_fars_arr[idx]) 

    # Save raw
    fpath = os.path.join(out_dir, build_depth_path(images_fnames[idx]))
    save_raw_16bit(depth_ * {DepthEstimatorUtils._DEPTH_SCALE_METERS_TO_SAVE}, fpath) #rescale to millimeters

    depth_paths.append(fpath)

print(images_fnames)
print(depth_paths)
        """
        
        py_fname = os.path.join(tmp_dir, "zoe_py_depth.py")
        with open(py_fname, "+w") as py_file:
            py_file.write(program)
        
        py_out = SubProcessUtils.conda_pyrun_from_file(DepthEstimatorUtils.Method.Zoe.name.lower(), py_fname)            
        backslash = "\\"
        Logger.debug(f"ZOEDEPTH: {py_out.stdout.split(f'{backslash}n')}", str(DepthEstimatorUtils))
        if len(py_out.stderr) > 1 or py_out.stderr != '':
            Logger.error(f"ZOEDEPTH ERR: {py_out.stderr}")
        
        color_paths = py_out.stdout.split('\n')[-3][2:-2].split("', '")
        depth_paths = py_out.stdout.split('\n')[-2][2:-2].split("', '")
        
        Logger.debug(f"Color paths: {color_paths}.", DepthEstimatorUtils.__name__)
        Logger.debug(f"Depth paths: {depth_paths}.", DepthEstimatorUtils.__name__)
        
        return (color_paths, depth_paths)

    @staticmethod
    def _depth_pro(path_color : str, out_path : str, tmp_dir : str) -> str:
        r"""Compute the depth from ZoeDepth (see submodule; external/ZoeDepth).

        Args:
            path_color (str): path to the color image
            out_path (str): Path to the estimated depth map.
            tmp_dir (str): Tmp directory to save tmp results.

        Returns:
            str: Path to the estimated depth map.
        """
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        DepthPro_dir = os.path.join(current_dir, "..", "..", "external/DepthPro/")
        Logger.debug(DepthPro_dir)
        program : str = f"""
from PIL import Image
import torch
import depth_pro

# Load model and preprocessing transform
config = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
config.checkpoint_uri = '{os.path.join(DepthPro_dir, "checkpoints/depth_pro.pt")}'

model, transform = depth_pro.create_model_and_transforms(
    config=config
)
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb('{path_color}') #f_px can be none, not a problem for later
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].

def save_raw_16bit(depth, fpath="raw.png"):
    import numpy as np
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    #depth = depth * 256  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)
    print("Saved raw depth to", fpath)

depth_numpy = depth.numpy()
fpath = "{out_path}"
save_raw_16bit(depth_numpy * {DepthEstimatorUtils._DEPTH_SCALE_METERS_TO_SAVE}, fpath) #rescale to millimeters
"""
        py_fname = os.path.join(tmp_dir, "depth_pro_py_depth.py")
        with open(py_fname, "+w") as py_file:
            py_file.write(program)
        
        py_out = SubProcessUtils.conda_pyrun_from_file(DepthEstimatorUtils.Method.DepthPro.name.lower(), py_fname)
        Logger.debug(f"DepthPro OUT: {py_out.stdout}", str(DepthEstimatorUtils))
        if len(py_out.stderr) > 1 or py_out.stderr != '':
            Logger.error(f"DepthPro ERR: {py_out.stderr}")
        
        return out_path
        
    @staticmethod
    def _depth_anything_3(color_dir : str, out_dir : str, tmp_dir : str, camera_params_path : str | None = None) -> tuple[list[str], list[str]]:
        r"""Compute the depth from DepthAnything3 (see submodule; external/DepthAnything3).

        Args:
            color_dir (str): path to the color image directory
            out_dir (str): Directory where the estimated depth maps will be saved.
            tmp_dir (str): Tmp directory to save tmp results.
            camera_params_path (str): Path to the camera parameter json file.

        Returns:
            tuple[list[str], list[str]]: (Color file paths, Depth file paths)
        """
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(current_dir, "..", "..", "external/DepthAnything3/")
        Logger.debug(src_dir)
        program : str = DepthEstimatorUtils._PROGRAM_GENERAL_IMPORT + DepthEstimatorUtils._PROGRAM_GENERAL_DATA_LOADING + DepthEstimatorUtils._PROGRAM_GENERAL_IMG_UTILS + DepthEstimatorUtils._get_program_load_data(color_dir, camera_params_path) + f"""
# ----------------- MODEL CODE ----------------- 

from depth_anything_3.api import DepthAnything3
device = torch.device("cuda")
cache_dir = os.path.join('{src_dir}', 'cache_models')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
model = DepthAnything3.from_pretrained(
    "depth-anything/DA3-BASE",
    cache_dir=cache_dir
)
model = model.to(device=device)

# Run inference.
prediction = model.inference(
    images_fnames,
    intrinsics=intrinsics_arr,
    extrinsics=extrinsics_arr,
    process_res=process_res,
)

_DEPTH_SCALE_METERS_TO_SAVE = {DepthEstimatorUtils._DEPTH_SCALE_METERS_TO_SAVE}
depths = prediction.depth
out_dir = "{out_dir}"
if not os.path.exists(out_dir): os.makedirs(out_dir)

depth_paths : list[str] = []
if len(depths) <= 0:
    raise ValueError("There is no depth")
else:
    for idx in range(len(depths)): 
        fpath = os.path.join(out_dir, build_depth_path(images_fnames[idx]))
        depth_numpy = depths[idx]
        if sizes_arr is not None:
            size =  sizes_arr[idx]
            depth_numpy = resize(depth_numpy, size)
        save_raw_16bit(depth_numpy * _DEPTH_SCALE_METERS_TO_SAVE, fpath) #rescale to millimeters
        
        depth_paths.append(fpath)

print(images_fnames)
print(depth_paths)
"""
        py_fname = os.path.join(tmp_dir, "depth_anything_3_py_depth.py")
        with open(py_fname, "+w") as py_file:
            py_file.write(program)
        
        py_out = SubProcessUtils.conda_pyrun_from_file(DepthEstimatorUtils.Method.DepthAnything3.name.lower(), py_fname)
        backslash = "\\"
        Logger.debug(f"DepthAnything3: {py_out.stdout.split(f'{backslash}n')}", str(DepthEstimatorUtils))
        if len(py_out.stderr) > 1 or py_out.stderr != '':
            Logger.error(f"DepthAnything3 ERR: {py_out.stderr}")
        
        color_paths = py_out.stdout.split('\n')[-3][2:-2].split("', '")
        depth_paths = py_out.stdout.split('\n')[-2][2:-2].split("', '")
        
        Logger.debug(f"Color paths: {color_paths}.", DepthEstimatorUtils.__name__)
        Logger.debug(f"Depth paths: {depth_paths}.", DepthEstimatorUtils.__name__)
        
        return (color_paths, depth_paths)

class SubProcessUtils:    
    import subprocess
    
    @staticmethod
    def conda_env_exists(env_name: str) -> bool:
        """
        Check if a conda environment exists.
        
        Args:
            env_name: Name of the conda environment to check
            
        Returns:
            bool: True if environment exists, False otherwise
        """
        try:
            # Get list of all conda environments
            result = SubProcessUtils.subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True,
                text=True,
                shell=False
            )
            
            if result.returncode != 0:
                # Try alternative command for older conda versions
                result = SubProcessUtils.subprocess.run(
                    ["conda", "info", "--envs", "--json"],
                    capture_output=True,
                    text=True,
                    shell=False
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to list conda environments: {result.stderr}"
                    )
            
            # Parse JSON output
            env_data = json.loads(result.stdout)
            
            # The JSON structure varies between conda versions
            if "envs" in env_data:
                # Newer conda versions
                envs = env_data["envs"]
            elif "environment" in env_data:
                # Some versions use different keys
                envs = list(env_data["environment"].keys())
            else:
                # Fallback: check if we can find the environment in stdout
                result = SubProcessUtils.subprocess.run(
                    ["conda", "env", "list"],
                    capture_output=True,
                    text=True,
                    shell=False
                )
                envs = result.stdout.splitlines()
            
            # Check if environment exists
            env_name_lower = env_name.lower()
            
            # Convert to string and check
            if isinstance(envs, list):
                for env_path in envs:
                    if isinstance(env_path, str):
                        # Extract env name from path
                        path_parts = env_path.strip().split(os.sep)
                        if path_parts:
                            current_env = path_parts[-1]
                            if current_env.lower() == env_name_lower:
                                return True
            
            # Also check in the plain text output for older conda versions
            if isinstance(envs, list) and all(isinstance(e, str) for e in envs):
                for line in envs:
                    if env_name_lower in line.lower() and not line.strip().startswith('#'):
                        return True
            
            return False
            
        except (json.JSONDecodeError, KeyError, SubProcessUtils.subprocess.SubprocessError) as e:
            raise RuntimeError(
                f"Error checking conda environment '{env_name}': {str(e)}"
            )
    
    @staticmethod
    def conda_pyrun_from_str(env_name : str, py_prog : str):
        if not SubProcessUtils.conda_env_exists(env_name):
            raise RuntimeError("Impossible to run conda environment.")
            
        args = ["conda", "run", "-n", env_name, "python", py_prog]
        Logger.debug(f"Run with args: {' '.join(args)}")
        result = SubProcessUtils.subprocess.run(
            args,
            capture_output=True,
            text=True
        )
        return result
    
    @staticmethod
    def conda_pyrun_from_file(env_name : str, py_fname : str):
        if not os.path.exists(py_fname):
            raise FileNotFoundError("Python file does not exist.")
            
        if not SubProcessUtils.conda_env_exists(env_name):
            raise RuntimeError("Impossible to run conda environment.")
            
        args = ["conda", "run", "-n", env_name, "python", py_fname]
        Logger.debug(f"Run with args: {' '.join(args)}")
        result = SubProcessUtils.subprocess.run(
            args,
            capture_output=True,
            text=True
        )
        return result

class ImageUtils:
    """Utility functions for image format handling, loading, saving, and YUV conversions."""

    class YUVFormat(Enum):
        """List popular YUVFormats.
        """
        NONE = 0
        yuv420p = 1
        gray16le = 2
        yuv444p = 3
        yuv420p10le = 4
        yuv420p16le = 5
        yuv444p10le = 6
        yuv444p16le = 7

        @property
        def code(self) -> str:
            """Extracts the YUV code of the format.
            The YUV code is a string that just tells about the YUV encoding format.
            Example: `YUVFormat.yuv420p.code -> "YUV420"` 

            Raises:
                KeyError: If the YUV format is not recognized.

            Returns:
                str: the YUV code.
            """
            if self.name.startswith("yuv420"):
                return "YUV420"
            elif self.name.startswith("yuv444"):
                return "YUV444"
            elif self.name.startswith("gray"):
                return "YUV400"
            else:
                raise KeyError("Invalid YUV format.")
        
        @property
        def bit(self) -> int:
            """The number of bits for each data element.
            Example: `YUVFormat.yuv420p10le.bit -> 10`

            Raises:
                KeyError: If the YUV format is not recognized.

            Returns:
                int: nb of bit.
            """
            if self.name.startswith("yuv"):
                right = self.name.split("p")[1]
                if right.endswith("le"):
                    return int(right.split("le")[0])
                else:
                    return 8
            elif self.name.startswith("gray"):
                right = self.name.split("gray")[1]
                if right.endswith("le"):
                    return int(right.split("le")[0])
                else:
                    return 8
            else:
                raise KeyError("Invalid YUV format.")

    @staticmethod
    def get_yuv_fmt(bit : int, yuv_code : str) -> YUVFormat:
        """Retrieve the complete YUVFormat from nb of bits and the YUV code (e.g., YUV420).

        Args:
            bit (int): nb of bits.
            yuv_code (str): YUV code (E.g., YUV420, YUV400, etc.).

        Returns:
            YUVFormat: Complete YUVFormat.
        """
        if(bit == 8 and yuv_code == "YUV420"):
            return ImageUtils.YUVFormat.yuv420p
        elif(bit == 10 and yuv_code == "YUV420"):
            return ImageUtils.YUVFormat.yuv420p10le
        elif(bit == 16 and yuv_code == "YUV420"):
            return ImageUtils.YUVFormat.yuv420p16le
        elif(bit == 8 and yuv_code == "YUV444"):
            return ImageUtils.YUVFormat.yuv444p
        elif(bit == 16 and yuv_code == "YUV400"):
            return ImageUtils.YUVFormat.gray16le
        
        return ImageUtils.YUVFormat.NONE

    @staticmethod
    def read_from_normal_file(path : str, dtype : str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f'image could not be loaded. Path: {path}')
        
        if len(img.shape) == 2:
            img = img.reshape((*img.shape, 1))
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        elif img.shape[2] == 4: 
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        
        img = ArrayUtils.convert_to_dtype(img, dtype)
        
        return img

    @staticmethod
    def load_exr_depth(path) -> np.ndarray:
        """
        .. deprecated:: 

        Load EXR depth map using OpenCV with proper configuration
        Args:
            path (_type_): _description_

        Raises:
            FileNotFoundError: _description_
            IOError: _description_

        Returns:
            np.ndarray: _description_
        """
        
        # EXR requires special handling
        try:
            # Enable EXR support in OpenCV
            #cv2.setErrorVerbosity(0)
            depth = cv2.imread(
                path, 
                cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
            )
            
            if depth is None:
                raise FileNotFoundError('Depth image could not be loaded.')

            # Handle multi-channel EXR (take first channel)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
                
            return depth.astype(np.float32)
        except Exception as e:
            #print(f"Error loading EXR {path}: {str(e)}")
            raise IOError("ImageUtils: cannot load EXR file. Check the following exception: " + str(e))
        return np.zeros(())
    
    @staticmethod
    def get_image_slangpy_format(arr : np.ndarray) -> spy.Format:
        """Returns the Image slangpy.format based on the numpy array.

        Args:
            arr (np.ndarray): numpy array of the image

        Raises:
            ValueError: If the images has less than 3 dimensions.

        Returns:
            spy.Format: The slangpy.format correspoint on the nb of channels and the dtype.
        """
        if(len(arr.shape) < 3): raise ValueError('Image must have at least 3 dimensions.')
        
        C = arr.shape[2]
        dtype = arr.dtype
        return ArrayUtils.get_array_slangpy_format(C, dtype)
    
    @staticmethod
    def set_array_to_image_format(in_arr : np.ndarray, channels : int, default_bck_value : float | list[float] = 1.0) -> np.ndarray:
        """Set a numpy array to the image format (i.e., 3 dimensions (H, W, C))

        Args:
            in_arr (np.ndarray): input numpy array
            channels (int): Given number of channels.

        Raises:
            RuntimeError: If nb of channels is not {1, 3, 4}
            RuntimeError: If the array has more than 3 dimensions.
            RuntimeError: If the array has less than 2 dimensions.

        Returns:
            np.ndarray: Numpy array in the image format (H, W, C).
        """
        dtype = in_arr.dtype
        has_changed = False
        if(channels != 1 and channels != 3 and channels != 4): raise RuntimeError('Undefined image behavior.')

        if(len(in_arr.shape) > 3):
            raise RuntimeError('Undefined image behavior.')
        elif(len(in_arr.shape) < 2):
            raise RuntimeError('Undefined image behavior.')
        elif(len(in_arr.shape) == 2):
            in_arr = np.reshape(in_arr, (in_arr.shape[0], in_arr.shape[1], 1))
            has_changed = True

        if(in_arr.shape[2] == 1 and channels >= 3):
            in_arr = np.repeat(in_arr, 3, axis=2)
            has_changed = True
            
        if(in_arr.shape[2] == 3 and channels == 4):
            zeros = np.ones((in_arr.shape[0], in_arr.shape[1], 1), dtype=dtype)
            in_arr = np.concatenate([in_arr, zeros], axis = 2)
            has_changed = True
        elif(in_arr.shape[2] == 3 and channels == 1):
            in_arr = in_arr[:,:,0:1]
            has_changed = True
            
        if channels == 4:
            alpha_mask = in_arr[:,:,3] <= 0.0
            if isinstance(default_bck_value, float): default_bck_value = [default_bck_value, default_bck_value, default_bck_value, 0.0]
            elif isinstance(default_bck_value, list) and len(default_bck_value) != channels: raise ValueError("Cannot apply background value, wrong number of channels.") 
            in_arr[alpha_mask] = default_bck_value
            has_changed = True
        
        if has_changed: 
            return in_arr.astype(dtype=dtype)
        
        return in_arr
    
    @staticmethod
    def read_img(path : str, resolution : tuple[int, int], yuv_format : YUVFormat = YUVFormat.NONE, dtype = "float32", convert_2_rgb : bool = False) -> np.ndarray:
        path_ext = StrUtils.get_file_extension(path)
        img : np.ndarray 
        if path_ext == "yuv":
            img = ImageUtils.read_from_yuv_file(path, resolution, yuv_format, dtype, convert_2_rgb=convert_2_rgb)
        elif path_ext in ["png", "jpeg", "jpg", "PNG", "JPG", "JPEG"]:
            img = ImageUtils.read_from_normal_file(path, dtype)
        elif path_ext == "exr":
            img = ImageUtils.load_exr_depth(path)
        else:
            raise RuntimeError(f"Does not know how to read the following image extensions: {path_ext}.")
            
        if img.shape[1] != resolution[0] or img.shape[0] != resolution[1]: raise ValueError('image resolution mismatch.')
        
        return img
    
    @staticmethod
    def save_img(img : np.ndarray, path : str,channels : int, out_dtype : str = 'uint8', out_yuv_format : YUVFormat = YUVFormat.yuv420p):
        """Save a numpy array (reflecting an image) in the given image extension (through path), dtype or yuv format.

        .. warning:: save to EXR file format not implemented yet.
        
        Args:
            img (np.ndarray): Numpy array to save
            path (str): Path where to store the image.
            channels (int): Number of channels. (To check if the image is in the corract image format.)
            out_dtype (str, optional): Output dtype. Used when saving to *normal* image file format (i.e., png, jpeg, etc.). Defaults to 'uint8'.
            out_yuv_format (YUVFormat, optional): Output YUV format. Used when saving to YUV image file format. Defaults to YUVFormat.yuv420p.

        Raises:
            ValueError: If less than 3 dimensions or the number of channels is not correct. 
            NotImplementedError: EXR file format not implemented yet.
            KeyError: if the file extension is not recognized.
        """
        if(len(img.shape) < 3) or (channels != img.shape[2]): raise ValueError("ImageUtils: cannot save image because the image does meet the requirements.")

        img = ArrayUtils.convert_to_dtype(img, out_dtype)

        img_ext = StrUtils.get_file_extension(path)
        if(img_ext == "yuv"):
            ImageUtils.save_img_to_yuv_file_format(img, path, out_yuv_format)
        elif(img_ext == "exr"):
            raise NotImplementedError("")
        elif(img_ext == "jpg" or img_ext == "jpeg" or img_ext == "png"):
            ImageUtils.save_img_to_normal_file_format(img, path)
        else:
            raise KeyError("ImageUtils: cannot recognize the image extension: " + img_ext)

    @staticmethod
    def save_img_to_normal_file_format(img : np.ndarray, path : str):
        """Save an image to a normal file format (i.e., png, jpeg, etc.).

        Args:
            img (np.ndarray): input numpy array to save
            path (str): path where to store the image.
        """
        if img.shape[2] == 1:
          img = np.stack([img[:,:,0],img[:,:,0],img[:,:,0]], axis=2)
          
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(path, img)

    @staticmethod
    def save_img_to_yuv_file_format(img : np.ndarray, path : str, yuv_format : YUVFormat):
        """Save an image to a YUV file format.

        Args:
            img (np.ndarray): input numpy array to save
            path (str): path where to store the image.
            yuv_format (YUVFormat): YUV format in which the image is stored.

        Raises:
            ValueError: If the array does not have the right image shape.
            ValueError: If U or V channel is missing.
            ValueError: Unsupported YUV formats. Check supported YUV formats in ImageUtils.YUVFormat.
        """
        if(img.shape[2] >= 4):
            img = img[:,:,0:3]
        elif(img.shape[2] == 2):
            raise ValueError('Invalid image shape for YUV conversion.')
        elif(img.shape[2] == 1):
            img = img.repeat(3, 2)
        
        path = StrUtils.build_yuvview_path(path).format(img.shape[1], img.shape[0], yuv_format.name)
        
        match yuv_format:
            case ImageUtils.YUVFormat.yuv444p:
                yuv_img = yuvio.from_rgb(img, yuv_format.name, value_range="full")

            case ImageUtils.YUVFormat.yuv420p:
                yuv_img = yuvio.from_rgb(img, "yuv444p", value_range="full")
                
                if yuv_img.u is None or yuv_img.v is None: raise ValueError('Missing U or V data in YUV image.')
                
                new_u = ArrayUtils.downscale(yuv_img.u, 2, True)
                new_v = ArrayUtils.downscale(yuv_img.v, 2, True)
                yuv_img_ = yuvio.frame((yuv_img.y, new_u, new_v), 'yuv420p') 
                yuv_img = yuv_img_

            case ImageUtils.YUVFormat.gray16le:
                yuv_img = yuvio.frame((img[:,:,0], None, None), 'gray16le') 

            case default:
                raise ValueError('Unsupported YUV format.')
        
        yuvio.imwrite(path, yuv_img)

    @staticmethod
    def read_from_yuv_file(path : str, resolution : tuple[int, int], yuv_format : YUVFormat, dtype : str = "float32", convert_2_rgb : bool = True) -> np.ndarray:
        """Read image data from a YUV file. 
        It requires the resolution, the YUV format for reading the file.

        Args:
            path (str): path to the YUV file.
            resolution (tuple[int, int]): Resolution of the image.
            yuv_format (YUVFormat): YUV format of the image.
            dtype (str, optional): Target numpy dtype string. Defaults to "float32".
            convert_2_rgb (bool, optional): If it needs to be converted to RGB, YUV->RGB color space. Defaults to True.

        Raises:
            KeyError: If the YUV format is not recognized.

        Returns:
            np.ndarray: Return the YUV data in a numpy array.
        """
        max_value = 1.0
        res : yuvio.core.YUVFrame | np.ndarray = np.array(())
        Logger.debug(f"Reading YUV file: {path}", str(ImageUtils))

        match yuv_format:
            case ImageUtils.YUVFormat.yuv444p:
                max_value = 2**8 - 1
                res = yuvio.imread(path, resolution[0], resolution[1], yuv_format.name) 
            
            case ImageUtils.YUVFormat.yuv420p:
                max_value = 2**8 - 1
                yuv_color = yuvio.imread(path, resolution[0], resolution[1], yuv_format.name)
                new_u = ArrayUtils.upsample(yuv_color.u, 2)
                new_v = ArrayUtils.upsample(yuv_color.v, 2)
                res = yuvio.frame((yuv_color.y, new_u, new_v), ImageUtils.YUVFormat.yuv444p.name)
            
            case ImageUtils.YUVFormat.yuv420p10le:
                max_value = 2**10 - 1
                yuv_color = yuvio.imread(path, resolution[0], resolution[1], yuv_format.name)
                new_u = ArrayUtils.upsample(yuv_color.u, 2)
                new_v = ArrayUtils.upsample(yuv_color.v, 2)
                res = yuvio.frame((yuv_color.y, new_u, new_v), ImageUtils.YUVFormat.yuv444p10le.name)
            
            case ImageUtils.YUVFormat.yuv420p16le:
                max_value = 2**16 - 1
                yuv_color = yuvio.imread(path, resolution[0], resolution[1], yuv_format.name)
                new_u = ArrayUtils.upsample(yuv_color.u, 2)
                new_v = ArrayUtils.upsample(yuv_color.v, 2)
                res = yuvio.frame((yuv_color.y, new_u, new_v), ImageUtils.YUVFormat.yuv444p16le.name) 
            
            case ImageUtils.YUVFormat.gray16le:
                max_value = 2**16 - 1
                res = yuvio.imread(path, resolution[0], resolution[1], yuv_format.name)
        
        if convert_2_rgb and isinstance(res, yuvio.core.YUVFrame):
            return yuvio.to_rgb(res, specification='bt709', value_range='limited').astype(dtype) / max_value
        elif (isinstance(res, np.ndarray)):
            return res.astype(dtype) / max_value
        elif (isinstance(res, yuvio.core.YUVFrame)):
            y = res.y
            u = res.u if res.u is not None else np.zeros_like(res.y)
            v = res.v if res.v is not None else np.zeros_like(res.y)
            res_ = np.stack([y, u, v], axis=2)
            return res_.astype(dtype) / max_value

        raise KeyError("ImageUtils: impossible to read YUV file because the YUV format is not recognized: " + yuv_format.name)

    @staticmethod
    def convert_normalized_mpeg_disparity_to_depth(depth : np.ndarray, near : float, far : float) -> np.ndarray:
        r"""
        Convert normalized MPEG disparity ($d \in [0, 1]$) to depth values ($D\in[D_{min},D_{max}]$).
        Since depth cannot be directly saved in an integer data format (e.g., YUV, png, etc.), depth values are converted to MPEG disparity format.
        The MPEG disparity format sets the nearest depth value to $2^N - 1$ and the farthest to $1$. The null value is kept for marking the disparity as invalid.  
        
        One can convert MPEG disparity values to depth values using:
        $D = (D_{min}D_{max})/(D_{min} + d (D_{max} - D_{min}))$

        Inverserly, the disparity can be obtained with:
        $d = ((D_{min}D_{max})/(D(2^{N}-1))- D_{min})/(D_{min}D_{max})$

        More details can be found in *Chapter 3.1.2 Depth maps* of [1].

        .. references::
        [1] *'Dury, S., Bonatto, D., Teratani, M., & Lafruit, G. (2022). 3D Computer Graphics: View Synthesis Tool for VR Immersive Video.'*

        Args:
            depth (np.ndarray): Disparity numpy array
            near (float): Nearest disparity value
            far (float): Farthest dispairty value

        Returns:
            np.ndarray: Depth values between $[D_{min},D_{max}]$
        todo: check this
        """
        if (far >= 1000.0) :
            depth = near / (depth + 1e-7)
        else:
            depth = (far * near) / (near + depth * (far - near) + 1e-7)
        return depth
    
    @staticmethod
    def convert_depth_to_normalized_mpeg_disparity(depth : np.ndarray, near : float, far : float) -> np.ndarray:
        r"""
        Convert depth values ($D\in[D_{min},D_{max}]$) to normalized MPEG disparity ($d \in [0, 1]$).
        Since depth cannot be directly saved in an integer data format (e.g., YUV, png, etc.), depth values are converted to MPEG disparity format.
        The MPEG disparity format sets the nearest depth value to $2^N - 1$ and the farthest to $1$. The null value is kept for marking the disparity as invalid.  
        
        One can convert MPEG disparity values to depth values using:
        $D = (D_{min}D_{max})/(D_{min} + d (D_{max} - D_{min}))$.

        Inverserly, the disparity can be obtained with:
        $d = (((D_{min}D_{max})/D) - D_{min})/(D_{max}-D_{min})$

        More details can be found in *Chapter 3.1.2 Depth maps* of [1].

        .. references::
        [1] *'Dury, S., Bonatto, D., Teratani, M., & Lafruit, G. (2022). 3D Computer Graphics: View Synthesis Tool for VR Immersive Video.'*

        Args:
            depth (np.ndarray): RVS depth values in a numpy array.
            near (float): Nearest disparity value
            far (float): Farthest disparity value

        Returns:
            np.ndarray: MPEG disparity values between $[1, 2^N-1]$ (0 for invalid values).
        """
        if (far >= 1000.0) :
            depth = near / (depth + 1e-7)
        else:
            depth = (((far * near) / (depth + 1e-7)) - near) / (far - near) 
        return depth
    
    @staticmethod
    def ssim(arr_1 : np.ndarray, arr_2 : np.ndarray, k_1 : float = 0.01, k_2 : float = 0.03, win_size : int = 5) -> float:
        """Compute the Structural Similarity Index Measure (SSIM [1]) between two images (numpy arrays).

        .. References::
        [1] Wang, Zhou; Bovik, A.C.; Sheikh, H.R.; Simoncelli, E.P. (2004-04-01). "Image quality assessment: from error visibility to structural similarity". IEEE Transactions on Image Processing. 13 (4): 600-612.

        Args:
            arr_1 (np.ndarray): Input numpy array: HxWxC
            arr_2 (np.ndarray): Input numpy array: HxWxC
            k_1 (float, optional): Constant variable in the SSIM algorithm. Defaults to 0.01.
            k_2 (float, optional): Constant variable in the SSIM algorithm. Defaults to 0.03.

        Raises:
            ValueError: If the arrays are not encoded in the same format.
            ValueError: If the tensors do not have the same dimensions.

        Returns:
            float: SSIM value.
        """
        if arr_1.shape != arr_2.shape: raise ValueError("Impossible to run SSIM with tensors of different dimensions.")
        if arr_1.dtype != arr_2.dtype: raise ValueError("Cannot compute SSIM if arrays do not have the same encoding.")
        
        # ------------------- SSIM INIT ------------------- 
        ssim_ = 0.0

        arr_1 = ArrayUtils.convert_to_dtype(arr_1, "uint8")
        arr_2 = ArrayUtils.convert_to_dtype(arr_2, "uint8")

        luma_1 = cv2.cvtColor(arr_1, cv2.COLOR_RGB2YCrCb)[:,:,0] # keep only the luma
        if not isinstance(luma_1, np.ndarray): raise ValueError("Impossible to convert RGB to luma.")
        luma_2 = cv2.cvtColor(arr_2, cv2.COLOR_RGB2YCrCb)[:,:,0] # keep only the luma
        if not isinstance(luma_2, np.ndarray): raise ValueError("Impossible to convert RGB to luma.")
     
        ssim_ = ssim(luma_1, luma_2, win_size=5, data_range=255)
        if not isinstance(ssim_, float): raise ValueError("Something went wrong during SSIM computation.")

        return ssim_
    
    @staticmethod
    def lpips(tensor_1 : torch.Tensor, tensor_2 : torch.Tensor):
        """Compute the perceptual similarity (LPIPS [1, 2]) between two images (3xHxW) or two batch of images (Nx3xHxW).

        .. References::
        [1] The Unreasonable Effectiveness of Deep Features as a Perceptual Metric Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. In CVPR, 2018. \n
        [2] [github repo](https://github.com/richzhang/PerceptualSimilarity?tab=readme-ov-file)

        Raises:
            ValueError: If the tensors do not have the same dimensions.

        Args:
            tensor_1 (torch.Tensor): CxHxW or NxCxWxH. Both tensors must have same dimensions.
            tensor_2 (torch.Tensor): CxHxW or NxCxWxH. Both tensors must have same dimensions.

        Returns:
            float: LPIPS value.
        """
        if tensor_1.shape != tensor_2.shape: raise ValueError("Impossible to run LPIPS with tensors of different dimensions.")

        tensor_1 = TorchUtils.img_to_batch(tensor_1)
        tensor_2 = TorchUtils.img_to_batch(tensor_2)
        
        tensor_1 = TorchUtils.to_lpips(tensor_1)
        tensor_2 = TorchUtils.to_lpips(tensor_2)

        lpips_model = TorchUtils.LPIPS_MODEL
        lpips_ = lpips_model.forward(tensor_1[:,0:3,:,:], tensor_2[:,0:3,:,:])
        if isinstance(lpips_, torch.Tensor): lpips_ = lpips_.item()
        return lpips_

    @staticmethod
    def resize(img : np.ndarray, size : tuple[int, int], dtype="float32", enable_blur : bool = True) -> np.ndarray:
        assert size[0] > 0
        assert size[1] > 0
        
        shape      = img.shape
        is_smaller = size[0] <= img.shape[0] or size[1] <= img.shape[1]
        inter_mode = cv2.INTER_AREA if is_smaller else cv2.INTER_LINEAR
        pre_blur   = is_smaller and enable_blur
        post_blur  = (not is_smaller) and enable_blur
        
        if pre_blur: img = cv2.GaussianBlur(img, (5, 5), 2)
        img_ = cv2.resize(img, size, interpolation=inter_mode)
        #if post_blur: img_ = cv2.GaussianBlur(img_, (5, 5), 2)
        
        return ImageUtils.set_array_to_image_format(img_, shape[2])

class TorchUtils:
    import lpips
    
    """Utility functions for all torch-related stuff.
    """
    LPIPS_MODEL = lpips.LPIPS(net='alex', verbose=False)

    @staticmethod
    def to_lpips(tensor : torch.Tensor) -> torch.Tensor:
        """Set a normalized tensor to the LPIPS range [-1, 1]. 

        Args:
            tensor (torch.Tensor): the tensor on which the operation will be performed.

        Returns:
            torch.Tensor: the new tensor.
        """
        return (tensor - 0.5)*2.0 
    
    @staticmethod
    def img_to_batch(tensor : torch.Tensor) -> torch.Tensor:
        """Converts a tensor to a batch-like tensor. An image batch has 4 dimensions: NxCxHxW.
        If the tensor has already 4 dimensions then, it returns it.
        If not, it adds the bacth dimension. (CxHxW -> 1xCxHxW)

        Args:
            tensor (torch.Tensor): the tensor on which the operation will be performed.

        Returns:
            torch.Tensor: the new tensor.
        """
        if len(tensor.shape) > 4: return tensor
        return tensor.reshape((1, *tensor.shape))

class SlangReflectionUtils:
    """
    .. deprecated::
    """
    
    @staticmethod
    def extract_uniforms_from_prog_layout(program_layout : spy.ProgramLayout, group_per_entry_point = True) -> dict:
        res = {}
        
        entry_points = program_layout.entry_points
        for idx_entry_point in range(len(entry_points)):
            e_p = entry_points[idx_entry_point]
            res.update(SlangReflectionUtils.extract_uniforms_from_entry_point_layout(e_p, group_per_entry_point = group_per_entry_point))

        return res


    @staticmethod
    def extract_uniforms_from_entry_point_layout(entry_point_layout : spy.EntryPointLayout, group_per_entry_point = True) -> dict:
        res = {}
        params = entry_point_layout.parameters
        for param in params:
            res.update(SlangReflectionUtils.extract_uniforms_from_params(param))
        if group_per_entry_point:
            return {entry_point_layout.name : res}
        else:
            return res
    
    @staticmethod
    def extract_uniforms_from_params(param : spy.VariableLayoutReflection) -> dict:
        param_name = param.name
        param_kind = param.type_layout.kind
        #print(param_name)
        #print(param_kind)
        
        match param_kind:
            case spy.TypeReflection.Kind.none:
                return {}
            case spy.TypeReflection.Kind.array:
                for param_ in param.type_layout.fields:
                    return {param_name : SlangReflectionUtils.extract_uniforms_from_params(param_)}
                
            case spy.TypeReflection.Kind.constant_buffer:
                param_ = param.type_layout.element_type_layout
                return {param_name : SlangReflectionUtils.extract_uniforms_from_type(param_)}
            
            case spy.TypeReflection.Kind.parameter_block:
                param_ = param.type_layout.element_type_layout
                return {param_name : SlangReflectionUtils.extract_uniforms_from_type(param_)}
            
            case spy.TypeReflection.Kind.vector:
                return {param.name : list[SlangReflectionUtils.extract_uniforms_from_scalar(param.type_layout)]}
            
            case spy.TypeReflection.Kind.matrix:
                return {param.name : np.ndarray}
            
            case spy.TypeReflection.Kind.scalar:
                return {param.name :SlangReflectionUtils.extract_uniforms_from_scalar(param.type_layout)}
            
            case spy.TypeReflection.Kind.resource:
                return {param.name : spy.Texture}
            
            case spy.TypeReflection.Kind.sampler_state:
                return {param.name : spy.Sampler}
            
            case spy.TypeReflection.Kind.struct:
                res = {}
                for param_ in param.type_layout.fields: 
                    res.update(SlangReflectionUtils.extract_uniforms_from_params(param_))
                return {param.name : res}
        
        raise TypeError('Undefined Slang variable type.')
        return {}
    
    @staticmethod
    def extract_uniforms_from_type(param : spy.TypeLayoutReflection):
        param_name = param.name
        param_kind = param.kind
        
        match param_kind:
            case spy.TypeReflection.Kind.none:
                return type(None)
            
            case spy.TypeReflection.Kind.array:
                return SlangReflectionUtils.extract_uniforms_from_scalar(param)
            
            case spy.TypeReflection.Kind.vector:
                return SlangReflectionUtils.extract_uniforms_from_scalar(param)
            
            case spy.TypeReflection.Kind.matrix:
                return np.ndarray
            
            case spy.TypeReflection.Kind.scalar:
                return SlangReflectionUtils.extract_uniforms_from_scalar(param)
            
            case spy.TypeReflection.Kind.struct:
                res = {}
                for param_ in param.fields: 
                    res.update(SlangReflectionUtils.extract_uniforms_from_params(param_))
                return res
        
        raise TypeError('Undefined Slang type.')
        return type(None)

    @staticmethod
    def extract_uniforms_from_scalar(param : spy.TypeLayoutReflection) -> type:
        param.type.kind
        spy.TypeReflection.Kind.texture_buffer
        spy.TypeReflection.Kind.sampler_state
        match param.type.scalar_type:
            case spy.TypeReflection.ScalarType.bool:
                return bool
            
            case spy.TypeReflection.ScalarType.int16:
                return int
            
            case spy.TypeReflection.ScalarType.int32:
                return int
            
            case spy.TypeReflection.ScalarType.int64:
                return int
            
            case spy.TypeReflection.ScalarType.int8:
                return int
            
            case spy.TypeReflection.ScalarType.float16:
                return float
            
            case spy.TypeReflection.ScalarType.float32:
                return float
            
            case spy.TypeReflection.ScalarType.float64:
                return float
            
            case spy.TypeReflection.ScalarType.uint8:
                return int
            
            case spy.TypeReflection.ScalarType.uint16:
                return int
            
            case spy.TypeReflection.ScalarType.uint32:
                return int
            
            case spy.TypeReflection.ScalarType.uint64:
                return int
            
            case spy.TypeReflection.ScalarType.void:
                return type(None)
            
            case spy.TypeReflection.ScalarType.none:
                return type(None)
            
        raise TypeError('Undefined Slang scalar type.')
        return type(None)
    

    @staticmethod
    def check_type_uniforms(uniforms : dict, typed_uniform_must : dict) -> bool:
        #print(uniforms, typed_uniform_must)
        for key in uniforms.keys():
            if(not key in typed_uniform_must):
                raise KeyError('Uniform not found in template.')
            
            if (type(uniforms[key]) is dict) and (type(typed_uniform_must[key]) is dict):
                if(not SlangReflectionUtils.check_type_uniforms(uniforms[key], typed_uniform_must[key])):
                    return False

            elif (type(uniforms[key]) is _TypedDictMeta) and (type(typed_uniform_must[key]) is dict):
                if(not SlangReflectionUtils.check_type_uniforms(uniforms[key].__annotations__, typed_uniform_must[key])): 
                    return False

            if type(uniforms[key]) == type(uniforms[key]):
                return uniforms[key] == uniforms[key]

            elif type(uniforms[key]) == UnionType and type(uniforms[key]) == type:
                return issubclass(typed_uniform_must[key], uniforms[key])
            
            elif type(uniforms[key]) == type and type(uniforms[key]) == UnionType:
                return issubclass(uniforms[key], typed_uniform_must[key])
            
        return True

class SpaceTransformerUtils:
    """Class for utility functions to compute [Rt] matrix.
    """

    RADPERDEG = 0.01745329252

    Rot_T     = tuple[float, float, float] 
    Pos_T     = tuple[float, float, float]

    @staticmethod
    def rotation_matrix_to_euler_angles(R : np.ndarray) -> Rot_T:
        yaw   = 0.0
        pitch = 0.0
        roll  = 0.0
        
        def AllmostZero(v):
            return abs(v) < 1e-7
        
        if AllmostZero(R[0,0]) and AllmostZero( R[1,0] ) :
            yaw = np.arctan2( R[1,2], R[0,2] )
            if R[2,0] < 0.0:
                pitch = np.pi/2
            else:
                pitch = -np.pi/2
            roll = 0.0
        else:
            yaw = np.arctan2( R[1,0], R[0,0] )
            if AllmostZero( R[0,0] ) :
                pitch = np.arctan2( -R[2,0], R[1,0] / np.sin(yaw) )
            else:
                pitch = np.arctan2( -R[2,0], R[0,0] / np.cos(yaw) )
            
            roll = np.arctan2( R[2,1], R[2,2] )
        
        euler = np.array( [yaw, pitch, roll] )
        
        euler = np.rad2deg(euler)
        return (euler[0], euler[1], euler[2])
    
    @staticmethod
    def transform_point_from_to(point : Pos_T, pos_cam : Pos_T, rot_cam : Rot_T, pos_cam_to : Pos_T, rot_cam_to : Rot_T) -> np.ndarray:
        """Trasform a point from a camera system coordinate to another.

        Args:
            point (Pos_T): The point to apply the transformation to.
            pos_cam (Pos_T): Position of the input camera.
            rot_cam (Rot_T): Rotation of the input camera.
            pos_cam_to (Pos_T): Position of the target camera.
            rot_cam_to (Rot_T): Rotation of the target camera.

        Returns:
            np.ndarray: Transformerd point.
        """
        R_i = SpaceTransformerUtils.get_rotation_matrix_from_to(rot_cam, rot_cam_to)
        t_i = SpaceTransformerUtils.get_translation_matrix_from_to(pos_cam, pos_cam_to, rot_cam_to)

        point_cam_to = np.matmul(R_i, point) + t_i
        return point_cam_to

    @staticmethod
    def pinhole_unproject_point(point : Pos_T, focal : tuple[float,float], pp : tuple[float,float] = (0, 0)) -> Pos_T:
        """Unproject a point from a pinhole camera.

        Args:
            point (Pos_T): The XYZ coordinate of the point.
            focal (tuple[float,float]): Focal length of the pinhole.
            pp (tuple[float,float]): Principal point of the pinhole.

        Returns:
            Pos_T: Unprojected point.
        """
        return (
            point[0],
            - (point[1] - pp[0]) * point[0] / focal[0],
            - (point[2] - pp[1]) * point[0] / focal[1],
        ) # z, y, x

    @staticmethod
    def get_rotation_matrix_from_to(rot_1 : tuple[float, float, float], rot_2 : tuple[float, float, float]) -> np.ndarray:
        r"""Compute the rotation 3x3 matrix, i.e., R, from one point to another.
        $R = R_2^T \cdot R_1$
        Args:
            rot_1 (tuple[float, float, float]): Rotation angles of point 1.
            rot_2 (tuple[float, float, float]): Rotation angles of point 2.

        Returns:
            np.ndarray: The 3x3 rotation matrix.
        """
        rot_mat_1 = SpaceTransformerUtils.get_rotation_matrix(rot_1)
        rot_mat_2 = SpaceTransformerUtils.get_rotation_matrix(rot_2)
        
        #return rot_mat_2.T * rot_mat_1
        return np.matmul(rot_mat_2.T, rot_mat_1)

    @staticmethod
    def get_translation_matrix_from_to(pos_1 : tuple[float, float, float], pos_2 : tuple[float, float, float], rot_2 : tuple[float, float, float]) -> np.ndarray:
        r"""Compute the 3x1 translation vector (i.e., t) from one point to another.
        $t = -R_2^T \cdot (p_1 - p_2)$

        Args:
            pos_1 (tuple[float, float, float]): Position vector of point 1
            pos_2 (tuple[float, float, float]): Position vector of point 2
            rot_2 (tuple[float, float, float]): Rotation angles of point 1

        Returns:
            np.ndarray: The 3x1 translation vector.
        """
        pos_vec_1 = np.array(pos_1)
        pos_vec_2 = np.array(pos_2)
        rot_mat_2 = SpaceTransformerUtils.get_rotation_matrix(rot_2)
        
        #return -rot_mat_2.T * (pos_vec_1 - pos_vec_2)
        return np.matmul(-rot_mat_2.T, pos_vec_1 - pos_vec_2)
    
    @staticmethod
    def get_rotation_angles_from_rot_mat(rot_mat : np.ndarray) -> Rot_T:
        rot_angles : SpaceTransformerUtils.Rot_T = (0, 0, 0)
        
        r_32 = rot_mat[2,1].item()
        r_33 = rot_mat[2,2].item()
        r_31 = rot_mat[2,0].item()
        r_21 = rot_mat[1,0].item()
        r_11 = rot_mat[0,0].item()
        
        euler_x = atan2(r_32, r_33)
        euler_y = atan2(-r_31, (r_32**2 + r_33**2)**0.5)
        euler_z = atan2(r_21, r_11)
        
        rot_angles = (
            euler_z / SpaceTransformerUtils.RADPERDEG,
            euler_y / SpaceTransformerUtils.RADPERDEG,
            euler_x / SpaceTransformerUtils.RADPERDEG,
        )
        
        return rot_angles
        
    
    @staticmethod
    def get_rotation_matrix(rot : tuple[float, float, float]) -> np.ndarray:
        """Compute rotation matrix from the world to the point.

        Args:
            rot (tuple[float, float, float]): Rotation angles

        Returns:
            np.ndarray: The 3x3 rotation matrix.
        """
        return SpaceTransformerUtils.euler_angles_to_rot_matrix(SpaceTransformerUtils.RADPERDEG * np.array(rot))
    
    @staticmethod
    def euler_angles_to_rot_matrix(rotation : np.ndarray) -> np.ndarray:
        return np.matmul(
            SpaceTransformerUtils.rotation_matrix_from_rotation_around_z(rotation[0]),
            np.matmul(
                SpaceTransformerUtils.rotation_matrix_from_rotation_around_y(rotation[1]), 
                SpaceTransformerUtils.rotation_matrix_from_rotation_around_x(rotation[2])
                )
        )

    @staticmethod
    def rotation_matrix_from_rotation_around_x(rx : float) -> np.ndarray:
        return np.array(
            [
				[1.0, 0.0, 0.0],
				[0.0, cos(rx), -sin(rx)],
				[0.0, sin(rx), cos(rx)]
            ], dtype="float32"
        )

    @staticmethod
    def rotation_matrix_from_rotation_around_y(ry : float) -> np.ndarray:
        return np.array(
            [
				[cos(ry), 0.0, sin(ry)],
				[0.0, 1.0, 0.0],
				[-sin(ry), 0.0, cos(ry)]
            ], dtype="float32"
        )

    @staticmethod
    def rotation_matrix_from_rotation_around_z(rz : float) -> np.ndarray:
        return np.array(
            [
				[cos(rz), -sin(rz), 0.0],
				[sin(rz), cos(rz), 0.0],
				[0.0, 0.0, 1.0]
            ], dtype="float32"
        )

class UniformUtils:
    """Utility functions for applying uniforms to Slang shader objects."""

    @staticmethod
    def apply_uniforms_to_shader(program_layout : spy.ProgramLayout, sh_o : spy.ShaderObject, uniforms : TypeUtils.UniformGPUData_t):
        """
        Send the CPU uniform data to the shader program.
        The uniform data is described by dictionnaries where the keys (str) are the variable names and the values are the data.
        The values can be themself an uniform dict. This is used to send struct data to the shader program.

        This function goes through the entry points of the program and check if the parameter is *available* in the uniform dict.
        If so, we send the uniform value at the right uniform shader location. 

        Args:
            program_layout (spy.ProgramLayout): The Slangpy program layout. It gives information about the entry point names, the required uniforms, etc.
            sh_o (spy.ShaderObject): The slangpy shader object from which the uniforms are sent/written.
            uniforms (TypeUtils.UniformGPUData_t): The uniform table.
        """
        cursor = spy.ShaderCursor(sh_o)
        
        entry_points = program_layout.entry_points
        
        for idx_entry_point in range(len(entry_points)):
            param_names = [param.name for param in entry_points[idx_entry_point].parameters]
            uniforms_ = param_names & uniforms.keys()
            for param in uniforms_:
                Logger.debug(f"Uniform match: {param}", str(UniformUtils))
                cursor.find_entry_point(idx_entry_point).find_field(param).write(uniforms[param])


class IOUtils:
    """Utility functions for file and directory operations."""

    @staticmethod
    def do_file_exist(fname : str) -> bool:
        """Check if the filename exists.

        Args:
            fname (str): Path to the file.

        Raises:
            FileNotFoundError: If the file is not found or does not exist.

        Returns:
            bool: always return True if it succeeds. If not, it raises an Error.
        """
        if(not pathlib.Path.exists(pathlib.Path(fname).parent.resolve())):
            raise FileNotFoundError(f'File path does not exist. fname: {fname}')
        return True
    
    @staticmethod
    def create_dirs(dir_name : str):
        """Create all directories. Equivalent to mkdirs.

        Args:
            fdir (str): Path to the directory name.
        """
        pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def find_files_in_dir(dir_name : str, fname_regex : str) -> list[str]:
        """Find all the files that match the given regex present in the directory.

        Args:
            dir_name (str): Directory where to find the files.
            fname_regex (str): Regex string that the files should match.

        Returns:
            list[str]: List of paths of all the files that match the regex.
        """
        regex = re.compile(fname_regex)
        filenames = []
        for root, dirs, files in os.walk(dir_name):
            for file in files:
                if regex.match(file):
                    filenames.append(file)
        return filenames
    
    @staticmethod
    def read_toml(path : str) -> dict:
        """Read a TOML file.
        First, check if the file exists.

        Args:
            path (str): path to the TOML file.

        Returns:
            dict: TOML data.
        """
        IOUtils.do_file_exist(path)

        with open(path, 'rb') as file:
            data = tomllib.load(file)
        return data
    
    @staticmethod
    def save_json(path :str, data : dict):
        """Save data in the json format.
        First, check if the file exists.

        Args:
            path (str): Path where to write the JSON file.
            data (dict): JSON data.
        """
        IOUtils.do_file_exist(path)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_json(path : str) -> dict:
        """Load a JSON file as a dict.
        First, check if the file exists.

        Args:
            path (str): Path to the JSON file.

        Returns:
            dict: JSON data.
        """
        IOUtils.do_file_exist(path)
        with open(path, "r") as f:
            return json.load(f)

class RandomUtils:
    """Utility functions related to random generation.
    """

    @staticmethod
    def set_seed(seed : int):
        """Set the random seed of Python and numpy.

        Args:
            seed (int): seed.
        """
        random.seed(seed)
        np.random.seed(seed)

class Singleton(type):
    """Singleton class that can later be used to define a class as a singleton.
    """

    _instances = {}
    def __call__(cls, *args, **kwargs):
        """Create a new class instance if the class does not exist yet in the directories of _instances.

        Returns:
            _type_: Return the instance of the desired class.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            Logger.debug(f"CREATE SINGLETON: {cls._instances[cls].__class__}", str(cls.__class__))
        return cls._instances[cls]


class GPUUtils:
    """Utility functions for the GPU.
    """

    @staticmethod
    def print_info(device : spy.Device):
        """Print information about a slangy GPU.

        Args:
            device (spy.Device): Slangpy device reflecting (most of the times) a GPU.
        """
        desc = device.desc
        info = device.info
        
        Logger.info("-"*10 + " Graphics Info" + "-"*10)
        Logger.info(f"Running Graphics API: {info.api_name}")
        Logger.info(f"Type Device: {desc.type}")
        Logger.info("-"*10 + " GPU Info" + "-"*10)
        Logger.info(f"GPU Name: {info.adapter_name}")
        #Logger.info(f"GPU GHz: {str(info.timestamp_frequency / 1e6)}")
        

class LogType(Enum):
    """List of log types. 
    It can be used to log information at a certain level.
    """
    NORMAL  = 0
    DEBUG   = 1
    INFO    = 2
    WARNING = 3
    ERROR   = 4
    RESULT  = 5
    STATE   = 6

class LogLevel(Enum):
    """List of log levels.
    If ALL, it logs all the logs.
    If NORMAL, it does not log DEBUG log type.
    If NO_STATE, it does not log the STATE logs (used for optimization).
    """
    NONE    = 0
    ALL     = 1
    NORMAL  = 2
    NO_STATE= 3

class PrintColors(Enum):
    """Color Definitions for colored prints.
    """
    PURPLE = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[33m'

class Logger(metaclass = Singleton):
    """Singleton-based logger with color-coded output and log levels."""

    _ASCII_ART = """
                            .:::::::::::..         .:::::.  .::::::::::::..      .::::::::::::.     
             .:-=-:.       =%%%%%%%%%%%%%%+.      -%%%%%.  :%%%%%%%%%%%%%%#:.   .%%%%%%%%%%%%%%#:.  
           .#%%%##%%%+.    =%%%%%%%%%%%%%%%%*.    -%%%%%.  :%%%%%%%%%%%%%%%%*.  .%%%%%%%%%%%%%%%%*. 
            ...    .+%%-.  =%%%%%.....:*%%%%%%.   -%%%%%.  :%%%%%:  ..:%%%%%%:  .%%%%%=.....*%%%%%+.
                     *%%:  =%%%%%.      .#%%%%#.  -%%%%%.  :%%%%%:     =%%%%%.  .%%%%%=     .#%%%%%.
                     :%%#. =%%%%%.       :%%%%%-  -%%%%%.  :%%%%%:  ..=%%%%%:.  .%%%%%=     .#%%%%#.
             .=####=.:%%%. =%%%%%.       .%%%%%=  -%%%%%.  :%%%%%%%%%%%%%+:.    .%%%%%+---=*%%%%%%- 
           .*%%%+::=#%%%%. =%%%%%.       .%%%%%=  -%%%%%.  :%%%%%%%%%%%%%%%%=.  .%%%%%%%%%%%%%%%%-. 
          .#%%#.    .%%%#. =%%%%%.       :%%%%%:  -%%%%%.  :%%%%%:    .:%%%%%*. .%%%%%%%%%%%%%%:.   
          =%%%.     .%%%-  =%%%%%.      .%%%%%#.  -%%%%%.  :%%%%%:      *%%%%%. .%%%%%=  .#%%%%#.   
          +%%%.    .*%%#.  =%%%%%:::::=#%%%%%#.   -%%%%%.  :%%%%%:....:+%%%%%#. .%%%%%=   .#%%%%#.  
          .%%%.   .=%%%.   =%%%%%%%%%%%%%%%%+.    -%%%%%.  :%%%%%%%%%%%%%%%%%:. .%%%%%=    .#%%%%#. 
           .#%#-.-#%%*.    =%%%%%%%%%%%%%#-.      -%%%%%.  :%%%%%%%%%%%%%%%=.   .%%%%%=     .#%%%%%.
              :+*+=..                                                                   
    """

    _LICENSE_MSG = """
Copyright (C) 2026 "Université Libre de Bruxelles (ULB)". All rights reserved.

This program is licensed under "AGPL-3.0-or-later".
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact:
    - Armand Losfeld - armand.lfd.pro@proton.me
    - Daniele Bonatto - daniele.bonatto@ulb.be
    """
    
    def __init__(self, log_level : list[LogLevel] = [LogLevel.NORMAL]):
        """Set the log level to normal.

        Args:
            log_level (list[LogLevel], optional): desired log level. Defaults to [LogLevel.NORMAL].
        """
        self.log_level  = log_level
        self.start_t    = datetime.datetime.now()

        self._log(LogType.INFO , self._ASCII_ART + "\n" + self._LICENSE_MSG, PrintColors.OKBLUE)
    
    def _log(self, log_type : LogType, msg : str, log_color : PrintColors = PrintColors.ENDC):
        """Logs timing, the log type, and the message if the log type statisfied the log level.

        Args:
            log_type (LogType): Log type of the msg.
            msg (str): Msg to log.
            log_color (PrintColors, optional): Color to add to highlight the log type. Defaults to PrintColors.ENDC.
        """
        if (LogLevel.NONE in self.log_level):
            return

        if (
            ((LogLevel.NORMAL in self.log_level) and (log_type == LogType.DEBUG)) or
            ((LogLevel.NO_STATE in self.log_level) and (log_type == LogType.STATE)) 
            ):
            return
        
        print(f"[Time: {((datetime.datetime.now() - self.start_t))}] | LOGGER [{log_color.value} {log_type.name} {PrintColors.ENDC.value}]: {msg}")

    @staticmethod
    def log(log_type : LogType, msg : str, log_color : PrintColors = PrintColors.ENDC):
        """Base static function to log message without keeping a Logger instance.

        Args:
            log_type (LogType): Log type of the msg.
            msg (str): Msg to log.
            log_color (PrintColors, optional): Color to add to highlight the log type. Defaults to PrintColors.ENDC.
        """
        Logger()._log(log_type, msg, log_color=log_color)

    @staticmethod
    def debug(msg : str, class_name : str = "UNKNOWN"):
        Logger.log(LogType.DEBUG, f"{class_name} - {msg}", PrintColors.PURPLE)

    @staticmethod
    def info(msg : str):
        Logger.log(LogType.INFO, msg, PrintColors.OKBLUE)

    @staticmethod
    def result(msg : str):
        Logger.log(LogType.RESULT, msg, PrintColors.OKGREEN)
    
    @staticmethod
    def normal(msg : str):
        Logger.log(LogType.NORMAL, msg, PrintColors.ENDC)

    @staticmethod
    def warning(msg : str):
        Logger.log(LogType.WARNING, msg, PrintColors.YELLOW)

    @staticmethod
    def error(msg : str):
        Logger.log(LogType.ERROR, msg, PrintColors.FAIL)

    @staticmethod
    def state(msg : str):
        Logger.log(LogType.STATE, msg, PrintColors.OKCYAN)

class MPEGJSONUtils:
    """Utility functions and definitions for the MPEG JSON camera file.
    """

    HEADER_CAMERA_JSON = {
        "Version": "2.0",
        "Content_name": "cameras.json",
        "BoundingBox_center": [0, 0, 0],
        "Fps": 1,
        "Frames_number": 1,
        "Informative": {
            "Converted_by": "VSRS_to_JSON.m",
            "Original_units": "mm",
            "New_units": "m"
        },
    }

class PAMUtils:
    Point   = list[Union[float, int]]
    Points  = np.ndarray
    Idx     = int

    class Cluster:
        Idx     = int

        def __init__(self, medoid_idx : Idx):
            self.medoid_idx = medoid_idx
            self.point_indices = []

        def add_point(self, point_idx : Idx):
            self.point_indices.append(point_idx)

        def __repr__(self):
            return f"Cluster(medoid_idx={self.medoid_idx}, point_indices={self.point_indices})"


    class DistanceMetric(Enum):
        EUCLIDEAN = 1
        MANHATTAN = 2

    @staticmethod
    def compute_distance_matrix(points : Points, metric=DistanceMetric.EUCLIDEAN) -> np.ndarray:
        """
        Compute the distance matrix for a list of points.

        Args:
            points: List of camera positions (each position is a list or array of coordinates).
            metric: DistanceMetric enum (EUCLIDEAN or MANHATTAN).

        Returns:
            A 2D numpy array where D[i,j] is the distance between points[i] and points[j].
        """
        n = len(points)
        D = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if metric == PAMUtils.DistanceMetric.EUCLIDEAN:
                    D[i, j] = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                elif metric == PAMUtils.DistanceMetric.MANHATTAN:
                    D[i, j] = np.sum(np.abs(np.array(points[i]) - np.array(points[j])))
                else:
                    raise ValueError("Unsupported distance metric.")

        return D

    @staticmethod
    def build_initial_medoids(points : Points, k : int, distance_matrix : np.ndarray) -> list[Idx]:
        """
        Select initial medoids using the BUILD phase of the PAM algorithm.

        Args:
            points: List of camera positions.
            k: Number of clusters (medoids to select).
            distance_matrix: Precomputed distance matrix.

        Returns:
            List of indices of the initial medoids.
        """
        n = len(points)
        medoids = []

        # Step 1: Select the first medoid (minimizes total distance)
        total_distances = np.sum(distance_matrix, axis=1)
        first_medoid_idx = np.argmin(total_distances)
        medoids.append(first_medoid_idx)

        # Step 2: Select remaining medoids
        for _ in range(1, k):
            best_reduction = -1
            best_medoid_idx = -1

            for candidate_idx in range(n):
                if candidate_idx in medoids:
                    continue  # Skip already selected medoids

                # Calculate the reduction in total distance if candidate_idx is chosen
                reduction = 0
                for i in range(n):
                    current_dist = min(distance_matrix[i, medoid] for medoid in medoids)
                    new_dist = min(current_dist, distance_matrix[i, candidate_idx])
                    reduction += (current_dist - new_dist)

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_medoid_idx = candidate_idx

            medoids.append(best_medoid_idx)

        return medoids
    
    @staticmethod
    def assign_points_to_medoids(points : Points, medoids : list[Idx], distance_matrix : np.ndarray) -> list[Cluster]:
        """
        Assign each point to the nearest medoid.

        Args:
            points: List of camera positions.
            medoids: List of medoid indices.
            distance_matrix: Precomputed distance matrix.

        Returns:
            List of Cluster objects.
        """
        clusters = [PAMUtils.Cluster(medoid_idx) for medoid_idx in medoids]

        for point_idx in range(len(points)):
            # Find the nearest medoid
            nearest_medoid_idx = medoids[0]
            min_distance = distance_matrix[point_idx, nearest_medoid_idx]

            for medoid_idx in medoids[1:]:
                dist = distance_matrix[point_idx, medoid_idx]
                if dist < min_distance:
                    min_distance = dist
                    nearest_medoid_idx = medoid_idx

            # Assign the point to the nearest medoid's cluster
            for cluster in clusters:
                if cluster.medoid_idx == nearest_medoid_idx:
                    cluster.add_point(point_idx)
                    break

        return clusters
    
    @staticmethod
    def compute_total_cost(clusters : list[Cluster], distance_matrix : np.ndarray) -> float:
        """
        Compute the total cost (sum of distances from each point to its medoid).

        Args:
            clusters: List of Cluster objects.
            distance_matrix: Precomputed distance matrix.

        Returns:
            Total cost (float).
        """
        total_cost = 0.0

        for cluster in clusters:
            medoid_idx = cluster.medoid_idx
            for point_idx in cluster.point_indices:
                total_cost += distance_matrix[point_idx, medoid_idx]

        return total_cost
    
    @staticmethod
    def optimize_medoids(points : Points, clusters : list[Cluster], distance_matrix : np.ndarray, max_iter : int = 100) -> list[Cluster]:
        """
        Optimize medoids by swapping to reduce total cost.

        Args:
            points: List of camera positions.
            clusters: List of Cluster objects.
            distance_matrix: Precomputed distance matrix.
            max_iter: Maximum number of iterations.

        Returns:
            List of optimized Cluster objects.
        """
        n = len(points)
        k = len(clusters)
        medoids = [cluster.medoid_idx for cluster in clusters]
        improved = True
        iter_count = 0

        while improved and iter_count < max_iter:
            improved = False
            best_cost = PAMUtils.compute_total_cost(clusters, distance_matrix)

            for m in range(k):
                current_medoid = medoids[m]
                for h in range(n):
                    if h in medoids:
                        continue  # Skip current medoids

                    # Temporarily swap medoid m with point h
                    medoids[m] = h
                    # Reassign points to nearest medoid
                    new_clusters = PAMUtils.assign_points_to_medoids(points, medoids, distance_matrix)
                    new_cost = PAMUtils.compute_total_cost(new_clusters, distance_matrix)

                    # If cost improves, keep the swap
                    if new_cost < best_cost:
                        best_cost = new_cost
                        clusters = new_clusters
                        improved = True
                    else:
                        # Revert the swap
                        medoids[m] = current_medoid

            iter_count += 1

        Logger.debug(f"PAM SWAP phase converged after {iter_count} iterations.", str(PAMUtils))

        return clusters
    
    @staticmethod
    def WCSS(clusters : list[Cluster], distance_matrix : np.ndarray) -> float:
        """_summary_

        Args:
            clusters (list[Cluster]): List of clusters.
            distance_matrix (np.ndarray): Distance matrix where d[i,j] is the distance between the point i and j.

        Returns:
            float: The WCSS metric of the clustering/partioning.
        """
        wcss_ = 0
        for cluster in clusters:
            for p_idx in cluster.point_indices:
                wcss_ += distance_matrix[p_idx, cluster.medoid_idx].item()
        return wcss_
    
    @staticmethod
    def fit(points : Points, k : int, metric=DistanceMetric.EUCLIDEAN, max_iter:int=100) -> tuple[list[Cluster], list[Idx], np.ndarray]:
        D = PAMUtils.compute_distance_matrix(points, metric)
        medoids     = PAMUtils.build_initial_medoids(points, k, D)
        clusters    = PAMUtils.assign_points_to_medoids(points, medoids, D)
        clusters    = PAMUtils.optimize_medoids(points, clusters, D, max_iter)
        return clusters, medoids, D
    
    @staticmethod
    def plot(points : Points, clusters : list[Cluster]):
        # Plot
        colors = [(random.random(), random.random(), random.random()) for _ in range(len(clusters))]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i, cluster in enumerate(clusters):
            cluster_points = points[cluster.point_indices]
            
            if cluster_points.shape[1] < 3:
                zeros = np.zeros((cluster_points.shape[0], 3 - cluster_points.shape[1]))
                cluster_points = np.concatenate([cluster_points, zeros], axis=1)

            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                cluster_points[:, 2], # pyright: ignore[reportArgumentType]
                color=colors[i],
                label=f'Cluster {i+1}'
            )

            medoid_point = points[cluster.medoid_idx, :]
            if medoid_point.shape[0] < 3:
                zeros = np.zeros((3 - medoid_point.shape[0]))
                medoid_point = np.concatenate([medoid_point, zeros], axis=0)

            # Plot medoid
            ax.scatter(
                medoid_point[0],
                medoid_point[1],
                medoid_point[2],
                c='k',
                marker='x',
                linewidths=3
            )

        plt.title(f'PAM Clustering ({len(clusters)} clusters)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') # type: ignore
        ax.legend()
        plt.show()
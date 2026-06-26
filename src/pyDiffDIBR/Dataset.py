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
import os
import re
from glob import glob
import numpy as np
from typing import Mapping, Any, Literal, Sequence

from ._utils import ImageUtils, Logger, ArrayUtils, SpaceTransformerUtils, MPEGJSONUtils, IOUtils, PAMUtils, StrUtils, DepthEstimatorUtils
from .Image import Image
from .Camera import Camera


class Dataset:
    """Core dataset class managing a collection of cameras."""

    def __init__(self, cameras: list[Camera], batch_size: int = 1, shuffle: bool = True):
        """
        Args:
            cameras (list[Camera]): List of cameras composing the dataset.
            batch_size (int, optional): the batch size; when ’batch()’ is called it will return this number of camera. Defaults to 1.
            shuffle (bool, optional): if true, when ’batch()’ is called it will return N random cameras. If not, then it returns the N sequential cameras. Defaults to True.
        """
        self.cameras = cameras
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reset_permutation()

        for info in self.info: Logger.debug(info, str(self.__class__))

    @property
    def info(self) -> list[str]:
        """Returns a list of information about the dataset.
        It is used for debbuging purposes.

        Returns:
            list[str]: List of information.
        """
        info = []
        info.append("-"*5 + "Dataset" +"-"*5)
        info.append(f"Nb Cams: {len(self.cameras)}")
        info.append(f"Batch size: {self.batch_size}")
        return info
    
    def reset_permutation(self):
        """Reset the permutation of camera index used for extracting a batch from it.
        """
        self.permutation = np.random.permutation(len(self.cameras)) if self.shuffle else np.arange(len(self.cameras))
        self.current_idx = 0

    def batch(self) -> list[Camera] | None:
        """Return next batch of cameras.

        Returns:
            list[Camera] | None: batch of cameras.
        """
        if self.current_idx >= len(self.cameras):
            return None
        start = self.current_idx
        end = min(start + self.batch_size, len(self.cameras))
        self.current_idx = end
        return [self.cameras[i] for i in self.permutation[start:end]]

    def __len__(self):
        return len(self.cameras)

    def get_camera_by_name(self, name: str) -> Camera:
        """Retrieve a camera by name.

        Args:
            name (str): camera name.

        Raises:
            KeyError: If no camera was found.

        Returns:
            Camera: The camera if found.
        """
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(f"Camera '{name}' not found.")

    def __iter__(self):
        yield from list(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]

    def to_json(self) -> dict[str, Any]:
        """Returns the dataset as json dict following the "camera" json file format of MPEG.
        It also adds the MPEG json headers (e.g., 'Version', etc.).

        Returns:
            dict[str, Any]: MPEG json dict.
        """
        json_str = MPEGJSONUtils.HEADER_CAMERA_JSON
        json_str["cameras"] = [cam.cam_params for cam in self.cameras]
        json_str["cameras"].sort(key = lambda camera: camera["Name"]) # Sort camera, to always have same output
        return json_str

class ColorDataset:
    """Color dataset class. This should only be used to load a dataset of only color images that will be lated transformed to a RGBD dataset."""

    def __init__(self, cameras: list[Camera], batch_size: int = 1, shuffle: bool = True):
        """
        Args:
            cameras (list[Camera]): List of cameras composing the dataset.
        """
        self.cameras = cameras
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def _load_depth_from_cache(self, path : str, resolution : tuple[int, int], depth_range : tuple[float, float]) -> np.ndarray:
        disp = ImageUtils.read_img(path, resolution, convert_2_rgb = False)
        if disp.shape[2] > 1:
            disp = disp[:,:,0:1] 
        depth = ImageUtils.convert_normalized_mpeg_disparity_to_depth(disp, depth_range[0], depth_range[1])
        return depth
    
    def _generate_depths_from_colors(self, cameras : list[Camera], depths : Sequence[np.ndarray | None], estimation_method : DepthEstimatorUtils.Method = DepthEstimatorUtils.Method.DepthAnything3, sigmoid : bool = True) -> list[np.ndarray]:
        assert len(cameras) == len(depths), "Cameras and initial depths should have same size"
        
        colors : list[np.ndarray] = []
        params : list[Mapping] = []
        for idx in range(len(cameras)):
            if depths[idx] is not None:
                continue    
            color_arr = cameras[idx].image_pair.color.arr
            if sigmoid: color_arr = ArrayUtils.sigmoid(color_arr)    
            colors.append(color_arr)
            params.append(cameras[idx].cam_params)
        
        if len(colors) == 0:
            # Depth is already loaded
            return depths #type: ignore
        
        estimated_depths = DepthEstimatorUtils.from_arrays(colors, out_dir=None, estimation_method = estimation_method, parameters=params) # Save it later in MPEG disp encoding
        if len(estimated_depths) != len(colors):
            raise ValueError("Not same number of estimated depths and color inputs.")
        all_depths = []
        idx_ = 0
        for idx in range(len(cameras)):
            all_depths.append(depths[idx])
            if depths[idx] is None:
                all_depths[idx] = estimated_depths[idx_] * DepthEstimatorUtils._DEPTH_SCALE_MM_TO_METERS
                idx_ += 1
        return all_depths
    
    def _cache_depth(self, path : str, depth_arr : np.ndarray, depth_range : tuple[float, float]):
        IOUtils.create_dirs(os.path.dirname(path))
        
        disp = ImageUtils.convert_depth_to_normalized_mpeg_disparity(depth_arr, depth_range[0], depth_range[1])
        ImageUtils.save_img(disp, path, 1, out_dtype='uint16')
    
    def to_rgbd(self, estimation_method : DepthEstimatorUtils.Method = DepthEstimatorUtils.Method.DepthAnything3, batch_size : int | None = None, shuffle : bool | None = None, depth_paths : list[str] | None = None, ignore_cache : bool = False, inv_sigmoid : bool = False) -> Dataset:
        
        cameras = self.cameras
        depths : list[np.ndarray | None] = [None for _ in range(len(cameras))]
        
        # ---------------------- LOAD CACHED DEPTH MAP ---------------------- 
        if depth_paths is not None and not ignore_cache:
            for i in range(len(cameras)):
                depth_path : str = depth_paths[i]
                if os.path.exists(depth_path) and not ignore_cache:
                    depths[i] = self._load_depth_from_cache(depth_path, cameras[i].cpu_params.resolution, cameras[i].cpu_params.depth_range)
        
        # ---------------------- GENERATE NEW DEPTH MAP ----------------------
        all_depths = self._generate_depths_from_colors(cameras, depths, estimation_method, sigmoid=inv_sigmoid) 
        
        #----------------------- CHECK IF DEPTH IS CORRECT ----------------------- 
        for idx, d in enumerate(all_depths): all_depths[idx] = np.clip(all_depths[idx], cameras[idx].cpu_params.depth_range[0] * 1.1, cameras[idx].cpu_params.depth_range[1] * 0.9)
        
        # ------------------------ CACHE DEPTH MAP ------------------------
        if depth_paths is not None:
            for i in range(len(cameras)):
                if depths[i] is None:
                    self._cache_depth(depth_paths[i], all_depths[i], cameras[i].cpu_params.depth_range)
        
        # ------------------------ Processing DEPTH MAP ------------------------
        for i in range(len(cameras)):
            depth = all_depths[i]
            if inv_sigmoid:
                depth = ArrayUtils.inv_sigmoid(depth, shift = cameras[i].cpu_params.depth_range[0], scale = cameras[i].cpu_params.depth_range[1] - cameras[i].cpu_params.depth_range[0])
                
            cameras[i].image_pair.depth = Image(depth, cameras[i].image_pair.depth.requires_grad, cameras[i].image_pair.depth.name, cameras[i].image_pair.depth.always_reload, type_range = cameras[i].image_pair.depth.type_range)
            
        batch_size  = self.batch_size if batch_size is None else batch_size 
        shuffle     = self.shuffle if shuffle is None else shuffle 
        return Dataset(cameras, batch_size=batch_size, shuffle=shuffle)

class DatasetProcessingUtils:
    
    @staticmethod
    def downscale(dataset : Dataset, scale : float = 2.0) -> Dataset:
        new_cams : list[Camera] = []
        for cam in dataset:
            new_cam = cam.__copy__()
            new_cam.downscale(scale)
            new_cams.append(new_cam)
        return Dataset(new_cams, shuffle=dataset.shuffle, batch_size=dataset.batch_size)
        
    @staticmethod
    def upscale(dataset : Dataset, scale : float = 2.0) -> Dataset:
        new_cams : list[Camera] = []
        for cam in dataset:
            new_cam = cam.__copy__()
            new_cam.upscale(scale)
            new_cams.append(new_cam)
        return Dataset(new_cams, shuffle=dataset.shuffle, batch_size=dataset.batch_size)
    
class DatasetIO:
    """Handles dataset serialization and deserialization."""

    @staticmethod
    def save_to_json(dataset: Dataset, path: str):
        """Save the dataset into a MPEG dataset config file.

        Args:
            dataset (Dataset): the dataset.
            path (str): json filename.
        """
        IOUtils.save_json(path, dataset.to_json())

    @staticmethod
    def load_from_json(json_path: str, data_dict: dict[str, tuple[str, str]] | None, batch_size: int = 4, shuffle: bool = True) -> Dataset:
        """Loads a dataset from a json file and a dict of the camera names with their color and depth path.

        Args:
            json_path (str): MPEG json file path.
            data_dict (dict[str, tuple[str, str]]): dict where the keys are the camera names that must be loaded; the values are the paths to the color and depth image file.
            batch_size (int): The batch size of the returned dataset.
            shuffle (bool, optional): If true, the dataset is shuffled before picking a batch. Defaults to True.

        Raises:
            KeyError: If a camera of ’data_dict’ is not found in the dataset config file.

        Returns:
            Dataset: the dataset
        """
        params = IOUtils.load_json(json_path)["cameras"]

        camera_params= {}
        for c in range(len(params)): camera_params[params[c]["Name"]] = params[c]
        
        cameras = []
        data_dict_ = data_dict
        if not isinstance(data_dict_, dict):
            data_dict_ = {cam["Name"] : None for cam in params}

        for cam_name, image_paths in data_dict_.items():
            if cam_name not in camera_params:
                raise KeyError(f"Camera '{cam_name}' not found in JSON parameters")
                
            # Create camera with intrinsic parameters
            camera = Camera(cam_name, camera_params[cam_name], requires_grad=False, always_reload=False)

            yuv_format = [ImageUtils.YUVFormat.NONE, ImageUtils.YUVFormat.NONE]
            if ("BitDepthColor" in camera_params[cam_name].keys() and "ColorSpace" in camera_params[cam_name].keys()):
                yuv_format[0] = ImageUtils.get_yuv_fmt(camera_params[cam_name]["BitDepthColor"], camera_params[cam_name]["ColorSpace"])
            if ("BitDepthDepth" in camera_params[cam_name].keys() and "DepthColorSpace" in camera_params[cam_name].keys()):
                yuv_format[1] = ImageUtils.get_yuv_fmt(camera_params[cam_name]["BitDepthDepth"], camera_params[cam_name]["DepthColorSpace"])
            
            # Add all image pairs for this camera
            if isinstance(image_paths, list) or isinstance(image_paths, tuple): 
                color_path, depth_path = image_paths
                camera.load_image_pair(color_path, depth_path, yuv_format=(yuv_format[0],yuv_format[1]))
            else:
                camera.load_empty_pair()
            
            cameras.append(camera)

        return Dataset(cameras, batch_size, shuffle)
        
    @staticmethod
    def from_checkpoint(ckpt_dir : str, batch_size: int = 4, shuffle: bool = True, img_format : str = "trained_{}_{}.yuv", json_fname : str = "camera.json") -> Dataset:
        IOUtils.do_file_exist(ckpt_dir)
        
        color_files = [] 
        depth_files = []
        regex_str = r"trained_{}_(.*).*\..*"
        def get_cam_name(path : str, regex_str:str, idx_group = 0):
            #return path.split(f"{type}_")[1].split(split_str)[0]
            match_ =  re.search(regex_str, path)
            if match_ is None:
                raise RuntimeError("no match found")
            return match_.group(idx_group + 1)
        
        if StrUtils.get_file_extension(img_format) == 'yuv':
            img_format = StrUtils.build_yuvview_path(img_format)
        
            color_files = glob(os.path.join(ckpt_dir, img_format.format("color", "*", "*", "*", "*"))) # type, name, w, h, format
            depth_files = glob(os.path.join(ckpt_dir, img_format.format("depth", "*", "*", "*", "*")))
            regex_str = r"trained_{}_(.*)_(.*)x(.*)_(.*).*.\..*"
        else:
            color_files = glob(os.path.join(ckpt_dir, img_format.format("color", "*"))) # type, name
            depth_files = glob(os.path.join(ckpt_dir, img_format.format("depth", "*")))
            
        if len(color_files) != len(depth_files):
            raise RuntimeError("Checkpoint should have same number of color/depth images.")
        elif len(color_files) < 1:
            raise RuntimeError("Checkpoint is empty.")
        
        data_dict = {}
        for i in range(len(color_files)):
            color_file = color_files[i]
            color_cam_name = get_cam_name(color_file, regex_str.format('color'))
            
            found = False
            depth_f : str  = ""
            for j in range(len(depth_files)):
                depth_file = depth_files[j]
                depth_cam_name = get_cam_name(depth_file, regex_str.format('depth'))
                if (found := (color_cam_name == depth_cam_name)):
                    depth_f = depth_file
                    break
            if not found:
                raise RuntimeError(f"Could not find a depth image linked to the color image {color_file}.")
                
            data_dict[color_cam_name] = [color_file, depth_f]
        
        return DatasetIO.load_from_json(os.path.join(ckpt_dir, json_fname), data_dict, batch_size=batch_size, shuffle=shuffle)

class DatasetSplitUtils:
    """Utility for splitting datasets into subsets."""
    @staticmethod
    def split(dataset: Dataset, ratios: list[float]) -> list[Dataset]:
        """Split a dataset into N sub-datasets. This is particularly useful when you want to split a dataset into a training, validation, and testing set.

        Args:
            dataset (Dataset): The input full dataset.
            ratios (list[float]): Ratio size of each sub-dataset. The sum must be 1.0. 

        Raises:
            ValueError: If the ratio sizes do not sum to 1.0.

        Returns:
            list[Dataset]: _description_
        """
        if abs(sum(ratios) - 1.0) > 1e-4:
            raise ValueError("Ratios must sum to 1.0.")
        splits : list[Dataset] = []
        start = 0
        for ratio in ratios:
            end = start + int(ratio * len(dataset))
            subset = [dataset.cameras[i] for i in dataset.permutation[start:end]]
            splits.append(Dataset(subset, dataset.batch_size, dataset.shuffle))
            start = end
        return splits

class DatasetSpatialUtils:
    """Utility for spatial operations on cameras."""

    _MinMaxType = tuple[float,  float]
    
    @staticmethod
    def dist_cam_centers(cam_1 : Camera, cam_2 : Camera) -> float:
        return np.linalg.norm(np.array(cam_1.cpu_params.position) - np.array(cam_2.cpu_params.position)).item()
    
    @staticmethod
    def dist_principal_point_3D(cam_1 : Camera, cam_2 : Camera, sigmoid_p1 : bool = False, sigmoid_p2 : bool = False) -> float:
        p_1 = cam_1.cam_center(sigmoid=sigmoid_p1)
        p_1 = SpaceTransformerUtils.pinhole_unproject_point(p_1, cam_1.cpu_params.focal)
        p_1 = SpaceTransformerUtils.transform_point_from_to(p_1, cam_1.cpu_params.position, cam_1.cpu_params.rotation, cam_2.cpu_params.position, cam_2.cpu_params.rotation)

        p_2 = cam_2.cam_center(sigmoid=sigmoid_p2)
        p_2 = SpaceTransformerUtils.pinhole_unproject_point(p_2, cam_2.cpu_params.focal)

        return np.linalg.norm(np.array(p_1) - np.array(p_2)).item()

class DatasetCameraSelectorUtils:
    T_Ranks = dict[int, int] #cam_idx, rank 

    class SelectionBasedType(Enum):
        Camera_Distance     = 1
        Camera_Direction    = 2
        Principal_point_3D  = 3

    @staticmethod
    def select_input_views_from_cam_distance(
            input_cams : list[Camera], target_cam : Camera,
            nb_cams : int,
        ) -> list[int]:
        """_summary_

        Args:
            input_cams (list[Camera]): List of input cameras.
            target_cam (Camera): The target camera.
            nb_cams (int): The number of cameras to select.

        Returns:
            list[int]: Idx of the cameras to use.
        """
        return DatasetCameraSelectorUtils.select_from_rank(
            DatasetCameraSelectorUtils.rank_input_views_from_cam_distance(
                input_cams, target_cam,
            ), nb_cams
        )
        
    @staticmethod
    def select_input_views_from_3D(
            input_cams : list[Camera], target_cam : Camera,
            nb_cams : int,
            sigmoid_input : bool, sigmoid_target : bool,
        ) -> list[int]:
        """_summary_

        Args:
            input_cams (list[Camera]): List of input cameras.
            target_cam (Camera): The target camera.
            nb_cams (int): The number of cameras to select.

        Returns:
            list[int]: Idx of the cameras to use.
        """
        return DatasetCameraSelectorUtils.select_from_rank(
            DatasetCameraSelectorUtils.rank_input_views_from_principal_point_3D(
                input_cams, target_cam,
                sigmoid_input, sigmoid_target
            ), nb_cams
        )
    
    @staticmethod
    def select_input_views(
            input_cams : list[Camera], target_cam : Camera,
            nb_cams : int,
            sigmoid_input : bool, sigmoid_target : bool,
            selection_types : list[SelectionBasedType] = [SelectionBasedType.Camera_Distance, SelectionBasedType.Camera_Direction, SelectionBasedType.Principal_point_3D],
        ) -> list[int]:
        """_summary_

        Args:
            input_cams (list[Camera]): List of input cameras.
            target_cam (Camera): The target camera.
            nb_cams (int): The number of cameras to select.

        Returns:
            list[int]: Idx of the cameras to use.
        """
        return DatasetCameraSelectorUtils.select_from_rank(DatasetCameraSelectorUtils.rank_input_view(input_cams, target_cam, sigmoid_input, sigmoid_target, selection_types), nb_cams)
    
    @staticmethod
    def rank_input_view(
        input_cams : list[Camera], target_cam : Camera,
        sigmoid_input : bool, sigmoid_target : bool,
        selection_types : list[SelectionBasedType] = [SelectionBasedType.Camera_Distance, SelectionBasedType.Camera_Direction, SelectionBasedType.Principal_point_3D],
    ) -> T_Ranks:
        ranks = []
        for selection_type in selection_types:
            if(selection_type == DatasetCameraSelectorUtils.SelectionBasedType.Camera_Distance):
                ranks.append(DatasetCameraSelectorUtils.rank_input_views_from_cam_distance(input_cams, target_cam))
            elif(selection_type == DatasetCameraSelectorUtils.SelectionBasedType.Camera_Direction):
                ranks.append(DatasetCameraSelectorUtils.rank_input_views_from_cam_direction(input_cams, target_cam))
            elif(selection_type == DatasetCameraSelectorUtils.SelectionBasedType.Principal_point_3D):
                ranks.append(DatasetCameraSelectorUtils.rank_input_views_from_principal_point_3D(input_cams, target_cam, sigmoid_input, sigmoid_target))
            else:
                raise ValueError("SelectionType does not exist.")
        return DatasetCameraSelectorUtils.normalize_ranks(ranks, [6, 2, 2])
    
    @staticmethod
    def select_input_views_from_cam_direction(
            input_cams : list[Camera], target_cam : Camera,
            nb_cams : int,
        ) -> list[int]:
        """_summary_

        Args:
            input_cams (list[Camera]): List of input cameras.
            target_cam (Camera): The target camera.
            nb_cams (int): The number of cameras to select.

        Returns:
            list[int]: Idx of the cameras to use.
        """
        return DatasetCameraSelectorUtils.select_from_rank(
            DatasetCameraSelectorUtils.rank_input_views_from_cam_direction(
                input_cams, target_cam
            ), nb_cams
        )
        
    @staticmethod
    def from_score_to_rank(scores : np.ndarray | list) -> T_Ranks:
        cam_idx_sorted = np.argsort(scores)  # Ascending: smallest first
        return {int(cam.item()):r for r,cam in enumerate(cam_idx_sorted)}
    
    @staticmethod
    def normalize_ranks(list_ranks : list[T_Ranks], weights : list[int] | None = None) -> T_Ranks:
        rank = list_ranks.pop()
        
        if weights is None: 
            weights = [1 for _ in range(len(list_ranks))]
        total_weights = int(np.sum(weights).item())
        
        for idx_rank, ranks in enumerate(list_ranks):
            for idx_cam, r in ranks.items():
                rank[idx_cam] += (weights[idx_rank] * r)
        for idx_cam, r in rank.items(): rank[idx_cam] = int(rank[idx_cam] / (total_weights))
        return rank 
    
    @staticmethod
    def select_from_rank(ranks : T_Ranks, number : int) -> list[int]:
        return [idx_cam for idx_cam, rank in sorted(ranks.items(), key=lambda item : item[1])][:number]
    
    @staticmethod
    def rank_input_views_from_cam_direction(
            input_cams : list[Camera], target_cam : Camera,
        ) -> T_Ranks:
        """_summary_

        Args:
            input_cams (list[Camera]): List of input cameras.
            target_cam (Camera): The target camera.
            nb_cams (int): The number of cameras to select.

        Returns:
            list[int]: Idx of the cameras to use.
        """

        # 1 if facing in same direction, -1 if opposite
        alignment_with_target = np.array([0.0 for _ in input_cams])
        for idx in range(len(input_cams)):
            R = SpaceTransformerUtils.get_rotation_matrix_from_to(input_cams[idx].cpu_params.rotation, target_cam.cpu_params.rotation)
            alignment_with_target[idx] = -R[0, 0] 

        return DatasetCameraSelectorUtils.from_score_to_rank(alignment_with_target)
    
    @staticmethod
    def rank_input_views_from_cam_distance(
            input_cams : list[Camera], target_cam : Camera,
        ) -> T_Ranks:
        """_summary_

        Args:
            input_cams (list[Camera]): List of input cameras.
            target_cam (Camera): The target camera.
            nb_cams (int): The number of cameras to select.

        Returns:
            list[int]: Idx of the cameras to use.
        """
        dist = np.array([0.0 for _ in input_cams])

        for idx in range(len(input_cams)):
            cam = input_cams[idx]
            
            dist[idx] = DatasetSpatialUtils.dist_cam_centers(cam, target_cam)
        
        return DatasetCameraSelectorUtils.from_score_to_rank(dist)
        
    @staticmethod
    def rank_input_views_from_principal_point_3D(
            input_cams : list[Camera], target_cam : Camera,
            sigmoid_input : bool, sigmoid_target : bool,
        ) -> T_Ranks:
        """_summary_

        Args:
            input_cams (list[Camera]): List of input cameras.
            target_cam (Camera): The target camera.
            nb_cams (int): The number of cameras to select.

        Returns:
            list[int]: Idx of the cameras to use.
        """
        dist = np.array([0.0 for _ in input_cams])

        for idx in range(len(input_cams)):
            cam = input_cams[idx]
            
            dist[idx] = DatasetSpatialUtils.dist_principal_point_3D(cam, target_cam, sigmoid_p1=sigmoid_input, sigmoid_p2=sigmoid_target,)
        
        return DatasetCameraSelectorUtils.from_score_to_rank(dist)

class DatasetTrainingUtils:
    """Utility for preparing cameras for training."""
    @staticmethod
    def copy_model_camera_for_training(camera: Camera, rand_color_ratio: float, rand_depth_ratio: float) -> Camera:
        """Copy and prepare a camera for a training.
        - Copy the camera parameters, and the texture data.
        - apply uniform noise to the color image.
        - apply uniform noise and gaussian noise to the depth image. Scale of each is half of the rand_depth ratio.
        - reload the texture to the gpu

        Args:
            camera (Camera): The camera to copy for training.
            rand_color_ratio (float): Ratio of the color noise. If 0, no noise. If 0.5, half is noise hafl the original data. If 1.0, completely random.
            rand_depth_ratio (float): Ratio of the depth noise. If 0, no noise. If 0.5, half is noise hafl the original data. If 1.0, completely random.

        Returns:
            Camera: a new camera instance.
        """
        new_cam = camera.__copy__(True, camera.nb_chunks, False)
        new_cam.image_pair = camera.image_pair.__copy__(True, f"trainable_{camera.name}", False)
        new_cam.image_pair.color.rand(rand_color_ratio)
        new_cam.image_pair.depth.rand(rand_depth_ratio/2, new_cam.cpu_params.depth_range)
        new_cam.image_pair.color.arr = ArrayUtils.inv_sigmoid(new_cam.image_pair.color.arr)
        new_cam.image_pair.depth.arr = ArrayUtils.inv_sigmoid(new_cam.image_pair.depth.arr, shift = new_cam.cpu_params.depth_range[0], scale = new_cam.cpu_params.depth_range[1] - new_cam.cpu_params.depth_range[0])
        new_cam.image_pair.color.reload_gpu_texture()
        new_cam.image_pair.depth.reload_gpu_texture()
        return new_cam
    
    @staticmethod
    def find_model_cameras_with_PAM(dataset : Dataset, nb_cams : int, ratio_rand_color : float, ratio_rand_depth : float, plot : bool = False, project_center : bool = False, sigmoid = False) -> list[Camera]:
        """Use the memoids found by the PAM algorithm as the model cameras.

        Args:
            dataset (Dataset): The dataset of camera.
            nb_cams (int): Nb of cameras to find (in PAM=number of clusters).
            rand_color_ratio (float): Ratio of the color noise. If 0, no noise. If 0.5, half is noise hafl the original data. If 1.0, completely random.
            rand_depth_ratio (float): Ratio of the depth noise. If 0, no noise. If 0.5, half is noise hafl the original data. If 1.0, completely random.

        Returns:
            list[Camera]: All model cameras (that correspond to the memoids of the dataset).
        """

        camera_positions = []
        origin = (0, 0, 0)
        for cam in dataset:
            if not project_center:
                camera_positions.append(cam.cpu_params.position)
            else:
                p_in        = cam.cam_center(sigmoid=sigmoid)
                p_proj_in   = SpaceTransformerUtils.pinhole_unproject_point(p_in, cam.cpu_params.focal, cam.cpu_params.principal_point)
                p_proj_in   = SpaceTransformerUtils.transform_point_from_to(p_proj_in, cam.cpu_params.position, cam.cpu_params.rotation, origin, origin)

                camera_positions.append(p_proj_in)

        #camera_positions = [cam.cpu_params.position for cam in dataset]
        camera_positions = np.array(camera_positions)

        clusters, memoids, D = PAMUtils.fit(camera_positions, nb_cams)
        if(plot): 
            PAMUtils.plot(camera_positions, clusters)
        Logger.result(f"WCSS result of the clustering: {PAMUtils.WCSS(clusters, D)}")

        cams = [DatasetTrainingUtils.copy_model_camera_for_training(dataset[idx], ratio_rand_color, ratio_rand_depth) for idx in memoids]

        return cams
        
    
    @staticmethod
    def sample_from_ranks(
        total_elements : int, nb_elements : int, ranks : DatasetCameraSelectorUtils.T_Ranks,
        temperature : float = 1.0,
    ) -> list[int]:
        
        sorted_ranks_ = np.array([rank for _, rank in sorted(ranks.items(), key = lambda item : item[0])])  # Ascending: smallest first
        sorted_ranks_ = (len(sorted_ranks_) - sorted_ranks_) / len(sorted_ranks_) # Higher is Better score, lower is worst score

        prob = ArrayUtils.softmax(sorted_ranks_, temperature)
            
        if np.count_nonzero(prob) < nb_elements or np.isnan(prob).any():
            chosen_cams = np.argsort(sorted_ranks_)[:nb_elements]
        else:
            chosen_cams = np.random.choice(total_elements, size=nb_elements, p = prob, replace=False)
        return chosen_cams.tolist()

    @staticmethod
    def sample_model_views(
            input_cams : list[Camera], target_cam : Camera,
            nb_cams : int,
            sigmoid_input : bool, sigmoid_target : bool,
            temperature : float = 1.0,
            selection_types : list[DatasetCameraSelectorUtils.SelectionBasedType] = [DatasetCameraSelectorUtils.SelectionBasedType.Camera_Distance, DatasetCameraSelectorUtils.SelectionBasedType.Camera_Direction, DatasetCameraSelectorUtils.SelectionBasedType.Principal_point_3D],
        ) -> list[int]:
        """_summary_

        Args:
            input_cams (list[Camera]): List of input cameras.
            target_cam (Camera): The target camera.
            nb_cams (int): The number of cameras to select.
            temperature (float): Controls the sampling distribution. Lower temperature makes closer elements more likely. Must be > 0.

        Returns:
            list[int]: Idx of the cameras to use.
        """
        return DatasetTrainingUtils.sample_from_ranks(len(input_cams), nb_cams, DatasetCameraSelectorUtils.rank_input_view(input_cams, target_cam, sigmoid_input, sigmoid_target, selection_types), temperature)
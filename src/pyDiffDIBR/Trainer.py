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
from enum import Enum
from typing import Literal
import numpy as np
from functools import reduce

from ._utils import ArrayUtils, ImageUtils, IOUtils, Logger
from .Image import RenderedImagePair
from .Camera import Camera
from .Dataset import Dataset, DatasetTrainingUtils, DatasetCameraSelectorUtils
from .Loss import (
    AnisotropicTotalVariation,
    L1Norm,
    L2Norm,
    Loss,
    LossDataParameterDict,
    LossUtils,
    MaskedL1Norm,
    MaskedL2Norm,
    DepthConsistency
)
from .Model import DiffDIBR
from .Monitoring import Monitor
from .Optimizer import Optimizer
from .RenderingApp import RenderingObject, RenderingObjectType, RenderingStageType
from .RenderingDevice import RenderingDevice, SubmitId

_TBatch = list[Camera]

class Trainer(ABC):

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def train(self, batch : _TBatch, epoch : int):
        pass

    @abstractmethod
    def validate(self, batch : _TBatch):
        pass

    @abstractmethod
    def eval(self, out_dir : str | None, img_format : str = "{}_{}.yuv",validation : bool = True, testing : bool = True, training : bool = False):
        pass

    #@abstractmethod
    #def save(self, out_path : str | None, *args : Any, **kwargs : Any):
    #    pass

class RandomInputTrainingSamplingMethod(Enum):
    none        = 0
    Fixed       = 1
    Adaptative  = 2

class RandomInputTrainingSampling:
    
    def __init__(
            self,
            temperature : float, 
            method : RandomInputTrainingSamplingMethod | Literal['none', 'Fixed', 'Adaptative'] = RandomInputTrainingSamplingMethod.Fixed
        ):
        self._method : RandomInputTrainingSamplingMethod = method if not isinstance(method, str) else RandomInputTrainingSamplingMethod[method]
        self._temperature = temperature

    def input_cams(self, all_cams : list[Camera], nb_cams : int, target_cam : Camera, **args) -> list[int]:
        cams : list[Camera]

        if self._method == RandomInputTrainingSamplingMethod.none:
            return self._get_input_cams(all_cams, nb_cams, target_cam)
        elif self._method == RandomInputTrainingSamplingMethod.Fixed:
            temp = self._temperature
            return self._sample_input_cams(all_cams, nb_cams, target_cam, temp)
        elif self._method == RandomInputTrainingSamplingMethod.Adaptative:
            step        = args.get('step', 0)
            max_step    = args.get('max_step', 1)

            temp = self._adaptative_temperature(step, max_step)
            return self._sample_input_cams(all_cams, nb_cams, target_cam, temp)
        
        raise ValueError(f"Does not recognize following view sampling method: {self._method}")

    def _sample_input_cams(self, all_cams : list[Camera], nb_cams : int, target_cam : Camera, temperature : float) -> list[int]:
        return DatasetTrainingUtils.sample_model_views(all_cams, target_cam, nb_cams, True, False, max(1e-3, temperature))
    
    def _get_input_cams(self, all_cams : list[Camera], nb_cams : int, target_cam : Camera) -> list[int]:
        return DatasetCameraSelectorUtils.select_input_views(all_cams, target_cam, nb_cams, True, False)
    
    def _adaptative_temperature(self, step, max_step) -> float:
        return self._temperature * (max_step - step) / max_step


class SimpleTrainer(Trainer):
    _L2_NORM_COLOR_LABEL          = "l2_color"
    _L1_NORM_COLOR_LABEL          = "l1_color"
    _ATV_INPUT_DEPTH_LABEL        = "atv_input_depth"
    _ATV_BLENDED_DEPTH_LABEL      = "atv_blended_depth"
    _L1_NORM_DEPTH_LABEL          = "l1_depth"

    def __init__(
        self,
        nb_steps : int, nb_epochs : int,
        nb_logs : int, fname_csv : str | None,
        l2_color : float, l1_color : float, depth_reg : float, depth_consistency : float,
        training_set: Dataset, validation_set: Dataset, # test_set: Dataset, no need of test set in MVS context
        diff_dibr : DiffDIBR, optimizer : Optimizer,
        dropout : float, rnd_training_sampling_temperature : float, rnd_training_sampling_method : RandomInputTrainingSamplingMethod | str,
        ckpt_dir : str | None = None
    ):
        self.loss : list[Loss] = [
            MaskedL2Norm(
                l2_color / float(training_set.batch_size) / (l2_color + l1_color + depth_reg + depth_consistency),
                [
                    RenderingObject(RenderingObjectType.Color, RenderingStageType.Target),
                    RenderingObject(RenderingObjectType.Color, RenderingStageType.BlendedResult)
                ],
                label=SimpleTrainer._L2_NORM_COLOR_LABEL,
                slang_fname=L2Norm._KERNEL_FNAME,
                fwd_bwd_kernel_entry_point_name=MaskedL2Norm._KERNEL_FWD_BWD_COLOR_EP
            ),
            MaskedL1Norm(
                l1_color / float(training_set.batch_size) / (l2_color + l1_color + depth_reg + depth_consistency),
                [
                    RenderingObject(RenderingObjectType.Color, RenderingStageType.Target),
                    RenderingObject(RenderingObjectType.Color, RenderingStageType.BlendedResult)
                ],
                label=SimpleTrainer._L1_NORM_COLOR_LABEL,
                slang_fname=L2Norm._KERNEL_FNAME,
                fwd_bwd_kernel_entry_point_name=MaskedL1Norm._KERNEL_FWD_BWD_COLOR_EP
            ),
            AnisotropicTotalVariation(
                depth_reg / (l2_color + l1_color + depth_reg + depth_consistency),
                RenderingObject(RenderingObjectType.Depth, RenderingStageType.Input),
                label=SimpleTrainer._ATV_INPUT_DEPTH_LABEL,
                slang_fname=AnisotropicTotalVariation._KERNEL_FNAME,
                fwd_bwd_kernel_entry_point_name=AnisotropicTotalVariation._KERNEL_FWD_BWD_DEPTH_EP
            ),
            DepthConsistency(
                depth_consistency / float(training_set.batch_size * (1.0 + diff_dibr.nb_active_cameras) ) / (l2_color + l1_color + depth_reg + depth_consistency),
                [
                    RenderingObject(RenderingObjectType.Depth, RenderingStageType.IntermediateResult),
                    RenderingObject(RenderingObjectType.Depth, RenderingStageType.BlendedResult),
                    RenderingObject(RenderingObjectType.Quality, RenderingStageType.IntermediateResult),
                    RenderingObject(RenderingObjectType.Quality, RenderingStageType.BlendedResult)
                ],
                label=SimpleTrainer._L1_NORM_DEPTH_LABEL,
                slang_fname=DepthConsistency._KERNEL_FNAME,
                fwd_bwd_kernel_entry_point_name=DepthConsistency._KERNEL_FWD_BWD_EP
            ),
        ]

        # Remove unnecessary loss
        for loss_ in self.loss:
            if loss_.scale <= 0.0:
                self.loss.remove(loss_)

        self.nb_epochs : int = nb_epochs
        self.nb_steps = nb_steps

        self.diff_dibr = diff_dibr

        self.training_set = training_set
        self.validation_set = validation_set

        self.monitor = Monitor(RenderingDevice().device, int((nb_steps * nb_epochs) // (nb_logs + 1e-7)), False, out_fname=fname_csv)

        self.ckpt_dir = ckpt_dir

        self.dropout = dropout
        rnd_training_sampling_method = rnd_training_sampling_method if not isinstance(rnd_training_sampling_method, str) else RandomInputTrainingSamplingMethod[rnd_training_sampling_method]
        self.rnd_sampling = RandomInputTrainingSampling(rnd_training_sampling_temperature, rnd_training_sampling_method)

        self.nb_active_cams = self.diff_dibr.nb_active_cameras

        self.optimizer = optimizer

    def __call__(self):
        for epoch in range(self.nb_epochs):
            Logger.info("-" * 10 + f" Epoch: {epoch} " + "-" * 10 )

            self.training_set.reset_permutation()
            self.validation_set.reset_permutation()
            
            if epoch != 0 : self.diff_dibr.simplify_mesh()

            while ((batch := self.training_set.batch()) is not None):
                Logger.info("-" * 10 + f" Training Batch {self.training_set.current_idx} " + "-" * 10 )
                if self.nb_steps <= 1:
                    self.train(batch, epoch)
                else:
                    self.train_multi_step(batch, epoch, self.nb_steps)

            Logger.info("-" * 10 + " Validation Step" + "-" * 10 )
            while ((batch := self.validation_set.batch()) is not None):
                self.validate(batch)

            if self.ckpt_dir is not None:
                self.diff_dibr._save_ckpt(self.ckpt_dir + f"ckpt-e{epoch}/")

    def train(self, batch : _TBatch, epoch : int, reload : bool = True, release : bool = True):
        if reload:
            self.diff_dibr._reload_batch(batch, True)

        loss = {"training_" : self.loss}

        active_cam_idx_list : list[list[int]] = [
                self.rnd_sampling.input_cams(self.diff_dibr.model_cameras, self.nb_active_cams, cam, step=epoch+1, max_step=self.nb_epochs)
                for cam in batch
            ]
        all_active_cameras_idx = reduce(np.union1d, np.array(active_cam_idx_list))
        all_active_cameras_idx = all_active_cameras_idx.tolist()

        self.diff_dibr.bwd_app.clear_grads()
        self.clear_loss()

        for o in range(len(batch)):
            active_cam_idx = active_cam_idx_list[o]

            self.diff_dibr.fwd(o, active_cam_idx)

            self.backward_loss(0, o, SimpleTrainer._L2_NORM_COLOR_LABEL)
            self.backward_loss(0, o, SimpleTrainer._L1_NORM_COLOR_LABEL)

            self.backward_loss(0, o, SimpleTrainer._ATV_BLENDED_DEPTH_LABEL)

            for i in active_cam_idx:
                self.backward_loss(i, o, SimpleTrainer._L1_NORM_DEPTH_LABEL)

            self.diff_dibr.bwd(o, active_cam_idx)

        for i in all_active_cameras_idx:
            self.backward_loss(i, 0, SimpleTrainer._ATV_INPUT_DEPTH_LABEL)

        self.optimizer.step()

        self.monitor.log(loss = loss)

        if release:
            self.release(batch)
            self.diff_dibr.update_model_cpu_storage()

    def train_multi_step(self, batch : _TBatch, epoch : int, nb_steps : int):
        self.diff_dibr._reload_batch(batch, True)

        for step in range(nb_steps):
            self.train(batch, epoch, reload=False, release=False)

        self.release(batch)

    def release(self, batch : _TBatch):
        self.diff_dibr._release_batch(batch)
        self.release_loss()

    def validate(self, batch : _TBatch):
        self.diff_dibr._reload_batch(batch, True)
        
        loss_obj : list[Loss] = []
        for l_type in [SimpleTrainer._L2_NORM_COLOR_LABEL, SimpleTrainer._L1_NORM_COLOR_LABEL]:
            try:
                loss_obj.append(LossUtils.find_loss_by_label(self.loss, l_type))
            except Exception as _:
                continue
                
        loss = {"val_": loss_obj}

        self.clear_loss()

        for o in range(len(batch)):
            self.diff_dibr.fwd(o)
            self.backward_loss(0, o, SimpleTrainer._L2_NORM_COLOR_LABEL)
            self.backward_loss(0, o, SimpleTrainer._L1_NORM_COLOR_LABEL)

        self.monitor.log(force=True, loss = loss)

        self.release(batch)

    def _eval_batch(self, batch : _TBatch, label : str):
        if len(batch) != len(self.diff_dibr.bwd_app._target_views):
            raise ValueError("Batch is not loaded.")

        mse_ = {}
        psnr_ = {}
        ssim_ = {}
        lpips_ = {}
        for o in range(len(batch)):
            out_cam = batch[o]

            out_color = self.diff_dibr.get_result_color(o).to_numpy()
            out_depth = self.diff_dibr.get_result_depth(o).to_numpy()
            image_pair = RenderedImagePair.from_array(out_color, out_depth, False, "", False)

            target = out_cam.image_pair.color.arr[:,:,0:3]
            out_color_ = out_color[:,:,0:3]

            target_tensor = out_cam.image_pair.color.to_torch()
            out_color_tensor = image_pair.color.to_torch()

            mse     = ArrayUtils.mse(target    , out_color_)
            psnr    = ArrayUtils.psnr(target   , out_color_)
            ssim    = ImageUtils.ssim(target   , out_color_)
            lpips   = ImageUtils.lpips(target_tensor, out_color_tensor)

            mse_[f'{label}_mse_{batch[o].name}'] = mse
            psnr_[f'{label}_psnr_{batch[o].name}'] = psnr
            ssim_[f'{label}_ssim_{batch[o].name}'] = ssim
            lpips_[f'{label}_lpips_{batch[o].name}'] = lpips

        self.monitor.log(force=True, is_result=True, mse = mse_, psnr = psnr_, ssim = ssim_, lpips = lpips_)

    def _save_out_batch(self, batch : _TBatch, out_path : str, label : str):
        if len(batch) != len(self.diff_dibr.bwd_app._target_views):
            raise ValueError("Batch is not loaded.")

        for o in range(len(batch)):
            out_cam = batch[o]

            out_color = self.diff_dibr.get_result_color(o).to_numpy()
            out_color[:,:,3] = 1 #ignore alpha channel
            out_depth = self.diff_dibr.get_result_depth(o).to_numpy()
            image_pair = RenderedImagePair.from_array(out_color, out_depth, False, "", False)

            image_pair.save_color(out_path.format(f"{label}_color", out_cam.name))
            image_pair.save_depth(out_path.format(f"{label}_depth", out_cam.name), depth_range=out_cam.cpu_params.depth_range)

    def _infer_evaluate_save_batch(self, batch : _TBatch, out_dir : str | None, img_format : str, label : str):
        Logger.info("-" * 10 + f" Evaluate and Save {label} set " + "-" * 10)

        self.diff_dibr._reload_batch(batch, True)

        for o in range(len(batch)):
            self.diff_dibr.fwd(o)

        self._eval_batch(batch, label)

        if out_dir is not None:
            IOUtils.create_dirs(out_dir)
            self._save_out_batch(batch, out_dir + img_format, label)

        self.release(batch)

    def eval(
        self, out_dir : str | None, img_format : str = "{}_{}.yuv",
        validation : bool = True, testing : bool = True, training : bool = False
    ):
        if validation:
            self.validation_set.reset_permutation()
            while ((batch := self.validation_set.batch()) is not None):
                self._infer_evaluate_save_batch(batch, out_dir, img_format, "validation")

        if training:
            self.training_set.reset_permutation()
            while ((batch := self.training_set.batch()) is not None):
                self._infer_evaluate_save_batch(batch, out_dir, img_format, "training")

    def backward_loss(self, input_idx : int, output_idx : int, label_loss : str, wait_submit_id : SubmitId = None) -> SubmitId:
        try:
            loss = LossUtils.find_loss_by_label(self.loss, label_loss)
        except Exception as e:
            Logger.debug("Cannot find loss but continue training.", str(Trainer))
            return None

        loss_data = self.diff_dibr._retrieve_loss_data(input_idx, output_idx, loss)

        if(isinstance(loss, L2Norm)):
            if(len(loss_data) != 2): 
                raise ValueError("Impossible to call L2-norm with more/less than two variables.")

            return loss.fwd_bwd(
                loss_data[0],
                loss_data[1],
                wait_submit_id=wait_submit_id
            )
        elif(isinstance(loss, L1Norm)):
            if(len(loss_data) != 2): 
                raise ValueError("Impossible to call L1-norm with more/less than two variables.")

            return loss.fwd_bwd(
                loss_data[0],
                loss_data[1],
                wait_submit_id=wait_submit_id
            )
        elif(isinstance(loss, MaskedL1Norm)):
            if(len(loss_data) != 2): 
                raise ValueError("Impossible to call L1-norm with more/less than two variables.")
            
            mask_l1 : dict = {}
            for l_data in loss_data:
                metadata = l_data["metadata"]
                if metadata is not None:
                    mask_l1 = metadata.get("mask", mask_l1)
                    break

            return loss.fwd_bwd(
                loss_data[0],
                loss_data[1],
                mask_l1,
                wait_submit_id=wait_submit_id
            )
        elif(isinstance(loss, MaskedL2Norm)):
            if(len(loss_data) != 2): 
                raise ValueError("Impossible to call L1-norm with more/less than two variables.")
            
            mask_l2 : dict = {}
            for l_data in loss_data:
                metadata = l_data["metadata"]
                if metadata is not None:
                    mask_l2 = metadata.get("mask", mask_l2)
                    break

            return loss.fwd_bwd(
                loss_data[0],
                loss_data[1],
                mask_l2,
                wait_submit_id=wait_submit_id
            )
        elif(isinstance(loss, AnisotropicTotalVariation)):
            if(len(loss_data) != 1): raise ValueError("Impossible to call ATV with more/less than one variable.")
            
            metadata = loss_data[0]["metadata"]
            scale_sigmoid : float = 1.0
            shift_sigmoid : float = 0.0
            if metadata is not None:
                scale_sigmoid = metadata.get("scale_sigmoid", scale_sigmoid)
                shift_sigmoid = metadata.get("shift_sigmoid", shift_sigmoid)
            
            return loss.fwd_bwd(
                loss_data[0],
                wait_submit_id= wait_submit_id,
                scale_sigmoid = scale_sigmoid,
                shift_sigmoid = shift_sigmoid,
            )
        elif(isinstance(loss, DepthConsistency)):
            if(len(loss_data) != 4): raise ValueError("Impossible to call DepthConsistency with more/less than four variables.")
            return loss.fwd_bwd(
                loss_data[0], #render_depth
                loss_data[2], #blended_depth
                loss_data[1], #render_quality
                loss_data[3], #blended_quality
                wait_submit_id= wait_submit_id,
            )
        else:
            raise NotImplementedError("This loss was not yet implemented.")

    def release_loss(self):
        for loss in self.loss:
            for obj in loss.objects:
                if obj.is_result():
                    loss._release_loss()

    def clear_loss(self) -> SubmitId:
        idx_ = None
        for loss in self.loss:
            idx_ = loss.clear()
        return idx_

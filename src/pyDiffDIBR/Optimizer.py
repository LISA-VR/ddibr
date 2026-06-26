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
from typing import TypedDict, Any, Sequence

import numpy as np
import slangpy as spy

from ._utils import Logger
from .RenderingDevice import RenderingDevice, KernelProgramDict, SubmitId
from .Model import ModelParameterDict
from .LearningRate import LearningRate

class OptimizerState(ABC):
    def __init__(self, lr : LearningRate | Sequence[LearningRate], lr_buffer : spy.Buffer | None = None, step : float = 0.0):
        self.lr : LearningRate | Sequence[LearningRate] = lr
        self._lr_buffer : None | spy.Buffer = lr_buffer
        self.step_counter : float = step
    
    @property
    def lr_buffer(self) -> spy.Buffer:
        if(self._lr_buffer is None): raise RuntimeError("Cannot update GPU buffer because it does not exist.")
        return self._lr_buffer

    def update_gpu(self):
        if self._lr_buffer is None: raise RuntimeError("Cannot update GPU buffer because it does not exist.")
        if(isinstance(self.lr, list) or isinstance(self.lr, Sequence)):
            lrs = np.array([[lr_()] for lr_ in self.lr], dtype="float32")
            self._lr_buffer.copy_from_numpy(lrs)

RegisteredParameterInformation = TypedDict("RegisteredParameterInformation", {
    "kernel_prog"           : KernelProgramDict,
    "model_param"           : ModelParameterDict,
    "optimizer_state"       : OptimizerState,
})

class Optimizer(ABC):
    _KERNEL_OPTIMIZER_FNAME = "unknown.slang"
    _KERNEL_OPTIMIZER_EP    = "tbd"

    def __init__(self):
        self.default_lr : LearningRate
        self.device_handler : RenderingDevice
        self.entry_point_name_format : str
        self.kernel_prog_model_params : dict[str, RegisteredParameterInformation] = {} # Name of model param + dict 

    @abstractmethod
    def register_parameter(self, model_param : ModelParameterDict, program_filename : str = _KERNEL_OPTIMIZER_FNAME, entry_point_name : str = _KERNEL_OPTIMIZER_EP, lr : LearningRate | None = None):
        raise NotImplementedError("You should not call a function of an abstract class.")

    @abstractmethod
    def register_grouped_parameters(self, model_param : ModelParameterDict, program_filename : str = _KERNEL_OPTIMIZER_FNAME, entry_point_name : str = _KERNEL_OPTIMIZER_EP, lr : Sequence[LearningRate] | None = None):
        raise NotImplementedError("You should not call a function of an abstract class.")      

    @staticmethod
    def get_kernel_model_opti_params(kernel_prog_model_params : dict[str, RegisteredParameterInformation], name_param : str) -> RegisteredParameterInformation:
        kernel_model = kernel_prog_model_params.get(name_param)
        if(kernel_model is None): raise KeyError("Optimizer: no parameter with this name was found")

        return kernel_model

    def step(self, wait_submit_id : SubmitId = None) -> SubmitId:
        Logger.debug(f"Optimized Step", str(Optimizer))

        command_encoder = self.device_handler.create_command_encoder()
        for name_param in self.kernel_prog_model_params.keys():
            self._step_param(name_param, command_encoder)
        return self.device_handler.submit_command(command_encoder, wait_submit_id)

    @abstractmethod
    def _step_param(self, name_param : str, command_encoder : spy.CommandEncoder):
        raise NotImplementedError("You should not call a function of an abstract class.")

class GradientDescentState(OptimizerState):
    def __init__(self, lr : LearningRate | Sequence[LearningRate], lr_buffer : spy.Buffer | None = None, step : float = 0.0):
        super().__init__(lr, lr_buffer, step)

GradientDescentUniformsDict = TypedDict("GradientDescentUniformsDict", {
    "learning_rate"        : float,
    "bufferLR" : spy.Buffer
}, total = False)

class GradientDescent(Optimizer):
    _KERNEL_OPTIMIZER_FNAME = "LearningKernels.slang"
    _KERNEL_SGD_EP_TEX4     = "gradient_descent_tex4_main"
    _KERNEL_SGD_EP_TEX1     = "gradient_descent_tex1_main"

    def __init__(self, device_handler : RenderingDevice, default_lr : LearningRate):
        super().__init__()

        self.device_handler = device_handler
        self.default_lr = default_lr

        Logger.debug(f"GradientDescent Optimizer creation.", str(self.__class__))

    def register_parameter(self, model_param : ModelParameterDict, program_filename : str = _KERNEL_OPTIMIZER_FNAME, entry_point_name : str = _KERNEL_SGD_EP_TEX4, lr : LearningRate | None = None):
        kernel_prog = self.device_handler.load_kernel(program_filename, entry_point_name)
        self.kernel_prog_model_params[model_param["name"]] = RegisteredParameterInformation(
            {
                "kernel_prog" : kernel_prog,
                "model_param" : model_param,
                "optimizer_state" : GradientDescentState(lr if lr != None else self.default_lr)
            }
        )

    def register_grouped_parameters(self, model_param : ModelParameterDict, program_filename : str = _KERNEL_OPTIMIZER_FNAME, entry_point_name : str = _KERNEL_SGD_EP_TEX4, lr : Sequence[LearningRate] | None = None):
        kernel_prog = self.device_handler.load_kernel(program_filename, entry_point_name)
        
        size = 1 
        for i in model_param["size"]: size *= i

        zero_arr = np.zeros((size,1), dtype="float32")
        lr_buff = self.device_handler.create_buffer(zero_arr, label = f"lr_buffer_{model_param['name']}")

        self.kernel_prog_model_params[model_param['name']] = RegisteredParameterInformation(
            {
                "kernel_prog" : kernel_prog,
                "model_param" : model_param, 
                "optimizer_state" : GradientDescentState(lr if lr != None else [self.default_lr for _ in range(size)], lr_buff)
            }
        )

    def _step_param(self, name_param : str, command_encoder : spy.CommandEncoder):
        Logger.debug(f"Optimize param: {name_param}.", str(self.__class__))

        kernel_model = self.get_kernel_model_opti_params(self.kernel_prog_model_params, name_param)

        thread_count = [kernel_model["model_param"]["size"][0], kernel_model["model_param"]["size"][1], 1]

        opti_data = GradientDescentUniformsDict()
        if (isinstance(kernel_model["optimizer_state"].lr, LearningRate)):
           opti_data["learning_rate"] = kernel_model["optimizer_state"].lr(step=kernel_model["optimizer_state"].step_counter)
        elif (isinstance(kernel_model["optimizer_state"], GradientDescentState)):
            kernel_model["optimizer_state"].update_gpu()
            opti_data["bufferLR"] = kernel_model["optimizer_state"].lr_buffer

        kernel_model["kernel_prog"]["kernel"].dispatch(
            thread_count=thread_count,
            command_encoder=command_encoder,
            learn_data = kernel_model["model_param"]["uniform"],
            opti_data = opti_data
        ) 

        command_encoder.clear_buffer(kernel_model["model_param"]["grad"])

        kernel_model["optimizer_state"].step_counter += 1.0

class AdamState(OptimizerState):
    def __init__(
        self, 
        lr : LearningRate | Sequence[LearningRate], 
        m1_buffer : spy.Buffer, m2_buffer : spy.Buffer,
        lr_buffer : spy.Buffer | None = None, step : float = 1.0, 
        beta_1 : float = 0.9, beta_2 : float = 0.999,
        ):
        super().__init__(lr, lr_buffer, step)

        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.m1_buffer = m1_buffer
        self.m2_buffer = m2_buffer

ADAMUniformsDict = TypedDict("ADAMUniformsDict", {
    "EPS"       : float,
    "beta_1"    : float,
    "beta_2"    : float,
    "lr"        : float,
    "iteration" : float,
    "bufferM1" : spy.Buffer,
    "bufferM2" : spy.Buffer,
    "bufferLR" : spy.Buffer
}, total = False)

class Adam(Optimizer):
    _KERNEL_OPTIMIZER_FNAME = "LearningKernels.slang"
    _KERNEL_ADAM_EP_TEX4     = "adam_tex4_main"
    _KERNEL_ADAM_EP_TEX1     = "adam_tex1_main"
    _KERNEL_ADAM_EP_SMALL_BUFFER     = "adam_small_buffer_main"

    def __init__(self, device_handler : RenderingDevice, default_lr : LearningRate, beta_1 : float = 0.9, beta_2 : float = 0.999, EPS : float = 1e-8):
        super().__init__()

        self.device_handler = device_handler
        self.default_lr = default_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.EPS = EPS

        Logger.debug(f"ADAM Optimizer creation.", str(self.__class__))

    def register_parameter(self, model_param : ModelParameterDict, program_filename : str = _KERNEL_OPTIMIZER_FNAME, entry_point_name : str = _KERNEL_ADAM_EP_TEX4,  lr : LearningRate | None = None):
        kernel_prog = self.device_handler.load_kernel(program_filename, entry_point_name)
        
        size = 1 
        for i in model_param["size"]: size *= i

        zero_arr = np.zeros((size,1), dtype="float32")
        m1 = self.device_handler.create_buffer(zero_arr, label = f"m1_buffer_{model_param['name']}")
        m2 = self.device_handler.create_buffer(zero_arr, label = f"m2_buffer_{model_param['name']}")

        self.kernel_prog_model_params[model_param['name']] = RegisteredParameterInformation(
            {
                "kernel_prog" : kernel_prog,
                "model_param" : model_param, 
                "optimizer_state" : AdamState(
                    self.default_lr if (lr is None) else lr,
                    m1, m2,
                    lr_buffer=None, beta_1=self.beta_1, beta_2=self.beta_2,
                    step=1.0
                )
            }
        )

    def register_grouped_parameters(self, model_param : ModelParameterDict, program_filename : str = _KERNEL_OPTIMIZER_FNAME, entry_point_name : str = _KERNEL_ADAM_EP_TEX4, lr : Sequence[LearningRate] | None = None):
        kernel_prog = self.device_handler.load_kernel(program_filename, entry_point_name)
        
        size = 1 
        for i in model_param["size"]: size *= i

        zero_arr = np.zeros((size,1), dtype="float32")
        m1 = self.device_handler.create_buffer(zero_arr, label = f"m1_buffer_{model_param['name']}")
        m2 = self.device_handler.create_buffer(zero_arr, label = f"m2_buffer_{model_param['name']}")
        lr_buff = self.device_handler.create_buffer(zero_arr, label = f"lr_buffer_{model_param['name']}")

        self.kernel_prog_model_params[model_param['name']] = RegisteredParameterInformation(
            {
                "kernel_prog" : kernel_prog,
                "model_param" : model_param, 
                "optimizer_state" : AdamState(
                    [self.default_lr for _ in range(size)] if (lr is None) else lr,
                    m1, m2,
                    lr_buffer=lr_buff, beta_1=self.beta_1, beta_2=self.beta_2,
                    step=1.0
                )
            }
        )

    def _step_param(self, name_param : str, command_encoder : spy.CommandEncoder):
        Logger.debug(f"Optimize param: {name_param}.", str(self.__class__))

        kernel_model = self.get_kernel_model_opti_params(self.kernel_prog_model_params, name_param)

        thread_count = [kernel_model["model_param"]["size"][0], kernel_model["model_param"]["size"][1], 1]

        if (not isinstance(kernel_model["optimizer_state"], AdamState)): raise TypeError("Cannot use another optimizer state than the Adam one.")

        opti_data = ADAMUniformsDict({
                "EPS"       : self.EPS,
                "beta_1"    : self.beta_1,
                "beta_2"    : self.beta_2,
                "iteration" : kernel_model["optimizer_state"].step_counter,
                "bufferM1"  : kernel_model["optimizer_state"].m1_buffer,
                "bufferM2"  : kernel_model["optimizer_state"].m2_buffer,
        })
        if (isinstance(kernel_model["optimizer_state"].lr, LearningRate)):
           opti_data["lr"] = kernel_model["optimizer_state"].lr(step=kernel_model["optimizer_state"].step_counter)
        elif (isinstance(kernel_model["optimizer_state"].lr_buffer, spy.Buffer)):
            kernel_model["optimizer_state"].update_gpu()
            opti_data["bufferLR"] = kernel_model["optimizer_state"].lr_buffer
            

        kernel_model["kernel_prog"]["kernel"].dispatch(
            thread_count=thread_count,
            command_encoder=command_encoder,
            learn_data = kernel_model["model_param"]["uniform"],
            opti_data = opti_data
        ) 

        command_encoder.clear_buffer(kernel_model["model_param"]["grad"])

        kernel_model["optimizer_state"].step_counter += 1.0

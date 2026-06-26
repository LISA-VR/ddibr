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

from typing import Callable, Any
from time  import time
import io
import csv
import numpy as np
import slangpy as spy
import matplotlib.pyplot as plt
import scienceplots

from ._utils import IOUtils, Logger
from .Window import Window
from .Loss import Loss

class Monitor:
    TIME_STR = "Time (s)"
    FPS_STR = "FPS"
    STEP_STR = "Step"

    def __init__(self, device : spy.Device, monitoring_step : int, use_renderdoc : bool = False, out_fname : str | None = None, window : Window | None = None):
        self.device = device
        self.monitoring_step = monitoring_step if (monitoring_step > 0) else 1
        self.use_renderdoc = use_renderdoc

        self.prev_step = 0
        self.step = 0
        self.capturing = False

        self.fieldnames : list[str] = []
        self.data : list[dict[str, Any]] = []

        self.window = window

        self.must_save = (not (out_fname is None) and IOUtils.do_file_exist(out_fname))
        self.first_row = True
        
        if (self.must_save and isinstance(out_fname, str)):
            self.out_fname : str = out_fname
            self.csv_file = open(self.out_fname, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self._init_csv()
            
        self.prev_time : float = time()
        self.curr_time : float = time()
    
    def _add_name_to_variables(self, name):
        if not (name in self.fieldnames): self.fieldnames.append(name)

    def update_loss(self, name : str, val : spy.TextureView):
        l = val.texture.to_numpy()
        self._add_name_to_variables(name)
        self.data[-1][name] = np.mean(l[:,:,0:3]).item()
    
    def update_scalar(self, name : str, val : float | int):
        self._add_name_to_variables(name)
        self.data[-1][name] = val
    
    def update_buffer(self, name : str, val : spy.Buffer):
        self._add_name_to_variables(name)
        self.data[-1][name] = str(val.to_numpy().tolist())
    
    def update_none(self, name : str, val : None):
        self._add_name_to_variables(name)
        self.data[-1][name] = None

    def update_list(self, key : str, val : list) -> list[str]:
        names = []
        for value in val:
            if isinstance(value, spy.TextureView): 
                key_ = key + value.texture.desc.label
                names.append(key_)
                self.update_loss(key_, value)
            elif isinstance(value, Loss):
                for loss_tex in value._textures: 
                    names.append(key + loss_tex.desc.label)
                    self.update_scalar(key + loss_tex.desc.label, value.mean(loss_tex.desc.label))
            elif isinstance(value, list):
                names += self.update_list(key, value)
        return names

    def update_dict(self, val : dict) -> list[str]:
        names = []
        for key in val.keys():
            value = val[key]
            names.append(key)
            if isinstance(value, dict): 
                names.pop()
                names += self.update_dict(value)
            elif isinstance(value, list): 
                names.pop()
                names += self.update_list(key, value)
            elif isinstance(value, spy.TextureView): self.update_loss(key, value)
            elif isinstance(value, spy.Buffer): self.update_buffer(key, value)
            elif isinstance(value, float) or isinstance(value, int): self.update_scalar(key, value)
            elif value is None: self.update_none(key, None)
        
        return names

    def reset(self): 
        self.step = -1

    def wait(self):
        self.device.wait()

    def log(self, force : bool = False, displayed_frame : spy.Texture | None = None, is_result : bool = False, **vars):
        self.step += 1 if not force else 0

        if(isinstance(self.window,Window)): self.window()

        self.end_capture()
        
        if ((self.step % self.monitoring_step) != 0 and self.step != self.monitoring_step//2 and not force):
            return
        self.wait()

        if(isinstance(self.window,Window) and isinstance(displayed_frame, spy.Texture)): self.window.update_frame(displayed_frame)

        self.prev_time = self.curr_time
        self.curr_time = time()
        if len(self.data) == 0: self.data.append({})
        elif self.data[-1][Monitor.STEP_STR] < self.step: self.data.append({})
        self.update_scalar(Monitor.STEP_STR, self.step)
        self.update_scalar(Monitor.TIME_STR, self.curr_time - self.prev_time)
        self.update_scalar(Monitor.FPS_STR, (self.step - self.prev_step) / (self.curr_time - self.prev_time))
        self.prev_step = self.step

        names = self.update_dict(vars)

        self.print_logs(is_result, Monitor.TIME_STR, Monitor.FPS_STR, *names)
        self.save_logs()

        self.start_capture()

    def end_capture(self):
        if spy.renderdoc.is_available() and self.capturing and self.use_renderdoc:
            self.capturing = False
            spy.renderdoc.end_frame_capture()

    def start_capture(self):
        if spy.renderdoc.is_available() and not (self.capturing) and self.use_renderdoc:
            spy.renderdoc.start_frame_capture(self.device)
            self.capturing = True

    def print_logs(self, is_result : bool, *names):
        _log = Logger.result if is_result else Logger.state 
        _log("-"*10 + "Step {:06.0f}".format(self.step) + "-"*10)
        for name, val in self.data[-1].items():
            if(name in names): _log("{}: {}".format(name, val))
        print(flush=True)


    def plot_logs(self,*names, ylog : bool = True):
        fig = None
        for name in names:
            if not (name in self.fieldnames):
                continue

            if fig is None:
                plt.style.use(['science', 'bright'])    
                fig = plt.figure(figsize=(7,5))
                plt.xlabel("Iterations",fontsize=25)
                plt.ylabel("Loss", fontsize=25)
                if(ylog) : plt.yscale("log")
    
            x, y = [], []
            for data in self.data:
                if name not in data: continue
                if Monitor.STEP_STR not in data: continue
                x.append(data[Monitor.STEP_STR])
                y.append(data[name])
            x_ = np.array(x)
            y_ = np.array(y)
            y_[y_ is None] = y_[[*(y_ == None)[1:], False]]
            plt.plot(x_, y_, linestyle='solid',label=name, linewidth=3)
        plt.legend(fontsize=22)
        plt.show()
        print()

    def _init_csv(self):
        self.csv_file.truncate(0)  # Clear the file
        self.csv_file.seek(0)      # Reset cursor to the start
        self.csv_writer.writeheader()

    def save_logs(self):
        if(not self.must_save):
            return

        if self.csv_writer.fieldnames == self.fieldnames:
            self._init_csv()
            self.csv_writer.writerows(self.data)
        else:
            self.csv_writer.writerow(self.data[-1])
        self.csv_file.flush()
            
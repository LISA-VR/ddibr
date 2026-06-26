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

import argparse
from time import time_ns
import glob, regex, os

from pyDiffDIBR.RenderingDevice import RenderingDevice
from pyDiffDIBR._utils import Logger, LogLevel
from pyDiffDIBR.Camera import Camera
from pyDiffDIBR.RenderingApp import RenderingApp
from pyDiffDIBR.Window import FreeViewpointNavigatorWindow, Window, TextureType
from pyDiffDIBR.Dataset import DatasetIO, DatasetTrainingUtils
from pyDiffDIBR.Model import DiffDIBR
from pyDiffDIBR._constants import DEFAULT_TOML_DIR

def main():
    arg_parser = argparse.ArgumentParser(prog="FreeViewpoint - DiffDIBR")
    arg_parser.add_argument("--path",help="Path to dataset.",required=True)
    arg_parser.add_argument("--cameras",help="Camera names",required=True, nargs='+')
    arg_parser.add_argument("--format",help="Camera name format",required=False, type=str, default="{}_{}.png")
    arg_parser.add_argument("--nb_cams_synthesis",help="Nb of cameras to use for the view syntheis",type=int, default=5)
    arg_parser.add_argument("--toml",help="TOML config file",type=str, default=f"{DEFAULT_TOML_DIR}/default_front_facing.toml")
    arg_parser.add_argument("--json_fname",help="Json fname",type=str, default="camera.json")
    arg_parser.add_argument("--debug",help="Set debug level",action='store_true', default=False)
    
    arg_parser.add_argument("--width",help="Window width",type=int, default=1920)
    arg_parser.add_argument("--height",help="Window height",type=int, default=1080)

    args = arg_parser.parse_args()

    _logger = Logger([LogLevel.ALL if args.debug else LogLevel.NORMAL])
    rd = RenderingDevice(debug = args.debug, defines={"NB_SAMPLES" : str(RenderingApp.DEFAULT_NB_SAMPLES_MS)})#, "__TRAINING__" : "1"})

    path_dataset = args.path

    win = FreeViewpointNavigatorWindow(rd, args.width, args.height, "Free-Viewpoint Navigator")

    # ------------------------- DATA ------------------------- 
    data_dict = {}
    if args.cameras is not None:
        for cam_name in args.cameras: 
            data_dict[cam_name] = [path_dataset + args.format.format("color",cam_name, "yuv444p"),path_dataset + args.format.format("depth",cam_name, "gray16le")]
    else:
        color_fnames = glob.glob(os.path.join(path_dataset, args.format.format("color", "*", "yuv444p")))
        color_fnames = sorted(color_fnames)
        regex_str = regex.Regex(args.format.format("color", "(.*)", "yuv444p"))
        for color_fname in color_fnames:
            cam_name = regex.findall(regex_str, color_fname)[0]
            _logger.info(f"Load cam: {cam_name}")
            data_dict[cam_name] = [path_dataset + args.format.format("color",cam_name, "yuv444p"),path_dataset + args.format.format("depth",cam_name, "gray16le")]
    
    dataset = DatasetIO.load_from_json(path_dataset + args.json_fname, data_dict, 0, False)
    trainable_cams = [DatasetTrainingUtils.copy_model_camera_for_training(cam, 0.0, 0.0) for cam in dataset.cameras]
    
    if len(trainable_cams) == 0: raise ValueError('No input cameras')
    
    dibr_app = DiffDIBR(trainable_cams, args.nb_cams_synthesis, False, args.toml)
    
    def change_viewpoint(win : FreeViewpointNavigatorWindow, camera : Camera):
        win._CAMERA.cpu_params = camera.cpu_params.__copy__()
        scale = max(win.window.width / win._CAMERA.cpu_params.resolution[0], win.window.height / win._CAMERA.cpu_params.resolution[1])
        if scale > 1.0:
            win._CAMERA.cpu_params.upscale(scale)
        else:
            win._CAMERA.cpu_params.downscale(scale)
        win._CAMERA.cpu_params.resolution = (win.window.width, win.window.height)
        win._CAMERA.update_gpu()

    change_viewpoint(win, trainable_cams[int(len(trainable_cams)//2)])
    dibr_app._reload_batch([win._CAMERA])

    frame = 0
    start_time = time_ns()
    while not Window._QUIT:
        if (frame%60 == 0):
            _logger.result(f"FPS: {frame/((time_ns() - start_time) / 1e9):.3f}")
        
        dibr_app.fwd(0) # todo: only update active cams every 60frames
        
        match win._TEXTURE_TYPE:
            case TextureType.RenderColor:
                win.update_frame(dibr_app.get_result_color(0))
            case TextureType.RenderDepth:
                win.update_frame(dibr_app.get_result_depth(0))
            case TextureType.RenderQuality:
                win.update_frame(dibr_app.get_result_quality(0))

        win()
        frame += 1

if __name__ == "__main__":
    main()    
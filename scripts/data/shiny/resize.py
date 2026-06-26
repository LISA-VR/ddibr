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

import cv2
from glob import glob
import os

scenes_res = {
    "cd": 2,
    "crest": 4,
    "food": 4,
    "giants": 4,
    "pasta": 4,
    "seasoning": 4,
    "lab": 2,
    "tools": 4,
}

for scene, scale in scenes_res.items():
    path = f"~/.cache/nerfbaselines/datasets/shiny/{scene}/images/"
    new_path = f"~/.cache/nerfbaselines/datasets/shiny/{scene}/images_{scale}/"
    if not os.path.exists(new_path): os.makedirs(new_path)

    filenames = glob(path+"*.[jpJP][npNP][egEG]")
    for fname in filenames:
        print(fname)
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert img is not None
        size = (int(img.shape[1] / scale), int(img.shape[0] / scale))
        img_ = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        new_fname = os.path.join(new_path, os.path.basename(fname))
        cv2.imwrite(new_fname, img_)

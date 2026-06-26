/*
This file is based on/modified from "RVS software"
which is Copyright (c) 2010-2018 ITU/ISO/IEC. All rights reserved.

Original code is licensed under the BSD 3-Clause "New" or "Revised" License.
Modifications and Slang port are licensed under the GNU Affero General Public License v3.0.

For the BSD 3-Clause terms of the original work, see https://gitlab.com/mpeg-i-visual/rvs/-/blob/d757a353c7f84d228ac30c179d6ff9de356c6101/LICENSE.
For AGPL-3.0-or-later terms see below.

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
*/

import _Utils;
import Projection;
import DiffTex;
import Uniforms;
import ShaderStageOut;
import SpaceTransformer;

[BackwardDifferentiable]
float3 loadXYZ(DifferentiableTexture<1> depthTexture, no_diff SamplerState s, float2 xy, no_diff DiffCameraParameters camParams) {
    return float3(
        xy,
        sigmoid_shift_scale(depthTexture.sample(s, xy / camParams.imgSize).x, 1.0, camParams.min_max_depth[0], camParams.min_max_depth[1] - camParams.min_max_depth[0]).x
    );
}

[BackwardDifferentiable]
float3 unproject_xy_to_world(float3 xyz, DiffCameraParameters camParams, no_diff int2 idx_block = int2(0, 0)) {
    float3 eucl;
    float2 focal = camParams.gFocal(idx_block);
    float2 princ_p = camParams.gPrincPoint(idx_block);

    if (true) { //* until there is another projection system
        eucl = PerspectiveProjection.unproject(xyz, focal, princ_p);
    }
    
    return eucl;
};

[BackwardDifferentiable]
float3 rotate_translate_world(float3 xyz, no_diff float3x3 R, no_diff float3 t) {
    float3 eucl_ = mul(R, xyz) + t;

    return eucl_;
};

[BackwardDifferentiable]
float3 rotate_translate_world(float3 xyz, SpaceTransform transform, no_diff int2 idx_block = int2(0, 0)) {
    float3x3 R = transform.g_R(idx_block);
    float3   t = transform.g_t(idx_block);

    float3 eucl_ = mul(R, xyz) + t;

    return eucl_;
};

[BackwardDifferentiable]
float4 project_xyz_to_camera(float3 xyz, no_diff CameraParameters camParams) {
    float4 pos;
    float2 focal = no_diff camParams.gFocal();
    float2 princ_p = no_diff camParams.gPrincPoint();

    if (true) { //* until there is another projection system
        pos = PerspectiveProjection.project(xyz, focal, princ_p);
    }

    return pos;
};
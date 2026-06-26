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

struct PerspectiveProjection {

    [Differentiable] static
    float3 unproject(float3 xyz, float2 focal, float2 princ_p) {
        return float3(
            xyz.z,
            -(xyz.z / focal.x) * (xyz.x - princ_p.x),
            -(xyz.z / focal.y) * (xyz.y - princ_p.y)
        );
    };

    [Differentiable] static
    float4 project(float3 zxy, float2 focal, float2 princ_p) {

        float2 uv = float2(
            -(focal.x * zxy.y / zxy.x) + princ_p.x,
            -(focal.y * zxy.z / zxy.x) + princ_p.y
        );

        float depth = zxy.x;
        return float4(uv, depth, 1.0);
    };
}
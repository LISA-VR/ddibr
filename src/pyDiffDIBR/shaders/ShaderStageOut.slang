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

struct VertexShader_OUT {
    float3 uv           : TEXCOORD;
    float4 position     : SV_Position;
    float3 input_pos : INPUT_POSITION;
    float3 output_pos : OUTPUT_POSITION;
    float2 size_img     : SIZE_IMG;
    float2 depth_range : DEPTH_RANGE;
    //float3 cam_dir : CAMERA_DIRECTION;
};

struct GeometryShader_OUT {
    float3 uv       : TEXCOORD;
    float4 position : SV_Position;
    float quality   : QUALITY;
    float normal    : NORMAL;
    uint primitiveID : SV_PrimitiveID;
};

struct FragmentShader_OUT {
    float4 frag_color   : SV_Target<0>;
    float frag_depthmap : SV_Target<1>;
    float frag_quality  : SV_Target<2>;
    float frag_mask     : SV_Target<3>;
    float frag_normal   : SV_Target<4>;
};

struct DiffFragmentShader_OUT {
};
/*
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

[shader("compute")]
[NumThreads(32, 32, 1)]
void main_copy_tex1_to_tex2(
    uint3 threadIdx: SV_DispatchThreadID,
    uniform Texture2D<float1> tex_1,
    uniform RWTexture2D<float4> tex_4,
    uniform int2 size
) {
    if ((threadIdx.x < 0 || threadIdx.x >= size.x) || (threadIdx.y < 0 || threadIdx.y >= size.y)) {
        return;
    }

    tex_4.Store(
        threadIdx.xy,
        float4(tex_1.Load(int3(threadIdx.xy, 0)).x)
    );
};